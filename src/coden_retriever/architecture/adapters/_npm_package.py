"""Shared scaffolding for `package.json`-driven adapters (JavaScript + TypeScript).

Owns everything the two adapters do identically:

- **Workspace + package.json discovery** — declared `workspaces` globs OR a
  recursive scan that catches pnpm/Lerna monorepos.
- **Effective-root auto-descend** through a `src/` wrapper layer.
- **One-level root-barrel trace** so `require('..')` / `import '..'` from
  `test/` and `examples/` attribute to the package the root entry re-exports.
- **Specifier resolution** — bare specifiers via the workspace `name` map,
  relative specifiers walked up the parent chain to the deepest registered
  package dir. Critical correctness property: a path SEGMENT named like a
  workspace at a different location never falsely matches.
- **Top-level import walking** — ES `import_statement` (covers `import type ...`
  and `import { type X }` transparently because tree-sitter exposes them as
  ordinary `import_statement` nodes with a `source` field), `require()` in
  variable declarators, and bare-statement `require()`.
- **Public-symbol extraction** from `export_statement` — function / class /
  const / `export { renamed }`. Language-specific export forms (TS interface,
  type alias, enum) plug in via the `_extra_export_symbols` hook.

Subclasses set the four `BaseTreeSitterAdapter` constants and may override:

- `_grammar_for_file` — when one language uses multiple tree-sitter grammars
  (TypeScript: `typescript` for `.ts/.cts/.mts`, `tsx` for `.tsx`).
- `_extra_export_symbols` — to fold language-specific export inner-types into
  the public facade (TS adds `interface_declaration`, `type_alias_declaration`,
  `enum_declaration`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.protocol import (
    ImportRef,
    InFunctionImport,
)
from ._base import (
    BaseTreeSitterAdapter,
    _find_first_existing,
    _node_text,
    _safe_parse,
    _safe_read_bytes,
)


_FUNCTION_LIKE_TYPES: frozenset[str] = frozenset({
    "function_declaration",
    "function_expression",
    "function",
    "arrow_function",
    "generator_function",
    "generator_function_declaration",
})
# Why: tree-sitter-{javascript,typescript} name anonymous function expressions
# `function_expression` and declarations `function_declaration`; generators get
# their own node types; arrow functions are syntactically distinct. We count
# parameters across all of them for `export default <fn>` support.


_WORKSPACE_SCAN_SKIP: frozenset[str] = frozenset({
    # Vendored / generated — never source-of-truth workspaces.
    "node_modules", ".git", "dist", "build", "out", "coverage", "__pycache__",
    # Test scaffolding. Real-repo finding (Ghost): nested package.json files
    # under test/fixtures/themes/ get discovered as workspaces and pollute
    # the package graph with non-architectural noise (theme test-fixtures).
    # Pruning test/tests/fixtures/examples here filters the entire subtree.
    "test", "tests", "__tests__", "fixtures", "__fixtures__",
    "examples", "samples", "__mocks__", "spec", "specs",
})
# Why: directories that either can't host source-of-truth workspaces or
# typically contain non-architectural scaffolding. Hidden dirs (.github,
# .vscode, …) are filtered separately by the leading-dot check.


@dataclass(frozen=True)
class _Replacement:
    """One absolute-path replacement template from a `paths` entry.

    `template` is already joined to its resolution root (baseUrl or the
    tsconfig dir); the `*` placeholder (if present) is kept intact so the
    matcher can substitute the captured middle in one pass.
    """
    template: str
    has_wildcard: bool


@dataclass(frozen=True)
class _PathPattern:
    """One `compilerOptions.paths` key, pre-split for fast matching.

    Wildcardless patterns store the whole key in `prefix` and treat any
    specifier comparison as exact. Wildcard patterns split on the `*`:
    a specifier matches when it starts with `prefix` AND ends with `suffix`.
    """
    prefix: str
    suffix: str
    has_wildcard: bool
    replacements: tuple[_Replacement, ...]


@dataclass(frozen=True)
class _CompiledPaths:
    """Cached tsconfig data needed at every resolution call.

    Bundling `extensions` + `index_basenames` lets the free-function resolver
    stay adapter-agnostic — the adapter bakes its own constants in once,
    then resolution code never has to consult the adapter instance.
    """
    patterns: tuple[_PathPattern, ...]
    base_url: Path | None
    extensions: frozenset[str]
    index_basenames: tuple[str, ...]


class NpmPackageAdapter(BaseTreeSitterAdapter):
    """Shared base for adapters whose package model is `package.json`-driven."""

    def __init__(self) -> None:
        super().__init__()
        self._cache_root_package_target: str | None = None
        self._cache_package_by_name: dict[str, str] = {}
        self._cache_tsconfig: _CompiledPaths | None = None

    def _post_layout(self, effective_root: Path, audit_root: Path) -> None:
        # tsconfig must populate BEFORE the barrel trace: a root entry like
        # `module.exports = require("@/lib")` needs alias resolution too.
        # `audit_root` may differ from `effective_root` when src/ auto-descend
        # kicks in; tsconfig may live at either level.
        self._cache_tsconfig = _load_tsconfig(
            effective_root, audit_root, self.EXTENSIONS, self.INDEX_BASENAMES,
        )
        self._cache_root_package_target = self._compute_root_package_target(
            effective_root,
        )
        self._cache_package_by_name = self._compute_package_by_name(
            effective_root,
            self._cache_package_by_path,
            self._cache_root_package_target,
        )

    def _compute_package_by_name(
        self,
        effective_root: Path,
        package_by_path: dict[Path, str],
        root_target: str | None,
    ) -> dict[str, str]:
        """npm name → audit-package name, used to resolve bare specifiers.

        Workspaces: each workspace's `package.json::name` maps to itself.
        Single-package: the root's `package.json::name` (if any) maps to the
        cached barrel target so `import '<own-name>'` resolves the same as
        `import '..'`.
        """
        name_to_pkg = {name: name for name in package_by_path.values()}
        root_pkg_name = _read_package_name(effective_root)
        if root_pkg_name and root_target and root_pkg_name not in name_to_pkg:
            name_to_pkg[root_pkg_name] = root_target
        return name_to_pkg

    def _compute_root_package_target(
        self, effective_root: Path,
    ) -> str | None:
        """One-level barrel trace: root entry → registered package, or None.

        Why: every standard npm library has `test/foo.js` / `examples/bar.ts`
        doing `require('..')` or `import '..'`, which Node resolves to
        `<root>/<package.json::main>` (defaulting to the first language-specific
        index file when `main` is absent). Following the barrel re-export turns
        otherwise-invisible test→lib edges into proper graph edges.
        """
        main_path = self._root_entry_path(effective_root)
        if main_path is None:
            return None
        parser = self._get_parser(self._grammar_for_file(main_path))
        if parser is None:
            return None
        source_bytes = _safe_read_bytes(main_path)
        if source_bytes is None:
            return None
        tree = _safe_parse(parser, source_bytes, main_path)
        if tree is None:
            return None
        specifier = _find_barrel_specifier(tree, source_bytes)
        if specifier is None:
            return None
        return _resolve_specifier(
            specifier, main_path, effective_root, self._cache_package_by_path,
            tsconfig=self._cache_tsconfig,
        )

    def _root_entry_path(self, effective_root: Path) -> Path | None:
        """Resolve the file `import '..'` would land on, else None.

        Honors `package.json::main` when set; falls back to the first existing
        `INDEX_BASENAMES` entry. The fallback lets the TS adapter find
        `index.ts` without re-implementing this method.
        """
        declared = _read_package_main(effective_root)
        if declared is not None:
            candidate = effective_root / declared
            return candidate if candidate.is_file() else None
        return _find_first_existing(effective_root, self.INDEX_BASENAMES)

    def _compute_effective_root(self, root: Path) -> Path:
        """Auto-descend into `src/` when `root` has no source at top level."""
        if _dir_has_source_directly(root, self.EXTENSIONS):
            return root
        src = root / "src"
        if src.is_dir() and _dir_has_source_recursive(src, self.EXTENSIONS):
            return src
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for each npm package within `effective_root`.

        Two modes:
          - **Workspaces** (the root's `package.json::workspaces` is declared).
            Each glob expansion that contains a `package.json` with a `name`
            becomes a first-class package. Names come from `package.json::name`
            (so `apps/portal/` can be `@scope/portal`).
          - **Single-package fallback**. Each direct subdirectory of
            `effective_root` that contains source recursively. Names = dir basenames.
        """
        workspace_pairs = _discover_workspaces(effective_root)
        if workspace_pairs:
            return workspace_pairs
        try:
            sub = list(effective_root.iterdir())
        except OSError:
            return ()
        pairs = [
            (c, c.name) for c in sub
            if c.is_dir() and _dir_has_source_recursive(c, self.EXTENSIONS)
        ]
        return tuple(sorted(pairs, key=lambda p: p[1]))

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Walk top-level statements, collect ES imports + require calls.

        `in_function_imports` is always `()` — Node/TS imports must be top-level,
        and the `require()`-inside-function cycle-workaround pattern is rare in
        idiomatic code.
        """
        stmt_count, imports = self._walk_top_level_imports(
            tree, source_bytes, file, effective_root,
            per_stmt=self._collect_top_level_imports,
        )
        return stmt_count, imports, ()

    def _collect_top_level_imports(
        self,
        node: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> list[ImportRef]:
        """Imports produced by ONE top-level statement (may be 0 or more refs)."""
        nt = node.type
        line = node.start_point[0] + 1
        if nt == "import_statement":
            spec = _import_source_specifier(node, source_bytes)
            return [] if spec is None else [
                self._make_import_ref(spec, line, file, effective_root)
            ]
        if nt in ("lexical_declaration", "variable_declaration"):
            return self._collect_requires_from_declarators(
                node, source_bytes, file, effective_root, line,
            )
        if nt == "expression_statement":
            for inner in node.named_children:
                spec = _require_call_specifier(inner, source_bytes)
                if spec is not None:
                    return [self._make_import_ref(spec, line, file, effective_root)]
            return []
        return []

    def _collect_requires_from_declarators(
        self,
        decl_node: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
        line: int,
    ) -> list[ImportRef]:
        """For `const x = require("a"), y = require("b")` → two refs, one statement."""
        out: list[ImportRef] = []
        for declarator in decl_node.named_children:
            if declarator.type != "variable_declarator":
                continue
            value = declarator.child_by_field_name("value")
            if value is None:
                continue
            spec = _require_call_specifier(value, source_bytes)
            if spec is None:
                continue
            out.append(self._make_import_ref(spec, line, file, effective_root))
        return out

    def _make_import_ref(
        self, specifier: str, line: int, file: Path, effective_root: Path,
    ) -> ImportRef:
        target_pkg = _resolve_specifier(
            specifier, file, effective_root, self._cache_package_by_path,
            root_target=self._cache_root_package_target,
            package_by_name=self._cache_package_by_name,
            tsconfig=self._cache_tsconfig,
        )
        return ImportRef(target_module=specifier, target_package=target_pkg, line=line)

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Walk top-level export statements, collect named symbols + param counts."""
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            if child.type != "export_statement":
                continue
            if _is_reexport(child):
                continue
            self._add_exports_from_statement(child, source_bytes, symbols)
        return symbols

    def _add_exports_from_statement(
        self,
        export_node: Any,
        source_bytes: bytes,
        symbols: dict[str, int],
    ) -> None:
        """Update `symbols` in place with public names + param counts from one export."""
        if _has_default_keyword(export_node):
            value = _default_export_value(export_node)
            params = _count_function_params(value) if value is not None else 0
            symbols.setdefault("default", params)
            return
        for inner in export_node.children:
            self._add_exports_from_inner(inner, source_bytes, symbols)

    def _add_exports_from_inner(
        self,
        inner: Any,
        source_bytes: bytes,
        symbols: dict[str, int],
    ) -> None:
        """Dispatch on inner node type to populate `symbols` for one export form."""
        it = inner.type
        if it == "function_declaration":
            name = _identifier_text(inner.child_by_field_name("name"), source_bytes)
            if name:
                symbols.setdefault(name, _count_function_params(inner))
        elif it == "class_declaration":
            name = _identifier_text(inner.child_by_field_name("name"), source_bytes)
            if name:
                symbols.setdefault(name, 0)
        elif it in ("lexical_declaration", "variable_declaration"):
            for declarator in inner.named_children:
                if declarator.type != "variable_declarator":
                    continue
                name = _identifier_text(
                    declarator.child_by_field_name("name"), source_bytes,
                )
                if name:
                    symbols.setdefault(name, 0)
        elif it == "export_clause":
            for spec in inner.named_children:
                if spec.type != "export_specifier":
                    continue
                alias_node = spec.child_by_field_name("alias")
                name_node = spec.child_by_field_name("name")
                final = _identifier_text(alias_node or name_node, source_bytes)
                if final:
                    symbols.setdefault(final, 0)
        else:
            self._extra_export_symbols(inner, source_bytes, symbols)

    def _extra_export_symbols(
        self,
        inner: Any,
        source_bytes: bytes,
        symbols: dict[str, int],
    ) -> None:
        """Hook for language-specific export inner-types. Default: no-op.

        Called for every `export_statement` child node that the shared
        dispatcher didn't already handle. TypeScript uses this to fold in
        `interface_declaration`, `type_alias_declaration`, and `enum_declaration`.
        """
        del inner, source_bytes, symbols


def _dir_has_source_directly(directory: Path, extensions: frozenset[str]) -> bool:
    """True if `directory` directly contains a file with one of `extensions`."""
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() in extensions:
                return True
    except OSError:
        return False
    return False


def _dir_has_source_recursive(directory: Path, extensions: frozenset[str]) -> bool:
    """True if any file under `directory` (recursively) has one of `extensions`."""
    try:
        for entry in directory.rglob("*"):
            if entry.is_file() and entry.suffix.lower() in extensions:
                return True
    except OSError:
        return False
    return False


def _import_source_specifier(node: Any, source_bytes: bytes) -> str | None:
    """Extract the source string from an `import_statement` node.

    Handles `import X from "..."`, `import "..."`, `import type { X } from "..."`,
    and `import { type X } from "..."` uniformly — tree-sitter exposes all of
    them as `import_statement` with a `source` field. The TS-only
    `import x = require("...")` form has NO `source` field and yields None here.
    """
    source = node.child_by_field_name("source")
    if source is None or source.type != "string":
        return None
    return _read_string_content(source, source_bytes)


def _require_call_specifier(node: Any, source_bytes: bytes) -> str | None:
    """If `node` is a `require(...)` call with a string first arg, return the specifier."""
    if node.type != "call_expression":
        return None
    fn = node.child_by_field_name("function")
    if fn is None or fn.type != "identifier":
        return None
    if _node_text(fn, source_bytes) != "require":
        return None
    args = node.child_by_field_name("arguments")
    if args is None:
        return None
    for arg in args.named_children:
        if arg.type == "string":
            return _read_string_content(arg, source_bytes)
        return None
    return None


def _read_string_content(string_node: Any, source_bytes: bytes) -> str | None:
    """Extract the content of a tree-sitter string node (no quotes)."""
    for child in string_node.named_children:
        if child.type == "string_fragment":
            return _node_text(child, source_bytes)
    return None


def _resolve_specifier(
    specifier: str,
    file: Path,
    effective_root: Path,
    package_by_path: dict[Path, str],
    root_target: str | None = None,
    package_by_name: dict[str, str] | None = None,
    tsconfig: _CompiledPaths | None = None,
) -> str | None:
    """Resolve a specifier to a project package name, else None.

    Three branches, tried in order:

    - **Relative specifier** (`./x`, `../y`): joined to the file's directory,
      then walked up the parent chain looking for the DEEPEST registered
      package dir in `package_by_path`. When the resolved path lands AT the
      effective root (e.g. `import '..'`), fall back to `root_target`.
      The ancestor-walk is critical for monorepos where a path part name can
      collide with a workspace npm name at a totally different location
      (e.g. `apps/ghost/` the dir vs. `ghost/core/` the workspace).
    - **Aliased specifier** (matched against `compilerOptions.paths` or
      rooted at `baseUrl`): substituted to an absolute file path via
      `_resolve_alias`, then funneled into the same ancestor-walk as relative
      imports. Skipped silently when `tsconfig` is None — no tsconfig.json at
      the effective root means no alias rewriting happens at all.
    - **Bare specifier** (`lodash`, `@scope/foo`, `react/jsx-runtime`): looked
      up in `package_by_name`. Matches `import '<own-name>'` from a
      single-package repo's tests AND `import '@scope/workspace'` from
      monorepo sibling packages. Subpath specifiers map to their root package
      (e.g. `@scope/foo/sub` → `@scope/foo`).
    """
    if specifier.startswith("."):
        joined_str = os.path.normpath(os.path.join(str(file.parent), specifier))
        return _walk_up_to_package(
            Path(joined_str), effective_root, package_by_path, root_target,
        )
    if tsconfig is not None:
        resolved = _resolve_alias(specifier, tsconfig)
        if resolved is not None:
            return _walk_up_to_package(
                resolved.parent, effective_root, package_by_path, root_target,
            )
    if not package_by_name:
        return None
    root = _bare_specifier_root(specifier)
    if root is None:
        return None
    return package_by_name.get(root)


def _walk_up_to_package(
    start: Path,
    effective_root: Path,
    package_by_path: dict[Path, str],
    root_target: str | None,
) -> str | None:
    """Walk ancestors of `start` for the deepest registered package, else None.

    When `start == effective_root` (the `import '..'` case), returns
    `root_target` — the barrel target the root entry re-exports. Returns
    None when `start` lies outside `effective_root` or no registered package
    is found before the walk terminates.
    """
    try:
        start.relative_to(effective_root)
    except ValueError:
        return None
    if start == effective_root:
        return root_target
    cur = start
    while True:
        if cur in package_by_path:
            return package_by_path[cur]
        if cur == effective_root or cur == cur.parent:
            return None
        cur = cur.parent


def _bare_specifier_root(specifier: str) -> str | None:
    """First package-name segment of a bare specifier.

    `@scope/foo/sub` → `@scope/foo`. `react/jsx-runtime` → `react`.
    Empty / malformed → None.
    """
    if not specifier:
        return None
    if specifier.startswith("@"):
        parts = specifier.split("/", 2)
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
        return None
    head = specifier.split("/", 1)[0]
    return head or None


def _read_package_main(effective_root: Path) -> str | None:
    """Return `<effective_root>/package.json::main` if usable, else None.

    Why no default: the JS/TS adapters diverge on the right default index name
    (`index.js` vs. `index.ts`). The caller falls back through
    `_find_first_existing(root, self.INDEX_BASENAMES)` so each adapter naturally
    looks for its own preferred entry file without conditional logic here.
    """
    data = _read_package_json(effective_root / "package.json")
    main = data.get("main") if isinstance(data, dict) else None
    return main if isinstance(main, str) and main else None


def _read_package_name(pkg_dir: Path) -> str | None:
    """Return `pkg_dir/package.json::name` if usable, else None."""
    data = _read_package_json(pkg_dir / "package.json")
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def _read_package_json(path: Path) -> dict[str, Any] | None:
    """Read + parse `path` as JSON, returning the dict on success, None otherwise."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _load_tsconfig(
    effective_root: Path,
    audit_root: Path,
    extensions: frozenset[str],
    index_basenames: tuple[str, ...],
) -> _CompiledPaths | None:
    """Load tsconfig from effective_root, falling back to audit_root.

    Two locations because of `src/` auto-descend: a Next.js project keeps
    `tsconfig.json` at the project root, but the adapter's effective_root
    auto-descends one level into `src/`. Checking effective_root first
    keeps a tsconfig that lives next to the source files (uncommon but
    legal) from being shadowed by a stale ancestor tsconfig.

    Missing file, broken JSON, no `compilerOptions`, or a `compilerOptions`
    block that declares neither `paths` nor `baseUrl` all return None — the
    resolver branches that consume the result skip themselves cleanly when
    they see None, so no other call site needs to care.
    """
    seen: set[Path] = set()
    for candidate_dir in (effective_root, audit_root):
        if candidate_dir in seen:
            continue
        seen.add(candidate_dir)
        data = _read_tsconfig_json(candidate_dir / "tsconfig.json")
        if data is None:
            continue
        compiled = _compile_tsconfig_paths(
            data, candidate_dir, extensions, index_basenames,
        )
        if compiled is not None:
            return compiled
    return None


def _read_tsconfig_json(path: Path) -> dict[str, Any] | None:
    """Read `tsconfig.json` tolerating JSONC `//` and `/*…*/` comments.

    Next.js's `create-next-app` template ships a `tsconfig.json` with line
    comments; strict `json.loads` rejects it. Fast path tries strict first
    (zero overhead for the comment-free case), falls back to the stripped
    text only when the first parse fails.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(_strip_jsonc_comments(raw))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _strip_jsonc_comments(raw: str) -> str:
    """Strip `//` and `/*…*/` comments while preserving string contents.

    Hand-rolled char-walk (no regex) so escape sequences inside strings stay
    intact: `"\\""` mustn't be misread as opening a second string, and a `//`
    inside a URL string mustn't be treated as a line comment.
    """
    out: list[str] = []
    i = 0
    n = len(raw)
    in_string = False
    string_quote = ""
    while i < n:
        ch = raw[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt == "/":
                end = raw.find("\n", i + 2)
                i = n if end == -1 else end
                continue
            if nxt == "*":
                end = raw.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _compile_tsconfig_paths(
    data: dict[str, Any],
    tsconfig_dir: Path,
    extensions: frozenset[str],
    index_basenames: tuple[str, ...],
) -> _CompiledPaths | None:
    """Normalize `compilerOptions.paths` + `baseUrl` into a `_CompiledPaths`.

    Returns None when the config declares neither `paths` nor `baseUrl` —
    the resolver short-circuits on None, saving a per-resolution lookup
    against an empty pattern list.

    Per the TS spec, `paths` keys are resolved relative to `baseUrl` when
    present, else relative to the tsconfig dir. Patterns are sorted
    longest-prefix-first so the most-specific match wins when several
    overlap (e.g. `@core/*` matches before `@*`).
    """
    options = data.get("compilerOptions")
    if not isinstance(options, dict):
        return None
    paths_raw = options.get("paths")
    base_url_raw = options.get("baseUrl")

    base_dir: Path | None = None
    if isinstance(base_url_raw, str) and base_url_raw:
        base_dir = Path(os.path.normpath(os.path.join(str(tsconfig_dir), base_url_raw)))

    resolve_root = base_dir if base_dir is not None else tsconfig_dir

    patterns: list[_PathPattern] = []
    if isinstance(paths_raw, dict):
        for key, value in paths_raw.items():
            if not isinstance(key, str) or not key:
                continue
            replacements = _normalize_replacements(value, resolve_root)
            if not replacements:
                continue
            patterns.append(_make_path_pattern(key, replacements))

    patterns.sort(key=lambda p: len(p.prefix), reverse=True)

    if not patterns and base_dir is None:
        return None
    return _CompiledPaths(
        patterns=tuple(patterns),
        base_url=base_dir,
        extensions=extensions,
        index_basenames=index_basenames,
    )


def _normalize_replacements(
    value: Any, resolve_root: Path,
) -> tuple[_Replacement, ...]:
    """Convert one `paths` value (list of replacement strings) to `_Replacement`s.

    Each replacement is joined to `resolve_root` upfront so the resolver only
    has to do an O(1) string substitution per attempt — no per-resolution
    path manipulation. The `*` placeholder is preserved through normpath,
    which doesn't recognize it as a special character.
    """
    if not isinstance(value, list):
        return ()
    out: list[_Replacement] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            continue
        absolute = os.path.normpath(os.path.join(str(resolve_root), raw))
        out.append(_Replacement(template=absolute, has_wildcard="*" in raw))
    return tuple(out)


def _make_path_pattern(
    key: str, replacements: tuple[_Replacement, ...],
) -> _PathPattern:
    """Split a `paths` key on `*` into prefix/suffix; non-wildcard keys match exactly."""
    if "*" in key:
        prefix, _, suffix = key.partition("*")
        return _PathPattern(
            prefix=prefix, suffix=suffix,
            has_wildcard=True, replacements=replacements,
        )
    return _PathPattern(
        prefix=key, suffix="",
        has_wildcard=False, replacements=replacements,
    )


def _resolve_alias(specifier: str, compiled: _CompiledPaths) -> Path | None:
    """Resolve `specifier` against `paths` first, then `baseUrl`, else None.

    Returns the first existing file (after extension/index walk) across all
    replacements of all matching patterns. Multi-candidate `paths` entries
    like `["./src/*", "./vendor/*"]` work because the file-existence check
    naturally selects whichever target actually contains the source.
    """
    for pattern in compiled.patterns:
        candidates = _try_path_pattern(specifier, pattern)
        if candidates is None:
            continue
        for candidate in candidates:
            resolved = _resolve_candidate_to_file(
                Path(candidate), compiled.extensions, compiled.index_basenames,
            )
            if resolved is not None:
                return resolved
    if compiled.base_url is not None:
        return _resolve_candidate_to_file(
            compiled.base_url / specifier,
            compiled.extensions, compiled.index_basenames,
        )
    return None


def _try_path_pattern(
    specifier: str, pattern: _PathPattern,
) -> tuple[str, ...] | None:
    """Match `specifier` against one pattern; return all candidate paths or None.

    None means the pattern didn't match at all — the caller should try the
    next pattern. An empty tuple is impossible here: a pattern without
    replacements is filtered out at compile time.
    """
    if pattern.has_wildcard:
        if not specifier.startswith(pattern.prefix):
            return None
        if not specifier.endswith(pattern.suffix):
            return None
        end = len(specifier) - len(pattern.suffix)
        middle = specifier[len(pattern.prefix):end]
        return tuple(
            r.template.replace("*", middle, 1) if r.has_wildcard else r.template
            for r in pattern.replacements
        )
    if specifier != pattern.prefix:
        return None
    return tuple(r.template for r in pattern.replacements)


def _resolve_candidate_to_file(
    candidate: Path,
    extensions: frozenset[str],
    index_basenames: tuple[str, ...],
) -> Path | None:
    """Apply extension + index resolution to `candidate`. None if nothing matches.

    Mirrors TypeScript's module-resolution probe order: exact file → file with
    each language extension → directory containing an index file. We don't
    enforce TS's exact extension priority order (`.ts` before `.tsx`) because
    `_find_first_existing` already encodes the adapter-preferred order via
    `INDEX_BASENAMES` and conflicts at the same stem are extremely rare in
    real codebases.
    """
    if candidate.is_file():
        return candidate
    for ext in extensions:
        with_ext = Path(str(candidate) + ext)
        if with_ext.is_file():
            return with_ext
    if candidate.is_dir():
        index = _find_first_existing(candidate, index_basenames)
        if index is not None:
            return index
    return None


def _discover_workspaces(effective_root: Path) -> tuple[tuple[Path, str], ...]:
    """Discover npm-package boundaries under `effective_root`, two strategies.

    1. **Declared workspaces** — read `package.json::workspaces` (supports both
       npm/yarn array form and Yarn `{packages: [...]}` form). Resolve each
       glob; keep dirs with a parseable `package.json::name`.
    2. **Recursive scan** — used when no `workspaces` field is declared. This
       catches pnpm monorepos (config in `pnpm-workspace.yaml`), Lerna repos
       declaring their own way, and any project sprinkling `package.json` files
       throughout. Walks the tree pruning `node_modules`, hidden dirs, and
       common build outputs; every non-root dir with a named `package.json` is
       a workspace.

    Returns `()` if either strategy finds zero sub-packages (caller falls back
    to directory-based discovery — the single-package mode).
    """
    data = _read_package_json(effective_root / "package.json")
    declared_globs = _workspace_globs(data.get("workspaces") if data else None)
    if declared_globs:
        return _expand_declared_workspaces(effective_root, declared_globs)
    return _scan_recursive_workspaces(effective_root)


def _expand_declared_workspaces(
    effective_root: Path, globs: list[str],
) -> tuple[tuple[Path, str], ...]:
    """Expand each `package.json::workspaces` glob, dedup by name."""
    seen: dict[str, tuple[Path, str]] = {}
    for pattern in globs:
        for ws_dir in _safe_glob(effective_root, pattern):
            if not ws_dir.is_dir():
                continue
            name = _read_package_name(ws_dir)
            if name is None:
                continue
            seen.setdefault(name, (ws_dir, name))
    return tuple(sorted(seen.values(), key=lambda p: p[1]))


def _scan_recursive_workspaces(
    effective_root: Path,
) -> tuple[tuple[Path, str], ...]:
    """Walk `effective_root`, returning every non-root dir with a named package.json."""
    seen: dict[str, tuple[Path, str]] = {}
    stack: list[Path] = [effective_root]
    is_root = True
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if not is_root and (cur / "package.json").is_file():
            name = _read_package_name(cur)
            if name is not None:
                seen.setdefault(name, (cur, name))
        is_root = False
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _WORKSPACE_SCAN_SKIP:
                continue
            stack.append(entry)
    return tuple(sorted(seen.values(), key=lambda p: p[1]))


def _workspace_globs(value: Any) -> list[str]:
    """Normalize `workspaces`: list-of-strings OR `{packages: list-of-strings}`."""
    if isinstance(value, list):
        return [s for s in value if isinstance(s, str) and s]
    if isinstance(value, dict):
        pkgs = value.get("packages")
        if isinstance(pkgs, list):
            return [s for s in pkgs if isinstance(s, str) and s]
    return []


def _safe_glob(root: Path, pattern: str) -> list[Path]:
    """Run `root.glob(pattern)` swallowing OSError/ValueError. Empty on failure."""
    try:
        return list(root.glob(pattern))
    except (OSError, ValueError):
        return []


def _find_barrel_specifier(tree: Any, source_bytes: bytes) -> str | None:
    """First top-level barrel target — `module.exports = require("X")` or `export ... from "X"`.

    Returns the literal specifier string (relative or otherwise) so the caller
    can re-resolve it through `_resolve_specifier`. Only the first match is
    returned: multi-target barrels are out of scope for v1.
    """
    for child in tree.root_node.children:
        nt = child.type
        if nt == "expression_statement":
            spec = _module_exports_require_specifier(child, source_bytes)
            if spec is not None:
                return spec
        elif nt == "export_statement" and _is_reexport(child):
            spec = _import_source_specifier(child, source_bytes)
            if spec is not None:
                return spec
    return None


def _module_exports_require_specifier(
    expr_stmt: Any, source_bytes: bytes,
) -> str | None:
    """If `expr_stmt` is `module.exports = require("X")`, return `"X"`, else None."""
    for inner in expr_stmt.named_children:
        if inner.type != "assignment_expression":
            continue
        left = inner.child_by_field_name("left")
        right = inner.child_by_field_name("right")
        if left is None or right is None:
            continue
        if not _is_module_exports(left, source_bytes):
            continue
        return _require_call_specifier(right, source_bytes)
    return None


def _is_module_exports(node: Any, source_bytes: bytes) -> bool:
    """True if `node` is a `member_expression` reading `module.exports`."""
    if node.type != "member_expression":
        return False
    obj = node.child_by_field_name("object")
    prop = node.child_by_field_name("property")
    if obj is None or prop is None:
        return False
    if obj.type != "identifier" or _node_text(obj, source_bytes) != "module":
        return False
    if prop.type != "property_identifier" or _node_text(prop, source_bytes) != "exports":
        return False
    return True


def _is_reexport(export_node: Any) -> bool:
    """True if this `export_statement` re-exports from another module."""
    return any(child.type == "from" for child in export_node.children)


def _has_default_keyword(export_node: Any) -> bool:
    """True if any child is the `default` keyword."""
    return any(child.type == "default" for child in export_node.children)


def _default_export_value(export_node: Any) -> Any | None:
    """Return the named child following the `default` keyword in an export_statement."""
    seen_default = False
    for child in export_node.children:
        if child.type == "default":
            seen_default = True
            continue
        if seen_default and child.is_named:
            return child
    return None


def _identifier_text(node: Any | None, source_bytes: bytes) -> str | None:
    """Return the text of an identifier-like node, or None."""
    if node is None:
        return None
    return _node_text(node, source_bytes)


def _count_function_params(node: Any) -> int:
    """Param count for any function-like node — 0 for non-functions."""
    if node is None or node.type not in _FUNCTION_LIKE_TYPES:
        return 0
    params = node.child_by_field_name("parameters")
    if params is not None and params.type == "formal_parameters":
        return len(params.named_children)
    if node.type == "arrow_function" and node.child_by_field_name("parameter") is not None:
        return 1
    return 0
