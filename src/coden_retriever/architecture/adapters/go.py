"""Go adapter: tree-sitter only.

A "package" here is any directory under the module root that contains at
least one `.go` file (excluding `vendor/`, `testdata/`, hidden dirs, and
`node_modules/`). Single-module packages are named by their POSIX-relative
path from the module root — `internal/auth/middleware`, `cmd/server`,
`doc`. Workspace packages carry the full module path prefix —
`example.com/mod_a/internal/auth`, `github.com/x/repo/cmd/server`. Files
directly at the module root attribute to `package=None` — included in
n_files/total_loc/oversized totals, excluded from the package-level graph
(mirrors the Python/JS/TS adapters).

The effective root depends on layout:

1. `root/go.work` exists → effective = `root` (Go workspace). Each `use`
   directive points at a module whose own `go.mod` is read for the module
   path; cross-module imports use the full module-path prefix.
2. `root/go.mod` exists → effective = `root` (single module).
3. `root/` has no `go.mod` but a single direct subdirectory does →
   auto-descend one level on that child.

Import resolution: one branch — module-rooted absolute imports.
Single-module: `import "<module-path>/internal/x"` → `internal/x`.
Workspace: each `use`-declared module path is prefix-matched
longest-first; on match the relative tail is checked against the
qualified package set.

Public-symbol extraction aggregates capitalized top-level identifiers
across every non-test `.go` file in the package directory. `_test.go`
files participate in the import graph (their imports are real edges)
but are excluded from the public facade.

`in_function_imports` is always `()` — Go's grammar disallows imports
inside function bodies (they must follow the `package` clause and
precede any declarations).

What v1 deliberately does NOT do:

- `go.work` `replace` / `replace ()` blocks — only `use` directives are
  read for source discovery.
- `go.work.sum` companion file — not source-relevant.
- Members outside the workspace root (`use ../sibling`) — silently dropped.
- Build tags / `//go:build` filtering — every file is parsed regardless
  of build constraints.
- `replace` / `exclude` / `require` directives in member `go.mod` files —
  only `module` is read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.protocol import (
    ImportRef,
    InFunctionImport,
)
from ._base import (
    BaseTreeSitterAdapter,
    _node_text,
)


_GO_SCAN_SKIP: frozenset[str] = frozenset({
    "vendor", "testdata", "node_modules",
})
# Why: directories that must never count as Go packages. `vendor/` holds
# copies of external deps the Go toolchain vendors locally; `testdata/`
# is the Go convention for test inputs (the toolchain ignores them);
# `node_modules/` is filtered for projects that ship a JS toolchain
# alongside their Go source. Hidden dirs (`.git`, `.github`) are filtered
# separately via the leading-dot check.

_MODULE_DIRECTIVE: str = "module"
# Why: the keyword that introduces the module-path declaration in
# `go.mod`. Always near the top of the file, before `go`/`require`/etc.

_TEST_SUFFIX: str = "_test.go"
# Why: Go test-file naming convention. Symbols in `*_test.go` are
# helpers for `go test`, not architectural public API — excluded from
# the public-facade walk but included in the import graph.

_MODULE_SCAN_LINES: int = 10
# Why: `go.mod` declares the `module` directive in the first few lines,
# before any other directive. Ten lines comfortably covers it even with
# leading whitespace or comments.

_GO_WORK_FILENAME: str = "go.work"
# Why: the well-known Go workspace manifest. Its presence at the audit
# root marks workspace mode.

_GO_WORK_BLOCK_OPEN_PATTERN: re.Pattern[str] = re.compile(r"^use\s*\(")
# Why: matches the opening of a `use ( ... )` block in `go.work`. Handles
# `use(`, `use (`, and `use\t(`. Single-line block `use ( ./mod )` is
# tokenized inline after the opener matches.


@dataclass(frozen=True)
class _GoWorkspaceMember:
    """One Go workspace member resolved on disk."""
    module_root: Path     # dir containing the module's go.mod
    module_path: str      # value of `module ...` directive


class GoAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for Go modules."""

    LANGUAGE = "go"
    EXTENSIONS = frozenset({".go"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//",)

    def __init__(self) -> None:
        super().__init__()
        self._cache_workspace_members: tuple[_GoWorkspaceMember, ...] = ()
        self._cache_module_count: int = 0
        self._cache_module_paths: dict[Path, str] = {}
        self._cache_module_paths_sorted: tuple[tuple[Path, str], ...] = ()
        self._cache_registered_module_roots: frozenset[Path] = frozenset()
        self._cache_dropped_out_of_root_count: int = 0

    def _post_layout(self, effective_root: Path, audit_root: Path) -> None:
        """Populate per-member module paths + pre-sorted resolver lookup table.

        Workspace mode: one dict entry per member, keyed by `module_root`.
        Single-module mode: one entry keyed by `effective_root` (so the
        resolver branch sees a uniform shape).

        `_cache_module_paths_sorted` is built ONCE here as a tuple ordered
        longest-`module_path`-first, so `_resolve_go_import` doesn't sort
        on every import statement (round-2 delta #4).
        """
        del audit_root
        if self._cache_workspace_members:
            self._cache_module_paths = {
                m.module_root: m.module_path
                for m in self._cache_workspace_members
            }
        else:
            module_path = _read_module_path(effective_root / "go.mod")
            self._cache_module_paths = (
                {effective_root: module_path} if module_path is not None else {}
            )
        self._cache_module_count = len(self._cache_workspace_members)
        self._cache_registered_module_roots = frozenset(
            m.module_root for m in self._cache_workspace_members
        )
        self._cache_module_paths_sorted = tuple(
            sorted(
                self._cache_module_paths.items(),
                key=lambda kv: len(kv[1]),
                reverse=True,
            )
        )

    def _compute_effective_root(self, root: Path) -> Path:
        """Workspace shape wins; else auto-descend one wrapper level for `go.mod`."""
        if (root / _GO_WORK_FILENAME).is_file():
            return root
        if (root / "go.mod").is_file():
            return root
        try:
            children = list(root.iterdir())
        except OSError:
            return root
        mod_children = [
            c for c in children
            if c.is_dir() and (c / "go.mod").is_file()
        ]
        if len(mod_children) == 1:
            return mod_children[0]
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every Go package under `effective_root`.

        Writes `self._cache_workspace_members` as a side effect — see the
        plan's "Critical ordering" section. `_post_layout` reads that
        cache to populate the per-member module-path dict + sorted
        lookup table.

        Single-module mode: any directory containing ≥1 `.go` file
        directly registers under its POSIX-relative path. Files at the
        module root attribute to `None`.

        Workspace mode: each `use`-declared member's source tree is
        walked; names are qualified with the full module-path prefix
        (`example.com/mod_a/internal/auth`). Nested child manifests are
        skipped unless they're in the declared members list (defends
        against vendored modules inside a member).
        """
        use_paths = _read_go_work_uses(effective_root / _GO_WORK_FILENAME)
        if use_paths is not None:
            members, dropped = _expand_go_work_members(
                effective_root, use_paths,
            )
            self._cache_workspace_members = members
            self._cache_dropped_out_of_root_count = dropped
            registered = frozenset(m.module_root for m in members)
            return _discover_workspace_pairs(members, registered)
        self._cache_workspace_members = ()
        self._cache_dropped_out_of_root_count = 0
        return _discover_single_module_pairs(effective_root)

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Collect every `import_declaration` at the file's top level.

        Each `import_declaration` — whether a single `import "x"` or a
        grouped `import (...)` block — counts as ONE top-level statement.
        Refs are flattened across the block.
        """
        stmt_count, imports = self._walk_top_level_imports(
            tree, source_bytes, file, effective_root,
            per_stmt=self._imports_for_top_level_statement,
        )
        return stmt_count, imports, ()

    def _imports_for_top_level_statement(
        self,
        node: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> list[ImportRef]:
        """Refs produced by ONE top-level statement (only `import_declaration` matters)."""
        del file, effective_root
        if node.type != "import_declaration":
            return []
        return self._collect_import_refs(
            node, source_bytes, node.start_point[0] + 1,
        )

    def _collect_import_refs(
        self, decl: Any, source_bytes: bytes, line: int,
    ) -> list[ImportRef]:
        """Walk an `import_declaration` for every `import_spec` it contains."""
        out: list[ImportRef] = []
        for child in decl.children:
            if child.type == "import_spec":
                ref = self._ref_from_spec(child, source_bytes, line)
                if ref is not None:
                    out.append(ref)
            elif child.type == "import_spec_list":
                for spec in child.named_children:
                    if spec.type != "import_spec":
                        continue
                    ref = self._ref_from_spec(spec, source_bytes, line)
                    if ref is not None:
                        out.append(ref)
        return out

    def _ref_from_spec(
        self, spec: Any, source_bytes: bytes, line: int,
    ) -> ImportRef | None:
        """Build an `ImportRef` from one `import_spec` node, or `None` if malformed."""
        path_node = spec.child_by_field_name("path")
        if path_node is None or path_node.type != "interpreted_string_literal":
            return None
        raw = _node_text(path_node, source_bytes)
        if len(raw) < 2 or raw[0] != '"' or raw[-1] != '"':
            return None
        specifier = raw[1:-1]
        target = _resolve_go_import(
            specifier,
            self._cache_module_paths_sorted,
            self._cache_package_names,
        )
        return ImportRef(target_module=specifier, target_package=target, line=line)

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Capitalized top-level identifiers in one Go file.

        The base aggregates these across every file returned by
        `_facade_source_files` — Go has no index convention, so the
        package facade is the union of exports across all source files.
        """
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            _collect_exports(child, source_bytes, symbols)
        return symbols

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Like the base but drop files inside a nested Go-module subtree.

        Override mirrors the module-boundary guard in
        `_discover_package_roots` so the file count stays consistent with
        the package count. Workspace-aware: files inside a declared `use`
        member are kept; files inside an unregistered nested `go.mod` are
        dropped (defends against sibling submodules in a monorepo when a
        `go.work` only declares some of them). Single-module mode passes
        `frozenset()` so any nested `go.mod` becomes a boundary.
        """
        files = super().project_files(root, excludes)
        effective = self._cache_effective
        registered = self._cache_registered_module_roots
        return [
            f for f in files
            if not _crosses_module_boundary(f, effective, registered)
        ]

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every non-test `.go` file directly in `package_root`, sorted by name.

        Override because Go has no index-file convention — the whole
        package contributes. `*_test.go` files participate in the import
        graph but not the public facade. Sub-dirs are independent
        packages so the walk is non-recursive.
        """
        try:
            entries = sorted(package_root.iterdir(), key=lambda p: p.name)
        except OSError:
            return ()
        return tuple(
            e for e in entries
            if e.is_file()
            and e.suffix.lower() == ".go"
            and not e.name.endswith(_TEST_SUFFIX)
        )


def _crosses_module_boundary(
    file: Path,
    effective_root: Path,
    registered_module_roots: frozenset[Path],
) -> bool:
    """True if `file` lies inside an unregistered nested Go-module subtree.

    Walks up from `file.parent` toward `effective_root`. If an ancestor
    contains a `go.mod` AND is NOT in `registered_module_roots`, the file
    is on the wrong side of a module boundary. Workspace mode passes the
    declared `use` set; single-module mode passes `frozenset()` so any
    nested `go.mod` (always a separate module in Go) becomes a boundary.
    """
    cur = file.parent
    while cur != effective_root and effective_root in cur.parents:
        if (cur / "go.mod").is_file() and cur not in registered_module_roots:
            return True
        cur = cur.parent
    return False


def _dir_has_go_files(entries: list[Path]) -> bool:
    """True if any entry in `entries` is a regular `.go` file."""
    return any(
        e.is_file() and e.suffix.lower() == ".go"
        for e in entries
    )


def _is_exported(name: str) -> bool:
    """Go's exact export rule: first character is ASCII uppercase."""
    if not name:
        return False
    ch = name[0]
    return "A" <= ch <= "Z"


def _collect_exports(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add `node`'s exported top-level identifier (if any) to `symbols`."""
    nt = node.type
    if nt == "function_declaration" or nt == "method_declaration":
        name = _exported_identifier(node.child_by_field_name("name"), source_bytes)
        if name is not None:
            symbols.setdefault(name, _count_params(node))
    elif nt == "type_declaration":
        for spec in node.named_children:
            if spec.type != "type_spec":
                continue
            name = _exported_identifier(
                spec.child_by_field_name("name"), source_bytes,
            )
            if name is not None:
                symbols.setdefault(name, 0)
    elif nt == "const_declaration" or nt == "var_declaration":
        _collect_value_spec_names(node, source_bytes, symbols)


def _collect_value_spec_names(
    decl: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add every exported identifier from a const/var declaration's specs."""
    for spec in decl.named_children:
        if spec.type not in ("const_spec", "var_spec"):
            continue
        for child in spec.named_children:
            if child.type != "identifier":
                continue
            text = _node_text(child, source_bytes)
            if _is_exported(text):
                symbols.setdefault(text, 0)


def _exported_identifier(node: Any | None, source_bytes: bytes) -> str | None:
    """Return the identifier text iff it's exported (uppercase first char)."""
    if node is None:
        return None
    text = _node_text(node, source_bytes)
    return text if _is_exported(text) else None


def _count_params(fn_node: Any) -> int:
    """Total parameter count for a function or method declaration.

    Includes ordinary + variadic params (`parameter_declaration` and
    `variadic_parameter_declaration`). Receiver params (`(s *S)`) live in
    a separate `receiver` field and are NOT counted as function params.
    """
    params = fn_node.child_by_field_name("parameters")
    if params is None or params.type != "parameter_list":
        return 0
    return sum(
        1 for c in params.named_children
        if c.type in ("parameter_declaration", "variadic_parameter_declaration")
    )


def _read_module_path(go_mod_path: Path) -> str | None:
    """Extract the module path from a `go.mod` file's `module ...` directive.

    Scans the first `_MODULE_SCAN_LINES` lines; `module` is always declared
    before any other directive. Tolerates double-quoted and backtick-quoted
    forms (rare but legal). Returns `None` if the file is missing,
    unreadable, or has no `module` directive within the scan window.
    """
    try:
        text = go_mod_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines()[:_MODULE_SCAN_LINES]:
        stripped = line.strip()
        if not stripped.startswith(_MODULE_DIRECTIVE):
            continue
        rest = stripped[len(_MODULE_DIRECTIVE):].lstrip()
        if not rest:
            continue
        comment_idx = rest.find("//")
        if comment_idx >= 0:
            rest = rest[:comment_idx].rstrip()
        if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
            rest = rest[1:-1]
        elif len(rest) >= 2 and rest[0] == "`" and rest[-1] == "`":
            rest = rest[1:-1]
        return rest or None
    return None


def _resolve_go_import(
    specifier: str,
    module_paths_sorted: tuple[tuple[Path, str], ...],
    project_packages: frozenset[str],
) -> str | None:
    """Resolve a Go import path to a project package name, else `None`.

    Iterates `module_paths_sorted` longest-`module_path`-first so a member
    `example.com/foo/sub` wins over `example.com/foo` for the specifier
    `example.com/foo/sub/bar` (round-2 delta #12). Single-module mode
    sees a one-entry tuple; workspace mode sees one entry per declared
    `use`. On match the relative tail is looked up in `project_packages`
    — Go workspaces use the FULL module-path-prefixed name, so the
    qualified key is `<module_path>/<rel>`.
    """
    for _root, module_path in module_paths_sorted:
        prefix = module_path + "/"
        if not specifier.startswith(prefix):
            continue
        rel = specifier[len(prefix):]
        if not rel:
            continue
        qualified = f"{module_path}/{rel}"
        if qualified in project_packages:
            return qualified
        if rel in project_packages:
            return rel
        return None
    return None


def _read_go_work_uses(go_work: Path) -> tuple[str, ...] | None:
    """Return the list of `use` paths in `go.work`, or `None` if absent.

    Tolerates four forms (round-2 delta #11):
      use ./mod_a
      use\t./mod_b
      use ( ./mod_c ./mod_d )                   # single-line block
      use (
          ./mod_e
          ./mod_f
      )                                          # multi-line block

    `//` comments are stripped per line. Returns `None` when the file
    doesn't exist; returns `()` when present with no `use` entries.
    """
    if not go_work.is_file():
        return None
    try:
        text = go_work.read_text(encoding="utf-8")
    except OSError:
        return None
    entries: list[str] = []
    in_block = False
    for raw_line in text.splitlines():
        stripped = _strip_inline_comment(raw_line).strip()
        if not stripped:
            continue
        if in_block:
            if stripped == ")":
                in_block = False
                continue
            if stripped.endswith(")"):
                inner = stripped[:-1].strip()
                entries.extend(t for t in inner.split() if t)
                in_block = False
                continue
            entries.extend(t for t in stripped.split() if t)
            continue
        if _GO_WORK_BLOCK_OPEN_PATTERN.match(stripped):
            after_paren = stripped.split("(", 1)[1].strip()
            if after_paren.endswith(")"):
                inner = after_paren[:-1].strip()
                entries.extend(t for t in inner.split() if t)
            else:
                in_block = True
                if after_paren:
                    entries.extend(t for t in after_paren.split() if t)
            continue
        if stripped == _MODULE_DIRECTIVE:
            continue
        if stripped.startswith("use ") or stripped.startswith("use\t"):
            rest = stripped[len("use"):].strip()
            entries.extend(t for t in rest.split() if t)
    return tuple(entries)


def _strip_inline_comment(line: str) -> str:
    """Remove `//`-style comments from one line of `go.work`."""
    idx = line.find("//")
    return line if idx < 0 else line[:idx]


def _expand_go_work_members(
    workspace_root: Path, use_paths: tuple[str, ...],
) -> tuple[tuple[_GoWorkspaceMember, ...], int]:
    """Resolve `use` entries on disk; return (members, dropped_out_of_root_count).

    Each entry must point to a directory containing `go.mod`. Out-of-root
    entries (`use ../sibling`) are silently dropped, surfaced via the
    runner tripwire. Module path is read from each member's `go.mod`;
    members without a parseable `module` directive are silently skipped.
    """
    seen: dict[Path, _GoWorkspaceMember] = {}
    dropped = 0
    resolved_root = workspace_root.resolve()
    for raw in use_paths:
        candidate = (workspace_root / raw).resolve()
        if not _is_within(candidate, resolved_root):
            dropped += 1
            continue
        module_root = workspace_root / raw
        go_mod = module_root / "go.mod"
        if not go_mod.is_file():
            continue
        module_path = _read_module_path(go_mod)
        if module_path is None or module_root in seen:
            continue
        seen[module_root] = _GoWorkspaceMember(
            module_root=module_root, module_path=module_path,
        )
    return (
        tuple(sorted(seen.values(), key=lambda m: m.module_path)),
        dropped,
    )


def _is_within(candidate: Path, root: Path) -> bool:
    """True iff `candidate` resolves under `root`."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _discover_single_module_pairs(
    effective_root: Path,
) -> tuple[tuple[Path, str], ...]:
    """Today's single-module DFS — preserved verbatim for the fast path."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [effective_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if cur != effective_root and _dir_has_go_files(entries):
            rel = cur.relative_to(effective_root).as_posix()
            pairs.append((cur, rel))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _GO_SCAN_SKIP:
                continue
            stack.append(entry)
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _discover_workspace_pairs(
    members: tuple[_GoWorkspaceMember, ...],
    registered_module_roots: frozenset[Path],
) -> tuple[tuple[Path, str], ...]:
    """Discover `(dir, fully-qualified-name)` pairs across every workspace member."""
    pairs: list[tuple[Path, str]] = []
    for member in members:
        pairs.extend(_walk_member_packages(member, registered_module_roots))
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _walk_member_packages(
    member: _GoWorkspaceMember,
    registered_module_roots: frozenset[Path],
) -> list[tuple[Path, str]]:
    """DFS one Go workspace member's source tree, yielding qualified pairs."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [member.module_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if cur != member.module_root and _dir_has_go_files(entries):
            rel = cur.relative_to(member.module_root).as_posix()
            pairs.append((cur, f"{member.module_path}/{rel}"))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _GO_SCAN_SKIP:
                continue
            if (entry / "go.mod").is_file() and entry not in registered_module_roots:
                continue
            stack.append(entry)
    return pairs
