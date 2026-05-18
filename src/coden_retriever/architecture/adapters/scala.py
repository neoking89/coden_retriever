"""Scala adapter: tree-sitter only.

Scala builds atop sbt or mill — `build.sbt` for sbt, `build.sc` for mill —
and lays out sources under the conventional `src/main/scala/`. The effective
root is resolved the same way as the Java and Kotlin adapters: descend into
`src/main/scala/` when a manifest sits at `root`; allow one wrapper-level
descent when exactly one direct child carries a manifest; everything else
(multi-manifest siblings, multi-module parents, no manifest anywhere) falls
back to walking the audit root directly. A multi-module sbt parent therefore
reports zero packages and zero files; point the audit at one submodule for
a useful audit.

Scala diverges from Java and Kotlin in four load-bearing ways:

1. **Two package-declaration forms**: top-level `package a.b.c` AND
   brace-block `package a.b.c { … }`. Both use the same `package_clause`
   node with a `name:` field; the brace form also exposes a `body:` field
   (`template_body`) whose contents the facade walker must descend into.
   Type declarations inside a brace-block package are NOT direct children
   of the compilation unit — they live one level deep under
   `package_clause.body.template_body`.

2. **Package identity comes from the `package_clause` header**, not from
   the directory basename. Scala permits header / folder mismatch. A
   directory becomes a registered package iff at least one `.scala` file
   directly inside it declares a parseable `package_clause`. The
   package's name is the dotted identifier in the FIRST alphabetical
   filename's header. Files with no header attribute to `None`.

3. **In-function imports are real**. Unlike Java and Kotlin, Scala
   permits `import x.y.z` inside a `def` body — and the language uses
   this idiom regularly (cycle workarounds, scoped extension methods).
   This is the only JVM-family adapter where `in_function_imports` is
   non-empty. The collector walks the entire tree once, tracking the
   enclosing function name; imports under a `function_definition` or
   `function_declaration` ancestor become `InFunctionImport` records,
   imports outside any function become top-level refs.

4. **Brace-list imports** `import a.b.{X, Y}` count as ONE statement
   with ONE ref (target_package = `a.b`). Same for wildcard
   `import a.b._` and aliased `import a.b.{X => Y}`. The selector
   subtree's identifiers are imported member names, NOT package
   qualifiers — they are NOT appended to the resolver segments.

Import resolution: longest-first prefix match across registered package
names. Identical in shape to Java/Kotlin's resolver because Scala's
grammar can't tell us whether a trailing identifier is a class, a
sub-package, or a member — longest-first is the only correct strategy.
This handles every import form uniformly:

  import a.b.C                segments `a.b.C` → match `a.b`
  import a.b._                segments `a.b`   → match `a.b`
  import a.b.{X, Y}           segments `a.b`   → match `a.b`
  import a.b.{X => Y}         segments `a.b`   → match `a.b`
  import a.b.C.method         segments `a.b.C.method` → match `a.b`

Public-symbol extraction collects every top-level `class_definition`,
`object_definition`, `trait_definition` — and the same types nested ONE
level under a brace-block `package_clause.body`. For each type: its
identifier registers as a symbol; the primary-constructor arity (count of
`class_parameter` children of the `class_parameters` field) seeds the
class's param count; direct public `function_definition` /
`function_declaration` members register under their own names. Top-level
`function_definition`s (free functions, Scala 3 toplevel-fun style) also
register. `private` and `protected` exclude a declaration.

v2 closes three v1 gaps surfaced by real-world A/B testing against
`typelevel/cats-effect`:

5. **Chained-package** (`package a.b\npackage c\n`). `_read_package_name`
   now walks ALL leading sibling `package_clause` (and trailing
   `package_object`) nodes and concatenates segments with `.`. The
   effective package for the chain `package a.b\npackage c` is `a.b.c`.

6. **`package object foo { … }`** (Scala 3). The trailing `package_object`
   contributes its `name:` identifier to the chain (so a file with
   `package foo\npackage object bar { … }` registers as `foo.bar`). The
   facade walker descends one level into the `package_object.body`
   (template_body) — same policy as brace-block `package_clause` bodies.
   A file containing ONLY `package object bar { … }` with NO leading
   `package_clause` returns `None` — there is no parent context to
   attribute `bar` to, and registering it standalone would be misleading.

7. **sbt multi-subproject root** (`build.sbt` with `≥2 lazy val NAME = project*`
   declarations). The previous "explicit-fail" gate only fired on multiple
   sibling `build.sbt`/`build.sc` files. v2 also parses the root `build.sbt`
   with tree-sitter and counts `lazy val NAME = project` / `project.in(file(…))`
   / `(project).in(…)` declarations; ≥2 → zero packages, zero files (mirrors
   Java/Kotlin multi-module explicit-fail). The detection is sbt-only and
   does NOT cover `crossProject`, `projectMatrix`, or infix `project in file(…)`
   — those repos walk normally as before.

What v2 deliberately does NOT do (preserved current behavior; not flagged
in QA):

- Scala 3 `given` / `using` declarations as facade entries.
- `crossProject(JSPlatform, …)` / `projectMatrix.in(…)` / infix
  `project in file(…)` shapes in `build.sbt` — gap-3 only detects the
  bare `project` / `project.in(file(…))` shapes.
- Mill (`build.sc`) multi-subproject patterns — gap-3 is sbt-only.
- Scala.js / Scala Native dialect handling.
- `private[scope]` qualified-visibility — treated as `private` (per
  v1 out-of-scope; the `access_qualifier` sub-tree sits inside
  `access_modifier` but the first-token check still fires).
- `case class Companion` / companion-object member promotion onto the
  enclosing class.
- `type` alias declarations as facade symbols.
- Top-level `val` / `var` declarations as facade symbols.
- Test class suffixes `*Suite.scala`, `*Properties.scala` (not filtered;
  only `*Spec.scala` and `*Test.scala` are excluded from the facade).
- Aliased import selectors `import a.b.{X => Y}` — alias name dropped
  from `target_module` display (matches Kotlin's `as`-suffix policy).
- Arbitrarily-nested brace-block packages within one file
  (`package x { package y { … } }`) — the facade walker descends one
  level only.

Scope:
  This adapter audits Scala; it is one of ten adapters registered in
  `architecture/core/runner.py::_ADAPTERS` (see README's
  `## Supported Languages` section for the full list). Out of scope:
  sbt multi-subproject parents (explicit-fail — point the audit at one
  submodule), Mill (`build.sc`) build files as a manifest signal beyond
  the existing single-module case, Scala 3 `given` / `using`
  declarations as facade entries, and Scala.js / Scala Native dialect
  handling. C, C++, and Bash are architecture-unsupported across all
  adapters; the README documents why.
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


_SCALA_MANIFESTS: frozenset[str] = frozenset({
    "build.sbt",
    "build.sc",
})
# Why: filenames that mark an sbt (`build.sbt`) or mill (`build.sc`)
# Scala project root. Their presence triggers descent into the
# conventional `src/main/scala/` source root. Unlike the Java/Kotlin
# adapters, Maven/Gradle are deliberately excluded — Scala projects
# almost always use sbt or mill; mixed-build Scala code organized
# under `src/main/scala/` will still be found via the no-manifest
# fallback (effective = root) when no sbt/mill marker is present.

_SCALA_SRC_PARTS: tuple[str, str, str] = ("src", "main", "scala")
# Why: the conventional source-root path under an sbt/mill project.
# After locating the manifest, the adapter descends three more levels
# so package discovery starts at the directory whose subdirs map
# directly to package qualifiers (`com/foo/bar/` → first .scala file's
# `package com.foo.bar`).

_SCALA_SCAN_SKIP: frozenset[str] = frozenset({
    "target",
})
# Why: build-output directory used by sbt and mill. `project/` is sbt's
# meta-build directory but ONLY at the audit-root level — `src/main/
# scala/com/project/` would be a legitimate Scala package. The walker
# checks for `project/` only as a direct child of the audit root via
# the `_skip_project_dir` helper, NOT via this generic skip set.

_TEST_FILENAME_SUFFIXES: tuple[str, ...] = (
    "Spec.scala", "Test.scala",
)
# Why: ScalaTest's `*Spec` is the dominant test-class naming convention
# in the Scala ecosystem; `*Test` is the JVM-wide JUnit/MUnit suffix.
# `*Suite.scala` and `*Properties.scala` are intentionally NOT filtered
# in v1 — they appear in ScalaTest but are also used as legitimate
# domain class names (e.g., `BenchmarkSuite`).

_PUBLIC_TYPE_NODE_TYPES: frozenset[str] = frozenset({
    "class_definition",
    "object_definition",
    "trait_definition",
})
# Why: the top-level type-declaration node kinds whose default-public
# declarations feed the package facade. `case class X` parses as a
# `class_definition` with an anonymous `case` token sibling — same
# node type. `package_object` (Scala 3) is intentionally absent (see
# module docstring's "deliberately does NOT do" list).

_NON_PUBLIC_ACCESS_TOKENS: frozenset[str] = frozenset({
    "private", "protected",
})
# Why: token types that exclude a declaration from the public facade.
# Applied by inspecting the FIRST direct-child token of each
# `access_modifier` node inside a declaration's `modifiers` group.
# The `access_qualifier` subtree (`[scope]`) sits as a sibling of that
# token inside the same `access_modifier`, so `private[scope]` registers
# as private without any extra handling — matches the v1 out-of-scope
# `private[scope]` policy.

_PACKAGE_CONTAINER_WITH_BODY_NODE_TYPES: frozenset[str] = frozenset({
    "package_clause",
    "package_object",
})
# Why: top-level node kinds that contribute a segment to the chained
# package name in `_assemble_chained_package` AND whose `body:` field
# (when present) the facade walker descends into one level. `package_clause`
# carries `name:` as a `package_identifier` (dotted); `package_object`
# carries `name:` as a single `identifier`. Both share the descent policy
# in `_collect_top_or_braced`.

_SBT_MULTI_SUBPROJECT_THRESHOLD: int = 2
# Why: number of `lazy val NAME = project*` declarations at the root
# `build.sbt` level above which the adapter treats the audit root as a
# multi-module sbt parent and explicit-fails (zero packages, zero files).
# One `lazy val core = project` is a legitimate single-module sbt layout;
# two or more signals "point me at one submodule, not the parent."

_SBT_PROJECT_IDENTIFIER: str = "project"
# Why: the sbt-reserved identifier that names a single-platform subproject
# in `build.sbt` (e.g. `lazy val core = project.in(file("core"))`). v2
# detects exactly this identifier as the leftmost leaf of a `lazy val`'s
# value-chain. `crossProject`, `projectMatrix`, and `MyHelpers.project(...)`
# all bottom out at a DIFFERENT identifier and are intentionally NOT counted.

_LAZY_MODIFIER_TOKEN: str = "lazy"
# Why: the tree-sitter token type that marks a `val_definition` as lazy
# inside its `modifiers` group. Only lazy vals participate in the sbt
# subproject count — sbt's contract is `lazy val ... = project*`.

_BUILD_SBT_BASENAME: str = "build.sbt"
# Why: the manifest filename gap-3 parses to count subproject declarations.
# Mill (`build.sc`) is intentionally out of scope.


class ScalaAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for Scala projects."""

    LANGUAGE = "scala"
    EXTENSIONS = frozenset({".scala"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//",)

    def _compute_effective_root(self, root: Path) -> Path:
        """Descend into `<project>/src/main/scala/` when an sbt/mill manifest is present."""
        source_root = _scala_source_root(root)
        if source_root is not None:
            return source_root
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every Scala package under `effective_root`.

        A "Scala package" is any directory containing ≥1 `.scala` file
        whose `package_clause` parses successfully. The directory's
        package name is the dotted identifier in the FIRST alphabetical
        filename's header — never the directory basename. Files at
        `effective_root` itself never form a registered package — they
        attribute to `None` via the base's `_find_package` walk.

        Two module-boundary guards explicit-fail (return `()`) when the
        audit target is a multi-module sbt parent:
        - Multi-manifest siblings: a child directory whose own contents
          include an sbt/mill manifest is a separate submodule and is
          skipped during the walk.
        - Root-level `lazy val NAME = project` declarations: when the
          `effective_root` carries a `build.sbt` with ≥2 such decls
          (sbt's idiomatic multi-subproject style — see cats-effect),
          return `()` outright. Without this, pointing the audit at a
          single-file-multi-module sbt parent walks every submodule and
          silently produces a coherent but meaningless report.
        """
        if _is_sbt_multi_subproject_root(effective_root, self._get_parser):
            return ()
        pairs: list[tuple[Path, str]] = []
        stack: list[Path] = [effective_root]
        while stack:
            cur = stack.pop()
            try:
                entries = list(cur.iterdir())
            except OSError:
                continue
            if cur != effective_root:
                package_name = _package_name_for_dir(entries, self._get_parser)
                if package_name is not None:
                    pairs.append((cur, package_name))
            for entry in entries:
                if not entry.is_dir():
                    continue
                if entry.name.startswith("."):
                    continue
                if entry.name in _SCALA_SCAN_SKIP:
                    continue
                if cur == effective_root and entry.name == "project":
                    continue
                if _has_scala_manifest(entry):
                    continue
                stack.append(entry)
        return tuple(sorted(pairs, key=lambda p: p[1]))

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Single recursive walk: top-level imports + in-function imports.

        Scala buries imports under brace-block-package bodies AND
        function bodies, so the base helper `_walk_top_level_imports`
        (which iterates only `tree.root_node.children`) cannot be
        reused. The walk classifies each `import_declaration` by its
        ancestor chain: under a `function_definition` /
        `function_declaration` → in-function; otherwise top-level.
        """
        del file, effective_root
        stmt_count, top_refs, in_func = _collect_imports(
            tree.root_node, source_bytes, self._cache_package_names,
            enclosing_func=None,
        )
        return stmt_count, tuple(top_refs), tuple(in_func)

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Top-level public types (including brace-block-nested) plus their public members."""
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            _collect_top_or_braced(child, source_bytes, symbols)
        return symbols

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Like the base but drop files inside a nested sbt submodule subtree.

        Two module-boundary guards mirror `_discover_package_roots`:
        - Root `build.sbt` with `≥2 lazy val NAME = project*` decls →
          return `[]` BEFORE the super-call enumerates files. Without
          this check at the top, `iter_source_files(effective)` walks
          the multi-module subtree and emits N files paired with the
          zero packages from `_discover_package_roots` — breaking the
          "multi-module → zero packages AND zero files" invariant.
        - Sibling-manifest submodules: filter files via
          `_crosses_module_boundary` after `super().project_files()`.
        """
        if _is_sbt_multi_subproject_root(
            self._compute_effective_root(root), self._get_parser,
        ):
            return []
        files = super().project_files(root, excludes)
        effective = self._cache_effective
        return [
            f for f in files
            if not _crosses_module_boundary(f, effective)
        ]

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every non-test `.scala` file directly in `package_root`, sorted by name.

        Scala has no single-index convention — each `.scala` file
        contributes its public symbols. Sub-dirs are independent
        packages so the walk is non-recursive. `*Spec.scala` /
        `*Test.scala` files are excluded from the facade but kept in
        the import graph via `project_files`. Brace-block descent
        happens inside `_collect_public_symbols`, not here.
        """
        try:
            entries = sorted(package_root.iterdir(), key=lambda p: p.name)
        except OSError:
            return ()
        return tuple(
            e for e in entries
            if e.is_file()
            and e.suffix.lower() == ".scala"
            and not _is_test_filename(e.name)
        )


def _is_test_filename(name: str) -> bool:
    """True if `name` ends in a ScalaTest/MUnit test-class suffix."""
    return any(name.endswith(suffix) for suffix in _TEST_FILENAME_SUFFIXES)


def _scala_source_root(root: Path) -> Path | None:
    """Locate the Scala source root under `root`, with one-level wrapper descent.

    Returns `root/src/main/scala/` when a manifest is present at
    `root` and that directory exists. If `root` itself has no
    manifest but exactly one direct child does, recurse one level on
    that child. Every other arrangement returns `None` — callers fall
    back to walking `root` directly.
    """
    if _has_scala_manifest(root):
        return _src_main_scala_under(root)
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    manifest_children = [
        c for c in children
        if c.is_dir() and _has_scala_manifest(c)
    ]
    if len(manifest_children) == 1:
        return _src_main_scala_under(manifest_children[0])
    return None


def _has_scala_manifest(directory: Path) -> bool:
    """True iff `directory` contains any sbt/mill manifest."""
    return any(
        (directory / name).is_file()
        for name in _SCALA_MANIFESTS
    )


def _crosses_module_boundary(file: Path, effective_root: Path) -> bool:
    """True if any ancestor of `file` between it and `effective_root` carries a manifest."""
    cur = file.parent
    while cur != effective_root and effective_root in cur.parents:
        if _has_scala_manifest(cur):
            return True
        cur = cur.parent
    return False


def _src_main_scala_under(project_root: Path) -> Path | None:
    """Return `<project_root>/src/main/scala/` if that directory exists, else `None`."""
    candidate = project_root
    for part in _SCALA_SRC_PARTS:
        candidate = candidate / part
    return candidate if candidate.is_dir() else None


def _package_name_for_dir(
    entries: list[Path],
    get_parser: Any,
) -> str | None:
    """Read the `package_clause` from the first alphabetical `.scala` file in `entries`.

    Returns the dotted package name, or `None` if no `.scala` file in
    `entries` declares a parseable `package_clause`. Deterministic by
    sort key so dir-iteration order can't make the result flaky.
    """
    scala_files = sorted(
        (e for e in entries if e.is_file() and e.suffix.lower() == ".scala"),
        key=lambda p: p.name,
    )
    for scala_file in scala_files:
        name = _read_package_name(scala_file, get_parser)
        if name is not None:
            return name
    return None


def _read_package_name(
    file_path: Path, get_parser: Any,
) -> str | None:
    """Parse `file_path` and return its effective dotted package name, or `None`.

    Combines every leading sibling `package_clause` (and a trailing
    `package_object`, if present) under the compilation unit into one
    dotted name — so `package a.b\\npackage c\\n` yields `a.b.c` and
    `package foo\\npackage object bar { … }` yields `foo.bar`. A file
    starting with `package object bar { … }` without a preceding
    `package_clause` returns `None`: there is no parent context to
    attribute `bar` to, and registering it standalone would be
    misleading. Files with no header return `None`.
    """
    parser = get_parser("scala")
    if parser is None:
        return None
    try:
        source_bytes = file_path.read_bytes()
    except OSError:
        return None
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return None
    return _assemble_chained_package(tree.root_node, source_bytes)


def _assemble_chained_package(
    root_node: Any, source_bytes: bytes,
) -> str | None:
    """Return the dotted package name from a chain of leading package headers.

    Walks `root_node.children` in source order. Collects segments from
    each `package_clause` (via its `package_identifier` `name:` field)
    and a single trailing `package_object` (its `name:` is one
    `identifier`). The chain ends at the first non-trivia node whose
    type is not in `_PACKAGE_CONTAINER_WITH_BODY_NODE_TYPES`, or at the
    first `package_clause` carrying a `body:` (brace-block form, which
    already encodes its dotted name in a single `package_identifier`),
    or after a `package_object` (which always terminates the chain).
    Returns `None` if no `package_clause` is found, even when a
    `package_object` is present — that case has no parent context.
    """
    segments: list[str] = []
    saw_package_clause = False
    for child in root_node.children:
        if child.type == "package_clause":
            saw_package_clause = True
            name_node = child.child_by_field_name("name")
            if name_node is None or name_node.type != "package_identifier":
                return None
            segments.extend(_flatten_package_identifier(name_node, source_bytes))
            if child.child_by_field_name("body") is not None:
                break
            continue
        if child.type == "package_object":
            if not saw_package_clause:
                return None
            name_node = child.child_by_field_name("name")
            if name_node is None or name_node.type != "identifier":
                return None
            segments.append(_node_text(name_node, source_bytes))
            break
        if _is_trivia_node(child):
            continue
        break
    return ".".join(segments) if segments else None


def _is_trivia_node(node: Any) -> bool:
    """True for whitespace / comment / ERROR nodes between package headers.

    Tree-sitter sometimes interleaves trivia between sibling header
    nodes; the chain walker tolerates them rather than treating their
    appearance as a chain-terminator (which would silently revert to v1
    behaviour on real-world files with comments between headers).
    """
    return node.type in {"comment", "block_comment", "ERROR"} or node.is_extra


def _flatten_package_identifier(node: Any, source_bytes: bytes) -> list[str]:
    """Collect direct `identifier` children text from a `package_identifier` node."""
    return [
        _node_text(c, source_bytes)
        for c in node.children
        if c.type == "identifier"
    ]


def _collect_imports(
    node: Any,
    source_bytes: bytes,
    project_packages: frozenset[str],
    enclosing_func: str | None,
) -> tuple[int, list[ImportRef], list[InFunctionImport]]:
    """Single recursive walk: emits top-level + in-function imports.

    `enclosing_func` is the name of the nearest containing
    `function_definition` / `function_declaration`, or `None` when
    outside any function. Imports at top level (no enclosing function)
    are emitted as `ImportRef` + increment the statement count. Imports
    inside a function become `InFunctionImport` records keyed by that
    function's name. The walk recurses through brace-block packages,
    template bodies, and class bodies WITHOUT changing
    `enclosing_func` — only function nodes set or update it.
    """
    stmt_count = 0
    top_refs: list[ImportRef] = []
    in_func: list[InFunctionImport] = []
    for child in node.children:
        if child.type == "import_declaration":
            ref = _ref_from_import_declaration(
                child, source_bytes, project_packages,
            )
            if ref is None:
                continue
            if enclosing_func is None:
                top_refs.append(ref)
                stmt_count += 1
            else:
                in_func.append(InFunctionImport(
                    line=child.start_point[0] + 1,
                    function=enclosing_func,
                    import_text=_node_text(child, source_bytes),
                    target_package=ref.target_package,
                ))
            continue
        if child.type in ("function_definition", "function_declaration"):
            name_node = child.child_by_field_name("name")
            sub_enclosing = (
                _node_text(name_node, source_bytes)
                if name_node is not None else enclosing_func
            )
            sub_count, sub_top, sub_in = _collect_imports(
                child, source_bytes, project_packages, sub_enclosing,
            )
        else:
            sub_count, sub_top, sub_in = _collect_imports(
                child, source_bytes, project_packages, enclosing_func,
            )
        stmt_count += sub_count
        top_refs.extend(sub_top)
        in_func.extend(sub_in)
    return stmt_count, top_refs, in_func


def _ref_from_import_declaration(
    decl: Any, source_bytes: bytes, project_packages: frozenset[str],
) -> ImportRef | None:
    """Build an `ImportRef` from one `import_declaration` node, or `None`.

    The Scala grammar exposes import segments as multiple sibling
    children of the declaration itself, each tagged with field-name
    `'path'`. There is NO single qualified-name child. The trailing
    member-selection may be a `namespace_selectors` (`{X, Y}` /
    `{X => Y}`) or `namespace_wildcard` (`_`) — collapsed to one ref
    pointing at the dotted-path prefix.
    """
    segments = [
        _node_text(c, source_bytes)
        for c in decl.children_by_field_name("path")
        if c.type == "identifier"
    ]
    if not segments:
        return None
    is_wildcard = any(c.type == "namespace_wildcard" for c in decl.children)
    return ImportRef(
        target_module=_format_import_module(segments, is_wildcard),
        target_package=_resolve_scala_import(segments, project_packages),
        line=decl.start_point[0] + 1,
    )


def _format_import_module(segments: list[str], is_wildcard: bool) -> str:
    """Build a display string for `ImportRef.target_module`."""
    body = ".".join(segments)
    if is_wildcard:
        body = body + ".*"
    return body


def _resolve_scala_import(
    segments: list[str], project_packages: frozenset[str],
) -> str | None:
    """Resolve qualified-name segments to a registered package, longest-first.

    Mirrors Java/Kotlin's resolver: try every prefix length, longest
    first, return the first one that matches a registered package.
    Handles every Scala import form uniformly because the selector
    subtree (brace-list / wildcard / aliased) is dropped from the
    segments before resolution.
    """
    for n in range(len(segments), 0, -1):
        candidate = ".".join(segments[:n])
        if candidate in project_packages:
            return candidate
    return None


def _collect_top_or_braced(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Collect from one compilation-unit child OR descend a body-bearing package once.

    Direct `class_definition` / `object_definition` / `trait_definition`
    children contribute via `_collect_public_type`. Direct
    `function_definition` / `function_declaration` children contribute
    via `_collect_public_top_level_fun` (Scala 3 toplevel-fun parity).
    A `package_clause` with a `body:` field (brace-block form) OR a
    `package_object` (Scala 3) triggers ONE level of descent into the
    body's `template_body`; arbitrarily-nested brace-block packages
    within one file are NOT descended (v1 out-of-scope).
    """
    if node.type in _PUBLIC_TYPE_NODE_TYPES:
        _collect_public_type(node, source_bytes, symbols)
        return
    if node.type in ("function_definition", "function_declaration"):
        _collect_public_top_level_fun(node, source_bytes, symbols)
        return
    if node.type in _PACKAGE_CONTAINER_WITH_BODY_NODE_TYPES:
        body = node.child_by_field_name("body")
        if body is None:
            return
        for child in body.children:
            if child.type in _PUBLIC_TYPE_NODE_TYPES:
                _collect_public_type(child, source_bytes, symbols)
            elif child.type in ("function_definition", "function_declaration"):
                _collect_public_top_level_fun(child, source_bytes, symbols)


def _collect_public_type(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add a top-level public type's symbols (class + public members) to `symbols`.

    Visibility default is public (no `modifiers` child → public). The
    primary-constructor parameter count seeds the class entry before
    the body walk, so a `case class D(a: Int, b: String)` registers
    as `D` with param-count 2 even when its body is empty.
    """
    if _visibility_excludes(node):
        return
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _node_text(name_node, source_bytes)

    class_params = node.child_by_field_name("class_parameters")
    if class_params is not None:
        symbols.setdefault(
            class_name,
            _count_class_parameters(class_params),
        )

    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.children:
            _collect_public_member(member, source_bytes, symbols)

    symbols.setdefault(class_name, 0)


def _collect_public_member(
    member: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add one direct member of a public type to `symbols` (when itself public).

    Both `function_definition` (concrete) and `function_declaration`
    (abstract; trait methods without a body) register the function
    under its name with the `parameters:` arity.
    """
    if _visibility_excludes(member):
        return
    if member.type not in ("function_definition", "function_declaration"):
        return
    name_node = member.child_by_field_name("name")
    if name_node is None:
        return
    symbols.setdefault(
        _node_text(name_node, source_bytes),
        _count_function_params(member),
    )


def _collect_public_top_level_fun(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add a top-level public function (free function) to `symbols`.

    Scala 3 permits free functions at file scope (and inside
    brace-block packages). First occurrence wins on param count via
    `setdefault` — matches the aggregator's policy across facade
    source files.
    """
    if _visibility_excludes(node):
        return
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    symbols.setdefault(
        _node_text(name_node, source_bytes),
        _count_function_params(node),
    )


def _visibility_excludes(node: Any) -> bool:
    """True iff `node` carries a `modifiers` group with a non-public `access_modifier`.

    Walks `modifiers` children. For each `access_modifier` child,
    inspects its FIRST direct-child token type; if that type is in
    `_NON_PUBLIC_ACCESS_TOKENS`, the declaration is excluded. The
    `access_qualifier` `[scope]` subtree sits as a sibling of that
    token inside the same `access_modifier`, so `private[scope]`
    fires the exclusion without extra handling.
    """
    for child in node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type != "access_modifier":
                continue
            for token in mod.children:
                if token.type in _NON_PUBLIC_ACCESS_TOKENS:
                    return True
                break
        return False
    return False


def _count_class_parameters(class_params: Any) -> int:
    """Count `class_parameter` direct children of a `class_parameters` node."""
    return sum(
        1 for c in class_params.children
        if c.type == "class_parameter"
    )


def _count_function_params(node: Any) -> int:
    """Count `parameter` slots on a function declaration / definition.

    The grammar exposes the parameter list as the `parameters:` field
    → `parameters` node whose direct children include one `parameter`
    per slot. Scala has no varargs node-type — `Int*` modifies the
    parameter's `type`, so the count is correct without special
    handling.
    """
    params = node.child_by_field_name("parameters")
    if params is None:
        return 0
    return sum(
        1 for c in params.children
        if c.type == "parameter"
    )


def _is_sbt_multi_subproject_root(root: Path, get_parser: Any) -> bool:
    """True iff `root/build.sbt` declares ≥2 `lazy val NAME = project*` decls.

    Gap-3 explicit-fail gate: when sbt's idiomatic
    `lazy val core = project.in(file("core"))` pattern appears two or
    more times in the root manifest, the audit target is a multi-module
    parent and walking it would produce a coherent but meaningless
    cross-module report (see cats-effect 19-package case). Returns
    `False` when no `build.sbt` exists at `root` — the no-op path that
    preserves v1 behaviour on single-module sbt and on `src/main/scala/`
    effective roots where the manifest sits one level up.
    """
    build_sbt = root / _BUILD_SBT_BASENAME
    if not build_sbt.is_file():
        return False
    return _count_lazy_val_project_decls(build_sbt, get_parser) >= _SBT_MULTI_SUBPROJECT_THRESHOLD


def _count_lazy_val_project_decls(
    build_sbt_path: Path, get_parser: Any,
) -> int:
    """Count top-level `lazy val NAME = project*` declarations in `build_sbt_path`.

    Walks `root_node.children`; for each `val_definition` whose
    `modifiers` group contains a `lazy` token, descends the `value:`
    field's call-chain through `call_expression.function`,
    `field_expression.value`, and `parenthesized_expression` (positional
    inner). When the descent bottoms out at an identifier whose text is
    exactly `project`, increments the count. `crossProject(...)`,
    `projectMatrix.in(...)`, and `MyHelpers.project(...)` bottom out at
    a DIFFERENT leftmost identifier — not counted, as intended.

    Returns 0 (no-op) when the scala grammar is unavailable, the file
    can't be read, or parsing fails — keeps v1 behaviour and logs once
    when the grammar is missing so the contract drift is observable.
    """
    parser = get_parser("scala")
    if parser is None:
        logger.warning(
            "scala grammar unavailable; sbt multi-subproject guard "
            "disabled at %s", build_sbt_path,
        )
        return 0
    try:
        source_bytes = build_sbt_path.read_bytes()
    except OSError:
        return 0
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return 0
    return sum(
        1 for child in tree.root_node.children
        if _is_lazy_val_project_decl(child, source_bytes)
    )


def _is_lazy_val_project_decl(node: Any, source_bytes: bytes) -> bool:
    """True iff `node` is a `lazy val NAME = project*` declaration.

    Two conditions: (1) `node` is a `val_definition` whose `modifiers`
    group includes a `lazy` token; (2) its `value:` field's leftmost
    leaf (after unwrapping calls / field selections / parens) is an
    identifier whose text equals `_SBT_PROJECT_IDENTIFIER`.
    """
    if node.type != "val_definition":
        return False
    if not _has_lazy_modifier(node):
        return False
    value = node.child_by_field_name("value")
    if value is None:
        return False
    leaf = _unwrap_call_chain_leftmost(value)
    if leaf is None or leaf.type != "identifier":
        return False
    return _node_text(leaf, source_bytes) == _SBT_PROJECT_IDENTIFIER


def _has_lazy_modifier(val_def: Any) -> bool:
    """True iff `val_def`'s `modifiers` child contains a `lazy` token."""
    for child in val_def.children:
        if child.type != "modifiers":
            continue
        return any(m.type == _LAZY_MODIFIER_TOKEN for m in child.children)
    return False


def _unwrap_call_chain_leftmost(node: Any) -> Any | None:
    """Descend the leftmost edge of a call / field / paren chain.

    Tree-sitter Scala exposes `call_expression.function`,
    `field_expression.value`, and `parenthesized_expression`'s inner
    expression as the leftmost child needed to find the receiver of a
    fluent chain. The descent stops when the current node is none of
    these three kinds — that node is the leftmost leaf the caller
    examines for the `project` identifier check.
    """
    cur = node
    while cur is not None:
        if cur.type == "call_expression":
            cur = cur.child_by_field_name("function")
            continue
        if cur.type == "field_expression":
            cur = cur.child_by_field_name("value")
            continue
        if cur.type == "parenthesized_expression":
            cur = _first_non_punctuation_child(cur)
            continue
        return cur
    return None


def _first_non_punctuation_child(node: Any) -> Any | None:
    """Return the first child of `node` that is not a `(` / `)` punctuation token.

    `parenthesized_expression` does not expose its inner expression via
    a named field; positional iteration is the only option. Skipping
    the parenthesis tokens lands on the wrapped expression.
    """
    for child in node.children:
        if child.type in ("(", ")"):
            continue
        return child
    return None
