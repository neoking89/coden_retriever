"""Kotlin adapter: tree-sitter only.

Kotlin shares the JVM project conventions with Java — `pom.xml` for Maven,
`build.gradle` / `settings.gradle` (plus their Kotlin DSL `.kts` variants) for
Gradle, and `src/main/kotlin/` as the source root. The effective root is
resolved the same way as the Java adapter: descend into `src/main/kotlin/`
when a manifest is present; allow one wrapper-level descent when exactly one
direct child carries a manifest; everything else (multi-manifest siblings,
multi-module parents, no manifest anywhere) falls back to walking the audit
root directly. A multi-module Gradle parent therefore reports zero packages
and zero files; point the audit at a submodule for a useful audit.

Kotlin diverges from Java in three load-bearing ways:

1. **Package identity comes from the `package` header**, not from the
   directory basename. Kotlin permits header / folder mismatch (unlike the
   JVM-enforced Java contract). A directory becomes a registered package
   iff at least one `.kt` file directly inside it declares a `package`
   header. The package's name is the dotted identifier in the FIRST
   alphabetical filename's header. Files in directories with no header at
   all attribute to `None` (mirrors Java's effective-root files).

2. **Imports live inside an `import_list` wrapper**, not at file top
   level — the base helper `_walk_top_level_imports` iterates
   `tree.root_node.children` and would mis-count the whole `import_list`
   as one statement. `_walk_imports` therefore overrides directly and
   walks `import_header` nodes inside the `import_list`.

3. **Default visibility is public**. Declarations without an explicit
   `visibility_modifier` are public. `private` / `internal` /
   `protected` exclude a declaration from the facade. Other modifier
   categories (`class_modifier` for `data` / `sealed` / `enum`;
   `function_modifier` for `inline` / `suspend`; `inheritance_modifier`
   for `open` / `abstract`) never change visibility — the filter checks
   `visibility_modifier` by node type, not by raw text.

Public-symbol extraction collects every top-level `class_declaration` and
`object_declaration` plus their direct public methods / secondary
constructors / primary-constructor parameter count, AND every top-level
`function_declaration` (Kotlin permits free functions; Java does not). The
class's symbol entry folds in its first public constructor's arity — the
`primary_constructor` field (counting `class_parameter` children) wins
when present; otherwise the first `secondary_constructor` in the body
wins; otherwise param-count 0.

Import resolution: longest-first prefix match across registered package
names. Identical in shape to Java's `_resolve_java_import` because
Kotlin's grammar can't tell the resolver whether a trailing identifier is
a class, a sub-package, or a member — longest-first is the only correct
strategy. This handles every Kotlin import form uniformly:

  import a.b.C                segments `a.b.C` → match `a.b`
  import a.b.*                segments `a.b`   → match `a.b`
  import a.b.C.method         segments `a.b.C.method` → match `a.b`
  import a.b.C as Aliased     segments `a.b.C` → match `a.b`

`in_function_imports` is always `()` — Kotlin forbids local imports
grammatically.

What v1 deliberately does NOT do (preserved current behavior; not flagged
in QA):

- `companion object` member promotion onto the enclosing class. The
  inner `companion object` is walked as a nested member but its methods
  are not lifted onto the outer class.
- `expect` / `actual` multiplatform declarations.
- `typealias` declarations as facade symbols.
- Top-level property declarations (`val x = ...`). Parity with Java's
  "types only" facade rule.
- `inline class` / `value class` get the same treatment as ordinary
  `class_declaration` (the grammar models them with a `class_modifier`).
- `fun interface` (functional interfaces) — treated like normal interface.
- `sealed` / `open` / `abstract` classes are public by default and feed
  the facade like normal classes.
- `inner class` declarations inside a body are not collected — the walk
  is top-level only, matching Java's "top-level types only" convention.
- KSP / kapt generated `.kt` files — treated as ordinary sources.
- `.kts` build-script files — `EXTENSIONS` excludes them; the audit
  walks `.kt` only. Gradle `build.gradle.kts` is a manifest, not a
  source file.
- Gradle multi-project / composite-build workspaces (multiple
  `settings.gradle[.kts]` files across siblings) — same explicit-fail
  behavior as Java's multi-module parent.

Scope:
  This adapter audits Kotlin; it is one of ten adapters registered in
  `architecture/core/runner.py::_ADAPTERS` (see README's
  `## Supported Languages` section for the full list). Out of scope:
  Gradle multi-project / composite-build workspaces (explicit-fail —
  point the audit at one submodule) and Kotlin Multiplatform source
  layouts (`src/commonMain/kotlin/`, `src/jvmMain/kotlin/`, etc. — only
  `src/main/kotlin/` is recognized today). C, C++, and Bash are
  architecture-unsupported across all adapters; the README documents
  why.
"""
from __future__ import annotations

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


_KOTLIN_MANIFESTS: frozenset[str] = frozenset({
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
})
# Why: filenames that mark a Maven (`pom.xml`) or Gradle (`build.gradle`,
# `settings.gradle`, plus the Kotlin DSL `.kts` variants) project root.
# Identical set to `_JVM_MANIFESTS` in `java.py` — Kotlin shares Maven and
# Gradle infrastructure with Java, so dropping `pom.xml` would silently
# fail on Maven-Kotlin projects.

_KOTLIN_SRC_PARTS: tuple[str, str, str] = ("src", "main", "kotlin")
# Why: the conventional source-root path under a Maven/Gradle project
# whose primary language is Kotlin. Symmetric with Java's
# `src/main/java/` — both lay out package directories directly under
# this root.

_KOTLIN_SCAN_SKIP: frozenset[str] = frozenset({
    "build", "out", "target",
})
# Why: build-output directories that must never count as Kotlin
# packages — `build/` for Gradle, `out/` for IntelliJ IDEA, `target/`
# for mixed Maven-Kotlin projects. Hidden dirs (`.gradle`, `.idea`) are
# filtered separately via the leading-dot check.

_TEST_FILENAME_SUFFIXES: tuple[str, ...] = (
    "Test.kt", "Tests.kt",
)
# Why: JUnit / kotest convention — `*Test.kt` and `*Tests.kt`. Excluded
# from the public facade only; the import graph still walks them.

_PUBLIC_TYPE_NODE_TYPES: frozenset[str] = frozenset({
    "class_declaration",
    "object_declaration",
})
# Why: top-level type declarations that contribute a class-shaped
# symbol. `class_declaration` covers `class`, `interface`, `enum class`,
# `data class`, `sealed class`, `inline class`, and `value class` — the
# `kind` field discriminates but the public-symbol walk treats them
# uniformly. `object_declaration` covers top-level singletons.

_NON_PUBLIC_VISIBILITY: frozenset[str] = frozenset({
    "private", "internal", "protected",
})
# Why: visibility-modifier tokens that exclude a declaration from the
# public facade. `public` is the default and is omitted from this set.
# The filter scans `visibility_modifier` nodes specifically, so
# `class_modifier` (`data`, `sealed`, `enum`) and `inheritance_modifier`
# (`open`, `abstract`) never interfere.


class KotlinAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for Kotlin projects."""

    LANGUAGE = "kotlin"
    EXTENSIONS = frozenset({".kt"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//",)

    def _compute_effective_root(self, root: Path) -> Path:
        """Descend into `<project>/src/main/kotlin/` when a JVM manifest is present."""
        source_root = _kotlin_source_root(root)
        if source_root is not None:
            return source_root
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every Kotlin package under `effective_root`.

        A "Kotlin package" is any directory containing ≥1 `.kt` file whose
        `package` header parses successfully. The directory's package name
        is the dotted identifier in the FIRST alphabetical filename's
        header — never the directory basename. Files at `effective_root`
        itself never form a registered package — they attribute to `None`
        via the base's `_find_package` walk.

        Module-boundary guard: a child directory whose own contents
        include a JVM manifest is a separate Maven/Gradle module and is
        skipped. Mirrors Java's `_has_jvm_manifest` guard — without it,
        pointing the audit at a multi-project Gradle parent would walk
        every submodule's source tree and silently produce wrong output.
        """
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
                if entry.name in _KOTLIN_SCAN_SKIP:
                    continue
                if _has_kotlin_manifest(entry):
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
        """Collect every `import_header` inside the file's `import_list`.

        Kotlin wraps imports in an `import_list` node, so the base's
        `_walk_top_level_imports` (which iterates `root_node.children`)
        would see the whole list as a single statement. We override and
        walk inside the wrapper.
        """
        del file, effective_root
        import_list_node = _find_import_list(tree.root_node)
        if import_list_node is None:
            return 0, (), ()
        stmt_count = 0
        imports: list[ImportRef] = []
        for child in import_list_node.children:
            if child.type != "import_header":
                continue
            ref = _ref_from_import_header(
                child, source_bytes, self._cache_package_names,
            )
            if ref is None:
                continue
            stmt_count += 1
            imports.append(ref)
        return stmt_count, tuple(imports), ()

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Top-level public types + their direct public members + top-level functions."""
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            if child.type in _PUBLIC_TYPE_NODE_TYPES:
                _collect_public_type(child, source_bytes, symbols)
            elif child.type == "function_declaration":
                _collect_public_top_level_fun(child, source_bytes, symbols)
        return symbols

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Like the base but drop files inside a nested JVM-module subtree.

        Override mirrors the module-boundary guard in
        `_discover_package_roots`. Without it, pointing the audit at a
        Gradle multi-project parent would report N files across zero
        packages.
        """
        files = super().project_files(root, excludes)
        effective = self._cache_effective
        return [
            f for f in files
            if not _crosses_module_boundary(f, effective)
        ]

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every non-test `.kt` file directly in `package_root`, sorted by name.

        Kotlin has no single-index convention — each `.kt` file
        contributes its public symbols. Sub-dirs are independent
        packages so the walk is non-recursive. `*Test.kt` / `*Tests.kt`
        files are excluded from the facade but kept in the import graph
        via `project_files`.
        """
        try:
            entries = sorted(package_root.iterdir(), key=lambda p: p.name)
        except OSError:
            return ()
        return tuple(
            e for e in entries
            if e.is_file()
            and e.suffix.lower() == ".kt"
            and not _is_test_filename(e.name)
        )


def _is_test_filename(name: str) -> bool:
    """True if `name` ends in a JUnit / kotest test-class suffix."""
    return any(name.endswith(suffix) for suffix in _TEST_FILENAME_SUFFIXES)


def _kotlin_source_root(root: Path) -> Path | None:
    """Locate the Kotlin source root under `root`, with one-level wrapper descent.

    Returns `root/src/main/kotlin/` when a manifest is present at `root`
    and that directory exists. If `root` itself has no manifest but
    exactly one direct child does, recurse one level on that child.
    Every other arrangement returns `None` — callers fall back to
    walking `root` directly.
    """
    if _has_kotlin_manifest(root):
        return _src_main_kotlin_under(root)
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    manifest_children = [
        c for c in children
        if c.is_dir() and _has_kotlin_manifest(c)
    ]
    if len(manifest_children) == 1:
        return _src_main_kotlin_under(manifest_children[0])
    return None


def _has_kotlin_manifest(directory: Path) -> bool:
    """True iff `directory` contains any Maven/Gradle manifest."""
    return any(
        (directory / name).is_file()
        for name in _KOTLIN_MANIFESTS
    )


def _crosses_module_boundary(file: Path, effective_root: Path) -> bool:
    """True if any ancestor of `file` between it and `effective_root` carries a manifest."""
    cur = file.parent
    while cur != effective_root and effective_root in cur.parents:
        if _has_kotlin_manifest(cur):
            return True
        cur = cur.parent
    return False


def _src_main_kotlin_under(project_root: Path) -> Path | None:
    """Return `<project_root>/src/main/kotlin/` if that directory exists, else `None`."""
    candidate = project_root
    for part in _KOTLIN_SRC_PARTS:
        candidate = candidate / part
    return candidate if candidate.is_dir() else None


def _package_name_for_dir(
    entries: list[Path],
    get_parser: Any,
) -> str | None:
    """Read the `package` header from the first alphabetical `.kt` file in `entries`.

    Returns the dotted package name, or `None` if no `.kt` file in
    `entries` declares a parseable `package_header`. Deterministic by
    sort key so dir-iteration order can't make the result flaky.
    """
    kt_files = sorted(
        (e for e in entries if e.is_file() and e.suffix.lower() == ".kt"),
        key=lambda p: p.name,
    )
    for kt_file in kt_files:
        name = _read_package_header(kt_file, get_parser)
        if name is not None:
            return name
    return None


def _read_package_header(
    file_path: Path, get_parser: Any,
) -> str | None:
    """Parse `file_path` and return its `package` header as a dotted string, or `None`."""
    parser = get_parser("kotlin")
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
    for child in tree.root_node.children:
        if child.type != "package_header":
            continue
        for sub in child.children:
            if sub.type == "identifier":
                segments = _flatten_kotlin_qualified_name(sub, source_bytes)
                if segments:
                    return ".".join(segments)
        return None
    return None


def _find_import_list(root_node: Any) -> Any | None:
    """Return the `import_list` child of `root_node`, or `None` if no imports exist."""
    for child in root_node.children:
        if child.type == "import_list":
            return child
    return None


def _ref_from_import_header(
    header: Any, source_bytes: bytes, project_packages: frozenset[str],
) -> ImportRef | None:
    """Build an `ImportRef` from one `import_header` node, or `None`.

    Detects wildcard (`.*` sibling token) and alias (`import_alias`
    sibling) by scanning direct children. The qualified name lives in
    the sibling `identifier` node; flatten only that subtree so the
    alias's `type_identifier` never enters the segment list.
    """
    name_node: Any | None = None
    is_wildcard = False
    alias_name: str | None = None
    for child in header.children:
        if child.type == "identifier":
            name_node = child
        elif child.type == ".*":
            is_wildcard = True
        elif child.type == "import_alias":
            alias_name = _alias_name(child, source_bytes)
    if name_node is None:
        return None
    segments = _flatten_kotlin_qualified_name(name_node, source_bytes)
    if not segments:
        return None
    return ImportRef(
        target_module=_format_import_module(segments, is_wildcard, alias_name),
        target_package=_resolve_kotlin_import(segments, project_packages),
        line=header.start_point[0] + 1,
    )


def _alias_name(alias_node: Any, source_bytes: bytes) -> str | None:
    """Return the `type_identifier` text inside an `import_alias` node."""
    for child in alias_node.children:
        if child.type == "type_identifier":
            return _node_text(child, source_bytes)
    return None


def _flatten_kotlin_qualified_name(node: Any, source_bytes: bytes) -> list[str]:
    """Flatten a Kotlin dotted identifier into its segments.

    Kotlin grammar models the qualified path as `identifier →
    simple_identifier ('.' simple_identifier)*`. Collect every
    `simple_identifier` leaf in left-to-right order; punctuation `.`
    tokens are ignored.
    """
    segments: list[str] = []
    for child in node.children:
        if child.type == "simple_identifier":
            segments.append(_node_text(child, source_bytes))
    return segments


def _format_import_module(
    segments: list[str], is_wildcard: bool, alias_name: str | None,
) -> str:
    """Build a display string for `ImportRef.target_module`."""
    body = ".".join(segments)
    if is_wildcard:
        body = body + ".*"
    if alias_name is not None:
        body = body + " as " + alias_name
    return body


def _resolve_kotlin_import(
    segments: list[str], project_packages: frozenset[str],
) -> str | None:
    """Resolve qualified-name segments to a registered package, longest-first.

    Mirrors Java's resolver: try every prefix length, longest first,
    return the first one that matches a registered package. Handles
    qualified-type, wildcard, member, and aliased imports uniformly.
    """
    for n in range(len(segments), 0, -1):
        candidate = ".".join(segments[:n])
        if candidate in project_packages:
            return candidate
    return None


def _collect_public_type(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add a top-level public type's symbols (class + public members) to `symbols`.

    Visibility default is public (no `visibility_modifier` child →
    public). Public members register under their own name; secondary
    constructors fold into the class entry via `setdefault`. The
    primary-constructor parameter count seeds the class entry before
    the body walk, so a `data class D(val x: Int, val y: String)`
    registers as `D` with param-count 2 even though it has no body.
    """
    if _visibility_excludes(node):
        return
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _node_text(name_node, source_bytes)

    primary = node.child_by_field_name("primary_constructor")
    if primary is not None:
        symbols.setdefault(class_name, _count_primary_constructor_params(primary))

    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.children:
            _collect_public_member(member, class_name, source_bytes, symbols)

    symbols.setdefault(class_name, 0)


def _collect_public_member(
    member: Any, class_name: str, source_bytes: bytes,
    symbols: dict[str, int],
) -> None:
    """Add one direct member of a public type to `symbols` (when itself public)."""
    if _visibility_excludes(member):
        return
    if member.type == "function_declaration":
        name_node = member.child_by_field_name("name")
        if name_node is None:
            return
        symbols.setdefault(
            _node_text(name_node, source_bytes),
            _count_function_value_parameters(member),
        )
    elif member.type == "secondary_constructor":
        symbols.setdefault(
            class_name, _count_function_value_parameters(member),
        )


def _collect_public_top_level_fun(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add a top-level public function (free function) to `symbols`.

    Kotlin permits free functions at file scope; Java does not. First
    occurrence wins on param count via `setdefault` — matches the
    aggregator's policy across facade source files.
    """
    if _visibility_excludes(node):
        return
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    symbols.setdefault(
        _node_text(name_node, source_bytes),
        _count_function_value_parameters(node),
    )


def _visibility_excludes(node: Any) -> bool:
    """True iff `node` has a `visibility_modifier` whose token is non-public.

    Walks `modifiers` group children and matches by node type — only
    `visibility_modifier` counts. Other modifier categories
    (`class_modifier`, `function_modifier`, `inheritance_modifier`,
    `member_modifier`) never affect visibility. Declarations with no
    `modifiers` child at all are public by default — return False.
    """
    for child in node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type != "visibility_modifier":
                continue
            for token in mod.children:
                if token.type in _NON_PUBLIC_VISIBILITY:
                    return True
        return False
    return False


def _count_primary_constructor_params(primary: Any) -> int:
    """Count `class_parameter` children of a `primary_constructor` node."""
    return sum(
        1 for child in primary.children
        if child.type == "class_parameter"
    )


def _count_function_value_parameters(node: Any) -> int:
    """Count `parameter` slots on a function / secondary-constructor declaration.

    The grammar wraps parameters in `function_value_parameters`. Each
    `parameter` child is one slot. Kotlin has no varargs node-type — the
    `vararg` modifier sits on the parameter itself, so the count is
    correct without special handling.
    """
    params = node.child_by_field_name("parameters")
    if params is None:
        return 0
    return sum(
        1 for c in params.children
        if c.type == "parameter"
    )
