"""Java adapter: tree-sitter only.

A "package" here is any directory under the effective JVM source root that
contains at least one `.java` file directly (excluding `target/`, `build/`,
`out/`, hidden dirs). Packages are named by their POSIX-relative path with
`/` replaced by `.` — so `src/main/java/com/foo/bar/` registers as
`com.foo.bar`. Files directly at the effective root attribute to
`package=None` — included in n_files/total_loc/oversized totals, excluded
from the package-level graph (mirrors Python/JS/TS/Go/Rust).

The effective root depends on the layout:

1. `root/pom.xml` declares `<modules>` → effective = `root` (Maven
   multi-module workspace). Each declared `<module>` dir is walked under
   its own `src/main/java/`; bare dotted package names are unioned across
   modules.
2. `root/<manifest>` + `root/src/main/java/` → effective =
   `root/src/main/java/` (single-module).
3. `root/` has no manifest but a single direct child does → recurse one
   level on that child (mirrors Go/Rust one-level descent).

Any other shape (Gradle parent, sbt root, manifest without
`src/main/java/`, multi-manifest siblings with no `<modules>`) yields a
`None` source root — the audit walks `root` directly, the package-roots
walker refuses to descend into nested manifests, and the layout-warning
heuristic fires.

Import resolution: one branch — any qualified import whose dotted prefix is
a registered package name. The resolver walks prefixes longest-first
against `_cache_package_names`. In Maven workspace mode the registered
set unions across every `<module>`, so a `com.x.a` → `com.x.b` import
crosses modules naturally without resolver changes. This handles all
four Java import forms uniformly:

  import a.b.C;                  segments `a.b.C` → match `a.b`
  import a.b.*;                  segments `a.b`   → match `a.b`
  import static a.b.C.member;    segments `a.b.C.member` → match `a.b`
  import static a.b.C.*;         segments `a.b.C` → match `a.b`
  import a.b.Outer.Inner;        segments `a.b.Outer.Inner` → match `a.b`

External imports (`java.util.List`, `org.unknown.Thing`) yield
`target_package = None` because their prefixes don't match any registered
package.

Public-symbol extraction collects top-level `public` types
(`class_declaration`, `interface_declaration`, `enum_declaration`,
`record_declaration`) plus their direct public `method_declaration` and
`constructor_declaration` members. Methods register as separate symbols;
constructors fold into the class entry via `setdefault` — the first public
constructor's arity wins. Classes without any public constructor get
param-count 0.

`in_function_imports` is always `()` — the base's
`_walk_top_level_imports` only sees `root_node.children`, mirroring the
Go/Rust policy.

What v1 deliberately does NOT do (preserved current behavior; not flagged
in QA):

- Gradle multi-project / composite-build workspaces (`settings.gradle.kts`
  with `include(...)`). Non-declarative DSL manifests need a separate
  ticket; v1 ships Maven workspace only.
- sbt multi-module (`lazy val a = project.in(file("a"))`) — same Scala DSL
  reasoning; out of scope.
- Aggregator-only Maven modules (packaging=pom child without
  `src/main/java/`) are silently skipped. Matches Maven's own treatment.
- Polyglot or profile-conditional source dirs (`src/main/java-11/`,
  `src/main/scala/`, custom `<sourceDirectory>`) — modules using ONLY
  those are silently skipped from `n_modules`; a stderr tripwire surfaces
  the skip count.
- Module-name collisions across `<modules>` (two modules declaring
  `com.x.util`) merge into one graph node — itself a design smell. File
  attribution becomes deterministic via sorted-by-name `dict(pairs)` at
  `_base.py:200`: the lexically-larger path wins.
- JPMS `module-info.java` modules.
- Annotation processors and generated `.java` files (treated as ordinary
  sources).
- `package-info.java` files (no special treatment; included as ordinary
  `.java` files; the public-symbol walk finds nothing in them).
- Nested public types — `public static class Inner` inside `public class
  Outer` is NOT collected as a separate symbol. The walk is top-level
  only, matching Go's `tree.root_node.children` convention.
- `public @interface Marker` (annotation type declarations) — not in the
  captured node-type set.
- Lambdas (`x -> ...`) and anonymous inner classes (`new Runnable() {…}`)
  inside method bodies are not collected (consistent with the
  always-empty `in_function_imports`).
- `sealed` / `permits` keywords are tolerated by the parser but the
  adapter does not enumerate `permits` types.
- `var` local declarations are irrelevant — only public top-level types
  feed the facade.

Scope:
  This adapter audits Java; it is one of ten adapters registered in
  `architecture/core/runner.py::_ADAPTERS` (see README's
  `## Supported Languages` section for the full list). Maven multi-module
  is now supported via the `<modules>` XML element; Gradle multi-project
  and sbt remain explicit-fail (point at a single submodule). C, C++,
  and Bash are architecture-unsupported across all adapters; the README
  documents why.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..core.protocol import (
    ImportRef,
    InFunctionImport,
)
from ._base import (
    BaseTreeSitterAdapter,
    _node_text,
)


_JVM_MANIFESTS: frozenset[str] = frozenset({
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
})
# Why: filenames that mark a Maven (`pom.xml`) or Gradle (`build.gradle`,
# `settings.gradle`, plus the Kotlin DSL `.kts` variants) project root.
# Their presence triggers descent into the conventional `src/main/java/`
# source root.

_JAVA_SRC_PARTS: tuple[str, str, str] = ("src", "main", "java")
# Why: the conventional source-root path under a Maven/Gradle project.
# After locating the manifest, the adapter descends three more levels so
# package discovery starts at the directory whose subdirs map directly to
# package names (`com/foo/bar/` → `com.foo.bar`).

_JAVA_SCAN_SKIP: frozenset[str] = frozenset({
    "target", "build", "out",
})
# Why: build-output directories that must never count as Java packages —
# `target/` for Maven, `build/` for Gradle, `out/` for IntelliJ IDEA. Kept
# for symmetry with Go/Rust even though `iter_source_files` already
# filters them through `Config.SKIP_DIRS`. Hidden dirs (`.gradle`, `.idea`)
# are filtered separately via the leading-dot check.

_TEST_FILENAME_SUFFIXES: tuple[str, ...] = (
    "Test.java", "IT.java",
)
# Why: JUnit + Maven Failsafe conventions — `*Test.java` for unit tests
# and `*IT.java` for integration tests. Excluded from the public facade
# only (the import graph still sees them).

_PUBLIC_TYPE_NODE_TYPES: frozenset[str] = frozenset({
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
})
# Why: the top-level type-declaration node kinds whose `public` form feeds
# the package facade. `annotation_type_declaration` is intentionally
# absent — see the "deliberately does NOT do" list in the module
# docstring.

_MAVEN_POM_FILENAME: str = "pom.xml"
# Why: the Maven manifest. Workspace detection reads `<modules>` from this
# file only — Gradle parents (`build.gradle.kts`, `settings.gradle.kts`)
# are explicit-out-of-scope.

_MAVEN_MODULES_XPATH: str = ".//{*}modules/{*}module"
# Why: ElementTree XPath with `{*}` wildcard namespace handles the
# `xmlns="http://maven.apache.org/POM/4.0.0"` declaration without
# manual namespace registration (Python 3.8+ feature).


@dataclass(frozen=True)
class _JavaModule:
    """One Maven workspace module resolved on disk."""
    module_root: Path     # contains the module's pom.xml
    source_root: Path     # module_root / src/main/java/ (guaranteed to exist)


class JavaAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for Java projects."""

    LANGUAGE = "java"
    EXTENSIONS = frozenset({".java"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//",)

    def __init__(self) -> None:
        super().__init__()
        self._cache_modules: tuple[_JavaModule, ...] = ()
        self._cache_module_count: int = 0
        self._cache_registered_module_roots: frozenset[Path] = frozenset()
        self._cache_dropped_out_of_root_count: int = 0
        self._cache_skipped_polyglot_count: int = 0

    def _compute_effective_root(self, root: Path) -> Path:
        """Workspace shape wins; else descend into `<project>/src/main/java/`.

        Workspace mode: `root/pom.xml` declares `<modules>` → return
        `root` unchanged (each module's source root is walked separately).
        Single-module: descend via `_jvm_source_root` (one-level wrapper
        descent included).
        """
        if _read_maven_modules(root / _MAVEN_POM_FILENAME) is not None:
            return root
        source_root = _jvm_source_root(root)
        if source_root is not None:
            return source_root
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every Java package under `effective_root`.

        Writes `self._cache_modules` as a side effect — see the plan's
        "Critical ordering" section. `_post_layout` reads that cache to
        populate derived state.

        Single-module mode: any directory containing ≥1 `.java` file
        directly registers as a package (POSIX-relative path with `/` →
        `.`). Child dirs carrying a JVM manifest are skipped — pointing
        the audit at a multi-module parent (no `src/main/java/`) walks
        zero packages and trips the layout-warning heuristic.

        Maven workspace mode: each declared `<modules>` entry contributes
        every directory under its `src/main/java/` that contains `.java`
        files. Names are bare dotted (`com.foo.bar`); collisions across
        modules merge into one graph node. Nested child manifests are
        skipped unless they're in the declared modules list (defends
        against vendored/nested modules outside the workspace).
        """
        modules_names = _read_maven_modules(
            effective_root / _MAVEN_POM_FILENAME,
        )
        if modules_names is not None:
            modules, dropped, skipped = _expand_maven_modules(
                effective_root, modules_names,
            )
            self._cache_modules = modules
            self._cache_dropped_out_of_root_count = dropped
            self._cache_skipped_polyglot_count = skipped
            registered = frozenset(m.module_root for m in modules)
            return _discover_workspace_pairs(modules, registered)
        self._cache_modules = ()
        self._cache_dropped_out_of_root_count = 0
        self._cache_skipped_polyglot_count = 0
        return _discover_single_module_pairs(effective_root)

    def _post_layout(self, effective_root: Path, audit_root: Path) -> None:
        """Populate derived caches; emit a stderr tripwire on silent module drops."""
        del effective_root, audit_root
        self._cache_module_count = len(self._cache_modules)
        self._cache_registered_module_roots = frozenset(
            m.module_root for m in self._cache_modules
        )
        if self._cache_skipped_polyglot_count > 0:
            print(
                f"java adapter: {self._cache_skipped_polyglot_count} "
                f"declared <modules> dropped (no src/main/java/) — "
                f"likely polyglot or profile-conditional sources",
                file=sys.stderr,
            )

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Collect every top-level `import_declaration`. One declaration = one statement."""
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
        ref = _ref_from_import_declaration(
            node, source_bytes, self._cache_package_names,
        )
        return [ref] if ref is not None else []

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Top-level public types plus their direct public methods/constructors."""
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            _collect_public_type(child, source_bytes, symbols)
        return symbols

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Like the base but drop files inside a nested JVM-module subtree.

        Override mirrors the module-boundary guard in
        `_discover_package_roots` so the file count stays consistent with
        the package count. Workspace-aware: files inside a declared Maven
        module are kept; files inside an unregistered nested manifest are
        dropped. Single-module mode passes an empty registered set so the
        legacy "skip every nested manifest" behavior is preserved.
        """
        files = super().project_files(root, excludes)
        effective = self._cache_effective
        registered = self._cache_registered_module_roots
        return [
            f for f in files
            if not _crosses_module_boundary(f, effective, registered)
        ]

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every non-test `.java` file directly in `package_root`, sorted by name.

        Override because Java has no single-index convention — each `.java`
        file contributes its public types. Sub-dirs are independent
        packages so the walk is non-recursive. `*Test.java` / `*IT.java`
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
            and e.suffix.lower() == ".java"
            and not _is_test_filename(e.name)
        )


def _is_test_filename(name: str) -> bool:
    """True if `name` ends in a JUnit/Failsafe test-class suffix."""
    return any(name.endswith(suffix) for suffix in _TEST_FILENAME_SUFFIXES)


def _jvm_source_root(root: Path) -> Path | None:
    """Locate the JVM source root under `root`, with one-level wrapper descent.

    Returns `root/src/main/java/` when a manifest is present at `root` and
    that directory exists. If `root` itself has no manifest but exactly one
    direct child does, recurse one level on that child. Every other
    arrangement (manifest at `root` without `src/main/java/`, multiple
    manifest-bearing siblings, no manifest anywhere) returns `None` —
    callers fall back to walking `root` directly.
    """
    if _has_jvm_manifest(root):
        return _src_main_java_under(root)
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    manifest_children = [
        c for c in children
        if c.is_dir() and _has_jvm_manifest(c)
    ]
    if len(manifest_children) == 1:
        return _src_main_java_under(manifest_children[0])
    return None


def _has_jvm_manifest(directory: Path) -> bool:
    """True iff `directory` contains any Maven/Gradle manifest."""
    return any(
        (directory / name).is_file()
        for name in _JVM_MANIFESTS
    )


def _crosses_module_boundary(
    file: Path,
    effective_root: Path,
    registered_module_roots: frozenset[Path],
) -> bool:
    """True if `file` lies inside an unregistered nested JVM-manifest subtree.

    Walks up from `file.parent` toward `effective_root`. If an ancestor
    carries a JVM manifest AND is NOT in `registered_module_roots`, the
    file is on the wrong side of a module boundary and gets dropped.
    Workspace mode passes the declared `<modules>` set; single-module mode
    passes `frozenset()` and the predicate reduces to today's
    "any manifest ancestor crosses" rule.
    """
    cur = file.parent
    while cur != effective_root and effective_root in cur.parents:
        if _has_jvm_manifest(cur) and cur not in registered_module_roots:
            return True
        cur = cur.parent
    return False


def _src_main_java_under(project_root: Path) -> Path | None:
    """Return `<project_root>/src/main/java/` if that directory exists, else `None`.

    A manifest-bearing project without a real `src/main/java/` (e.g., a
    multi-module parent POM whose Java lives in submodules) returns
    `None`. Callers fall back to walking `project_root` directly, where
    the package-roots module-boundary guard then refuses to descend into
    sibling modules.
    """
    candidate = project_root
    for part in _JAVA_SRC_PARTS:
        candidate = candidate / part
    return candidate if candidate.is_dir() else None


def _dir_has_java_files(entries: list[Path]) -> bool:
    """True if any entry in `entries` is a regular `.java` file."""
    return any(
        e.is_file() and e.suffix.lower() == ".java"
        for e in entries
    )


def _read_maven_modules(pom_xml: Path) -> tuple[str, ...] | None:
    """Return the `<modules>/<module>` text content, or `None` for non-workspace POMs.

    Uses ElementTree's `{*}` wildcard namespace XPath so the Maven default
    `xmlns="http://maven.apache.org/POM/4.0.0"` declaration is handled
    without manual namespace registration. Returns `None` when the file
    is absent, malformed, or has no `<modules>` element. Returns `()`
    (empty tuple) when `<modules>` is declared but empty — caller treats
    as "workspace shape with zero usable members".
    """
    if not pom_xml.is_file():
        return None
    try:
        tree = ET.parse(pom_xml)  # noqa: S314 — local fixtures, not untrusted XML
    except (ET.ParseError, OSError):
        return None
    modules = tree.findall(_MAVEN_MODULES_XPATH)
    if not modules:
        return None
    return tuple(
        m.text.strip() for m in modules
        if m.text is not None and m.text.strip()
    )


def _expand_maven_modules(
    workspace_root: Path, names: tuple[str, ...],
) -> tuple[tuple[_JavaModule, ...], int, int]:
    """Resolve `<modules>` entries on disk; return (modules, dropped, skipped).

    `dropped` counts entries that resolve outside `workspace_root` (silent
    drop, surfaced via the runner tripwire). `skipped` counts entries that
    exist as dirs but lack `src/main/java/` — polyglot/profile-conditional
    sources or pure aggregator modules. Both kinds are silent omissions
    from `n_modules`; the polyglot count drives the per-adapter stderr
    tripwire in `_post_layout`.
    """
    seen: dict[Path, _JavaModule] = {}
    dropped = 0
    skipped = 0
    for name in names:
        candidate = (workspace_root / name).resolve()
        if not _is_within(candidate, workspace_root.resolve()):
            dropped += 1
            continue
        module_root = workspace_root / name
        source_root = module_root / _JAVA_SRC_PARTS[0] / _JAVA_SRC_PARTS[1] / _JAVA_SRC_PARTS[2]
        if not source_root.is_dir():
            skipped += 1
            continue
        if module_root in seen:
            continue
        seen[module_root] = _JavaModule(
            module_root=module_root, source_root=source_root,
        )
    return (
        tuple(sorted(seen.values(), key=lambda m: m.module_root.name)),
        dropped,
        skipped,
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
    """Today's single-module DFS — preserved verbatim for the single-pom fast path."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [effective_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if cur != effective_root and _dir_has_java_files(entries):
            rel = cur.relative_to(effective_root).as_posix()
            pairs.append((cur, rel.replace("/", ".")))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _JAVA_SCAN_SKIP:
                continue
            if _has_jvm_manifest(entry):
                continue
            stack.append(entry)
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _discover_workspace_pairs(
    modules: tuple[_JavaModule, ...],
    registered_module_roots: frozenset[Path],
) -> tuple[tuple[Path, str], ...]:
    """Discover `(dir, dotted-name)` pairs across every Maven module's source root."""
    pairs: list[tuple[Path, str]] = []
    for module in modules:
        pairs.extend(_walk_module_packages(module, registered_module_roots))
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _walk_module_packages(
    module: _JavaModule,
    registered_module_roots: frozenset[Path],
) -> list[tuple[Path, str]]:
    """DFS one Maven module's `src/main/java/`, yielding dotted-name pairs."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [module.source_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if cur != module.source_root and _dir_has_java_files(entries):
            rel = cur.relative_to(module.source_root).as_posix()
            pairs.append((cur, rel.replace("/", ".")))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _JAVA_SCAN_SKIP:
                continue
            if _has_jvm_manifest(entry) and entry not in registered_module_roots:
                continue
            stack.append(entry)
    return pairs


def _ref_from_import_declaration(
    decl: Any, source_bytes: bytes, project_packages: frozenset[str],
) -> ImportRef | None:
    """Build an `ImportRef` from one `import_declaration` node, or `None`.

    Detects `static` and wildcard (`.*`) by scanning direct children — the
    Java grammar models them as anonymous keyword/punctuation tokens
    siblings of the qualified name node, not as named fields. The
    qualified name is `scoped_identifier` or bare `identifier`.
    """
    is_static = False
    is_wildcard = False
    name_node: Any | None = None
    for child in decl.children:
        if child.type == "static":
            is_static = True
        elif child.type in ("*", "asterisk"):
            is_wildcard = True
        elif child.type in ("scoped_identifier", "identifier"):
            name_node = child
    if name_node is None:
        return None
    segments = _flatten_qualified_name(name_node, source_bytes)
    if not segments:
        return None
    return ImportRef(
        target_module=_format_import_module(segments, is_static, is_wildcard),
        target_package=_resolve_java_import(segments, project_packages),
        line=decl.start_point[0] + 1,
    )


def _flatten_qualified_name(node: Any, source_bytes: bytes) -> list[str]:
    """Flatten a Java qualified-name node into its dotted segments.

    Uses the grammar's named fields (`scope`, `name`) on `scoped_identifier`
    so whitespace and `.` punctuation never enter the segment list. Falls
    back to a bare `identifier` leaf.
    """
    if node.type == "identifier":
        return [_node_text(node, source_bytes)]
    if node.type == "scoped_identifier":
        scope = node.child_by_field_name("scope")
        name = node.child_by_field_name("name")
        segs = (
            _flatten_qualified_name(scope, source_bytes)
            if scope is not None else []
        )
        if name is not None:
            segs.append(_node_text(name, source_bytes))
        return segs
    return []


def _format_import_module(
    segments: list[str], is_static: bool, is_wildcard: bool,
) -> str:
    """Build a display string for `ImportRef.target_module`."""
    body = ".".join(segments)
    if is_wildcard:
        body = body + ".*"
    if is_static:
        body = "static " + body
    return body


def _resolve_java_import(
    segments: list[str], project_packages: frozenset[str],
) -> str | None:
    """Resolve qualified-name segments to a registered package, longest-first.

    Java's import grammar doesn't tell the resolver whether the trailing
    identifier is a class, a static member, or an inner type — so the
    resolver tries every prefix length, longest first, and returns the
    first one that matches a registered package. Mirrors Rust's
    `_resolve_rust_use` policy and handles all four import forms plus the
    inner-class case uniformly.
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

    The class symbol's param count is determined by the FIRST public
    constructor encountered (via `setdefault`); if no public constructor
    exists, the class is registered with param-count 0 after the body
    walk. Public methods become separate entries keyed by method name.
    """
    if node.type not in _PUBLIC_TYPE_NODE_TYPES:
        return
    if not _is_public(node):
        return
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _node_text(name_node, source_bytes)
    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.children:
            _collect_public_member(
                member, class_name, source_bytes, symbols,
            )
    symbols.setdefault(class_name, 0)


def _collect_public_member(
    member: Any, class_name: str, source_bytes: bytes,
    symbols: dict[str, int],
) -> None:
    """Add one direct member of a public type to `symbols` (when itself public).

    Methods register under their own name; constructors fold into the
    enclosing class name. Both use `setdefault` so the first occurrence
    wins on param count — deterministic across body iteration order.
    """
    if not _is_public(member):
        return
    if member.type == "method_declaration":
        name_node = member.child_by_field_name("name")
        if name_node is None:
            return
        symbols.setdefault(
            _node_text(name_node, source_bytes),
            _count_formal_params(member),
        )
    elif member.type == "constructor_declaration":
        symbols.setdefault(class_name, _count_formal_params(member))


def _is_public(node: Any) -> bool:
    """True iff `node` has a `modifiers` child whose children include `public`.

    Java models access modifiers as a single `modifiers` group node among
    the declaration's direct children. The group's children are anonymous
    keyword tokens (`public`, `static`, `final`, …); we scan for the
    `public` token specifically. Declarations without an explicit
    modifier (package-private) return False.
    """
    for child in node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type == "public":
                return True
        return False
    return False


def _count_formal_params(method_node: Any) -> int:
    """Count parameter slots on a `method_declaration` or `constructor_declaration`.

    `formal_parameter` is a regular argument; `spread_parameter` is the
    varargs form (`String... args`) — each counts as one slot, matching
    Go's variadic-counts-as-one policy.
    """
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return 0
    return sum(
        1 for c in params.named_children
        if c.type in ("formal_parameter", "spread_parameter")
    )
