"""C# adapter: tree-sitter only.

C# projects are anchored by `.sln` (solution) or `.csproj` (project) manifests
rather than the JVM `src/main/<lang>/` convention. The effective root depends
on the layout:

1. `root/<*.sln>` with ≥1 `Project(...)` row pointing at a `.csproj` → effective = root
   (solution workspace; each project is walked separately).
2. `root/<*.csproj>` present → effective = root (single C# project; packages
   nest directly under the project root, not under `src/main/cs/`).
3. `root/<child>/<*.csproj>` for exactly one child → effective = child.
4. Any other shape (no `.csproj` anywhere, multi-`.csproj` siblings without
   `.sln`, stray empty `.sln`) → effective = root; the package walker
   explicit-fails on multi-`.csproj` siblings.

C# diverges from Java/Kotlin in three load-bearing ways:

1. **Two namespace-declaration forms**: block-scoped `namespace X { … }`
   (`namespace_declaration` with `body:` field of type `declaration_list`),
   and file-scoped `namespace X;` (`file_scoped_namespace_declaration`,
   whose using-directives and type declarations live as DIRECT CHILDREN of
   the namespace node — there is no body field). Both forms expose `name:`
   as an `identifier` (single segment) or `qualified_name` (recursive
   `qualifier:` / `name:` shape). The walker descends into both forms
   uniformly.

2. **Package identity comes from the namespace header**, not from the
   directory basename — same model as Kotlin and Scala. A directory becomes
   a registered package iff at least one `.cs` file directly inside it
   declares a parseable namespace. The dir's package name is the dotted
   identifier from the FIRST alphabetical filename's namespace declaration.
   Files with no namespace attribute to `package=None`.

3. **`using` directives may appear at file scope OR inside either namespace
   form**. The walker collects from both locations and counts each
   `using_directive` as one top-level statement, matching the Java/Kotlin
   "one declaration = one statement" rule. Reusing the base's
   `_walk_top_level_imports` would mis-count: it would treat a whole
   `namespace_declaration` as one statement regardless of how many usings
   live inside its body.

Import resolution: longest-first prefix match across registered package names
(identical in shape to Java's `_resolve_java_import`). C# import forms:

  using a.b.C;                  segments `a.b.C`        → match `a.b`
  using static a.b.C.member;    segments `a.b.C.member` → match `a.b`
  using X = a.b.C;              segments `a.b.C`        → match `a.b`

C# has NO wildcard `using a.b.*` form — the namespace import IS the wildcard.

`in_function_imports` is always `()` — C# disallows `using` directives inside
method bodies grammatically. Local `using` *statements* are resource-disposal
(`using var x = …`) and a different node type entirely.

Public-symbol extraction collects every top-level `public` type
(`class_declaration`, `interface_declaration`, `struct_declaration`,
`record_declaration`, `enum_declaration`) — directly at file scope OR inside
either namespace form. For each public class, public `method_declaration` and
`constructor_declaration` members register separately; constructors fold into
the class entry via `setdefault`. Generated files (`*.g.cs`, `*.designer.cs`,
case-insensitive) and test files (`*Test.cs`, `*Tests.cs`) are excluded from
the facade — they remain in `project_files` so the import graph still walks
them.

What v1 deliberately does NOT do (preserved current behavior; not flagged in QA):

- Multi-dir namespace aggregation (`App.PkgA` spanning both `Foo/` and `Bar/`).
- `<TargetFrameworks>` multi-targeting in `.csproj`.
- Conditional compilation gating (`#if NETSTANDARD`).
- `partial class` declarations bridging facade content across files.
- Source generators emitting `.cs` files into `obj/` (`obj/` is in
  `Config.SKIP_DIRS` and pruned by the source walker).
- Top-level statements in `Program.cs` (C# 9+ Minimal API style).
- `record_declaration` primary-constructor arity — registered with param-count 0
  (v1 parity with Java records). The grammar exposes `parameters:` on
  `record_declaration`; v1 deliberately does not read it.
- `global using` directives — treated as ordinary usings; the `global` token
  doesn't change resolution.
- Aliased imports `using X = a.b.C;` — alias name dropped from `target_module`
  display (matches Kotlin's `as`-suffix policy).
- Nested types inside a public class (top-level only).
- `AssemblyInfo.cs` / `*.AssemblyAttributes.cs` filename filtering — handled
  by namespace-driven attribution and the public-modifier check, not by
  filename suffix.
- Solution Filter (`.slnf`) and `.slnx` formats — out of scope; `.sln` only.
- Non-C# rows in `.sln` (`.vcxproj`, `.shproj`, `.vbproj`, `.fsproj`, solution
  folders) are filtered by the project path's `.csproj` extension; only rows
  pointing at a `.csproj` engage workspace mode. The project-type GUID
  preceding each row is ignored — Microsoft has emitted multiple SDK-style
  GUIDs across template versions and hardcoding the set was fragile.
- Legacy Windows-1252 `.sln` encoding is handled via fallback after
  `utf-8-sig` fails; the fallback content is sanity-checked for either
  `Project(` or the `Microsoft Visual Studio Solution File` header before
  the parser runs.
- Parent-relative `..\\Shared\\Shared.csproj` paths are silently dropped as
  out-of-root (workspace members must live under the audit root).
- Namespace collisions across solution projects merge into one graph node
  (itself a design smell).
- Generic constraints (`where T : class`) — irrelevant to type/symbol counting.

Scope:
  This adapter audits C#; it is one of ten adapters registered in
  `architecture/core/runner.py::_ADAPTERS` (see README's
  `## Supported Languages` section for the full list). `.sln`
  workspaces are now supported; `.slnf` filter files, the new `.slnx`
  format, multi-`<TargetFrameworks>` with `#if`-gated sources, C# 9
  top-level statements in `Program.cs` (Minimal API style), `global
  using` literals, `record` primary-constructor arity, and `partial
  class` bodies bridging facade across files all remain out of scope.
  C, C++, and Bash are architecture-unsupported across all adapters;
  the README documents why.
"""
from __future__ import annotations

import re
from collections import Counter
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


_CSHARP_PROJECT_EXTENSIONS: frozenset[str] = frozenset({".csproj"})
# Why: file-extension set that identifies a C# project manifest. `.sln`
# solution files mark the multi-project root but are not project manifests
# themselves; the descent rule probes only for `.csproj`.

_CSHARP_SCAN_SKIP: frozenset[str] = frozenset({
    "bin", "obj",
})
# Why: build-output directories that must never count as C# packages. Both
# are already in `Config.SKIP_DIRS` (so the source walker prunes them) but
# the adapter mirrors them on its own descent for symmetry with the JVM
# adapters' explicit skip sets. Hidden dirs (`.vs`, `.idea`) are filtered
# separately via the leading-dot check.

_TEST_FILENAME_SUFFIXES: tuple[str, ...] = (
    "Test.cs", "Tests.cs",
)
# Why: xUnit / NUnit / MSTest conventions — `*Test.cs` and `*Tests.cs`.
# Excluded from the public facade only; the import graph still walks them.

_GENERATED_FILENAME_SUFFIXES: tuple[str, ...] = (
    ".g.cs", ".designer.cs",
)
# Why: codegen filename markers. `.g.cs` is the Roslyn source-generator
# suffix; `.designer.cs` is the WinForms / WPF designer codegen suffix.
# Matched case-insensitively to absorb Windows tooling that may emit
# `Form.Designer.cs` (capitalized D).

_PUBLIC_TYPE_NODE_TYPES: frozenset[str] = frozenset({
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "record_declaration",
    "enum_declaration",
})
# Why: the top-level type-declaration node kinds whose `public` form feeds
# the package facade. The C# grammar models each as its own node type.

_NAMESPACE_DECL_NODE_TYPES: frozenset[str] = frozenset({
    "namespace_declaration",
    "file_scoped_namespace_declaration",
})
# Why: the two namespace-declaration shapes the C# 10+ grammar emits.
# Block form has `body:` field (`declaration_list`); file-scoped form has
# its inner usings + types as direct children with no body field.

_SLN_EXTENSION: str = ".sln"
# Why: the Visual Studio solution file extension. Presence in `root`
# marks workspace mode (precedence over single-`.csproj` descent).

_SLN_PROJECT_LINE_PREFIX: str = "Project("
# Why: the literal prefix of every project row in a `.sln`. Used as a
# cheap pre-filter before the regex match — saves a regex on every
# non-project line.

_SLN_HEADER_FRAGMENT: str = "Microsoft Visual Studio Solution File"
# Why: sanity-check fragment for the cp1252 fallback path — distinguishes
# a real solution file decoded as Windows-1252 from a binary blob whose
# cp1252 round-trip yields garbage.

_SLN_PROJECT_LINE_PATTERN: re.Pattern[str] = re.compile(
    r'^Project\("\{[0-9A-Fa-f-]+\}"\)\s*=\s*'
    r'"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"\{[0-9A-Fa-f-]+\}"'
)
# Why: matches `Project("{<type-guid>}") = "<name>", "<rel-path>",
# "{<project-guid>}"` and captures (name, rel-path). The two GUIDs are
# part of the format but not load-bearing — project KIND is determined
# by the `<rel-path>` file extension downstream (`_CSPROJ_EXTENSION`).

_CSPROJ_EXTENSION: str = ".csproj"
# Why: the file-extension discriminator for C# project rows in a `.sln`.
# Replaces a hardcoded GUID set — Microsoft has emitted multiple
# SDK-style project-type GUIDs across template versions (canonical
# `{9A19103F-...-845BC087BFD0}` AND `{9A19103F-...-9A1E7A4F7556}` and
# probably more in the future), and the legacy `{FAE04EC0-...}` for
# pre-SDK projects. The path's extension is the canonical, stable
# signal that the row represents a C# project; solution folders, VB,
# F#, C++, and shared-project rows all carry a different extension
# (or no real file path at all).


@dataclass(frozen=True)
class _CSharpSolutionProject:
    """One Visual Studio solution project resolved on disk."""
    project_root: Path     # dir containing the `.csproj`
    csproj_path: Path      # the `.csproj` file itself


class CSharpAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for C# projects."""

    LANGUAGE = "c_sharp"
    EXTENSIONS = frozenset({".cs"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//",)

    def __init__(self) -> None:
        super().__init__()
        self._cache_solution_projects: tuple[_CSharpSolutionProject, ...] = ()
        self._cache_module_count: int = 0
        self._cache_registered_module_roots: frozenset[Path] = frozenset()
        self._cache_dropped_out_of_root_count: int = 0

    def _compute_effective_root(self, root: Path) -> Path:
        """Workspace precedence: `.sln` with >=1 `.csproj` project wins.

        Stray `.sln` with no `.csproj` rows (e.g. only `.vcxproj` C++
        projects or only solution folders) falls through to
        single-`.csproj` descent.
        """
        if _solution_projects_at(root):
            return root
        source_root = _csharp_source_root(root)
        if source_root is not None:
            return source_root
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every C# package under `effective_root`.

        Writes `self._cache_solution_projects` as a side effect — see the
        plan's "Critical ordering" section. `_post_layout` reads that
        cache to populate the registered-module-roots set.

        Single-module mode: any directory containing ≥1 `.cs` file whose
        namespace declaration parses successfully registers as a package
        (named by the most-common namespace; alphabetical filename
        tiebreaker). Child dirs carrying `.csproj` are skipped.

        Workspace mode: each declared solution project is walked as if it
        were its own single-module audit, with `project_root` as the
        inner effective root. Names are bare dotted namespaces;
        collisions across projects merge into one graph node per
        locked-decision #2.
        """
        solution_projects = _solution_projects_at(effective_root)
        if solution_projects is not None:
            self._cache_solution_projects = solution_projects
            self._cache_dropped_out_of_root_count = 0
            registered = frozenset(p.project_root for p in solution_projects)
            return _discover_workspace_pairs(
                solution_projects, registered, self._get_parser,
            )
        self._cache_solution_projects = ()
        self._cache_dropped_out_of_root_count = 0
        return _discover_single_project_pairs(effective_root, self._get_parser)

    def _post_layout(self, effective_root: Path, audit_root: Path) -> None:
        """Populate derived caches read by `project_files` + the runner tripwire."""
        del effective_root, audit_root
        self._cache_module_count = len(self._cache_solution_projects)
        self._cache_registered_module_roots = frozenset(
            p.project_root for p in self._cache_solution_projects
        )

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Walk file-scope `using_directive`s plus those inside either namespace form.

        Cannot reuse `_walk_top_level_imports`: that helper counts one
        statement per top-level node, so a namespace wrapping three usings
        would yield three refs but a count of one. The C# semantic is
        three statements, three refs.
        """
        del file, effective_root
        stmt_count = 0
        imports: list[ImportRef] = []
        for child in tree.root_node.children:
            sub_count, sub_refs = self._imports_from_node(child, source_bytes)
            stmt_count += sub_count
            imports.extend(sub_refs)
        return stmt_count, tuple(imports), ()

    def _imports_from_node(
        self, node: Any, source_bytes: bytes,
    ) -> tuple[int, list[ImportRef]]:
        """Collect usings from one compilation-unit child, descending into namespaces.

        Returns `(stmt_count, refs)`. Direct `using_directive` → one
        statement, one ref. `namespace_declaration` → iterate its `body:`
        `declaration_list`. `file_scoped_namespace_declaration` → iterate
        its direct children (no body field). Anything else → zero.
        """
        if node.type == "using_directive":
            ref = _ref_from_using_directive(
                node, source_bytes, self._cache_package_names,
            )
            return (1, [ref]) if ref is not None else (0, [])
        if node.type == "namespace_declaration":
            body = node.child_by_field_name("body")
            return _collect_inner_usings(
                body.children if body is not None else (),
                source_bytes, self._cache_package_names,
            )
        if node.type == "file_scoped_namespace_declaration":
            return _collect_inner_usings(
                node.children, source_bytes, self._cache_package_names,
            )
        return 0, []

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Top-level public types (including inside either namespace form)."""
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            _collect_at_or_under_namespace(child, source_bytes, symbols)
        return symbols

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Like the base but drop files inside a nested `.csproj` subtree.

        Workspace-aware: files inside a declared solution project are
        kept; files inside an unregistered nested `.csproj` are dropped.
        Single-module mode passes an empty registered set so the legacy
        "any nested .csproj crosses" rule is preserved.
        """
        files = super().project_files(root, excludes)
        effective = self._cache_effective
        registered = self._cache_registered_module_roots
        return [
            f for f in files
            if not _crosses_module_boundary(f, effective, registered)
        ]

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every non-test, non-generated `.cs` file directly in `package_root`.

        Sorted by name. Sub-dirs are independent packages so the walk is
        non-recursive. `*Test.cs` / `*Tests.cs` and `*.g.cs` / `*.designer.cs`
        files are excluded from the facade but kept in the import graph via
        `project_files`.
        """
        try:
            entries = sorted(package_root.iterdir(), key=lambda p: p.name)
        except OSError:
            return ()
        return tuple(
            e for e in entries
            if e.is_file()
            and e.suffix.lower() == ".cs"
            and not _is_test_filename(e.name)
            and not _is_generated_filename(e.name)
        )


def _is_test_filename(name: str) -> bool:
    """True if `name` ends in an xUnit / NUnit / MSTest test-class suffix."""
    return any(name.endswith(suffix) for suffix in _TEST_FILENAME_SUFFIXES)


def _is_generated_filename(name: str) -> bool:
    """True if `name` ends in a codegen suffix (case-insensitive)."""
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in _GENERATED_FILENAME_SUFFIXES)


def _csharp_source_root(root: Path) -> Path | None:
    """Locate the C# project root under `root`, with one-level wrapper descent.

    Returns `root` when a `.csproj` sits at `root` directly. If `root`
    itself has no `.csproj` but exactly one direct child does, returns
    that child. Every other arrangement (no `.csproj` anywhere, multiple
    `.csproj`-bearing children) returns `None` — callers fall back to
    walking `root` directly, where the package-roots module-boundary
    guard refuses to descend into sibling projects.
    """
    if _has_csharp_project(root):
        return root
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    project_children = [
        c for c in children
        if c.is_dir() and _has_csharp_project(c)
    ]
    if len(project_children) == 1:
        return project_children[0]
    return None


def _has_csharp_project(directory: Path) -> bool:
    """True iff `directory` contains a `.csproj` file directly."""
    try:
        entries = directory.iterdir()
    except OSError:
        return False
    return any(
        e.is_file() and e.suffix.lower() in _CSHARP_PROJECT_EXTENSIONS
        for e in entries
    )


def _crosses_module_boundary(
    file: Path,
    effective_root: Path,
    registered_module_roots: frozenset[Path],
) -> bool:
    """True if `file` lies inside an unregistered nested `.csproj` subtree.

    Walks up from `file.parent` toward `effective_root`. If an ancestor
    carries a `.csproj` AND is NOT in `registered_module_roots`, the
    file is on the wrong side of a project boundary and gets dropped.
    Workspace mode passes the declared solution-project set; single-module
    mode passes `frozenset()` and the predicate reduces to today's
    "any .csproj ancestor crosses" rule.
    """
    cur = file.parent
    while cur != effective_root and effective_root in cur.parents:
        if _has_csharp_project(cur) and cur not in registered_module_roots:
            return True
        cur = cur.parent
    return False


def _find_solution_file(root: Path) -> Path | None:
    """Return the first `*.sln` directly in `root` (alphabetical tiebreak), or `None`."""
    try:
        candidates = sorted(
            e for e in root.iterdir()
            if e.is_file() and e.suffix.lower() == _SLN_EXTENSION
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _read_sln_text(sln_path: Path) -> str | None:
    """Read a `.sln` with utf-8-sig → cp1252 fallback + sanity check.

    Modern VS emits UTF-8 with BOM (`utf-8-sig` absorbs it). Pre-2017
    emits Windows-1252 — cp1252 is a total encoding over 0x00-0xFF so it
    NEVER raises `UnicodeDecodeError`. To distinguish a real legacy
    `.sln` from a binary blob whose cp1252 round-trip yields garbage
    (round-2 delta #8), the fallback decode is sanity-checked for either
    `Project(` or the `Microsoft Visual Studio Solution File` header.
    """
    try:
        return sln_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        pass
    except OSError:
        return None
    try:
        text = sln_path.read_text(encoding="cp1252")
    except (OSError, UnicodeDecodeError):
        return None
    if _SLN_PROJECT_LINE_PREFIX not in text and _SLN_HEADER_FRAGMENT not in text:
        return None
    return text


def _parse_sln_project_paths(sln_text: str) -> list[tuple[str, str]]:
    """Parse a `.sln` body into `(rel_path, project_name)` tuples for C# projects.

    `.splitlines()` handles CRLF transparently. Backslashes in paths
    (Windows-style) are normalized to POSIX. Only project rows whose
    path ends in `.csproj` (case-insensitive) are returned — VB/F#/C++
    projects, shared projects, and solution folders carry different
    extensions (or non-file paths) and are filtered here.
    """
    out: list[tuple[str, str]] = []
    for raw_line in sln_text.splitlines():
        line = raw_line.strip()
        if not line.startswith(_SLN_PROJECT_LINE_PREFIX):
            continue
        m = _SLN_PROJECT_LINE_PATTERN.match(line)
        if m is None:
            continue
        name = m.group(1)
        rel_path = m.group(2).replace("\\", "/")
        if not rel_path.casefold().endswith(_CSPROJ_EXTENSION):
            continue
        out.append((rel_path, name))
    return out


def _expand_sln_projects(
    workspace_root: Path, project_rows: list[tuple[str, str]],
) -> tuple[_CSharpSolutionProject, ...]:
    """Resolve solution project paths on disk; dedupe; sort by project_root name.

    Out-of-root entries (parent-relative `..\\Shared\\Shared.csproj`) are
    silently dropped. Missing `.csproj` files are silently dropped.
    """
    seen: dict[Path, _CSharpSolutionProject] = {}
    resolved_root = workspace_root.resolve()
    for rel_path, _name in project_rows:
        candidate_csproj = (workspace_root / rel_path).resolve()
        if not _is_within(candidate_csproj, resolved_root):
            continue
        csproj_path = workspace_root / rel_path
        if not csproj_path.is_file():
            continue
        project_root = csproj_path.parent
        if project_root in seen:
            continue
        seen[project_root] = _CSharpSolutionProject(
            project_root=project_root, csproj_path=csproj_path,
        )
    return tuple(sorted(seen.values(), key=lambda p: p.project_root.name))


def _solution_projects_at(root: Path) -> tuple[_CSharpSolutionProject, ...] | None:
    """Find + parse a `.sln` at `root`; return its `.csproj` project rows.

    Returns `None` when no usable solution shape is present:
      - no `*.sln` at root, OR
      - `.sln` read failure (encoding both paths fail / file missing), OR
      - parsed but zero `.csproj` rows (round-2 delta #17: precedence
        rule keeps `_compute_effective_root` from engaging workspace
        mode for a stray empty `.sln`).
    """
    sln_path = _find_solution_file(root)
    if sln_path is None:
        return None
    text = _read_sln_text(sln_path)
    if text is None:
        return None
    rows = _parse_sln_project_paths(text)
    if not rows:
        return None
    projects = _expand_sln_projects(root, rows)
    return projects if projects else None


def _is_within(candidate: Path, root: Path) -> bool:
    """True iff `candidate` resolves under `root`."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _discover_single_project_pairs(
    effective_root: Path, get_parser: Any,
) -> tuple[tuple[Path, str], ...]:
    """Today's single-project DFS — preserved verbatim for the fast path."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [effective_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        package_name = _package_name_for_dir(entries, get_parser)
        if package_name is not None:
            pairs.append((cur, package_name))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _CSHARP_SCAN_SKIP:
                continue
            if _has_csharp_project(entry):
                continue
            stack.append(entry)
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _discover_workspace_pairs(
    projects: tuple[_CSharpSolutionProject, ...],
    registered_module_roots: frozenset[Path],
    get_parser: Any,
) -> tuple[tuple[Path, str], ...]:
    """Discover `(dir, dotted-namespace)` pairs across every solution project."""
    pairs: list[tuple[Path, str]] = []
    for project in projects:
        pairs.extend(
            _walk_project_packages(project, registered_module_roots, get_parser),
        )
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _walk_project_packages(
    project: _CSharpSolutionProject,
    registered_module_roots: frozenset[Path],
    get_parser: Any,
) -> list[tuple[Path, str]]:
    """DFS one solution project's tree, yielding namespace-attributed pairs."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [project.project_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        package_name = _package_name_for_dir(entries, get_parser)
        if package_name is not None:
            pairs.append((cur, package_name))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _CSHARP_SCAN_SKIP:
                continue
            if _has_csharp_project(entry) and entry not in registered_module_roots:
                continue
            stack.append(entry)
    return pairs


def _package_name_for_dir(
    entries: list[Path],
    get_parser: Any,
) -> str | None:
    """Pick the dominant namespace across `.cs` files in `entries`.

    Returns the dotted package name most frequently declared by files in
    this directory, with alphabetical filename as the tiebreaker. Returns
    `None` if no `.cs` file declares a parseable namespace. A pure
    first-alphabetical rule mis-buckets canonical C# layouts when a
    polyfill file (e.g. `Guard.cs` with `namespace JetBrains.Annotations`)
    sorts ahead of the dominant-namespace files. Dominance + alphabetical
    tiebreak is deterministic and matches the layout C# authors actually
    intend.
    """
    cs_files = sorted(
        (e for e in entries if e.is_file() and e.suffix.lower() == ".cs"),
        key=lambda p: p.name,
    )
    counts: Counter[str] = Counter()
    for cs_file in cs_files:
        name = _read_namespace_name(cs_file, get_parser)
        if name is not None:
            counts[name] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _read_namespace_name(
    file_path: Path, get_parser: Any,
) -> str | None:
    """Parse `file_path` and return its namespace as a dotted string, or `None`."""
    parser = get_parser("c_sharp")
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
        if child.type not in _NAMESPACE_DECL_NODE_TYPES:
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            return None
        segments = _flatten_qualified_name(name_node, source_bytes)
        return ".".join(segments) if segments else None
    return None


def _flatten_qualified_name(node: Any, source_bytes: bytes) -> list[str]:
    """Flatten a C# qualified-name node into its dotted segments.

    The grammar models `qualified_name` recursively: `qualifier:` (which
    is itself either `qualified_name` or `identifier`) + `name:` (always
    an `identifier`). A single-segment using like `using System;` carries
    the identifier directly with no `qualified_name` wrapper.
    """
    if node.type == "identifier":
        return [_node_text(node, source_bytes)]
    if node.type == "qualified_name":
        qualifier = node.child_by_field_name("qualifier")
        segments = (
            _flatten_qualified_name(qualifier, source_bytes)
            if qualifier is not None else []
        )
        name = node.child_by_field_name("name")
        if name is not None:
            segments.append(_node_text(name, source_bytes))
        return segments
    return []


def _collect_inner_usings(
    children: Any,
    source_bytes: bytes,
    project_packages: frozenset[str],
) -> tuple[int, list[ImportRef]]:
    """Count `using_directive` nodes among `children` and emit refs.

    Used for both `namespace_declaration.body.children` and
    `file_scoped_namespace_declaration.children` — same shape, same rule.
    """
    stmt_count = 0
    refs: list[ImportRef] = []
    for child in children:
        if child.type != "using_directive":
            continue
        ref = _ref_from_using_directive(child, source_bytes, project_packages)
        if ref is None:
            continue
        stmt_count += 1
        refs.append(ref)
    return stmt_count, refs


def _ref_from_using_directive(
    directive: Any, source_bytes: bytes, project_packages: frozenset[str],
) -> ImportRef | None:
    """Build an `ImportRef` from one `using_directive` node, or `None`.

    The C# grammar exposes the dotted target via the `name:` field
    (`identifier` or `qualified_name`), the alias via the `alias:` field
    (`name_equals` containing an `identifier`), and the `static` keyword
    as an anonymous sibling token. The `global using` prefix appears as
    a sibling `global` token — treated as an ordinary using.
    """
    name_node = directive.child_by_field_name("name")
    if name_node is None:
        return None
    segments = _flatten_qualified_name(name_node, source_bytes)
    if not segments:
        return None
    is_static = any(child.type == "static" for child in directive.children)
    alias_name = _alias_name(directive, source_bytes)
    return ImportRef(
        target_module=_format_import_module(segments, is_static, alias_name),
        target_package=_resolve_csharp_import(segments, project_packages),
        line=directive.start_point[0] + 1,
    )


def _alias_name(directive: Any, source_bytes: bytes) -> str | None:
    """Return the alias identifier text from a `using X = …;` directive, or `None`."""
    alias_node = directive.child_by_field_name("alias")
    if alias_node is None:
        return None
    for child in alias_node.children:
        if child.type == "identifier":
            return _node_text(child, source_bytes)
    return None


def _format_import_module(
    segments: list[str], is_static: bool, alias_name: str | None,
) -> str:
    """Build a display string for `ImportRef.target_module`."""
    body = ".".join(segments)
    if is_static:
        body = "static " + body
    if alias_name is not None:
        body = body + " as " + alias_name
    return body


def _resolve_csharp_import(
    segments: list[str], project_packages: frozenset[str],
) -> str | None:
    """Resolve qualified-name segments to a registered package, longest-first.

    Tries every prefix length, longest first, and returns the first one
    that matches a registered package. Mirrors Java's `_resolve_java_import`
    — C#'s grammar can't tell us whether a trailing identifier is a type,
    a static member, or a sub-namespace, so longest-first is the only
    correct strategy. Handles every using form uniformly.
    """
    for n in range(len(segments), 0, -1):
        candidate = ".".join(segments[:n])
        if candidate in project_packages:
            return candidate
    return None


def _collect_at_or_under_namespace(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Collect from one compilation-unit child OR descend a namespace body once.

    Direct `_PUBLIC_TYPE_NODE_TYPES` children contribute via
    `_collect_public_type`. A `namespace_declaration` triggers descent
    into its `body:` `declaration_list`. A `file_scoped_namespace_declaration`
    triggers descent into its own direct children (no body field).
    Arbitrarily-nested namespaces inside one file are NOT descended past
    one level (v1 out-of-scope).
    """
    if node.type in _PUBLIC_TYPE_NODE_TYPES:
        _collect_public_type(node, source_bytes, symbols)
        return
    if node.type == "namespace_declaration":
        body = node.child_by_field_name("body")
        if body is None:
            return
        for child in body.children:
            if child.type in _PUBLIC_TYPE_NODE_TYPES:
                _collect_public_type(child, source_bytes, symbols)
        return
    if node.type == "file_scoped_namespace_declaration":
        for child in node.children:
            if child.type in _PUBLIC_TYPE_NODE_TYPES:
                _collect_public_type(child, source_bytes, symbols)


def _collect_public_type(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add a top-level public type's symbols (class + public members) to `symbols`.

    The class symbol's param count is determined by the FIRST public
    constructor encountered (via `setdefault`); if no public constructor
    exists, the class registers with param-count 0 after the body walk.
    Public methods become separate entries keyed by method name.
    `enum_declaration` registers only the enum's identifier — enum
    members are not classes and do not feed the facade.
    """
    if not _is_public(node):
        return
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _node_text(name_node, source_bytes)
    if node.type != "enum_declaration":
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

    `method_declaration` registers under its own name; `constructor_declaration`
    folds into the enclosing class name. Both use `setdefault` so the first
    occurrence wins on param count — deterministic across body iteration order.
    """
    if not _is_public(member):
        return
    if member.type == "method_declaration":
        name_node = member.child_by_field_name("name")
        if name_node is None:
            return
        symbols.setdefault(
            _node_text(name_node, source_bytes),
            _count_parameters(member),
        )
    elif member.type == "constructor_declaration":
        symbols.setdefault(class_name, _count_parameters(member))


def _is_public(node: Any) -> bool:
    """True iff `node` has a `modifier` child wrapping a `public` keyword.

    The C# grammar models access modifiers as individual `modifier` nodes
    appearing as direct children of the declaration — NOT a single
    `modifiers` group as Java/Kotlin/Scala do. Each `modifier`'s first
    child is the keyword token (`public`, `private`, `static`, …).
    Declarations without an explicit modifier (internal by default for
    top-level types) return False.
    """
    for child in node.children:
        if child.type != "modifier":
            continue
        if any(token.type == "public" for token in child.children):
            return True
    return False


def _count_parameters(method_node: Any) -> int:
    """Count `parameter` slots on a method or constructor declaration.

    The C# grammar exposes the parameter list as the `parameters:` field
    → `parameter_list` node whose direct children include one `parameter`
    per slot. `params` (variadic) is a modifier on the parameter, not a
    separate node type, so the count is correct without special handling.
    """
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return 0
    return sum(
        1 for c in params.children
        if c.type == "parameter"
    )
