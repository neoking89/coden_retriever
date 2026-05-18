"""PHP adapter: tree-sitter only.

PHP projects are anchored by `composer.json` rather than the JVM
`src/main/<lang>/` convention. The effective root is computed by descending
one optional wrapper level when exactly one direct child carries a
`composer.json`, mirroring the C# / Java / Kotlin / Scala one-level-descent
rule:

1. `root/composer.json` present → effective = root.
2. `root/<child>/composer.json` for exactly one child → effective = child.
3. Any other shape → effective = root; the package walker explicit-fails on
   multi-`composer.json` siblings.

Composer manifest contents are NOT parsed. Only the file's existence drives
effective-root descent and the multi-composer module-boundary guard. The
autoload map (`psr-4`, `psr-0`, `classmap`, `files`) is not consumed by any
code path; reading it would be speculative work without a verified consumer.

PHP diverges from C# in three load-bearing ways:

1. **Two namespace-declaration shapes** under one node type
   (`namespace_definition`): block form `namespace X { … }` carries a
   `body:` field of type `compound_statement`; file-scoped form
   `namespace X;` has no `body:` field. Both expose `name:` as a
   `namespace_name` node holding `name` tokens interleaved with `\`
   separators.

2. **Top-level types have NO visibility modifier**. Every `class`,
   `interface`, `trait`, and `enum` at file scope is implicitly public —
   asymmetric with C#'s "internal by default" rule. The walker therefore
   registers every type-declaration without an `_is_public_type` check.
   Methods DO have an optional `visibility_modifier`; absent means
   public (PHP method default), present means the keyword decides.

3. **`use` declarations have four forms**, all handled by a single helper:

     use a\b\C;                      single-class import
     use a\b\{ X, Y };               grouped form (ONE statement, ONE ref)
     use function a\b\fn;            function import
     use const a\b\C;                const import
     use a\b\C as Alias;             aliased import

   `use` may appear at file scope OR inside a block-form namespace body.
   The walker collects from both locations. The grouped form's prefix is
   `namespace_name` as a DIRECT child of the declaration (not wrapped in a
   `namespace_use_clause`); the per-clause group entries do not contribute
   to the target package.

Import resolution: longest-first prefix match across registered package
names. The `\` namespace separator is converted to `.` to stay consistent
with the other nine adapters.

Public-symbol extraction collects top-level types (`class_declaration`,
`interface_declaration`, `trait_declaration`, `enum_declaration`) — at file
scope OR inside a block-form namespace body. For each non-enum type, public
method declarations contribute. A method named `__construct` folds into the
enclosing class's symbol via `setdefault`, mirroring C#'s
`constructor_declaration` policy — without this, every PHP package with N
classes registers 2N facade symbols vs N for the C#/Java precedents,
warping `depth_ratio` and breaking cross-adapter comparability.

`in_function_imports` is always `()` — PHP forbids `use` inside method
bodies grammatically.

What v1 deliberately does NOT do (preserved current behavior; not flagged in QA):

- composer.json autoload parsing (`psr-4`, `psr-0`, `classmap`, `files`,
  `autoload-dev`). All ignored.
- PSR-4 / PSR-0 driven fallback for files lacking a `namespace` header.
  Such files attribute to `package=None`, matching C#.
- Conditional multi-namespace-per-file brace blocks. Only the FIRST
  `namespace_definition` drives package attribution; symbols collected
  from later namespaces in the same file leak into the first namespace's
  facade.
- Trait `use` declarations INSIDE a class body (`class C { use T; }`).
  These are `use_declaration` (not `namespace_use_declaration`) and are
  ignored, same convention as Java.
- Top-level `function_definition` (free functions outside any class).
  Not registered in the facade; matches Java/C#.
- Multi-`composer.json` parents — explicit-fail with zero packages and
  zero files (same shape as C#'s multi-`.csproj` and Java's multi-module
  guards).
- PHP attributes `#[Attr]` count as comments in LOC due to the `#`
  line-comment prefix. Acceptable noise for v1.

Scope:
  This adapter audits PHP; it is one of ten adapters registered in
  `architecture/core/runner.py::_ADAPTERS` (see README's
  `## Supported Languages` section for the full list). Out of scope:
  multi-`composer.json` parents (explicit-fail — point at one project),
  parsing `composer.json` autoload maps (`psr-4`, `psr-0`,
  `autoload.classmap`, `autoload.files`, `autoload-dev`) and PSR-0
  legacy autoloading. C, C++, and Bash are architecture-unsupported
  across all adapters; the README documents why.
"""
from __future__ import annotations

from collections import Counter
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


_COMPOSER_MANIFEST: str = "composer.json"
# Why: filename that marks a PHP project root. Composer is the de-facto
# package manager for PHP; its manifest's existence anchors the audit root.

_PHP_SCAN_SKIP: frozenset[str] = frozenset({
    "vendor", "node_modules", "var", "cache", "build", "dist", "coverage",
})
# Why: directories that must never count as PHP packages. `vendor/` holds
# Composer-installed dependencies; `node_modules/` is JS tooling that
# sometimes lives in PHP repos; `var/`, `cache/` are Symfony framework
# runtime caches; `build/`, `dist/`, `coverage/` are CI output dirs.
# `tests/` is deliberately NOT skipped — PHP test files declare their own
# namespaces (e.g. `Symfony\Component\Console\Tests`) and ARE real
# packages by the namespace-driven definition; they're excluded from the
# PUBLIC FACADE only, via the `*Test.php` filename suffix in
# `_facade_source_files`.

_TEST_FILENAME_SUFFIXES: tuple[str, ...] = (
    "Test.php",
)
# Why: PHPUnit convention is `*Test.php`. Excluded from the public facade
# only; the import graph still walks them.

_PUBLIC_TYPE_NODE_TYPES: frozenset[str] = frozenset({
    "class_declaration",
    "interface_declaration",
    "trait_declaration",
    "enum_declaration",
})
# Why: the top-level type-declaration node kinds whose presence feeds the
# package facade. PHP top-level types have no visibility modifier and are
# always public-accessible, so the walker registers them all without a
# public/private check (asymmetric with `csharp._PUBLIC_TYPE_NODE_TYPES`
# whose handling requires an `_is_public` check on every node).

_CONSTRUCTOR_NAME: str = "__construct"
# Why: PHP's constructor method name. Folds into the enclosing class
# symbol via `setdefault`, mirroring C#'s `constructor_declaration`
# handling so `interface_area` (and downstream `depth_ratio`) stays
# comparable across adapters.


class PhpAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for PHP projects."""

    LANGUAGE = "php"
    EXTENSIONS = frozenset({".php"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//", "#")

    def _compute_effective_root(self, root: Path) -> Path:
        """Descend one level when exactly one child carries a `composer.json`."""
        source_root = _php_source_root(root)
        if source_root is not None:
            return source_root
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every PHP package under `effective_root`.

        A "PHP package" is any directory containing ≥1 `.php` file whose
        namespace declaration parses successfully. The directory's package
        name is the most-common namespace among its `.php` files, with
        alphabetical filename as the tiebreaker — never the directory
        basename. The effective root itself is a candidate: in canonical
        PHP layouts (e.g. Symfony Console) the central namespace lives at
        the `composer.json`-owning project root next to its sub-namespace
        directories, so excluding the root would silently drop the
        library's main API surface from the graph.

        Module-boundary guard: a child directory whose own contents include
        a `composer.json` is a separate PHP project and is skipped.
        Without this, pointing the audit at a multi-project parent would
        walk every submodule's source tree and silently produce wrong
        output.
        """
        pairs: list[tuple[Path, str]] = []
        stack: list[Path] = [effective_root]
        while stack:
            cur = stack.pop()
            try:
                entries = list(cur.iterdir())
            except OSError:
                continue
            package_name = _package_name_for_dir(entries, self._get_parser)
            if package_name is not None:
                pairs.append((cur, package_name))
            for entry in entries:
                if not entry.is_dir():
                    continue
                if entry.name.startswith("."):
                    continue
                if entry.name in _PHP_SCAN_SKIP:
                    continue
                if _has_composer_manifest(entry):
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
        """Walk file-scope `namespace_use_declaration`s plus those inside block namespaces.

        Cannot reuse `_walk_top_level_imports`: a `namespace_definition` in
        block form would be counted as one statement regardless of how
        many usings live inside its `compound_statement` body.
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
        """Collect usings from one compilation-unit child, descending into block namespaces.

        Direct `namespace_use_declaration` → one statement, one ref.
        `namespace_definition` with a `body:` field → iterate the body's
        `compound_statement.children` for inner usings (block form only;
        semicolon-form `namespace X;` has no body field and its usings
        appear as later siblings at the file scope, already walked).
        Anything else → zero.
        """
        if node.type == "namespace_use_declaration":
            ref = _ref_from_use_declaration(
                node, source_bytes, self._cache_package_names,
            )
            return (1, [ref]) if ref is not None else (0, [])
        if node.type == "namespace_definition":
            body = node.child_by_field_name("body")
            if body is None:
                return 0, []
            return _collect_inner_usings(
                body.children, source_bytes, self._cache_package_names,
            )
        return 0, []

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Top-level public types (including inside block-form namespaces)."""
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            _collect_at_or_under_namespace(child, source_bytes, symbols)
        return symbols

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Like the base but drop files inside a nested composer subtree.

        Mirrors the module-boundary guard in `_discover_package_roots` so
        the file count stays consistent with the package count. Without
        this, pointing the audit at a multi-project parent would report N
        files across zero packages.
        """
        files = super().project_files(root, excludes)
        effective = self._cache_effective
        return [
            f for f in files
            if not _crosses_module_boundary(f, effective)
        ]

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every non-test `.php` file directly in `package_root`.

        Sorted by name. Sub-dirs are independent packages so the walk is
        non-recursive. `*Test.php` files are excluded from the facade but
        kept in the import graph via `project_files`.
        """
        try:
            entries = sorted(package_root.iterdir(), key=lambda p: p.name)
        except OSError:
            return ()
        return tuple(
            e for e in entries
            if e.is_file()
            and e.suffix.lower() == ".php"
            and not _is_test_filename(e.name)
        )


def _is_test_filename(name: str) -> bool:
    """True if `name` ends in a PHPUnit test-class suffix."""
    return any(name.endswith(suffix) for suffix in _TEST_FILENAME_SUFFIXES)


def _php_source_root(root: Path) -> Path | None:
    """Locate the PHP project root under `root`, with one-level wrapper descent.

    Returns `root` when a `composer.json` sits at `root` directly. If
    `root` itself has no manifest but exactly one direct child does,
    returns that child. Every other arrangement (no manifest anywhere,
    multiple manifest-bearing children) returns `None` — callers fall
    back to walking `root` directly, where the package-roots
    module-boundary guard refuses to descend into sibling projects.
    """
    if _has_composer_manifest(root):
        return root
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    manifest_children = [
        c for c in children
        if c.is_dir() and _has_composer_manifest(c)
    ]
    if len(manifest_children) == 1:
        return manifest_children[0]
    return None


def _has_composer_manifest(directory: Path) -> bool:
    """True iff `directory` contains a `composer.json` file directly."""
    return (directory / _COMPOSER_MANIFEST).is_file()


def _crosses_module_boundary(file: Path, effective_root: Path) -> bool:
    """True if any ancestor of `file` between it and `effective_root` carries a manifest.

    Used by `PhpAdapter.project_files` to drop files belonging to a
    sibling Composer module that happens to live under the audit root —
    same boundary the package walker enforces inline.
    """
    cur = file.parent
    while cur != effective_root and effective_root in cur.parents:
        if _has_composer_manifest(cur):
            return True
        cur = cur.parent
    return False


def _package_name_for_dir(
    entries: list[Path],
    get_parser: Any,
) -> str | None:
    """Pick the dominant namespace across `.php` files in `entries`.

    Returns the dotted package name most frequently declared by files in
    this directory, with alphabetical filename as the tiebreaker. Returns
    `None` if no `.php` file declares a parseable namespace. The Counter
    + insertion-order tiebreak matches the C# adapter's policy and
    sidesteps the polyfill-file mis-bucketing that a first-alphabetical
    rule produces (e.g. a JetBrains.Annotations-style `Guard.php` sitting
    next to four `Symfony\Foo` files).
    """
    php_files = sorted(
        (e for e in entries if e.is_file() and e.suffix.lower() == ".php"),
        key=lambda p: p.name,
    )
    counts: Counter[str] = Counter()
    for php_file in php_files:
        name = _read_namespace_name(php_file, get_parser)
        if name is not None:
            counts[name] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _read_namespace_name(
    file_path: Path, get_parser: Any,
) -> str | None:
    """Parse `file_path` and return its FIRST namespace as a dotted string."""
    parser = get_parser("php")
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
        if child.type != "namespace_definition":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            return None
        segments = _flatten_namespace_name(name_node, source_bytes)
        return ".".join(segments) if segments else None
    return None


def _flatten_namespace_name(node: Any, source_bytes: bytes) -> list[str]:
    """Flatten a `namespace_name` node into its dotted segments.

    The grammar models `namespace_name` as a flat sequence of `name`
    tokens interleaved with anonymous `\\` separators. Filtering children
    to `type == "name"` is correct; the `\\` separators are skipped
    naturally.
    """
    if node.type != "namespace_name":
        return []
    return [
        _node_text(child, source_bytes)
        for child in node.children
        if child.type == "name"
    ]


def _collect_inner_usings(
    children: Any,
    source_bytes: bytes,
    project_packages: frozenset[str],
) -> tuple[int, list[ImportRef]]:
    """Count `namespace_use_declaration` nodes among `children` and emit refs.

    Used for `namespace_definition.body.children` (the
    `compound_statement` of a block-form namespace).
    """
    stmt_count = 0
    refs: list[ImportRef] = []
    for child in children:
        if child.type != "namespace_use_declaration":
            continue
        ref = _ref_from_use_declaration(child, source_bytes, project_packages)
        if ref is None:
            continue
        stmt_count += 1
        refs.append(ref)
    return stmt_count, refs


def _ref_from_use_declaration(
    decl: Any, source_bytes: bytes, project_packages: frozenset[str],
) -> ImportRef | None:
    """Build an `ImportRef` from one `namespace_use_declaration` node, or `None`.

    Handles four PHP use-statement shapes:

      use a\\b\\C;                     single (clause-wrapped qualified_name)
      use a\\b\\{ X, Y };              grouped (direct namespace_name + group)
      use function a\\b\\fn;           function variant (outer keyword)
      use const a\\b\\C;               const variant (outer keyword)
      use a\\b\\C as Alias;            aliased (clause carries aliasing_clause)

    All variants emit one `ImportRef`. The `function`/`const` keyword
    appears as a direct anonymous-token child of the declaration; the
    helper detects it for display only. The grouped form's per-clause
    `function`/`const` keywords inside `namespace_use_group_clause`
    children are deliberately omitted from the display string — the
    grouped marker `.{...}` flags the form without re-implementing
    per-clause render.
    """
    kind = _outer_use_kind(decl)
    segments, alias_name, is_grouped = _segments_and_alias_for_use(
        decl, source_bytes,
    )
    if not segments:
        return None
    return ImportRef(
        target_module=_format_import_module(
            segments, kind, alias_name, is_grouped,
        ),
        target_package=_resolve_php_import(segments, project_packages),
        line=decl.start_point[0] + 1,
    )


def _outer_use_kind(decl: Any) -> str | None:
    """Return `"function"` / `"const"` if the outer keyword is set, else `None`."""
    for child in decl.children:
        if child.type in ("function", "const"):
            return child.type
    return None


def _segments_and_alias_for_use(
    decl: Any, source_bytes: bytes,
) -> tuple[list[str], str | None, bool]:
    """Extract `(segments, alias_name, is_grouped)` for one use declaration.

    Single form: a `namespace_use_clause` child wraps a `qualified_name`
    (prefix + trailing `name`) and an optional `namespace_aliasing_clause`.
    Grouped form: a `namespace_name` direct child holds the prefix; the
    `namespace_use_group` siblings flag the grouped shape.
    """
    for child in decl.children:
        if child.type == "namespace_use_clause":
            segs, alias = _segments_and_alias_for_clause(child, source_bytes)
            return segs, alias, False
        if child.type == "namespace_name":
            return (
                _flatten_namespace_name(child, source_bytes),
                None,
                True,
            )
    return [], None, False


def _segments_and_alias_for_clause(
    clause: Any, source_bytes: bytes,
) -> tuple[list[str], str | None]:
    """Walk a `namespace_use_clause` → (segments, alias_name)."""
    segments: list[str] = []
    alias_name: str | None = None
    for child in clause.children:
        if child.type == "qualified_name":
            segments = _flatten_qualified_name(child, source_bytes)
        elif child.type == "name":
            segments = [_node_text(child, source_bytes)]
        elif child.type == "namespace_aliasing_clause":
            alias_name = _alias_name_from_clause(child, source_bytes)
    return segments, alias_name


def _flatten_qualified_name(node: Any, source_bytes: bytes) -> list[str]:
    """Flatten a `qualified_name` into its dotted segments.

    The grammar models `qualified_name` as `namespace_name_as_prefix`
    (holding a `namespace_name` and optional leading `\\` tokens) plus a
    trailing `name` child. The leading `\\` of a fully-qualified
    `use \\App\\X;` import is an anonymous token, naturally ignored by
    the `name`-token filter on `namespace_name`.
    """
    segments: list[str] = []
    for child in node.children:
        if child.type == "namespace_name_as_prefix":
            for prefix_child in child.children:
                if prefix_child.type == "namespace_name":
                    segments.extend(
                        _flatten_namespace_name(prefix_child, source_bytes),
                    )
        elif child.type == "name":
            segments.append(_node_text(child, source_bytes))
        elif child.type == "namespace_name":
            segments.extend(_flatten_namespace_name(child, source_bytes))
    return segments


def _alias_name_from_clause(
    clause: Any, source_bytes: bytes,
) -> str | None:
    """Return the alias identifier text from a `namespace_aliasing_clause`, or `None`."""
    for child in clause.children:
        if child.type == "name":
            return _node_text(child, source_bytes)
    return None


def _format_import_module(
    segments: list[str],
    kind: str | None,
    alias_name: str | None,
    is_grouped: bool,
) -> str:
    """Build a display string for `ImportRef.target_module`."""
    body = ".".join(segments)
    if is_grouped:
        body = body + ".{...}"
    if kind is not None:
        body = kind + " " + body
    if alias_name is not None:
        body = body + " as " + alias_name
    return body


def _resolve_php_import(
    segments: list[str], project_packages: frozenset[str],
) -> str | None:
    """Resolve qualified-name segments to a registered package, longest-first.

    Tries every prefix length, longest first, and returns the first one
    that matches a registered package. Mirrors C#'s `_resolve_csharp_import`
    — PHP's grammar can't tell us whether a trailing identifier is a
    class, a function, a const, or a sub-namespace, so longest-first is
    the only correct strategy. Handles every use form uniformly.
    """
    for n in range(len(segments), 0, -1):
        candidate = ".".join(segments[:n])
        if candidate in project_packages:
            return candidate
    return None


def _collect_at_or_under_namespace(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Collect from one compilation-unit child OR descend a block-namespace body.

    Direct `_PUBLIC_TYPE_NODE_TYPES` children contribute via
    `_collect_top_type`. A `namespace_definition` with a `body:` field
    triggers descent into its `compound_statement.children`. The
    semicolon-form `namespace X;` has no body field; later siblings at
    file scope are walked already by the outer iteration.
    """
    if node.type in _PUBLIC_TYPE_NODE_TYPES:
        _collect_top_type(node, source_bytes, symbols)
        return
    if node.type == "namespace_definition":
        body = node.child_by_field_name("body")
        if body is None:
            return
        for child in body.children:
            if child.type in _PUBLIC_TYPE_NODE_TYPES:
                _collect_top_type(child, source_bytes, symbols)


def _collect_top_type(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add a top-level type's symbols (type + public members) to `symbols`.

    The type symbol's param count is determined by the FIRST `__construct`
    method encountered (via `setdefault`); if no constructor exists, the
    type registers with param-count 0 after the body walk. Public methods
    become separate entries keyed by method name. `enum_declaration`
    registers only the enum's identifier — enum cases are not methods and
    do not feed the facade.
    """
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
    """Add one direct member of a type to `symbols` (when itself public).

    `method_declaration` registers under its own name; a method named
    `__construct` folds into the enclosing class name via `setdefault` so
    the class symbol's param count is the constructor's arity (parity
    with C#'s `constructor_declaration` policy).
    """
    if member.type != "method_declaration":
        return
    if not _is_public_method(member):
        return
    name_node = member.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _node_text(name_node, source_bytes)
    param_count = _count_parameters(member)
    if method_name == _CONSTRUCTOR_NAME:
        symbols.setdefault(class_name, param_count)
        return
    symbols.setdefault(method_name, param_count)


def _is_public_method(method: Any) -> bool:
    """True iff `method` is public.

    PHP method default visibility is public — absent `visibility_modifier`
    means public (asymmetric with Java's package-private default). The
    modifier's first child token is the keyword (`public`, `private`,
    `protected`).
    """
    for child in method.children:
        if child.type != "visibility_modifier":
            continue
        for token in child.children:
            if token.type == "public":
                return True
            if token.type in ("private", "protected"):
                return False
        return False
    return True


def _count_parameters(method_node: Any) -> int:
    """Count `simple_parameter` / `variadic_parameter` slots on a method.

    The PHP grammar exposes the parameter list as the `parameters:` field
    → `formal_parameters` node. `simple_parameter` is a regular argument;
    `variadic_parameter` is `...$args` — each counts as one slot, matching
    Java's spread-parameter policy.
    """
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return 0
    return sum(
        1 for c in params.children
        if c.type in ("simple_parameter", "variadic_parameter")
    )
