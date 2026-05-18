"""Shared scaffolding for tree-sitter-based language adapters.

Centralizes the parts every adapter does identically — file enumeration,
layout caching, LOC counting, parser instantiation, and the outer skeletons
of `analyze_file`/`package_public_facade`. Subclasses provide the language-
specific descent rule, import walker, and public-symbol collector through
the abstract hooks at the bottom of `BaseTreeSitterAdapter`.

The `StubAdapter` deliberately does NOT inherit — it ships zero behavior on
purpose, and inheriting a 100-line scaffold would obscure that.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tree_sitter import Parser

from ...language import LanguageLoader
from ...utils.source_walker import iter_source_files, path_hits_excludes
from ..core.protocol import (
    FileAnalysis,
    ImportRef,
    InFunctionImport,
    PackageFacade,
)

logger = logging.getLogger(__name__)


class BaseTreeSitterAdapter(ABC):
    """Shared base for tree-sitter-driven `LanguageAdapter` implementations.

    Subclasses set the four class attributes (`LANGUAGE`, `EXTENSIONS`,
    `INDEX_BASENAMES`, `LINE_COMMENT_PREFIXES`) and implement the abstract
    hooks (`_compute_effective_root`, `_discover_package_roots`,
    `_walk_imports`, `_collect_public_symbols`). Optional overrides:
    `_post_layout` for extra layout-cache fields, `_grammar_for_file` for
    multi-grammar languages, `_facade_source_files` for languages without
    an index-file convention.
    """

    LANGUAGE: str = ""
    EXTENSIONS: frozenset[str] = frozenset()
    INDEX_BASENAMES: tuple[str, ...] = ()
    LINE_COMMENT_PREFIXES: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._loader = LanguageLoader()
        self._parsers: dict[str, Parser] = {}
        self._failed_grammars: set[str] = set()
        self._cache_root: Path | None = None
        self._cache_effective: Path = Path()
        self._cache_package_roots: tuple[Path, ...] = ()
        self._cache_package_names: frozenset[str] = frozenset()
        self._cache_package_by_path: dict[Path, str] = {}

    def package_roots(self, root: Path) -> tuple[Path, ...]:
        """Top-level package directories within `root` (post auto-descend)."""
        self._ensure_layout(root)
        return self._cache_package_roots

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Enumerate source files under the effective root honoring excludes."""
        self._ensure_layout(root)
        effective = self._cache_effective
        result: list[Path] = []
        exclude_parts = {e for e in excludes if e}
        for path, _stat in iter_source_files(effective):
            if path.suffix.lower() not in self.EXTENSIONS:
                continue
            if path_hits_excludes(path, effective, exclude_parts):
                continue
            result.append(path)
        return result

    def analyze_file(self, file: Path, root: Path) -> FileAnalysis:
        """Single tree-walk: returns LOC, top imports, in-function imports, package info."""
        self._ensure_layout(root)
        effective = self._cache_effective
        package, package_root = _find_package(
            file, effective, self._cache_package_by_path,
        )

        source_bytes = _safe_read_bytes(file)
        if source_bytes is None:
            return _empty_file_analysis(file, package, package_root, loc=0)

        loc = self._count_loc(source_bytes)
        parsed = self._parse_with_bytes(file, source_bytes)
        if parsed is None:
            return _empty_file_analysis(file, package, package_root, loc=loc)
        _, tree = parsed

        stmt_count, imports, in_func = self._walk_imports(
            tree, source_bytes, file, effective,
        )
        return FileAnalysis(
            file=file,
            package=package,
            package_root=package_root,
            loc=loc,
            top_import_statements=stmt_count,
            imports=imports,
            in_function_imports=in_func,
        )

    def package_public_facade(self, package_root: Path) -> PackageFacade:
        """Aggregate public symbols across `_facade_source_files(package_root)`.

        Each source file is parsed once; its `_collect_public_symbols` result
        (a `name → param-count` dict) is merged via `setdefault` so the first
        file declaring a name wins on its param count. The final tuple is
        sorted; `public_params` is the sum across all retained symbols.
        """
        merged: dict[str, int] = {}
        for path in self._facade_source_files(package_root):
            parsed = self._parse_file(path)
            if parsed is None:
                continue
            source_bytes, tree = parsed
            for name, params in self._collect_public_symbols(tree, source_bytes).items():
                merged.setdefault(name, params)
        return PackageFacade(
            public_symbols=tuple(sorted(merged)),
            public_params=sum(merged.values()),
        )

    def _count_loc(self, source_bytes: bytes) -> int:
        """Non-blank, non-line-comment LOC for this language's comment style."""
        try:
            text = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = source_bytes.decode("utf-8", errors="replace")
        prefixes = self.LINE_COMMENT_PREFIXES
        count = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in prefixes):
                continue
            count += 1
        return count

    def _get_parser(self, grammar: str | None = None) -> Parser | None:
        """Lazy per-grammar parser cache. `None` if the grammar fails to load.

        `grammar=None` falls back to `self.LANGUAGE`. Adapters whose files map
        to multiple tree-sitter grammars (TypeScript: typescript + tsx) override
        `_grammar_for_file` and the base picks the right cached parser.
        """
        name = grammar if grammar is not None else self.LANGUAGE
        cached = self._parsers.get(name)
        if cached is not None:
            return cached
        if name in self._failed_grammars:
            return None
        language = self._loader.load(name)
        if language is None:
            logger.warning("%s tree-sitter grammar unavailable", name)
            self._failed_grammars.add(name)
            return None
        try:
            parser = Parser(language)
        except TypeError:
            parser = Parser()
            parser.set_language(language)  # type: ignore[attr-defined]
        self._parsers[name] = parser
        return parser

    def _grammar_for_file(self, file: Path) -> str:
        """Tree-sitter grammar name to use for `file`. Default: `self.LANGUAGE`.

        Override when one adapter must dispatch across grammars (e.g. TypeScript
        uses the `tsx` grammar for `.tsx` and the `typescript` grammar for the
        other three TS extensions).
        """
        return self.LANGUAGE

    def _ensure_layout(self, root: Path) -> None:
        """Populate the layout cache via the subclass-provided rules.

        After repopulating the shared cache, calls `_post_layout` so a
        subclass can populate adapter-specific cache fields in the same
        invalidation step (e.g. `NpmPackageAdapter` loads tsconfig +
        barrel target; `GoAdapter` reads `go.mod`).
        """
        if self._cache_root == root:
            return
        self._cache_root = root
        effective = self._compute_effective_root(root)
        pairs = self._discover_package_roots(effective)
        self._cache_effective = effective
        self._cache_package_roots = tuple(p for p, _ in pairs)
        self._cache_package_names = frozenset(n for _, n in pairs)
        self._cache_package_by_path = dict(pairs)
        self._post_layout(effective, root)

    def _post_layout(self, effective_root: Path, audit_root: Path) -> None:
        """Hook for adapter-specific layout cache. Default: no-op.

        Called once per `_ensure_layout` cache miss, AFTER the shared
        package-roots cache is populated. Override to read auxiliary
        config (tsconfig, go.mod, …). `effective_root` is post auto-descend;
        `audit_root` is the user-supplied path (the two may differ when the
        adapter descends through a `src/` wrapper).
        """
        del effective_root, audit_root

    def _parse_file(self, path: Path) -> tuple[bytes, Any] | None:
        """Read + parse `path` with the right grammar; `None` on any failure.

        Returns `(source_bytes, tree)` on success. The grammar is picked
        via `_grammar_for_file(path)` so multi-grammar adapters (TS) work
        transparently. Failures (missing file, unavailable grammar,
        parser exception) all collapse to `None` — callers check once.
        """
        source_bytes = _safe_read_bytes(path)
        if source_bytes is None:
            return None
        return self._parse_with_bytes(path, source_bytes)

    def _parse_with_bytes(
        self, path: Path, source_bytes: bytes,
    ) -> tuple[bytes, Any] | None:
        """Like `_parse_file` but with already-read bytes — skip the re-read."""
        parser = self._get_parser(self._grammar_for_file(path))
        if parser is None:
            return None
        tree = _safe_parse(parser, source_bytes, path)
        if tree is None:
            return None
        return source_bytes, tree

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Files whose top-level exports contribute to the package facade.

        Default: a one-element tuple holding the first `INDEX_BASENAMES`
        entry that exists in `package_root`, or `()` if none does — the
        Python / JS / TS convention. Languages without an index file
        (Go) override this to return every contributing source file.
        """
        index_path = _find_first_existing(package_root, self.INDEX_BASENAMES)
        return () if index_path is None else (index_path,)

    def _walk_top_level_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
        per_stmt: Callable[[Any, bytes, Path, Path], list[ImportRef]],
    ) -> tuple[int, tuple[ImportRef, ...]]:
        """Iterate top-level statements via `per_stmt`; count statements + flatten refs.

        Statements that yield zero refs are not counted (matches the
        existing JS/TS/Go behavior — `import {}` from an empty named
        import contributes neither a count nor a ref).
        """
        stmt_count = 0
        imports: list[ImportRef] = []
        for child in tree.root_node.children:
            refs = per_stmt(child, source_bytes, file, effective_root)
            if refs:
                stmt_count += 1
                imports.extend(refs)
        return stmt_count, tuple(imports)

    @abstractmethod
    def _compute_effective_root(self, root: Path) -> Path:
        """Resolve the effective audit root (post auto-descend, if applicable)."""

    @abstractmethod
    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(package_dir, package_name)` pairs under `effective_root`.

        Python uses dir basenames. JavaScript uses `package.json::name` when
        workspaces are declared (so `apps/portal/` can be named
        `@scope/portal`), otherwise falls back to dir basenames.
        """

    @abstractmethod
    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Walk the parsed tree → (top-statement count, imports, in-function imports)."""

    @abstractmethod
    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Return `{public-symbol-name: param-count}` for one facade source file.

        The base aggregates across `_facade_source_files` using `setdefault`,
        so this hook only needs to walk one tree. Empty dict → no contribution.
        """


def _find_package(
    file: Path,
    effective_root: Path,
    package_by_path: dict[Path, str],
) -> tuple[str | None, Path | None]:
    """Map `file` to the DEEPEST containing package, or (None, None).

    Walks the parent chain of `file` looking for the closest path that is a
    declared package root. This handles monorepo workspaces where package
    boundaries live at any depth (`apps/portal/`, `packages/i18n/`), not
    just at the effective-root level.
    """
    try:
        file.relative_to(effective_root)
    except ValueError:
        return None, None
    cur = file.parent
    while True:
        if cur in package_by_path:
            return package_by_path[cur], cur
        if cur == effective_root or cur == cur.parent:
            return None, None
        cur = cur.parent


def _node_text(node: Any, source: bytes) -> str:
    """Decode the source bytes spanned by `node`."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_first_existing(directory: Path, basenames: tuple[str, ...]) -> Path | None:
    """First entry in `basenames` whose `directory/basename` exists, else None."""
    for name in basenames:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _safe_read_bytes(path: Path) -> bytes | None:
    """Read `path` as bytes; return `None` on OSError (logged at debug)."""
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.debug("read failed for %s: %s", path, exc)
        return None


def _safe_parse(parser: Parser, source_bytes: bytes, path: Path) -> Any | None:
    """Parse `source_bytes`; return `None` on any parser error (logged at debug)."""
    try:
        return parser.parse(source_bytes)
    except Exception as exc:
        logger.debug("parse failed for %s: %s", path, exc)
        return None


def _empty_file_analysis(
    file: Path,
    package: str | None,
    package_root: Path | None,
    loc: int,
) -> FileAnalysis:
    """`FileAnalysis` skeleton used whenever the file can't be read or parsed."""
    return FileAnalysis(
        file=file,
        package=package,
        package_root=package_root,
        loc=loc,
        top_import_statements=0,
        imports=(),
        in_function_imports=(),
    )
