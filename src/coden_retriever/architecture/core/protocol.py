"""Adapter protocol + value types shared between core and adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ImportRef:
    """A top-of-file import edge."""
    target_module: str
    target_package: str | None
    line: int


@dataclass(frozen=True)
class InFunctionImport:
    """An `import` statement living inside a function body — cycle-workaround signal."""
    line: int
    function: str
    import_text: str
    target_package: str | None


@dataclass(frozen=True)
class FileAnalysis:
    """Everything the core needs from one source file. Adapter parses once.

    `top_import_statements` is the per-statement count (for the oversized-file
    rule and display). `imports` is the per-edge list (one entry per imported
    module — `import a, b` produces two ImportRefs but one statement).
    """
    file: Path
    package: str | None
    package_root: Path | None
    loc: int
    top_import_statements: int
    imports: tuple[ImportRef, ...]
    in_function_imports: tuple[InFunctionImport, ...]


@dataclass(frozen=True)
class PackageFacade:
    """Public-surface count for a package — drives the depth-ratio metric."""
    public_symbols: tuple[str, ...]
    public_params: int


class LanguageAdapter(Protocol):
    """One implementation per supported language. v1 ships Python + stub.

    The adapter owns ALL language-specific layout knowledge — the runner stays
    pure. Implementations may transparently re-root through conventional
    wrapper directories so the user can pass either the wrapper (`src/`) or
    the actual code root (`src/<pkg>/`) and get the same audit.
    """

    LANGUAGE: str

    def package_roots(self, root: Path) -> tuple[Path, ...]:
        """Return the top-level package directories the adapter would audit under `root`.

        Adapters encode their language's package-boundary convention here:
        Python yields directories containing `__init__.py`; a JS adapter would
        yield `packages/*/` (monorepo) or `src/*/` (single project); a Go
        adapter would yield directories with `.go` files under the module root.

        Adapters MAY descend through one wrapper layer when `root` itself is
        not a package and contains exactly one direct child that is — this is
        how the Python adapter handles the `src/<pkg>/` layout.
        """
        ...

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        """Return every source file under `root` to include in the audit."""
        ...

    def analyze_file(self, file: Path, root: Path) -> FileAnalysis:
        """Parse `file` once and return its imports, in-function imports, LOC, package."""
        ...

    def package_public_facade(self, package_root: Path) -> PackageFacade:
        """Return the public symbols and public-param count for a package's facade."""
        ...
