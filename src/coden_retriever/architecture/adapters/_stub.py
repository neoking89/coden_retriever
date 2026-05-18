"""Stub adapter for the polyglot-seam smoke test.

`coden architecture <any-path> --lang stub` runs through end-to-end with
zero findings; proves the adapter seam works without needing a second
fully-implemented language. v2 adapters (JS/TS, Go, ...) plug in here.
"""
from __future__ import annotations

from pathlib import Path

from ..core.protocol import FileAnalysis, PackageFacade


class StubAdapter:
    """Empty-result adapter — useful for verifying the seam, no real findings."""

    LANGUAGE = "stub"

    def package_roots(self, root: Path) -> tuple[Path, ...]:
        return ()

    def project_files(self, root: Path, excludes: tuple[str, ...]) -> list[Path]:
        return []

    def analyze_file(self, file: Path, root: Path) -> FileAnalysis:
        return FileAnalysis(
            file=file,
            package=None,
            package_root=None,
            loc=0,
            top_import_statements=0,
            imports=(),
            in_function_imports=(),
        )

    def package_public_facade(self, package_root: Path) -> PackageFacade:
        return PackageFacade(public_symbols=(), public_params=0)
