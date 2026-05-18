"""Per-file rule: two Blob variants, ORed.

Coupling-driven Blob: `loc > OVERSIZED_FILE_LOC AND top_imports > OVERSIZED_FILE_IMPORTS`.
Size-driven Blob:     `loc > OVERSIZED_FILE_LOC_HARD` (regardless of imports).

Pattern follows Marinescu's conjunctive detection strategies (ICSM 2004)
with DECOR's (Moha et al., TSE 2010) OR-union of smell variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    OVERSIZED_FILE_IMPORTS,
    OVERSIZED_FILE_LOC,
    OVERSIZED_FILE_LOC_HARD,
)
from .protocol import FileAnalysis


@dataclass(frozen=True)
class OversizedFile:
    """A file flagged by either the coupling- or size-driven Blob rule."""
    path: Path
    loc: int
    top_imports: int


def find_oversized_files(
    file_analyses: list[FileAnalysis],
    root: Path,
) -> list[OversizedFile]:
    """Return files flagged by either Blob rule, sorted by LOC descending.

    A file is flagged if EITHER:
      - `loc > OVERSIZED_FILE_LOC AND top_imports > OVERSIZED_FILE_IMPORTS`
        (the coupling-driven Blob — Marinescu's conjunctive form), OR
      - `loc > OVERSIZED_FILE_LOC_HARD` (the size-driven Blob — anchored to
        Nagappan & Ball's ~665 LOC defect-density knee).

    `path` is relative to `root` for display; if relativization fails (file
    outside root), the absolute path is kept.
    """
    results: list[OversizedFile] = []
    for fa in file_analyses:
        if not _is_oversized(fa):
            continue
        try:
            rel = fa.file.relative_to(root)
        except ValueError:
            rel = fa.file
        results.append(OversizedFile(
            path=rel,
            loc=fa.loc,
            top_imports=fa.top_import_statements,
        ))
    results.sort(key=lambda o: (-o.loc, str(o.path)))
    return results


def _is_oversized(fa: FileAnalysis) -> bool:
    """True if `fa` trips the coupling-driven OR size-driven Blob rule."""
    if fa.loc > OVERSIZED_FILE_LOC_HARD:
        return True
    return fa.loc > OVERSIZED_FILE_LOC and fa.top_import_statements > OVERSIZED_FILE_IMPORTS
