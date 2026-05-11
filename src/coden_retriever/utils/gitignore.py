"""Shared `.gitignore` support for the project indexer.

The indexer walkers (`cache/manager.py`, `search/engine.py`) previously
honoured only `Config.SKIP_DIRS` — a static allow-list of standard build
output directories. Project-specific ignores (`artifacts/`, fixture
output trees, extracted VSIXes, etc.) were walked anyway, which dominated
cold-start indexing cost on real repos.

This module loads `.gitignore` at the project root and returns a
`pathspec.PathSpec` the walkers use to prune directories and files in
addition to `SKIP_DIRS`. Nested `.gitignore` files are intentionally not
merged — the canonical project-level file is enough to cover the 90%
case without the complexity of per-directory spec composition.

Files outside the source tree (sensitive-values scanner, flag tooling)
deliberately do NOT use this helper: scanning gitignored paths is part
of their job.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)

_GITIGNORE_FILENAME = ".gitignore"


def load_gitignore_spec(source_dir: Path) -> pathspec.PathSpec | None:
    """Return a PathSpec matching the project's root `.gitignore`, or None.

    Returns None when the file is absent or unreadable; callers should
    treat that as "no additional exclusions beyond SKIP_DIRS".
    """
    gitignore = source_dir / _GITIGNORE_FILENAME
    if not gitignore.is_file():
        return None
    try:
        patterns = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", gitignore, exc)
        return None
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def is_ignored(
    spec: pathspec.PathSpec | None,
    source_dir: Path,
    path: Path,
    is_dir: bool,
) -> bool:
    """Check whether `path` is excluded by `spec` relative to `source_dir`.

    When `spec` is None, always returns False. Directory paths are
    matched with a trailing slash per gitwildmatch semantics so that
    patterns like `artifacts/codelldb-*/` anchor on directories only.
    """
    if spec is None:
        return False
    try:
        rel = path.relative_to(source_dir).as_posix()
    except ValueError:
        return False
    if is_dir and not rel.endswith("/"):
        rel = rel + "/"
    return spec.match_file(rel)
