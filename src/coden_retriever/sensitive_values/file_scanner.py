"""File scanner for whitelist-based text file discovery.

Walks a project directory, matches files against gitignore-style glob patterns,
and returns paths for sensitive value extraction. Respects SKIP_DIRS and
enforces a file size limit to avoid scanning huge binaries.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

# 1 MB limit prevents memory exhaustion from loading large binaries or logs.
# Matches cache manager threshold to maintain consistent file handling across codebase.
# Config/secret files are typically < 100KB, so 1MB provides 10x safety margin.
_MAX_FILE_SIZE_BYTES = 1_048_576


def scan_whitelist_files(
    root: Path,
    patterns: list[str],
) -> list[Path]:
    """Walk a directory and return files matching any of the given glob patterns.

    Args:
        root: Project root directory to walk.
        patterns: List of gitignore-style glob patterns (e.g. ``["*.env", "*.json"]``).

    Returns:
        List of matching file paths, deduplicated.
    """
    matched: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune directories that should be skipped (in-place to stop os.walk descent)
        dirnames[:] = [
            d for d in dirnames
            if d not in Config.SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if not _matches_any_pattern(filename, patterns):
                continue

            full_path = Path(dirpath) / filename
            try:
                size = full_path.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_SIZE_BYTES:
                logger.debug("Skipping oversized file: %s (%d bytes)", full_path, size)
                continue

            matched.append(full_path)

    return matched


def _matches_any_pattern(filename: str, patterns: list[str]) -> bool:
    """Check if a filename matches any of the glob patterns."""
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)
