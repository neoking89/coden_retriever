"""Shared source-file walker for indexer cold-start scans.

`CacheManager._scan_source_files` and `SearchEngine._collect_files` need
identical filesystem walks: os.walk from a root, prune SKIP_DIRS + hidden
+ gitignored dirs, then yield files whose extension is in LANGUAGE_MAP
and whose size is under the per-file cap. The two callers only differ in
how they consume the result (CacheManager wants mtime+size dicts;
SearchEngine wants a Path list), so both go through this single walker.

`path_hits_excludes` is the matching helper for the user-supplied
`--exclude` set — same semantics in every caller (architecture adapters,
runner-level layout check).
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from ..config import Config
from ..language import LANGUAGE_MAP
from .gitignore import is_ignored, load_gitignore_spec

# 1 MB cap. Files larger than this are almost always generated artefacts
# (bundles, minified JS, fixture dumps) that slow the indexer without
# adding search signal. Kept as a private constant because CacheManager
# and SearchEngine are the only legitimate consumers — other tools
# (sensitive-values scanner, watcher) have their own size policies.
_MAX_SOURCE_FILE_BYTES = 1_000_000


def iter_source_files(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    """Yield `(path, stat)` for every source file under `root` to index.

    Prunes `Config.SKIP_DIRS`, hidden directories (names starting with
    `.`), and gitignored paths; filters files by `LANGUAGE_MAP` extension
    and a 1 MB size cap. Unreadable files (stat errors) are silently
    skipped so a single permission issue doesn't abort the walk.
    """
    skip_dirs = Config.SKIP_DIRS
    skip_files = Config.SKIP_FILES
    gitignore_spec = load_gitignore_spec(root)

    for dirpath, dirs, filenames in os.walk(root):
        root_path = Path(dirpath)
        dirs[:] = [
            d for d in dirs
            if d not in skip_dirs
            and not d.startswith('.')
            and not is_ignored(gitignore_spec, root, root_path / d, is_dir=True)
        ]
        for name in filenames:
            if name in skip_files:
                continue
            path = root_path / name
            if path.suffix.lower() not in LANGUAGE_MAP:
                continue
            if is_ignored(gitignore_spec, root, path, is_dir=False):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > _MAX_SOURCE_FILE_BYTES:
                continue
            yield path, stat


def path_hits_excludes(path: Path, root: Path, excludes: set[str]) -> bool:
    """True if any path segment under `root` matches a caller-supplied exclude name."""
    if not excludes:
        return False
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in excludes for part in rel_parts)
