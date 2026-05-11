"""Git-history helpers for `--simple` map mode.

Engine-side counterpart of the research harvester at
`research/improving-code-map/process_metrics/harvest.py`. Used by `--simple` map
mode to rank candidate files by `change_count` (number of commits touching the
file), then refine the final ranking with per-object blame-derived commit counts.

Sync subprocess invocation — engine code is sync, and a single `git log` per
search is fast enough that we don't need the async wrapper from
`git/commands.py`.
"""
from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from ..constants import (
    SIMPLE_MAP_BLAME_TIMEOUT_SECONDS,
    SIMPLE_MAP_HISTORY_PROBE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Sentinel that delimits commits in `git log --pretty=format:...`. Any string
# that cannot appear in a numstat line works; matches the research harvester.
_COMMIT_SENTINEL: str = "--COMMIT--"
# A numstat row is exactly three tab-separated fields: added, removed, path.
_NUMSTAT_FIELDS: int = 3
# Numstat marks binary files with "-" in both numeric columns.
_BINARY_MARKER: str = "-"
# Header line in `git blame --line-porcelain`: <sha> <orig> <final> [group-size]
_BLAME_HEADER_RE = re.compile(r"^([0-9a-f]{40})\s+\d+\s+(\d+)(?:\s+(\d+))?$")
# Truthy config values returned by `git config --bool`.
_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
# Bound the harvest. Pathological histories (>5 min) fall back gracefully.
_GIT_LOG_TIMEOUT_SECONDS: float = 300.0


def _git_creationflags() -> int:
    creationflags = 0
    if sys.platform == "win32":
        # Prevent console popup when invoked from GUI / daemon contexts.
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return creationflags


def _run_git(
    repo_root: str,
    args: list[str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str] | None:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
            creationflags=_git_creationflags(),
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"git command failed for {repo_root} ({args}): {e}")
        return None


def history_is_locally_available(repo_root: str) -> bool:
    """True when git history can be queried without lazy network fetches.

    Promisor / partial clones can make innocent-looking history commands
    (`git log --numstat`, `git blame`) fetch missing objects on demand and look
    hung. `--simple` should fail closed to line-count ranking instead.
    """
    shallow = _run_git(
        repo_root,
        ["rev-parse", "--is-shallow-repository"],
        SIMPLE_MAP_HISTORY_PROBE_TIMEOUT_SECONDS,
    )
    if shallow and shallow.returncode == 0:
        if shallow.stdout.strip().lower() in _TRUE_VALUES:
            return False

    partial = _run_git(
        repo_root,
        [
            "config",
            "--get-regexp",
            r"^(remote\..*\.promisor|extensions\.partialclone)$",
        ],
        SIMPLE_MAP_HISTORY_PROBE_TIMEOUT_SECONDS,
    )
    if partial and partial.returncode == 0:
        for line in partial.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            key = parts[0].lower()
            value = parts[1].strip().lower() if len(parts) > 1 else ""
            if key.endswith(".promisor") and value in _TRUE_VALUES:
                return False
            if key == "extensions.partialclone":
                return False

    return True


def git_head_sha(repo_root: str) -> str | None:
    """Return the current HEAD commit SHA, or None when unavailable.

    None covers: not a git repo, git missing, empty repo (no commits), or any
    probe failure. Callers treat None as "no git identity to track" — the lite
    cache still works on non-git trees, it just can't invalidate change_count
    on commit changes (there are none).
    """
    proc = _run_git(
        repo_root,
        ["rev-parse", "HEAD"],
        SIMPLE_MAP_HISTORY_PROBE_TIMEOUT_SECONDS,
    )
    if proc is None or proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha if sha else None


def git_is_dirty(repo_root: str) -> bool:
    """True when the working tree has uncommitted or untracked changes.

    False outside a git repo or on probe failure — non-git trees never flip the
    dirty flag, so they don't churn the change_count cache.
    """
    proc = _run_git(
        repo_root,
        ["status", "--porcelain"],
        SIMPLE_MAP_HISTORY_PROBE_TIMEOUT_SECONDS,
    )
    if proc is None or proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def _parse_path(path_field: str) -> str:
    """Resolve numstat's rename notation to the post-rename path.

    Examples:
        src/file.py                  -> src/file.py
        src/{old.py => new.py}       -> src/new.py
        {src => other}/file.py       -> other/file.py
        old.py => new.py             -> new.py
    """
    if "=>" not in path_field:
        return path_field
    if "{" in path_field and "}" in path_field:
        before_brace, _, after_brace = path_field.partition("{")
        inside, _, after_close = after_brace.partition("}")
        _, _, new_part = inside.partition("=>")
        return before_brace + new_part.strip() + after_close
    _, _, new = path_field.partition("=>")
    return new.strip()


def harvest_change_count(repo_root: str) -> dict[str, int]:
    """Return per-file commit count keyed by repo-relative POSIX path.

    Returns an empty dict if `repo_root` is not a git repository, git is
    missing, or the harvest fails for any reason — callers fall back to
    line-count-only ranking.
    """
    if not history_is_locally_available(repo_root):
        logger.debug(
            "harvest_change_count: skipping history harvest for %s because the "
            "repository is shallow or promisor-backed",
            repo_root,
        )
        return {}

    proc = _run_git(
        repo_root,
        [
            "log",
            "--all",
            "--no-merges",
            "--numstat",
            # `--relative` makes numstat paths relative to repo_root, not
            # the git toplevel. Required when repo_root is a subdirectory
            # (e.g. `coden src`) so paths match `to_repo_relative_posix`.
            "--relative",
            f"--pretty=format:{_COMMIT_SENTINEL}",
        ],
        _GIT_LOG_TIMEOUT_SECONDS,
    )
    if proc is None:
        return {}

    if proc.returncode != 0:
        logger.debug(
            f"harvest_change_count: git log returned {proc.returncode} "
            f"for {repo_root} (likely not a git repo)"
        )
        return {}

    counts: dict[str, int] = defaultdict(int)
    for line in proc.stdout.splitlines():
        if not line or line.startswith(_COMMIT_SENTINEL):
            continue
        parts = line.split("\t")
        if len(parts) != _NUMSTAT_FIELDS:
            continue
        added_str, removed_str, path_field = parts
        if added_str == _BINARY_MARKER and removed_str == _BINARY_MARKER:
            continue
        path = _parse_path(path_field)
        counts[path] += 1

    return dict(counts)


def harvest_line_blame_commits(repo_root: str, file_path: str) -> dict[int, str] | None:
    """Return line-number -> commit-hash for one tracked file.

    `file_path` must be repo-root-relative (or relative to the searched
    subdirectory when `repo_root` itself points inside a repository, matching the
    `--relative` behavior used by `harvest_change_count`). Returns `None` on git
    failure so callers can distinguish command failure from a valid empty map.
    """
    proc = _run_git(
        repo_root,
        ["blame", "--line-porcelain", "--", file_path],
        SIMPLE_MAP_BLAME_TIMEOUT_SECONDS,
    )
    if proc is None:
        return None
    if proc.returncode != 0:
        logger.debug(
            "harvest_line_blame_commits: git blame returned %s for %s:%s",
            proc.returncode,
            repo_root,
            file_path,
        )
        return None

    line_commits: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        match = _BLAME_HEADER_RE.match(line)
        if not match:
            continue
        commit_hash = match.group(1)
        start_line = int(match.group(2))
        group_size = int(match.group(3) or "1")
        for line_no in range(start_line, start_line + group_size):
            line_commits[line_no] = commit_hash

    return line_commits


@functools.lru_cache(maxsize=8)
def _resolved_root(repo_root: str) -> Path:
    """The same `repo_root` string is passed 2k+ times per `_simple_map_search`,
    but `Path(...).resolve()` is a `_getfinalpathname` syscall on Windows."""
    return Path(repo_root).resolve()


def to_repo_relative_posix(file_path: str, repo_root: str) -> str | None:
    """Convert an absolute file path to the repo-relative POSIX form used by
    `harvest_change_count` keys. Returns None when the path is outside the
    repo (e.g. a symlinked dependency)."""
    try:
        rel = Path(file_path).resolve().relative_to(_resolved_root(repo_root))
    except ValueError:
        return None
    return rel.as_posix()
