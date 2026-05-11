"""GitHistoryContextService — deepened git_history_context implementation.

Encapsulates the blame/diff/rename pipeline behind a small service interface
fed via HistoryContextRequest. The MCP tool in inspection.py is a thin
orchestrator over this service.

run() reads as the sequence of pipeline steps; each step is a private method
or module helper kept under McCabe ~12 so per-step logic is testable in
isolation. Subprocess invocations are routed through the injected
GitBlameSource port.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from coden_retriever.git.blame_port import GitBlameSource

# 40-char SHA-1 prefix marks each block in `git blame --porcelain` output.
_BLAME_HASH_RE = re.compile(r'^[0-9a-f]{40}')

# Match a full hash on its own line in `git log --format=%H` output.
_LOG_HASH_RE = re.compile(r'^[0-9a-f]{40}$')

# Capture the new-file start line and length from a unified diff hunk header.
_HUNK_HEADER_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


@dataclass(frozen=True)
class HistoryContextRequest:
    """All inputs to GitHistoryContextService.run()."""

    file_path: str
    start_line: int
    end_line: int
    include_diff: bool = False
    include_line_blame: bool = False
    follow_renames: bool = False
    author: str | None = None
    since: str | None = None
    until: str | None = None


def _parse_date_to_timestamp(date_str: str) -> int | None:
    """Parse a date string into a Unix timestamp.

    Supports common formats like:
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM:SS
    - ISO 8601 format

    Returns None if parsing fails.
    """
    date_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]

    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue

    return None


def _format_timestamp(ts_str: str) -> str:
    try:
        timestamp = int(ts_str)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return "unknown"


def _filter_diff_to_range(diff_output: str, start_line: int, end_line: int) -> str:
    """Filter a diff to only include hunks that affect the specified line range."""
    lines = diff_output.splitlines()
    result_lines: list[str] = []
    in_relevant_hunk = False

    for line in lines:
        match = _HUNK_HEADER_RE.match(line)
        if match:
            hunk_start = int(match.group(1))
            hunk_len = int(match.group(2)) if match.group(2) else 1
            hunk_end = hunk_start + hunk_len - 1
            in_relevant_hunk = hunk_start <= end_line and hunk_end >= start_line
            if in_relevant_hunk:
                result_lines.append(line)
        elif in_relevant_hunk:
            result_lines.append(line)
        elif line.startswith('diff --git') or line.startswith('index ') or line.startswith('--- ') or line.startswith('+++ '):
            if not result_lines or not result_lines[-1].startswith('diff'):
                result_lines.append(line)

    return "\n".join(result_lines)


def _parse_porcelain(
    stdout: str, start_line: int, include_line_blame: bool
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Parse `git blame --porcelain` output into a commits map + line_blame list."""
    commits: dict[str, dict[str, Any]] = {}
    line_blame: list[dict[str, Any]] = []
    current_commit = ""
    current_line_num = start_line

    for line in stdout.splitlines():
        if _BLAME_HASH_RE.match(line):
            current_commit = line.split()[0]
            commits.setdefault(current_commit, {})
            if include_line_blame:
                line_blame.append({
                    "line": current_line_num,
                    "hash": current_commit[:8],
                })
                current_line_num += 1
        elif current_commit:
            _absorb_porcelain_metadata(line, commits[current_commit], line_blame, include_line_blame)

    return commits, line_blame


def _absorb_porcelain_metadata(
    line: str,
    commit_entry: dict[str, Any],
    line_blame: list[dict[str, Any]],
    include_line_blame: bool,
) -> None:
    """Mutate commit_entry / latest line_blame entry from a porcelain metadata line."""
    if line.startswith("author "):
        commit_entry["author"] = line[7:]
        if include_line_blame and line_blame:
            line_blame[-1]["author"] = line[7:]
    elif line.startswith("author-mail "):
        commit_entry["author_email"] = line[12:].strip("<>")
    elif line.startswith("author-time "):
        commit_entry["author_time"] = line[12:]
        if include_line_blame and line_blame:
            try:
                ts = int(line[12:])
                line_blame[-1]["date"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            except (ValueError, OSError):
                pass
    elif line.startswith("summary "):
        commit_entry["summary"] = line[8:]


def _commit_passes_filters(
    commit_info: dict[str, Any], req: HistoryContextRequest
) -> bool:
    """True if commit_info matches the author/since/until filters in req."""
    if req.author:
        haystack = (
            commit_info.get("author", "").lower()
            + " "
            + commit_info.get("author_email", "").lower()
        )
        if req.author.lower() not in haystack:
            return False

    try:
        commit_ts = int(commit_info.get("author_time", "0"))
    except (ValueError, TypeError):
        commit_ts = 0

    if req.since and commit_ts > 0:
        since_ts = _parse_date_to_timestamp(req.since)
        if since_ts is not None and commit_ts < since_ts:
            return False
    if req.until and commit_ts > 0:
        until_ts = _parse_date_to_timestamp(req.until)
        if until_ts is not None and commit_ts > until_ts:
            return False
    return True


def _apply_filters(
    commits: dict[str, dict[str, Any]], req: HistoryContextRequest
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Return either (filter_no_match_message, {}) or (None, filtered_commits)."""
    has_filter = bool(req.author or req.since or req.until)
    if not has_filter:
        return None, commits

    filtered = {
        h: info for h, info in commits.items() if _commit_passes_filters(info, req)
    }
    if filtered:
        return None, filtered

    desc: list[str] = []
    if req.author:
        desc.append(f"author='{req.author}'")
    if req.since:
        desc.append(f"since='{req.since}'")
    if req.until:
        desc.append(f"until='{req.until}'")
    return (
        {
            "message": f"No commits match the specified filters: {', '.join(desc)}",
            "total_commits_before_filter": len(commits),
        },
        {},
    )


def _build_base_result(
    req: HistoryContextRequest,
    sorted_commits: list[tuple[str, dict[str, Any]]],
    most_recent_hash: str,
    most_recent: dict[str, Any],
    oldest_hash: str,
    oldest: dict[str, Any],
    commit_message: str,
    commits_count: int,
) -> dict[str, Any]:
    """Assemble the base result dict (incl. legacy top-level fields and first_introduced)."""
    author_name = most_recent.get("author", "Unknown")
    short_hash = most_recent_hash[:8]
    date_str = _format_timestamp(most_recent.get("author_time", "0"))
    summary_line = (
        f"Lines {req.start_line}-{req.end_line} last modified by {author_name} "
        f"in commit {short_hash}: {most_recent.get('summary', 'No message')}"
    )

    result: dict[str, Any] = {
        "summary": summary_line,
        "most_recent": {
            "commit_hash": most_recent_hash,
            "short_hash": short_hash,
            "author": author_name,
            "author_email": most_recent.get("author_email", ""),
            "date": date_str,
            "commit_message": commit_message,
        },
        "commits_in_range": commits_count,
        "all_commits": [
            {
                "hash": h[:8],
                "author": info.get("author", "Unknown"),
                "summary": info.get("summary", ""),
                "date": _format_timestamp(info.get("author_time", "0")),
            }
            for h, info in sorted_commits
        ],
        # Keep legacy fields for backwards compatibility
        "commit_hash": most_recent_hash,
        "short_hash": short_hash,
        "author": author_name,
        "author_email": most_recent.get("author_email", ""),
        "date": date_str,
        "commit_message": commit_message,
    }

    if oldest_hash != most_recent_hash:
        result["first_introduced"] = {
            "commit_hash": oldest_hash,
            "short_hash": oldest_hash[:8],
            "author": oldest.get("author", "Unknown"),
            "date": _format_timestamp(oldest.get("author_time", "0")),
            "summary": oldest.get("summary", ""),
        }
    return result


def _attach_line_blame(
    result: dict[str, Any],
    line_blame: list[dict[str, Any]],
    commits: dict[str, dict[str, Any]],
) -> None:
    """Enrich each line_blame entry with author/date from commits, then attach to result."""
    for entry in line_blame:
        full_hash = next((h for h in commits if h.startswith(entry["hash"])), None)
        if not full_hash:
            continue
        commit_info = commits[full_hash]
        if "author" not in entry and "author" in commit_info:
            entry["author"] = commit_info["author"]
        if "date" not in entry and "author_time" in commit_info:
            try:
                ts = int(commit_info["author_time"])
                entry["date"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            except (ValueError, OSError):
                pass
    result["line_blame"] = line_blame


def _parse_rename_log(log_output: str) -> list[dict[str, str]]:
    """Parse `git log --follow --name-status --diff-filter=R` output into rename entries."""
    rename_history: list[dict[str, str]] = []
    current_hash = ""
    for line in log_output.strip().splitlines():
        if _LOG_HASH_RE.match(line):
            current_hash = line
            continue
        if not (line.startswith("R") and current_hash):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            rename_history.append({
                "commit": current_hash[:8],
                "old_name": parts[1],
                "new_name": parts[2],
            })
    return rename_history


class GitHistoryContextService:
    """Computes git history context for a file's line range."""

    def __init__(self, source: GitBlameSource) -> None:
        self._source = source

    async def run(self, req: HistoryContextRequest) -> dict[str, Any]:
        if not os.path.isfile(req.file_path):
            return {"error": f"File not found: {req.file_path}"}
        if req.start_line > req.end_line:
            req = replace(req, start_line=req.end_line, end_line=req.start_line)
        file_dir = str(Path(req.file_path).parent)

        repo_err = await self._check_git_repo(file_dir)
        if repo_err:
            return repo_err

        blame_err, stdout = await self._fetch_blame(req, file_dir)
        if blame_err:
            return blame_err

        commits, line_blame = _parse_porcelain(stdout, req.start_line, req.include_line_blame)
        if not commits:
            return {"error": "Could not parse git blame output"}

        filter_msg, commits = _apply_filters(commits, req)
        if filter_msg:
            return filter_msg

        sorted_commits = sorted(
            commits.items(),
            key=lambda x: int(x[1].get("author_time", "0")),
            reverse=True,
        )
        most_recent_hash, most_recent = sorted_commits[0]
        oldest_hash, oldest = sorted_commits[-1]

        commit_message = await self._resolve_commit_message(
            file_dir, most_recent_hash, most_recent.get("summary", "")
        )

        result = _build_base_result(
            req, sorted_commits,
            most_recent_hash, most_recent,
            oldest_hash, oldest,
            commit_message, len(commits),
        )

        if req.include_line_blame and line_blame:
            _attach_line_blame(result, line_blame, commits)
        if req.include_diff:
            await self._attach_diff(result, req, file_dir, most_recent_hash)
        if req.follow_renames:
            await self._attach_rename_history(result, req, file_dir)

        return result

    async def _check_git_repo(self, file_dir: str) -> dict[str, Any] | None:
        returncode, _stdout, stderr = await self._source.rev_parse_git_dir(file_dir)
        if returncode == 0:
            return None
        detail = stderr.strip() if stderr.strip() else f"git rev-parse failed in {file_dir}"
        return {"error": f"Not a git repository: {detail}"}

    async def _fetch_blame(
        self, req: HistoryContextRequest, file_dir: str
    ) -> tuple[dict[str, Any] | None, str]:
        returncode, stdout, stderr = await self._source.blame_porcelain(
            file_dir, req.file_path, req.start_line, req.end_line, req.follow_renames
        )
        if returncode != 0:
            if "no such path" in stderr.lower():
                return {"error": f"File not tracked by git: {req.file_path}"}, ""
            return {"error": f"git blame failed: {stderr.strip()}"}, ""
        if not stdout.strip():
            return {"error": "No blame information available for the specified lines"}, ""
        return None, stdout

    async def _resolve_commit_message(
        self, file_dir: str, commit_hash: str, fallback: str
    ) -> str:
        returncode, commit_msg, _ = await self._source.show_commit_message(file_dir, commit_hash)
        return commit_msg.strip() if returncode == 0 else fallback

    async def _attach_diff(
        self,
        result: dict[str, Any],
        req: HistoryContextRequest,
        file_dir: str,
        most_recent_hash: str,
    ) -> None:
        returncode, diff_output, _ = await self._source.show_commit_diff(
            file_dir, most_recent_hash, req.file_path
        )
        if returncode != 0 or not diff_output.strip():
            return
        filtered = _filter_diff_to_range(diff_output, req.start_line, req.end_line)
        result["diff"] = filtered if filtered else diff_output.strip()

    async def _attach_rename_history(
        self, result: dict[str, Any], req: HistoryContextRequest, file_dir: str
    ) -> None:
        returncode, log_output, _ = await self._source.log_renames(file_dir, req.file_path)
        if returncode != 0 or not log_output.strip():
            return
        rename_history = _parse_rename_log(log_output)
        if rename_history:
            result["rename_history"] = rename_history
            result["note"] = f"File was renamed {len(rename_history)} time(s)"
