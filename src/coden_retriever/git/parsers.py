"""Parsers for git command output formats.

This module provides parsing utilities for various git output formats
used by the git analysis tools.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def parse_shortlog(output: str) -> list[dict[str, Any]]:
    """Parse `git shortlog -sn` output into author stats.

    Input format:
        42  Alice <alice@example.com>
        15  Bob <bob@example.com>

    Args:
        output: Raw output from git shortlog -sn.

    Returns:
        List of dicts with 'name', 'email' (if available), and 'commits' keys.
    """
    results: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Format: "  42\tAlice <alice@example.com>" or "  42\tAlice"
        match = re.match(r"^\s*(\d+)\s+(.+)$", line)
        if match:
            commits = int(match.group(1))
            author_part = match.group(2).strip()

            # Try to extract email
            email_match = re.search(r"<([^>]+)>", author_part)
            if email_match:
                email = email_match.group(1)
                name = author_part[: email_match.start()].strip()
            else:
                email = None
                name = author_part

            results.append({
                "name": name,
                "email": email,
                "commits": commits,
            })

    return results


def parse_numstat(output: str) -> list[dict[str, Any]]:
    """Parse `git log --numstat` output into file change stats.

    Input format (per commit):
        <hash>
        10      5       src/file.py
        3       1       src/other.py

        <hash>
        ...

    Args:
        output: Raw output from git log --numstat --format="%H".

    Returns:
        List of dicts with 'commit', 'file', 'added', 'removed' keys.
    """
    results: list[dict[str, Any]] = []
    current_commit = ""

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check if this is a commit hash (40 hex chars)
        if re.match(r"^[0-9a-f]{40}$", line):
            current_commit = line
            continue

        # Parse numstat line: "10\t5\tfilename" or "-\t-\tfilename" (binary)
        parts = line.split("\t")
        if len(parts) == 3 and current_commit:
            added_str, removed_str, filepath = parts
            try:
                added = int(added_str) if added_str != "-" else 0
                removed = int(removed_str) if removed_str != "-" else 0
            except ValueError:
                added = 0
                removed = 0

            results.append({
                "commit": current_commit,
                "file": filepath,
                "added": added,
                "removed": removed,
            })

    return results


def parse_log_name_only(output: str) -> dict[str, int]:
    """Parse `git log --name-only` output into file commit counts.

    Input format:
        <commit info>

        file1.py
        file2.py

        <commit info>

        file1.py

    Args:
        output: Raw output from git log --format=format: --name-only.

    Returns:
        Dict mapping file path to number of commits affecting it.
    """
    file_counts: dict[str, int] = {}

    for line in output.splitlines():
        line = line.strip()
        if line and not line.startswith("commit "):
            # This is a filename
            file_counts[line] = file_counts.get(line, 0) + 1

    return file_counts


def parse_log_oneline(output: str) -> list[dict[str, Any]]:
    """Parse `git log --format` output into commit details.

    Expected format: "%H|%an|%ae|%at|%s" (hash|author|email|timestamp|subject)

    Args:
        output: Raw output from git log with the specified format.

    Returns:
        List of dicts with commit details.
    """
    results: list[dict[str, Any]] = []

    for line in output.strip().splitlines():
        if not line:
            continue

        parts = line.split("|", 4)
        if len(parts) >= 5:
            commit_hash, author, email, timestamp_str, subject = parts
            try:
                timestamp = int(timestamp_str)
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError):
                date_str = "unknown"

            results.append({
                "commit": commit_hash,
                "author": author,
                "email": email,
                "date": date_str,
                "subject": subject,
            })

    return results


def parse_function_log(output: str) -> list[dict[str, Any]]:
    """Parse `git log -L :function:file` output into function history.

    The -L option shows how a function evolved over commits.

    Args:
        output: Raw output from git log -L.

    Returns:
        List of dicts with commit info and diff for the function.
    """
    results: list[dict[str, Any]] = []
    current_commit: dict[str, Any] = {}
    current_diff_lines: list[str] = []
    in_diff = False

    for line in output.splitlines():
        # Commit header
        if line.startswith("commit "):
            # Save previous commit if exists
            if current_commit:
                current_commit["diff"] = "\n".join(current_diff_lines)
                results.append(current_commit)
                current_diff_lines = []

            current_commit = {
                "commit": line[7:].strip(),
                "author": "",
                "date": "",
                "subject": "",
                "lines_added": 0,
                "lines_removed": 0,
            }
            in_diff = False

        elif line.startswith("Author: ") and current_commit:
            # Parse "Author: Name <email>"
            author_part = line[8:].strip()
            email_match = re.search(r"<([^>]+)>", author_part)
            if email_match:
                current_commit["email"] = email_match.group(1)
                current_commit["author"] = author_part[: email_match.start()].strip()
            else:
                current_commit["author"] = author_part

        elif line.startswith("Date: ") and current_commit:
            # Parse date, try to normalize
            date_str = line[6:].strip()
            current_commit["date"] = date_str

        elif line.startswith("    ") and current_commit and not in_diff:
            # Commit message (indented)
            if not current_commit["subject"]:
                current_commit["subject"] = line.strip()

        elif line.startswith("diff --git") or line.startswith("@@"):
            in_diff = True
            current_diff_lines.append(line)

        elif in_diff:
            current_diff_lines.append(line)
            # Count additions and removals
            if line.startswith("+") and not line.startswith("+++"):
                current_commit["lines_added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_commit["lines_removed"] += 1

    # Save last commit
    if current_commit:
        current_commit["diff"] = "\n".join(current_diff_lines)
        results.append(current_commit)

    return results


def parse_file_dates(output: str) -> tuple[str | None, str | None]:
    """Parse git log output to find first and last commit dates for a file.

    Args:
        output: Raw output from git log --format="%at" (timestamps).

    Returns:
        Tuple of (first_date, last_date) in YYYY-MM-DD format, or None if empty.
    """
    timestamps: list[int] = []

    for line in output.strip().splitlines():
        line = line.strip()
        if line:
            try:
                timestamps.append(int(line))
            except ValueError:
                continue

    if not timestamps:
        return None, None

    # Git log returns newest first, so last in list is oldest
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    try:
        first_date = datetime.fromtimestamp(first_ts).strftime("%Y-%m-%d")
        last_date = datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d")
        return first_date, last_date
    except OSError:
        return None, None


def aggregate_file_stats(
    numstat_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate numstat results into per-file statistics.

    Args:
        numstat_results: Output from parse_numstat().

    Returns:
        Dict mapping file path to aggregated stats:
        - commits: number of commits
        - added: total lines added
        - removed: total lines removed
        - churned: added + removed
    """
    file_stats: dict[str, dict[str, Any]] = {}

    for entry in numstat_results:
        filepath = entry["file"]
        if filepath not in file_stats:
            file_stats[filepath] = {
                "commits": set(),
                "added": 0,
                "removed": 0,
            }

        file_stats[filepath]["commits"].add(entry["commit"])
        file_stats[filepath]["added"] += entry["added"]
        file_stats[filepath]["removed"] += entry["removed"]

    # Convert commit sets to counts and calculate churn
    for filepath, stats in file_stats.items():
        stats["commit_count"] = len(stats["commits"])
        stats["churned"] = stats["added"] + stats["removed"]
        del stats["commits"]  # Remove the set

    return file_stats
