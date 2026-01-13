"""
Precision inspection MCP tools for deep code analysis.

Provides read_source_range for reading specific line ranges with line numbers,
and git_history_context for understanding recent changes via git blame.
"""
import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from .file_edit import mark_file_as_read
from ..git.commands import run_git_command

logger = logging.getLogger(__name__)

# Language detection based on file extension
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".xml": "xml",
}


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


async def read_source_range(
    file_path: Annotated[
        str,
        Field(description="Absolute path to the file to read")
    ],
    start_line: Annotated[
        int,
        Field(description="Starting line number (1-based indexing)", ge=1)
    ],
    end_line: Annotated[
        int,
        Field(description="Ending line number (1-based, inclusive)", ge=1)
    ],
    context_lines: Annotated[
        int,
        Field(description="Number of lines to include before and after the range for context", ge=0, le=50)
    ] = 0,
    expand_to_scope: Annotated[
        bool,
        Field(description="If True, expand the range to include the full containing function/class scope")
    ] = False,
) -> dict[str, Any]:
    """Read a specific range of lines from a file with line numbers prepended.

    This is the 'zoom in' tool - use it when you've found a location via search
    or stacktrace and need to see the full context of a function or code block.

    WHEN TO USE:
    - When you've identified a specific line/range from a stacktrace or search result
    - To read the complete body of a function that was truncated in search results
    - To get surrounding context around an error location
    - Set expand_to_scope=True to automatically get the full function/class

    WHEN NOT TO USE:
    - To search for code (use code_search instead)
    - To find a symbol by name (use find_identifier instead)

    OUTPUT FORMAT:
    Returns lines with line numbers prepended (e.g., "  42 | x = x + 1").
    The line numbers match the original file, allowing accurate diff generation.
    Also includes language detection and containing entity information.
    """
    # Validate file exists
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    # Ensure start <= end
    if start_line > end_line:
        start_line, end_line = end_line, start_line

    # Detect language from file extension
    file_path_obj = Path(file_path)
    language = EXTENSION_TO_LANGUAGE.get(file_path_obj.suffix.lower(), "unknown")

    # Try to find containing entity scope if expand_to_scope is requested
    containing_entity: dict[str, Any] | None = None
    scope_start: int | None = None
    scope_end: int | None = None

    if expand_to_scope:
        try:
            # Import here to avoid circular imports and allow standalone use
            from ..cache import CacheManager
            from ..search import SearchEngine

            cache = CacheManager(file_path_obj.parent)
            # Try to load cached indices if available
            try:
                cached_indices = cache.load_or_rebuild()
                engine = SearchEngine.from_cached_indices(cached_indices)

                # Look for scopes that contain the requested line range
                # Try both the absolute path and relative variations
                scopes = engine._file_scopes.get(file_path, [])
                if not scopes:
                    # Try relative path
                    for key in engine._file_scopes:
                        if file_path.endswith(key) or key.endswith(file_path_obj.name):
                            scopes = engine._file_scopes[key]
                            break

                # Find the smallest scope containing the target lines
                best_scope = None
                best_scope_size = float('inf')
                for s_start, s_end, node_id in scopes:
                    if s_start <= start_line <= s_end or s_start <= end_line <= s_end:
                        scope_size = s_end - s_start
                        if scope_size < best_scope_size:
                            best_scope = (s_start, s_end, node_id)
                            best_scope_size = scope_size

                if best_scope:
                    scope_start, scope_end, node_id = best_scope
                    entity = engine._entities.get(node_id)
                    if entity:
                        containing_entity = {
                            "name": entity.name,
                            "type": entity.entity_type,
                            "scope_start": scope_start,
                            "scope_end": scope_end,
                        }
            except Exception as e:
                logger.debug(f"Could not load search engine for scope expansion: {e}")
        except ImportError:
            logger.debug("SearchEngine not available for scope expansion")

    def _read_range_sync() -> dict[str, Any]:
        try:
            content = file_path_obj.read_text(encoding="utf-8", errors="replace")
            # Register file as read for write/edit verification
            mark_file_as_read(file_path, content)
            lines = content.splitlines()
            total_lines = len(lines)

            # Determine effective range
            effective_start = start_line
            effective_end = end_line

            # Apply scope expansion if we found a containing entity
            if expand_to_scope and scope_start is not None and scope_end is not None:
                effective_start = min(effective_start, scope_start)
                effective_end = max(effective_end, scope_end)

            # Apply context lines
            if context_lines > 0:
                effective_start = max(1, effective_start - context_lines)
                effective_end = min(total_lines, effective_end + context_lines) if total_lines > 0 else effective_end

            # Clamp to file bounds
            clamped_start = max(1, min(effective_start, total_lines)) if total_lines > 0 else 1
            clamped_end = max(1, min(effective_end, total_lines)) if total_lines > 0 else 1

            if total_lines == 0:
                return {
                    "content": "",
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": 0,
                    "language": language,
                    "note": "File is empty"
                }

            # Calculate line number width for alignment
            max_line_num = clamped_end
            line_num_width = len(str(max_line_num))

            # Extract and format lines (convert to 0-based indexing)
            output_lines = []
            for i in range(clamped_start - 1, clamped_end):
                line_num = i + 1
                line_content = lines[i]
                formatted = f"{line_num:>{line_num_width}} | {line_content}"
                output_lines.append(formatted)

            result: dict[str, Any] = {
                "content": "\n".join(output_lines),
                "start_line": clamped_start,
                "end_line": clamped_end,
                "original_start": start_line,
                "original_end": end_line,
                "total_lines": total_lines,
                "language": language,
            }

            # Add context info if we added context
            if context_lines > 0:
                result["context_before"] = start_line - clamped_start
                result["context_after"] = clamped_end - end_line

            # Add containing entity if found
            if containing_entity:
                result["containing_entity"] = containing_entity

            # Add notes if we clamped or expanded the range
            notes = []
            if clamped_start != effective_start or clamped_end != effective_end:
                notes.append(f"Range clamped to file bounds (1-{total_lines})")
            if expand_to_scope and scope_start is not None:
                notes.append(f"Expanded to scope {scope_start}-{scope_end}")
            if notes:
                result["note"] = "; ".join(notes)

            return result

        except PermissionError:
            return {"error": f"Permission denied reading file: {file_path}"}
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}

    return await asyncio.to_thread(_read_range_sync)


async def read_source_ranges(
    file_path: Annotated[
        str,
        Field(description="Absolute path to the file to read")
    ],
    ranges: Annotated[
        str,
        Field(description="Comma-separated line ranges, e.g. '10-20,45-50,100-110'")
    ],
    context_lines: Annotated[
        int,
        Field(description="Number of lines to include before and after each range for context", ge=0, le=50)
    ] = 0,
) -> dict[str, Any]:
    """Read multiple discontinuous line ranges from a file in a single call.

    This is the 'multi-zoom' tool - use it when you need to see several
    non-adjacent code sections at once, such as:
    - A function definition AND its usages
    - Multiple related error locations from a stacktrace
    - A class definition AND its methods scattered through a file

    WHEN TO USE:
    - When you need to see 2+ non-adjacent code sections from the same file
    - When read_source_range would require multiple calls

    OUTPUT FORMAT:
    Returns each range separately with line numbers, plus a combined view.
    """
    # Validate file exists
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    # Parse ranges string
    parsed_ranges: list[tuple[int, int]] = []
    try:
        for part in ranges.split(","):
            part = part.strip()
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
            else:
                # Single line
                start = end = int(part)
            if start < 1:
                start = 1
            if start > end:
                start, end = end, start
            parsed_ranges.append((start, end))
    except ValueError as e:
        return {"error": f"Invalid range format: {ranges}. Use '10-20,45-50' format. {e}"}

    if not parsed_ranges:
        return {"error": "No valid ranges provided"}

    # Detect language
    file_path_obj = Path(file_path)
    language = EXTENSION_TO_LANGUAGE.get(file_path_obj.suffix.lower(), "unknown")

    def _read_ranges_sync() -> dict[str, Any]:
        try:
            content = file_path_obj.read_text(encoding="utf-8", errors="replace")
            # Register file as read for write/edit verification
            mark_file_as_read(file_path, content)
            lines = content.splitlines()
            total_lines = len(lines)

            if total_lines == 0:
                return {
                    "ranges": [],
                    "total_lines": 0,
                    "language": language,
                    "note": "File is empty"
                }

            range_results: list[dict[str, Any]] = []
            all_output_lines: list[str] = []

            for idx, (start, end) in enumerate(parsed_ranges):
                # Apply context
                effective_start = max(1, start - context_lines)
                effective_end = min(total_lines, end + context_lines)

                # Clamp to file bounds
                clamped_start = max(1, min(effective_start, total_lines))
                clamped_end = max(1, min(effective_end, total_lines))

                # Calculate line number width
                line_num_width = len(str(clamped_end))

                # Extract and format lines
                output_lines = []
                for i in range(clamped_start - 1, clamped_end):
                    line_num = i + 1
                    line_content = lines[i]
                    formatted = f"{line_num:>{line_num_width}} | {line_content}"
                    output_lines.append(formatted)

                range_result: dict[str, Any] = {
                    "range_index": idx,
                    "requested": f"{start}-{end}",
                    "start_line": clamped_start,
                    "end_line": clamped_end,
                    "content": "\n".join(output_lines),
                }

                if context_lines > 0:
                    range_result["context_before"] = start - clamped_start
                    range_result["context_after"] = clamped_end - end

                range_results.append(range_result)

                # Add separator for combined view
                if all_output_lines:
                    all_output_lines.append(f"{'-' * 40} [Range {idx + 1}: lines {clamped_start}-{clamped_end}]")
                else:
                    all_output_lines.append(f"[Range {idx + 1}: lines {clamped_start}-{clamped_end}]")
                all_output_lines.extend(output_lines)

            return {
                "ranges": range_results,
                "combined_content": "\n".join(all_output_lines),
                "total_ranges": len(range_results),
                "total_lines": total_lines,
                "language": language,
            }

        except PermissionError:
            return {"error": f"Permission denied reading file: {file_path}"}
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}

    return await asyncio.to_thread(_read_ranges_sync)


async def git_history_context(
    file_path: Annotated[
        str,
        Field(description="Absolute path to the file to analyze")
    ],
    start_line: Annotated[
        int,
        Field(description="Starting line number (1-based indexing)", ge=1)
    ],
    end_line: Annotated[
        int,
        Field(description="Ending line number (1-based, inclusive)", ge=1)
    ],
    include_diff: Annotated[
        bool,
        Field(description="Include the diff showing what changed in the most recent commit")
    ] = False,
    include_line_blame: Annotated[
        bool,
        Field(description="Include per-line blame information showing which commit changed each line")
    ] = False,
    follow_renames: Annotated[
        bool,
        Field(description="Track file renames to find history before the file was renamed")
    ] = False,
    author: Annotated[
        str | None,
        Field(description="Filter results to only show changes by this author (name or email)")
    ] = None,
    since: Annotated[
        str | None,
        Field(description="Only show changes after this date (e.g., '2024-01-01', '3 months ago')")
    ] = None,
    until: Annotated[
        str | None,
        Field(description="Only show changes before this date (e.g., '2024-12-31', 'yesterday')")
    ] = None,
) -> dict[str, Any]:
    """Get git blame information and commit messages for a line range.

    This is the 'time machine' tool - use it to understand who changed code
    and why, helping identify if a bug was introduced by a recent change.

    WHEN TO USE:
    - When debugging a regression to find when/why code changed
    - To understand the intent behind specific lines of code
    - To identify the author for follow-up questions
    - Set include_diff=True to see WHAT changed in the commit
    - Set include_line_blame=True to see per-line attribution
    - Set follow_renames=True to track history across file renames
    - Use author filter to find changes by a specific person
    - Use since/until filters to narrow down to a time period

    WHEN NOT TO USE:
    - For general code exploration (use code_search or code_map)
    - When git history doesn't matter for the task
    - For broad churn analysis (use find_hotspots instead)
    - For function-level evolution history (use code_evolution instead)

    OUTPUT:
    Returns a summary including author, commit hash, date, and commit message
    for the most recent change affecting the specified line range.
    Optionally includes diff, per-line blame, and rename history.
    When filters are applied, only matching commits are included.
    """
    # Validate file exists
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    # Ensure start <= end
    if start_line > end_line:
        start_line, end_line = end_line, start_line

    # Get the directory containing the file for git operations
    file_dir = str(Path(file_path).parent)

    # First, check if we're in a git repository
    returncode, stdout_check, stderr = await run_git_command(["rev-parse", "--git-dir"], file_dir)
    if returncode != 0:
        error_detail = stderr.strip() if stderr.strip() else f"git rev-parse failed in {file_dir}"
        return {"error": f"Not a git repository: {error_detail}"}

    # Run git blame with porcelain format for the line range
    blame_args = ["blame", "-L", f"{start_line},{end_line}", "--porcelain"]
    if follow_renames:
        # -C -C -C: detect copies/renames across files, even in different commits
        blame_args.extend(["-C", "-C", "-C"])
    blame_args.append(file_path)
    returncode, stdout, stderr = await run_git_command(blame_args, file_dir)

    if returncode != 0:
        if "no such path" in stderr.lower():
            return {"error": f"File not tracked by git: {file_path}"}
        return {"error": f"git blame failed: {stderr.strip()}"}

    if not stdout.strip():
        return {"error": "No blame information available for the specified lines"}

    # Parse porcelain output to extract commit info and per-line blame
    # Porcelain format: each block starts with commit hash, followed by metadata
    commits: dict[str, dict[str, Any]] = {}
    current_commit = ""
    line_blame: list[dict[str, Any]] = []
    current_line_num = start_line

    for line in stdout.splitlines():
        # Line starting with 40-char hash indicates new commit block
        if re.match(r'^[0-9a-f]{40}', line):
            parts = line.split()
            current_commit = parts[0]
            if current_commit not in commits:
                commits[current_commit] = {}
            # Track which line this blame entry is for
            if include_line_blame:
                line_blame.append({
                    "line": current_line_num,
                    "hash": current_commit[:8],
                })
                current_line_num += 1
        elif current_commit:
            # Parse metadata lines
            if line.startswith("author "):
                commits[current_commit]["author"] = line[7:]
                if include_line_blame and line_blame:
                    line_blame[-1]["author"] = line[7:]
            elif line.startswith("author-mail "):
                commits[current_commit]["author_email"] = line[12:].strip("<>")
            elif line.startswith("author-time "):
                commits[current_commit]["author_time"] = line[12:]
                if include_line_blame and line_blame:
                    try:
                        ts = int(line[12:])
                        line_blame[-1]["date"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    except (ValueError, OSError):
                        pass
            elif line.startswith("summary "):
                commits[current_commit]["summary"] = line[8:]

    if not commits:
        return {"error": "Could not parse git blame output"}

    # Apply filters to commits
    filtered_commits: dict[str, dict[str, Any]] = {}
    for commit_hash, commit_info in commits.items():
        # Author filter: check if author name or email contains the filter string
        if author:
            commit_author = commit_info.get("author", "").lower()
            commit_email = commit_info.get("author_email", "").lower()
            author_lower = author.lower()
            if author_lower not in commit_author and author_lower not in commit_email:
                continue

        # Date filters: check commit timestamp
        try:
            commit_ts = int(commit_info.get("author_time", "0"))
        except (ValueError, TypeError):
            commit_ts = 0

        if since and commit_ts > 0:
            # Parse 'since' date - try common formats
            since_ts = _parse_date_to_timestamp(since)
            if since_ts is not None and commit_ts < since_ts:
                continue

        if until and commit_ts > 0:
            # Parse 'until' date - try common formats
            until_ts = _parse_date_to_timestamp(until)
            if until_ts is not None and commit_ts > until_ts:
                continue

        filtered_commits[commit_hash] = commit_info

    # If filters removed all commits, report that
    if not filtered_commits and (author or since or until):
        filter_desc = []
        if author:
            filter_desc.append(f"author='{author}'")
        if since:
            filter_desc.append(f"since='{since}'")
        if until:
            filter_desc.append(f"until='{until}'")
        return {
            "message": f"No commits match the specified filters: {', '.join(filter_desc)}",
            "total_commits_before_filter": len(commits),
        }

    # Use filtered commits if filters were applied, otherwise use all commits
    commits = filtered_commits if (author or since or until) else commits

    # Sort commits by time (newest first)
    sorted_commits = sorted(
        commits.items(),
        key=lambda x: int(x[1].get("author_time", "0")),
        reverse=True
    )

    # Find the most recent commit
    most_recent_hash, most_recent = sorted_commits[0]

    # Find the oldest (first) commit for "when introduced"
    oldest_hash, oldest = sorted_commits[-1]

    # Get the full commit message for the most recent commit
    show_args = ["show", "-s", "--format=%B", most_recent_hash]
    returncode, commit_msg, _ = await run_git_command(show_args, file_dir)

    if returncode == 0:
        commit_message = commit_msg.strip()
    else:
        commit_message = most_recent.get("summary", "")

    # Format dates from Unix timestamp
    def format_timestamp(ts_str: str) -> str:
        try:
            timestamp = int(ts_str)
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return "unknown"

    date_str = format_timestamp(most_recent.get("author_time", "0"))

    # Build result
    author = most_recent.get("author", "Unknown")
    short_hash = most_recent_hash[:8]

    summary = f"Lines {start_line}-{end_line} last modified by {author} in commit {short_hash}: {most_recent.get('summary', 'No message')}"

    result: dict[str, Any] = {
        "summary": summary,
        "most_recent": {
            "commit_hash": most_recent_hash,
            "short_hash": short_hash,
            "author": author,
            "author_email": most_recent.get("author_email", ""),
            "date": date_str,
            "commit_message": commit_message,
        },
        "commits_in_range": len(commits),
        "all_commits": [
            {
                "hash": h[:8],
                "author": info.get("author", "Unknown"),
                "summary": info.get("summary", ""),
                "date": format_timestamp(info.get("author_time", "0")),
            }
            for h, info in sorted_commits
        ],
        # Keep legacy fields for backwards compatibility
        "commit_hash": most_recent_hash,
        "short_hash": short_hash,
        "author": author,
        "author_email": most_recent.get("author_email", ""),
        "date": date_str,
        "commit_message": commit_message,
    }

    # Add "first introduced" info if different from most recent
    if oldest_hash != most_recent_hash:
        result["first_introduced"] = {
            "commit_hash": oldest_hash,
            "short_hash": oldest_hash[:8],
            "author": oldest.get("author", "Unknown"),
            "date": format_timestamp(oldest.get("author_time", "0")),
            "summary": oldest.get("summary", ""),
        }

    # Add per-line blame if requested
    if include_line_blame and line_blame:
        # Enrich line blame entries with author info from commits dict
        for entry in line_blame:
            full_hash = None
            for h in commits:
                if h.startswith(entry["hash"]):
                    full_hash = h
                    break
            if full_hash and full_hash in commits:
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

    # Add diff if requested
    if include_diff:
        # Get the diff for the most recent commit affecting this file
        diff_args = [
            "show", most_recent_hash,
            "--format=",  # No commit message, just diff
            "-p",  # Patch format
            "--",
            file_path
        ]
        returncode, diff_output, _ = await run_git_command(diff_args, file_dir)

        if returncode == 0 and diff_output.strip():
            # Extract just the relevant lines from the diff
            # Filter to show only hunks that touch our line range
            filtered_diff = _filter_diff_to_range(diff_output, start_line, end_line)
            result["diff"] = filtered_diff if filtered_diff else diff_output.strip()

    # Add rename history if requested
    if follow_renames:
        # Get file rename history using git log --follow
        log_args = [
            "log", "--follow", "--name-status", "--format=%H",
            "--diff-filter=R",  # Only show renames
            "--", file_path
        ]
        returncode, log_output, _ = await run_git_command(log_args, file_dir)

        if returncode == 0 and log_output.strip():
            rename_history: list[dict[str, str]] = []
            lines = log_output.strip().splitlines()
            current_hash = ""

            for line in lines:
                if re.match(r'^[0-9a-f]{40}$', line):
                    current_hash = line
                elif line.startswith("R") and current_hash:
                    # Format: R100\told_name\tnew_name or R\told_name\tnew_name
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        old_name = parts[1]
                        new_name = parts[2]
                        rename_history.append({
                            "commit": current_hash[:8],
                            "old_name": old_name,
                            "new_name": new_name,
                        })

            if rename_history:
                result["rename_history"] = rename_history
                result["note"] = f"File was renamed {len(rename_history)} time(s)"

    return result


def _filter_diff_to_range(diff_output: str, start_line: int, end_line: int) -> str:
    """Filter a diff to only include hunks that affect the specified line range."""
    lines = diff_output.splitlines()
    result_lines: list[str] = []
    in_relevant_hunk = False
    hunk_header_pattern = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')

    for line in lines:
        # Check if this is a hunk header
        match = hunk_header_pattern.match(line)
        if match:
            hunk_start = int(match.group(1))
            hunk_len = int(match.group(2)) if match.group(2) else 1
            hunk_end = hunk_start + hunk_len - 1

            # Check if this hunk overlaps with our range
            in_relevant_hunk = (
                (hunk_start <= end_line and hunk_end >= start_line)
            )
            if in_relevant_hunk:
                result_lines.append(line)
        elif in_relevant_hunk:
            result_lines.append(line)
        elif line.startswith('diff --git') or line.startswith('index ') or line.startswith('--- ') or line.startswith('+++ '):
            # Always include file header lines
            if not result_lines or not result_lines[-1].startswith('diff'):
                result_lines.append(line)

    return "\n".join(result_lines)
