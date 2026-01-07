"""Git-aware code analysis tools.

This module provides MCP tools for analyzing code through its git history:
- find_hotspots: Identify frequently changed code areas (churn analysis)
- code_evolution: Trace how a specific code entity has evolved over time
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from coden_retriever.git.commands import get_git_root, is_git_repository, run_git_command
from coden_retriever.git.parsers import (
    aggregate_file_stats,
    parse_file_dates,
    parse_function_log,
    parse_log_oneline,
    parse_numstat,
)

logger = logging.getLogger(__name__)


async def find_hotspots(
    root_directory: Annotated[
        str,
        Field(description="Root directory of the git repository to analyze"),
    ],
    path_filter: Annotated[
        str | None,
        Field(
            description="Optional glob pattern to filter files (e.g., 'src/auth/**/*.py', '*.js')"
        ),
    ] = None,
    since: Annotated[
        str | None,
        Field(
            description="Start date for analysis (e.g., '2024-01-01', '3 months ago', '2024-06-01')"
        ),
    ] = None,
    until: Annotated[
        str | None,
        Field(description="End date for analysis (e.g., '2024-12-31', 'yesterday')"),
    ] = None,
    min_commits: Annotated[
        int,
        Field(
            description="Minimum number of commits for a file to qualify as a hotspot",
            ge=1,
        ),
    ] = 3,
    limit: Annotated[
        int,
        Field(description="Maximum number of hotspots to return", ge=1, le=100),
    ] = 20,
    include_authors: Annotated[
        bool,
        Field(description="Include author breakdown for each hotspot"),
    ] = True,
) -> dict[str, Any]:
    """Find code hotspots - files that change frequently.

    Hotspots often indicate:
    - Areas with bugs that need repeated fixes
    - Features under active development
    - Code that may benefit from refactoring
    - Critical business logic that evolves with requirements

    WHEN TO USE:
    - To identify which parts of the codebase change most often
    - To find potential candidates for refactoring
    - To understand which files have the most development activity
    - To identify areas that may need better test coverage

    WHEN NOT TO USE:
    - For non-git repositories
    - When you need line-level blame information (use git_history_context instead)
    - When you need to understand a specific function's history (use code_evolution instead)

    OUTPUT FORMAT:
    Returns a dictionary with:
    - analysis_period: Date range and total commits analyzed
    - hotspots: List of files ranked by change frequency, each with:
      - file: Path relative to repository root
      - commits: Number of commits affecting this file
      - unique_authors: Number of distinct contributors
      - lines_churned: Total lines added + removed
      - first_change: Date of earliest change in period
      - last_change: Date of most recent change
      - top_authors: (if include_authors=True) List of top contributors
    """
    # Validate root directory exists
    root_path = Path(root_directory)
    if not root_path.is_dir():
        return {"error": f"Directory not found: {root_directory}"}

    # Check if it's a git repository
    if not await is_git_repository(root_directory):
        return {"error": f"Not a git repository: {root_directory}"}

    # Get the git root (may be different from root_directory)
    git_root = await get_git_root(root_directory)
    if not git_root:
        return {"error": f"Could not determine git root for: {root_directory}"}

    # Build the git log command for file changes
    log_args = ["log", "--format=%H", "--numstat"]

    if since:
        log_args.extend(["--since", since])
    if until:
        log_args.extend(["--until", until])

    # Add path filter if specified
    if path_filter:
        log_args.append("--")
        log_args.append(path_filter)

    # Execute git log
    returncode, stdout, stderr = await run_git_command(log_args, git_root)
    if returncode != 0:
        return {"error": f"git log failed: {stderr.strip()}"}

    if not stdout.strip():
        return {
            "analysis_period": {
                "since": since,
                "until": until,
                "total_commits": 0,
            },
            "hotspots": [],
            "message": "No commits found in the specified period",
        }

    # Parse numstat output
    numstat_results = parse_numstat(stdout)

    # Aggregate per-file stats
    file_stats = aggregate_file_stats(numstat_results)

    # Apply path filter (glob matching) if specified and wasn't handled by git
    if path_filter:
        filtered_stats = {}
        for filepath, stats in file_stats.items():
            # Support simple directory prefix (e.g., "src") and glob patterns
            # fnmatch doesn't handle directory prefixes well, so check both
            is_match = (
                fnmatch.fnmatch(filepath, path_filter)  # Exact glob match
                or filepath.startswith(path_filter.rstrip("/") + "/")  # Directory prefix
                or fnmatch.fnmatch(filepath, path_filter.rstrip("/") + "/*")  # Pattern/*
            )
            if is_match:
                filtered_stats[filepath] = stats
        file_stats = filtered_stats

    # Filter by min_commits
    file_stats = {
        f: s for f, s in file_stats.items() if s["commit_count"] >= min_commits
    }

    # Sort by commit count (descending)
    sorted_files = sorted(
        file_stats.items(),
        key=lambda x: (x[1]["commit_count"], x[1]["churned"]),
        reverse=True,
    )[:limit]

    # Get commit count for the period
    count_args = ["rev-list", "--count", "HEAD"]
    if since:
        count_args.extend(["--since", since])
    if until:
        count_args.extend(["--until", until])
    returncode, count_out, _ = await run_git_command(count_args, git_root)
    total_commits = int(count_out.strip()) if returncode == 0 and count_out.strip() else 0

    # Build hotspots list
    hotspots: list[dict[str, Any]] = []

    for filepath, stats in sorted_files:
        hotspot: dict[str, Any] = {
            "file": filepath,
            "commits": stats["commit_count"],
            "lines_churned": stats["churned"],
            "lines_added": stats["added"],
            "lines_removed": stats["removed"],
        }

        # Get date range for this file
        date_args = ["log", "--format=%at"]
        if since:
            date_args.extend(["--since", since])
        if until:
            date_args.extend(["--until", until])
        date_args.extend(["--", filepath])

        returncode, date_out, _ = await run_git_command(date_args, git_root)
        if returncode == 0:
            first_date, last_date = parse_file_dates(date_out)
            hotspot["first_change"] = first_date
            hotspot["last_change"] = last_date

        # Get author stats if requested
        if include_authors:
            # Use git log instead of shortlog (shortlog doesn't work well non-interactively)
            author_args = ["log", "--format=%an", "--no-merges"]
            if since:
                author_args.extend(["--since", since])
            if until:
                author_args.extend(["--until", until])
            author_args.extend(["--", filepath])

            returncode, author_out, _ = await run_git_command(author_args, git_root)
            if returncode == 0 and author_out.strip():
                # Count commits per author
                author_counts: dict[str, int] = {}
                for author_name in author_out.strip().splitlines():
                    author_name = author_name.strip()
                    if author_name:
                        author_counts[author_name] = author_counts.get(author_name, 0) + 1

                # Sort by commit count
                sorted_authors = sorted(
                    author_counts.items(), key=lambda x: x[1], reverse=True
                )
                hotspot["unique_authors"] = len(sorted_authors)
                hotspot["top_authors"] = [
                    {"name": name, "commits": count} for name, count in sorted_authors[:5]
                ]
            else:
                hotspot["unique_authors"] = 0
                hotspot["top_authors"] = []
        else:
            hotspot["unique_authors"] = 0

        hotspots.append(hotspot)

    return {
        "analysis_period": {
            "since": since or "beginning",
            "until": until or "now",
            "total_commits": total_commits,
        },
        "hotspots": hotspots,
    }


async def code_evolution(
    root_directory: Annotated[
        str,
        Field(description="Root directory of the git repository"),
    ],
    entity_name: Annotated[
        str,
        Field(
            description="Name of the function, class, or method to trace (e.g., 'UserAuth.validate', 'process_payment', 'MyClass')"
        ),
    ],
    file_path: Annotated[
        str | None,
        Field(
            description="Specific file path (relative or absolute) if entity name is ambiguous across files"
        ),
    ] = None,
    since: Annotated[
        str | None,
        Field(
            description="Start date for history (e.g., '6 months ago', '2024-01-01')"
        ),
    ] = None,
    max_commits: Annotated[
        int,
        Field(description="Maximum number of commits to show", ge=1, le=50),
    ] = 10,
    include_diffs: Annotated[
        bool,
        Field(description="Include code diffs for each change (can be verbose)"),
    ] = False,
    diff_context_lines: Annotated[
        int,
        Field(description="Lines of context around changes in diffs", ge=0, le=20),
    ] = 3,
) -> dict[str, Any]:
    """Trace the evolution of a specific code entity over time.

    Shows how a function, class, or method has been modified across commits,
    helping understand:
    - Why the code looks the way it does today
    - Who has been responsible for changes
    - Whether the entity is stable or frequently modified
    - The progression of features or bug fixes

    WHEN TO USE:
    - To understand the history of a specific function or class
    - To see who has modified a piece of code and why
    - To trace the evolution of a feature or bug fix
    - To understand design decisions through commit messages

    WHEN NOT TO USE:
    - For broad file-level churn analysis (use find_hotspots instead)
    - For line-by-line blame information (use git_history_context instead)
    - When you don't know the entity name (search first)

    OUTPUT FORMAT:
    Returns a dictionary with:
    - entity: Name, file location, and overall statistics
    - evolution: Chronological list of commits that modified the entity
    - summary: High-level statistics about the entity's history
    """
    # Validate root directory
    root_path = Path(root_directory)
    if not root_path.is_dir():
        return {"error": f"Directory not found: {root_directory}"}

    # Check if it's a git repository
    if not await is_git_repository(root_directory):
        return {"error": f"Not a git repository: {root_directory}"}

    git_root = await get_git_root(root_directory)
    if not git_root:
        return {"error": f"Could not determine git root for: {root_directory}"}

    # Resolve file path if provided
    target_file: str | None = None
    if file_path:
        # Handle both absolute and relative paths
        file_p = Path(file_path)
        if file_p.is_absolute():
            target_file = str(file_p)
        else:
            target_file = str(Path(git_root) / file_path)

        if not Path(target_file).exists():
            return {"error": f"File not found: {file_path}"}

        # Convert to relative path for git
        try:
            target_file = str(Path(target_file).relative_to(git_root))
        except ValueError:
            return {"error": f"File is not within repository: {file_path}"}

    # Try function-level tracking first (git log -L)
    # This requires knowing the file path
    evolution_commits: list[dict[str, Any]] = []

    if target_file:
        # Use git log -L for function tracking (Git 2.15+)
        log_args = ["log", f"-L:{entity_name}:{target_file}"]
        if since:
            log_args.extend(["--since", since])
        log_args.extend(["-n", str(max_commits)])

        returncode, stdout, stderr = await run_git_command(log_args, git_root)

        if returncode == 0 and stdout.strip():
            # Parse the function-level log
            evolution_commits = parse_function_log(stdout)

            # Remove diffs if not requested
            if not include_diffs:
                for commit in evolution_commits:
                    commit.pop("diff", None)
        elif "no match" in stderr.lower() or "fatal" in stderr.lower():
            # Function not found or not supported, fall back to search
            pass

    # Fallback: Use git log -S to search for the entity name
    if not evolution_commits:
        # Search for the entity in the repository
        search_args = [
            "log",
            "--format=%H|%an|%ae|%at|%s",
            "-S",
            entity_name,
            "-n",
            str(max_commits),
        ]
        if since:
            search_args.extend(["--since", since])
        if target_file:
            search_args.extend(["--", target_file])

        returncode, stdout, stderr = await run_git_command(search_args, git_root)

        if returncode != 0:
            return {"error": f"git log failed: {stderr.strip()}"}

        if not stdout.strip():
            return {
                "error": f"Entity '{entity_name}' not found in repository history",
                "suggestion": "Try providing a file_path if the entity exists in a specific file",
            }

        # Parse the commit list
        commits = parse_log_oneline(stdout)

        for commit_info in commits:
            entry: dict[str, Any] = {
                "commit": commit_info["commit"][:8],
                "full_commit": commit_info["commit"],
                "date": commit_info["date"],
                "author": f"{commit_info['author']} <{commit_info['email']}>",
                "subject": commit_info["subject"],
            }

            # Get diff for this commit if requested
            if include_diffs:
                diff_args = [
                    "show",
                    commit_info["commit"],
                    f"--unified={diff_context_lines}",
                    "--format=",
                    "-p",
                ]
                if target_file:
                    diff_args.extend(["--", target_file])

                returncode, diff_out, _ = await run_git_command(diff_args, git_root)
                if returncode == 0:
                    entry["diff"] = diff_out.strip()

                    # Count lines changed
                    lines_added = sum(
                        1
                        for line in diff_out.splitlines()
                        if line.startswith("+") and not line.startswith("+++")
                    )
                    lines_removed = sum(
                        1
                        for line in diff_out.splitlines()
                        if line.startswith("-") and not line.startswith("---")
                    )
                    entry["lines_added"] = lines_added
                    entry["lines_removed"] = lines_removed

            evolution_commits.append(entry)

    if not evolution_commits:
        return {
            "error": f"No history found for entity '{entity_name}'",
            "suggestion": "Verify the entity name is correct and exists in the repository",
        }

    # Calculate summary statistics
    authors: dict[str, int] = {}
    total_added = 0
    total_removed = 0

    for commit in evolution_commits:
        author = commit.get("author", "").split("<")[0].strip()
        if author:
            authors[author] = authors.get(author, 0) + 1
        total_added += commit.get("lines_added", 0)
        total_removed += commit.get("lines_removed", 0)

    most_active = max(authors.items(), key=lambda x: x[1]) if authors else ("unknown", 0)

    # Get first introduction date if we have full history
    first_introduced = None
    if evolution_commits:
        # The last commit in our list is the oldest (git log returns newest first)
        oldest_commit = evolution_commits[-1]
        first_introduced = oldest_commit.get("date")

    return {
        "entity": {
            "name": entity_name,
            "file": target_file or "(multiple or unknown)",
            "total_commits": len(evolution_commits),
            "total_authors": len(authors),
        },
        "evolution": evolution_commits,
        "summary": {
            "first_introduced": first_introduced,
            "most_active_author": f"{most_active[0]} ({most_active[1]} commits)",
            "total_lines_added": total_added,
            "total_lines_removed": total_removed,
            "average_change_size": (
                f"{(total_added + total_removed) // len(evolution_commits)} lines"
                if evolution_commits
                else "0 lines"
            ),
        },
    }
