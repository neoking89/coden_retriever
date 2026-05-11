"""
Precision inspection MCP tools for deep code analysis.

Provides read_source_range for reading specific line ranges with line numbers,
and git_history_context for understanding recent changes via git blame.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from .file_edit import mark_file_as_read
from ..git.blame_port import SubprocessGitBlameSource
from ..git.history_context import GitHistoryContextService, HistoryContextRequest

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
    service = GitHistoryContextService(SubprocessGitBlameSource())
    return await service.run(HistoryContextRequest(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        include_diff=include_diff,
        include_line_blame=include_line_blame,
        follow_renames=follow_renames,
        author=author,
        since=since,
        until=until,
    ))
