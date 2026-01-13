"""
File Editing MCP Tools module.

Provides high-fidelity file manipulation tools following the Autonomous File Manipulation
Protocol (AFMP) for surgical code modifications. Implements:
- write_file: Create or overwrite files with absolute path resolution
- edit_file: SEARCH/REPLACE or SYMBOL-based surgical editing (Cline/RooCode + Windsurf style)
- delete_file: Remove files with read-before-delete verification
- undo_file_change: Revert the last change to a file (one-step undo per file)

All operations use absolute paths and require read-before-write verification.
Files must be read using read_source_range or read_source_ranges (from inspection.py)
before they can be edited or overwritten. This prevents accidental overwrites.

Security Note:
    By default, these tools can write to any absolute path the process has access to.
    For sandboxed environments, configure allowed paths via set_allowed_paths().

Architecture Note:
    This module uses SOLID-compliant classes from file_edit_core.py:
    - FileCache: Thread-safe LRU cache for tracking read files
    - PathPermissions: Path boundary checking and permission management
    - UndoManager: One-step undo buffer management
    - PatternMatcher: Regex pattern matching for SEARCH/REPLACE blocks
"""
import asyncio
import difflib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from .file_edit_core import (
    get_file_cache,
    get_path_permissions,
    get_pattern_matcher,
    get_undo_manager,
)

if TYPE_CHECKING:
    from ..models import CodeEntity

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================




def _validate_path_for_write(file_path: str) -> tuple[str, dict[str, Any] | None]:
    """Validate a path for write operations and resolve it.

    Delegates to PathPermissions class for path boundary checking.

    Args:
        file_path: The path to validate.

    Returns:
        Tuple of (resolved_absolute_path, error_dict_or_none).
        If error_dict is not None, the path is invalid.
    """
    return get_path_permissions().validate_path_for_write(file_path)


def _normalize_line_endings(content: str) -> str:
    """Normalize line endings to Unix-style (LF)."""
    return content.replace('\r\n', '\n').replace('\r', '\n')


def _is_path_allowed(abs_path: str) -> bool:
    """Check if a path is within the allowed paths for write operations.

    Delegates to PathPermissions class.

    Returns True if:
    - allowed_paths is empty (no restrictions)
    - The path is under one of the allowed directories
    """
    return get_path_permissions().is_path_allowed(abs_path)


def set_allowed_paths(paths: list[str]) -> None:
    """Configure allowed paths for write operations.

    Delegates to PathPermissions class.

    Args:
        paths: List of directory paths. Only files under these directories
               can be written. Empty list removes restrictions.
    """
    get_path_permissions().set_allowed_paths(paths)


def get_allowed_paths() -> list[str]:
    """Get the currently configured allowed paths.

    Delegates to PathPermissions class.
    """
    return get_path_permissions().get_allowed_paths()


def clear_read_cache() -> None:
    """Clear the read file cache. Useful for testing or session reset.

    Delegates to FileCache class.
    """
    get_file_cache().clear()


def clear_undo_buffer() -> None:
    """Clear the undo buffer. Useful for testing or session reset.

    Delegates to UndoManager class.
    """
    get_undo_manager().clear()


def _save_for_undo(abs_path: str, was_new_file: bool = False) -> None:
    """Save the current state of a file to the undo buffer.

    Delegates to UndoManager class.

    Args:
        abs_path: Absolute path to the file.
        was_new_file: If True, the file didn't exist before this operation.
    """
    get_undo_manager().save_for_undo(abs_path, was_new_file)


def has_undo_history(file_path: str) -> bool:
    """Check if a file has undo history available.

    Delegates to UndoManager class.
    """
    return get_undo_manager().has_history(file_path)


def mark_file_as_read(file_path: str, content: str) -> None:
    """Mark a file as having been read with its content.

    Delegates to FileCache class.

    This is called by read tools to register files for write verification.
    Uses LRU eviction when cache exceeds MAX_CACHE_SIZE.
    """
    get_file_cache().mark_as_read(file_path, content)


def get_cached_content(file_path: str) -> str | None:
    """Get cached content for a file if it was previously read.

    Delegates to FileCache class.
    """
    return get_file_cache().get_content(file_path)


def was_file_read(file_path: str) -> bool:
    """Check if a file was previously read in this session.

    Delegates to FileCache class.
    """
    return get_file_cache().was_read(file_path)


def _find_symbol_in_file(
    file_path: str,
    symbol_name: str,
    source_code: str,
) -> tuple["CodeEntity | None", list[str]]:
    """Find a symbol (function, class, method) in a file using Tree-sitter AST.

    Args:
        file_path: Absolute path to the file.
        symbol_name: Symbol to find. Can be:
            - "function_name" for top-level functions
            - "ClassName" for classes
            - "ClassName.method_name" for methods within a class
        source_code: The source code content to parse.

    Returns:
        Tuple of (matched_entity, available_symbols).
        If no match, entity is None and available_symbols lists what's available.
    """
    try:
        from ..parsers.tree_sitter_parser import RepoParser
    except ImportError:
        logger.warning("Tree-sitter parser not available for symbol lookup")
        return None, []

    parser = RepoParser()
    try:
        entities, _ = parser.parse_file(file_path, source_code)
    except Exception as e:
        logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
        return None, []

    if not entities:
        return None, []

    # Build list of available symbols for error messages
    available: list[str] = []
    for e in entities:
        if e.parent_class:
            available.append(f"{e.parent_class}.{e.name}")
        else:
            available.append(e.name)

    # Parse symbol_name for class.method pattern
    if "." in symbol_name:
        parts = symbol_name.split(".", 1)
        if len(parts) == 2:
            class_name, method_name = parts
            for entity in entities:
                if entity.name == method_name and entity.parent_class == class_name:
                    return entity, available
    else:
        # Look for exact match (function or class)
        for entity in entities:
            if entity.name == symbol_name and entity.parent_class is None:
                return entity, available

    return None, available


def _extract_symbol_source(
    source_code: str,
    entity: "CodeEntity",
) -> str:
    """Extract the full source code of a symbol from the file.

    Args:
        source_code: The full file content.
        entity: The CodeEntity with line range information.

    Returns:
        The source code for just that symbol (from line_start to line_end).
    """
    lines = source_code.split('\n')
    # line_start and line_end are 1-indexed
    start_idx = entity.line_start - 1
    end_idx = entity.line_end  # inclusive, so no -1 on end
    return '\n'.join(lines[start_idx:end_idx])


def _parse_edit_blocks(
    diff_content: str,
) -> list[tuple[int, str, str, str]]:
    """Parse SEARCH/REPLACE and SYMBOL/REPLACE blocks from diff content.

    Delegates to PatternMatcher class.

    Args:
        diff_content: The normalized diff content containing edit blocks.

    Returns:
        List of tuples: (position, block_type, search_content, replace_content)
        Sorted by position in the original diff_content.
    """
    blocks = get_pattern_matcher().parse_edit_blocks(diff_content)
    # Convert EditBlock objects to tuples for backwards compatibility
    return [(b.position, b.block_type, b.search_content, b.replace_content) for b in blocks]


def _find_closest_match(
    search_text: str,
    file_lines: list[str],
) -> tuple[float, int]:
    """Find the closest matching region in file_lines for search_text.

    Uses optimized similarity search with early termination and sampling
    for large files.

    Args:
        search_text: The text to search for.
        file_lines: The file content split into lines.

    Returns:
        Tuple of (best_similarity_ratio, best_start_line_index).
    """
    search_lines = search_text.split('\n')
    search_len = len(search_lines)
    num_positions = len(file_lines) - search_len + 1

    best_ratio = 0.0
    best_start = 0

    if num_positions <= 0:
        return best_ratio, best_start

    # Thresholds for optimization
    quick_threshold = 0.4  # Pre-filter threshold
    good_enough_threshold = 0.9  # Stop early if match is this good

    if num_positions > 100:
        # For large files: sample positions first to find promising regions
        sample_step = max(1, num_positions // 50)
        sampled_positions = list(range(0, num_positions, sample_step))

        # Find best region in samples
        best_sample_pos = 0
        best_sample_ratio = 0.0
        for i in sampled_positions:
            candidate = '\n'.join(file_lines[i:i + search_len])
            matcher = difflib.SequenceMatcher(None, search_text, candidate)
            if matcher.quick_ratio() > quick_threshold:
                ratio = matcher.ratio()
                if ratio > best_sample_ratio:
                    best_sample_ratio = ratio
                    best_sample_pos = i
                if ratio >= good_enough_threshold:
                    break

        # Search around the best sample position with full precision
        search_start = max(0, best_sample_pos - sample_step)
        search_end = min(num_positions, best_sample_pos + sample_step + 1)
        for i in range(search_start, search_end):
            candidate = '\n'.join(file_lines[i:i + search_len])
            matcher = difflib.SequenceMatcher(None, search_text, candidate)
            if matcher.quick_ratio() > max(quick_threshold, best_ratio - 0.1):
                ratio = matcher.ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                if ratio >= good_enough_threshold:
                    break
    else:
        # For smaller files: check all positions with quick_ratio filter
        for i in range(num_positions):
            candidate = '\n'.join(file_lines[i:i + search_len])
            matcher = difflib.SequenceMatcher(None, search_text, candidate)
            if matcher.quick_ratio() > max(quick_threshold, best_ratio - 0.1):
                ratio = matcher.ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                if ratio >= good_enough_threshold:
                    break

    return best_ratio, best_start


def _resolve_symbol_blocks(
    all_blocks: list[tuple[int, str, str, str]],
    file_path: str,
    file_content: str,
) -> tuple[list[tuple[str, str, str]], dict[str, Any] | None]:
    """Resolve SYMBOL blocks to their source code using Tree-sitter.

    Args:
        all_blocks: Parsed blocks from _parse_edit_blocks.
        file_path: Absolute path to the file being edited.
        file_content: The normalized file content.

    Returns:
        Tuple of (resolved_blocks, error_or_none).
        resolved_blocks: List of (search_text, replace_text, block_type).
        If error is not None, resolution failed.
    """
    resolved_blocks: list[tuple[str, str, str]] = []

    for _, block_type, content, replace_text in all_blocks:
        if block_type == 'symbol':
            entity, available = _find_symbol_in_file(file_path, content, file_content)
            if entity is None:
                error_response: dict[str, Any] = {
                    "error": f"Symbol '{content}' not found in file",
                    "hint": "Use ClassName.method_name for methods, or just function_name for top-level functions.",
                }
                if available:
                    error_response["available_symbols"] = available[:20]
                else:
                    error_response["hint"] = (
                        "No symbols could be parsed from this file. "
                        "Ensure the file has valid syntax and Tree-sitter support for this language."
                    )
                return [], error_response
            search_text = _extract_symbol_source(file_content, entity)
            resolved_blocks.append((search_text, replace_text, 'symbol'))
        else:
            resolved_blocks.append((content, replace_text, 'search'))

    return resolved_blocks, None


def _build_no_match_error(
    search_text: str,
    file_lines: list[str],
    block_type: str,
    block_idx: int,
) -> dict[str, Any]:
    """Build detailed error response when a search block doesn't match.

    Args:
        search_text: The text that failed to match.
        file_lines: The file content split into lines.
        block_type: Either 'search' or 'symbol'.
        block_idx: The 0-based index of the block.

    Returns:
        Error dict with helpful debugging information.
    """
    search_lines = search_text.split('\n')
    search_len = len(search_lines)
    block_label = "SYMBOL" if block_type == "symbol" else "SEARCH"

    error_msg: dict[str, Any] = {
        "error": f"{block_label} block {block_idx + 1} not found - no exact match in file",
        "block_type": block_type,
        "search_block_preview": search_text[:300] + "..." if len(search_text) > 300 else search_text,
        "hint": (
            "The resolved content must match character-for-character. "
            "Check whitespace, indentation, and invisible trailing spaces."
            if block_type == "search" else
            "The symbol was found but its source doesn't match the file. "
            "The file may have been modified since it was read."
        ),
    }

    # Find closest match for helpful error messages
    best_ratio, best_start = _find_closest_match(search_text, file_lines)

    if best_ratio > 0.5:
        # Show unified diff for reasonably close matches
        closest_match = '\n'.join(file_lines[best_start:best_start + search_len])
        diff_lines = list(difflib.unified_diff(
            search_text.splitlines(keepends=True),
            closest_match.splitlines(keepends=True),
            fromfile='your_search_block',
            tofile='actual_file_content',
            lineterm=''
        ))
        if diff_lines:
            error_msg["diff_hint"] = ''.join(diff_lines[:30])
            error_msg["match_location"] = f"Lines {best_start + 1}-{best_start + search_len}"
            error_msg["match_similarity"] = f"{best_ratio:.1%}"
        error_msg["suggestion"] = (
            f"The diff above shows exactly where your {block_label} block "
            "differs from the file. Re-read the file to get exact content."
        )
    else:
        # Fall back to partial line matching for very low similarity
        first_line = search_lines[0].strip() if search_lines else ""
        partial_matches = []
        for i, line in enumerate(file_lines):
            if first_line and first_line in line:
                partial_matches.append(f"Line {i+1}: {line[:100]}")
                if len(partial_matches) >= 5:
                    break
        if partial_matches:
            error_msg["possible_matches"] = partial_matches[:5]
            error_msg["suggestion"] = (
                "Found lines containing similar text. "
                "Re-read the file to get exact content."
            )

    return error_msg


async def write_file(
    file_path: Annotated[
        str,
        Field(description="MUST be the absolute path to the file to create or overwrite")
    ],
    content: Annotated[
        str,
        Field(description="The complete content to write to the file")
    ],
    create_directories: Annotated[
        bool,
        Field(description="If True, create parent directories if they don't exist")
    ] = True,
    overwrite: Annotated[
        bool,
        Field(description="If True, overwrite existing files. If False, fail if file exists")
    ] = True,
) -> dict[str, Any]:
    """Create a new file or overwrite an existing file with the specified content.

    This is the file creation tool - use it when you need to create a completely
    new file or fully replace an existing file's content.

    WHEN TO USE:
    - Creating new files that don't exist yet
    - When you need to completely replace a file's content
    - For configuration files, new modules, or scaffolding

    WHEN NOT TO USE:
    - For making targeted changes to existing files (use edit_file instead)
    - When you only need to modify specific lines or functions
    - The edit_file tool is preferred for surgical changes to avoid overwriting
      unrelated code

    IMPORTANT:
    - For EXISTING files: You MUST read the file first using a read tool
      before writing. This prevents accidentally overwriting important content.
    - All paths MUST be absolute paths
    - Follows v0's kebab-case naming preference for new files

    OUTPUT:
    Returns success status with file path and bytes written, or error details.
    """
    abs_path, error = _validate_path_for_write(file_path)
    if error:
        return error

    # Check if file exists and enforce read-before-write for existing files
    file_exists = os.path.isfile(abs_path)

    if file_exists:
        if not overwrite:
            return {"error": f"File already exists and overwrite=False: {abs_path}"}

        # Enforce read-before-write requirement (v0/Cline protocol)
        if not was_file_read(abs_path):
            return {
                "error": f"Read-before-write violation: You must read the file before overwriting it. "
                         f"Use read_source_range or similar to read '{abs_path}' first.",
                "hint": "This prevents accidentally overwriting important code. Read the file to "
                        "understand its current state before writing."
            }

    def _write_sync() -> dict[str, Any]:
        try:
            # Save current state for undo before making changes
            _save_for_undo(abs_path, was_new_file=not file_exists)

            # Create parent directories if requested
            parent_dir = Path(abs_path).parent
            parent_existed = parent_dir.exists()
            if create_directories:
                parent_dir.mkdir(parents=True, exist_ok=True)
            elif not parent_existed:
                return {"error": f"Parent directory does not exist: {parent_dir}"}

            # Normalize content and write
            normalized_content = _normalize_line_endings(content)

            with open(abs_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(normalized_content)

            bytes_written = len(normalized_content.encode('utf-8'))
            line_count = normalized_content.count('\n') + (1 if normalized_content and not normalized_content.endswith('\n') else 0)

            # Update the read cache with new content
            mark_file_as_read(abs_path, normalized_content)

            return {
                "status": "success",
                "message": f"File {'overwritten' if file_exists else 'created'}: {abs_path}",
                "file_path": abs_path,
                "bytes_written": bytes_written,
                "lines": line_count,
                "created_directories": create_directories and not parent_existed,
            }

        except PermissionError:
            return {"error": f"Permission denied writing to file: {abs_path}"}
        except OSError as e:
            return {"error": f"OS error writing file: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error writing file: {str(e)}"}

    return await asyncio.to_thread(_write_sync)


async def edit_file(
    file_path: Annotated[
        str,
        Field(description="MUST be the absolute path to the file to edit")
    ],
    diff_content: Annotated[
        str,
        Field(
            description="One or more edit blocks. Two formats supported:\n\n"
                       "TEXT-BASED (SEARCH/REPLACE):\n"
                       "<<<<<<< SEARCH\n"
                       "[exact content to find]\n"
                       "=======\n"
                       "[new content]\n"
                       ">>>>>>> REPLACE\n\n"
                       "AST-BASED (SYMBOL) - for functions/classes/methods:\n"
                       "<<<<<<< SYMBOL\n"
                       "[symbol_name or ClassName.method_name]\n"
                       "=======\n"
                       "[new content for entire symbol]\n"
                       ">>>>>>> REPLACE"
        )
    ],
    dry_run: Annotated[
        bool,
        Field(description="If True, only validate the edit without applying changes")
    ] = False,
) -> dict[str, Any]:
    """Apply surgical edits to a file using SEARCH/REPLACE or SYMBOL blocks.

    This is the precision editing tool following the Cline/RooCode protocol,
    enhanced with Windsurf-style AST-aware symbol targeting.

    TWO EDITING MODES:

    1. TEXT-BASED (SEARCH/REPLACE) - for exact text matching:
    ```
    <<<<<<< SEARCH
    [exact content to find - must match character-for-character]
    =======
    [new content to replace with]
    >>>>>>> REPLACE
    ```

    2. SYMBOL-BASED (AST-aware) - for replacing entire functions/classes/methods:
    ```
    <<<<<<< SYMBOL
    function_name
    =======
    def function_name(new_args):
        # entirely new implementation
    >>>>>>> REPLACE
    ```

    For methods within a class, use "ClassName.method_name" notation.

    WHEN TO USE TEXT-BASED:
    - Making small, targeted changes within functions
    - Changing specific lines or expressions
    - When you need precise control over what gets replaced

    WHEN TO USE SYMBOL-BASED:
    - Replacing an entire function implementation
    - Rewriting a method completely
    - Moving/refactoring code at the function level
    - When text matching is fragile due to whitespace/formatting

    WHEN NOT TO USE:
    - Creating entirely new files (use write_file instead)
    - For wholesale file replacement (use write_file instead)

    CRITICAL RULES:
    1. For SEARCH: content must match EXACTLY - character-for-character
    2. For SYMBOL: the symbol must exist and be unique
    3. Each block must be UNIQUE in the file
    4. You MUST read the file first to ensure your blocks match

    MULTIPLE EDITS:
    You can include multiple blocks (SEARCH or SYMBOL) in a single call.
    They are applied in order. If any block fails, the entire operation fails
    and no changes are made (atomic operation).

    OUTPUT:
    Returns success with number of changes applied, or detailed error if
    a block fails to match.
    """
    abs_path, error = _validate_path_for_write(file_path)
    if error:
        return error

    # Check file exists
    if not os.path.isfile(abs_path):
        return {"error": f"File not found: {abs_path}"}

    # Enforce read-before-write requirement
    if not was_file_read(abs_path):
        return {
            "error": f"Read-before-write violation: You must read the file before editing it. "
                     f"Use read_source_range or similar to read '{abs_path}' first.",
            "hint": "Reading the file first ensures your SEARCH blocks match the current "
                    "file state and prevents editing stale content."
        }

    def _edit_sync() -> dict[str, Any]:
        try:
            # Read in binary mode first to detect line endings accurately
            # (Python's text mode normalizes line endings, losing CRLF info)
            with open(abs_path, 'rb') as f:
                raw_content = f.read()

            # Detect original line ending style from raw bytes
            original_line_ending = '\r\n' if b'\r\n' in raw_content else '\n'

            # Decode and normalize for processing
            file_content = raw_content.decode('utf-8')

            # Normalize for processing
            normalized_content = _normalize_line_endings(file_content)
            normalized_diff = _normalize_line_endings(diff_content)

            # Parse edit blocks from diff content
            all_blocks = _parse_edit_blocks(normalized_diff)

            if not all_blocks:
                return {
                    "error": "No valid SEARCH/REPLACE or SYMBOL/REPLACE blocks found",
                    "hint": "Format must be:\n"
                           "<<<<<<< SEARCH\n[text]\n=======\n[replacement]\n>>>>>>> REPLACE\n"
                           "OR\n"
                           "<<<<<<< SYMBOL\n[symbol_name]\n=======\n[replacement]\n>>>>>>> REPLACE",
                    "received": diff_content[:500] + "..." if len(diff_content) > 500 else diff_content
                }

            # Resolve SYMBOL blocks to their source code using Tree-sitter
            resolved_blocks, resolve_error = _resolve_symbol_blocks(
                all_blocks, abs_path, normalized_content
            )
            if resolve_error:
                return resolve_error

            # Validate all blocks before applying any changes (atomic operation)
            modified_content = normalized_content
            applied_changes: list[dict[str, Any]] = []

            for idx, (search_text, replace_text, block_type) in enumerate(resolved_blocks):
                # Check for exact match
                if search_text not in modified_content:
                    file_lines = modified_content.split('\n')
                    return _build_no_match_error(search_text, file_lines, block_type, idx)

                # Check uniqueness - search_text should appear exactly once
                occurrences = modified_content.count(search_text)
                if occurrences > 1:
                    block_label = "SYMBOL" if block_type == "symbol" else "SEARCH"
                    return {
                        "error": f"{block_label} block {idx + 1} is ambiguous - found {occurrences} matches",
                        "block_type": block_type,
                        "search_block_preview": search_text[:200] + "..." if len(search_text) > 200 else search_text,
                        "hint": "Expand the block to include more surrounding context "
                                "to make it unique, or use multiple smaller edits."
                    }

                # Track the change location for reporting
                match_start = modified_content.find(search_text)
                line_number = modified_content[:match_start].count('\n') + 1

                # Apply the replacement (only first occurrence, which should be unique)
                modified_content = modified_content.replace(search_text, replace_text, 1)

                applied_changes.append({
                    "block": idx + 1,
                    "block_type": block_type,
                    "line": line_number,
                    "search_preview": (search_text[:50] + "...") if len(search_text) > 50 else search_text,
                    "lines_removed": search_text.count('\n') + 1,
                    "lines_added": replace_text.count('\n') + 1,
                })

            if dry_run:
                return {
                    "status": "dry_run",
                    "message": f"Validation passed - {len(resolved_blocks)} change(s) would be applied",
                    "file_path": abs_path,
                    "changes": applied_changes,
                    "preview": modified_content[:1000] + "..." if len(modified_content) > 1000 else modified_content,
                }

            # Save current state for undo AFTER validation passes and BEFORE writing
            # This ensures we don't overwrite undo history for failed edits
            _save_for_undo(abs_path, was_new_file=False)

            # Restore original line endings if needed
            if original_line_ending == '\r\n':
                modified_content = modified_content.replace('\n', '\r\n')

            # Write the modified content
            with open(abs_path, 'w', encoding='utf-8', newline='') as f:
                f.write(modified_content)

            # Update the read cache with new content
            mark_file_as_read(abs_path, _normalize_line_endings(modified_content))

            return {
                "status": "success",
                "message": f"Applied {len(resolved_blocks)} change(s) to {abs_path}",
                "file_path": abs_path,
                "changes_applied": len(resolved_blocks),
                "changes": applied_changes,
            }

        except PermissionError:
            return {"error": f"Permission denied editing file: {abs_path}"}
        except UnicodeDecodeError as e:
            return {"error": f"File encoding error (expected UTF-8): {e}"}
        except Exception as e:
            return {"error": f"Unexpected error editing file: {str(e)}"}

    return await asyncio.to_thread(_edit_sync)


async def delete_file(
    file_path: Annotated[
        str,
        Field(description="MUST be the absolute path to the file to delete")
    ],
    missing_ok: Annotated[
        bool,
        Field(description="If True, don't error if the file doesn't exist")
    ] = False,
) -> dict[str, Any]:
    """Delete a file from the filesystem.

    This tool permanently removes a file. Use with caution - deletion is not reversible
    (unless the file is under version control).

    WHEN TO USE:
    - Removing obsolete or temporary files
    - Cleaning up generated files
    - Removing files that are no longer needed after refactoring

    WHEN NOT TO USE:
    - When you're not sure if the file is still needed
    - For files that might be referenced elsewhere (search first!)

    IMPORTANT:
    - You MUST read the file first using a read tool before deleting it.
      This ensures you know what you're deleting and prevents accidental deletion.
    - All paths MUST be absolute paths
    - This operation is NOT reversible (except via version control)

    OUTPUT:
    Returns success status confirming deletion, or error details.
    """
    abs_path, error = _validate_path_for_write(file_path)
    if error:
        return error

    # Check if file exists
    if not os.path.exists(abs_path):
        if missing_ok:
            return {
                "status": "success",
                "message": f"File already does not exist: {abs_path}",
                "file_path": abs_path,
                "already_missing": True,
            }
        return {"error": f"File not found: {abs_path}"}

    # Prevent accidental directory deletion
    if os.path.isdir(abs_path):
        return {
            "error": f"Path is a directory, not a file: {abs_path}",
            "hint": "Use a different tool for directory removal, or remove files individually."
        }

    # Enforce read-before-delete requirement (similar to read-before-write)
    if not was_file_read(abs_path):
        return {
            "error": f"Read-before-delete violation: You must read the file before deleting it. "
                     f"Use read_source_range or similar to read '{abs_path}' first.",
            "hint": "Reading the file ensures you know what you're deleting and prevents "
                    "accidental removal of important files."
        }

    def _delete_sync() -> dict[str, Any]:
        try:
            # Save current state for undo before deleting
            _save_for_undo(abs_path, was_new_file=False)

            # Get file info before deletion
            file_size = os.path.getsize(abs_path)

            os.remove(abs_path)

            # Remove from read cache since file no longer exists
            get_file_cache().remove(abs_path)

            return {
                "status": "success",
                "message": f"File deleted: {abs_path}",
                "file_path": abs_path,
                "bytes_deleted": file_size,
            }

        except PermissionError:
            return {"error": f"Permission denied deleting file: {abs_path}"}
        except OSError as e:
            return {"error": f"OS error deleting file: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error deleting file: {str(e)}"}

    return await asyncio.to_thread(_delete_sync)


async def undo_file_change(
    file_path: Annotated[
        str,
        Field(description="MUST be the absolute path to the file to undo changes for")
    ],
) -> dict[str, Any]:
    """Revert the last change made to a file.

    This provides a one-step undo capability per file. Only the most recent change
    (write, edit, or delete) can be undone. Subsequent undo calls have no effect
    until a new change is made.

    WHEN TO USE:
    - When an edit broke something and you need to quickly revert
    - When you accidentally overwrote a file with wrong content
    - When you deleted a file by mistake
    - To escape from "linter loops" where each fix creates new problems

    WHEN NOT TO USE:
    - For undoing multiple sequential changes (only the last one can be undone)
    - When the file was modified outside of these tools (external changes)
    - When no changes have been made to the file in this session

    IMPORTANT:
    - Only the LAST change is stored - cannot undo multiple times
    - If the file was newly created, undo will DELETE it
    - If the file was deleted, undo will RESTORE it with its original content
    - Paths MUST be absolute

    OUTPUT:
    Returns success status with details about what was restored, or error if no
    undo history exists for the file.
    """
    abs_path, error = _validate_path_for_write(file_path)
    if error:
        return error

    def _undo_sync() -> dict[str, Any]:
        # Use UndoManager to retrieve undo entry
        undo_entry = get_undo_manager().pop_history(abs_path)
        if undo_entry is None:
            return {
                "error": f"No undo history available for: {abs_path}",
                "hint": "Undo history is only available after write_file, edit_file, or delete_file operations in this session."
            }

        try:
            if undo_entry.was_new_file:
                # File was newly created - undo means delete it
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    get_file_cache().remove(abs_path)
                    return {
                        "status": "success",
                        "message": f"Undo: Removed newly created file: {abs_path}",
                        "action": "deleted",
                        "file_path": abs_path,
                    }
                else:
                    return {
                        "status": "success",
                        "message": f"Undo: File already removed: {abs_path}",
                        "action": "no_change",
                        "file_path": abs_path,
                    }
            else:
                # Restore previous content
                if undo_entry.previous_content is None:
                    return {
                        "error": f"Undo history corrupted for: {abs_path}",
                    }

                # Check if file exists BEFORE writing (to determine restored vs recreated)
                file_existed = os.path.exists(abs_path)

                # Ensure parent directory exists (in case it was deleted)
                Path(abs_path).parent.mkdir(parents=True, exist_ok=True)

                with open(abs_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(undo_entry.previous_content)

                # Update read cache
                mark_file_as_read(abs_path, undo_entry.previous_content)

                bytes_restored = len(undo_entry.previous_content.encode('utf-8'))

                return {
                    "status": "success",
                    "message": f"Undo: {'Restored' if file_existed else 'Recreated'} file: {abs_path}",
                    "action": "restored",
                    "file_path": abs_path,
                    "bytes_restored": bytes_restored,
                }

        except PermissionError:
            return {"error": f"Permission denied during undo for: {abs_path}"}
        except OSError as e:
            return {"error": f"OS error during undo: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error during undo: {str(e)}"}

    return await asyncio.to_thread(_undo_sync)


def register_file_edit_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register file editing MCP tools on the given FastMCP instance.

    Args:
        mcp: FastMCP instance to register tools on.
        disabled_tools: Optional set of tool names to skip registration.
    """
    disabled = disabled_tools or set()

    all_tools = [
        write_file,
        edit_file,
        delete_file,
        undo_file_change,
    ]

    for func in all_tools:
        if func.__name__ not in disabled:
            mcp.tool()(func)
