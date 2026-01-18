"""Code flagging module for inserting [CODEN] comments.

Inserts language-appropriate comments above code objects based on
analysis results from hotspots, propagation cost, and clone detection.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from ..constants import FLAG_ANALYSIS_LIMIT, FLAG_MIN_LINES
from ..graph_utils import compute_coupling_hotspots
from ..language.definitions import LANGUAGE_MAP

if TYPE_CHECKING:
    import networkx as nx
    from ..models import CodeEntity

logger = logging.getLogger(__name__)

# Marker for coden-generated comments
CODEN_MARKER = "[CODEN]"

# Comment syntax by language
COMMENT_SYNTAX: dict[str, str] = {
    "python": "#",
    "javascript": "//",
    "typescript": "//",
    "go": "//",
    "rust": "//",
    "java": "//",
    "cpp": "//",
    "c": "//",
    "php": "//",
    "c_sharp": "//",
    "swift": "//",
    "kotlin": "//",
    "scala": "//",
    "ruby": "#",
    "shell": "#",
    "bash": "#",
}

# Languages that use # for comments
HASH_COMMENT_LANGUAGES = {"python", "ruby", "shell", "bash"}


def _get_comment_prefix(language: str) -> str:
    """Get the comment prefix for a language."""
    return COMMENT_SYNTAX.get(language, "//")


def _get_language_from_file(file_path: str) -> str | None:
    """Get language name from file extension."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(ext)


def _generate_hotspot_comment(item: dict, comment_prefix: str) -> list[str]:
    """Generate comment lines for a hotspot flag."""
    risk = item.get("risk_score", 0)
    fan_in = item.get("fan_in", 0)
    fan_out = item.get("fan_out", 0)
    complexity = item.get("complexity", 0)

    lines = [
        f"{comment_prefix} {CODEN_MARKER} HOTSPOT: Risk Score {risk:.2f} | Fan-in: {fan_in}, Fan-out: {fan_out} | Complexity: {complexity}",
        f"{comment_prefix} {CODEN_MARKER} This function is a coupling hotspot. Changes here may have wide impact.",
    ]
    return lines


def _generate_propagation_comment(item: dict, comment_prefix: str) -> list[str]:
    """Generate comment lines for a propagation cost flag."""
    cost = item.get("propagation_cost", 0)
    direct_deps = item.get("direct_deps", 0)
    transitive_deps = item.get("transitive_deps", 0)

    lines = [
        f"{comment_prefix} {CODEN_MARKER} PROPAGATION: Cost {cost:.1f}% | Direct deps: {direct_deps}, Transitive deps: {transitive_deps}",
        f"{comment_prefix} {CODEN_MARKER} High change propagation risk. Consider refactoring to reduce coupling.",
    ]
    return lines


def _generate_clone_comment(item: dict, comment_prefix: str) -> list[str]:
    """Generate comment lines for a clone flag."""
    similarity = item.get("similarity", 0) * 100
    clone_file = item.get("clone_file", "unknown")
    clone_func = item.get("clone_function", "unknown")
    clone_line = item.get("clone_line", 0)

    lines = [
        f"{comment_prefix} {CODEN_MARKER} CLONE: {similarity:.1f}% similar to {clone_file}::{clone_func} (line {clone_line})",
        f"{comment_prefix} {CODEN_MARKER} Consider extracting to shared utility to reduce duplication.",
    ]
    return lines


def _generate_echo_comment(item: dict, comment_prefix: str) -> list[str]:
    """Generate comment lines for an echo comment flag."""
    similarity = item.get("similarity_score", 0) * 100
    comment_text = item.get("comment_text", "")

    lines = [
        f"{comment_prefix} {CODEN_MARKER} ECHO: {similarity:.1f}% similar to identifier",
        f'{comment_prefix} {CODEN_MARKER} Original comment: "{comment_text}"',
        f"{comment_prefix} {CODEN_MARKER} This comment provides minimal value. Consider removing or elaborating.",
    ]
    return lines


def _generate_dead_code_comment(item: dict, comment_prefix: str) -> list[str]:
    """Generate comment lines for a dead code flag."""
    confidence = item.get("confidence", 0) * 100
    reason = item.get("reason", "No callers found")

    lines = [
        f"{comment_prefix} {CODEN_MARKER} DEAD_CODE: {confidence:.0f}% confidence | {reason}",
        f"{comment_prefix} {CODEN_MARKER} This function has no callers. Consider removing or documenting as entry point.",
    ]
    return lines


def _remove_dead_code_function(
    lines: list[str],
    start_line: int,
    end_line: int,
    language: str,
) -> list[str]:
    """Remove a function/method from the file.

    Args:
        lines: List of file lines
        start_line: 1-based start line of function
        end_line: 1-based end line of function
        language: Language for decorator handling

    Returns:
        Modified lines with function removed
    """
    if start_line <= 0 or end_line > len(lines) or start_line > end_line:
        return lines

    start_idx = start_line - 1  # Convert to 0-based
    end_idx = end_line  # exclusive

    # For Python, also remove decorators above the function
    if language == "python":
        start_idx = _find_decorator_start(lines, start_idx)

    # Remove the function
    result = lines[:start_idx] + lines[end_idx:]

    # Clean up double blank lines at removal point
    result = _clean_double_blank_lines(result, start_idx)

    return result


def _find_decorator_start(lines: list[str], func_start_idx: int) -> int:
    """Find the start of decorators above a function."""
    idx = func_start_idx
    while idx > 0:
        prev_line = lines[idx - 1].strip()
        if prev_line.startswith("@"):
            idx -= 1
        elif prev_line == "":
            # Check if there's a decorator before blank lines
            test_idx = idx - 1
            while test_idx > 0 and lines[test_idx - 1].strip() == "":
                test_idx -= 1
            if test_idx > 0 and lines[test_idx - 1].strip().startswith("@"):
                idx = test_idx
            else:
                break
        else:
            break
    return idx


def _clean_double_blank_lines(lines: list[str], position: int) -> list[str]:
    """Remove double blank lines at a specific position."""
    if position > 0 and position < len(lines):
        if lines[position - 1].strip() == "" and lines[position].strip() == "":
            return lines[:position] + lines[position + 1:]
    return lines


def _remove_echo_comment_at_line(
    lines: list[str],
    target_line: int,
    comment_prefix: str,
) -> list[str]:
    """Remove echo comment at the specified line.

    Args:
        lines: List of file lines
        target_line: 1-based line number to remove
        comment_prefix: Comment syntax for the language

    Returns:
        Modified list with echo comment removed
    """
    if target_line <= 0 or target_line > len(lines):
        return lines

    idx = target_line - 1  # Convert to 0-based

    # Check if line is a comment
    stripped = lines[idx].lstrip()
    if not stripped.startswith(comment_prefix):
        return lines

    # Skip CODEN markers (don't remove those directly)
    if CODEN_MARKER in stripped:
        return lines

    # Remove the line
    return lines[:idx] + lines[idx + 1:]


def _is_coden_comment(line: str, comment_prefix: str) -> bool:
    """Check if a line is a CODEN comment (not just contains the marker).

    Args:
        line: The line to check
        comment_prefix: The comment prefix for the language (e.g., "#" or "//")

    Returns:
        True if line is a comment containing [CODEN], False otherwise
    """
    stripped = line.lstrip()
    # Check if line starts with comment syntax and contains CODEN marker
    return stripped.startswith(comment_prefix) and CODEN_MARKER in stripped


def _remove_existing_coden_comments(lines: list[str], comment_prefix: str) -> list[str]:
    """Remove existing [CODEN] comments from lines.

    Args:
        lines: List of lines to filter
        comment_prefix: Comment syntax for the language

    Returns:
        Filtered list with CODEN comment lines removed
    """
    return [line for line in lines if not _is_coden_comment(line, comment_prefix)]


def _find_insertion_line(
    lines: list[str],
    target_line: int,
    language: str,
) -> int:
    """Find the correct line to insert comments before.

    Handles decorators (Python) and annotations (Java/Kotlin) by finding
    the first line of the decorated/annotated block.
    """
    if target_line <= 0 or target_line > len(lines):
        return max(0, target_line - 1)

    # Convert to 0-based index
    idx = target_line - 1

    # For Python, look for decorators above the target line
    if language == "python":
        while idx > 0:
            prev_line = lines[idx - 1].strip()
            if prev_line.startswith("@"):
                idx -= 1
            elif prev_line == "" or prev_line.startswith("#"):
                # Skip blank lines and comments above decorators
                test_idx = idx - 1
                while test_idx > 0 and (lines[test_idx - 1].strip() == "" or
                                        lines[test_idx - 1].strip().startswith("#")):
                    test_idx -= 1
                if test_idx > 0 and lines[test_idx - 1].strip().startswith("@"):
                    idx = test_idx
                else:
                    break
            else:
                break

    # For Java/Kotlin/C#, look for annotations
    elif language in {"java", "kotlin", "c_sharp"}:
        while idx > 0:
            prev_line = lines[idx - 1].strip()
            if prev_line.startswith("@"):
                idx -= 1
            elif prev_line == "":
                # Skip blank lines above annotations
                test_idx = idx - 1
                while test_idx > 0 and lines[test_idx - 1].strip() == "":
                    test_idx -= 1
                if test_idx > 0 and lines[test_idx - 1].strip().startswith("@"):
                    idx = test_idx
                else:
                    break
            else:
                break

    return idx


def _insert_comments_at_line(
    lines: list[str],
    insert_idx: int,
    comment_lines: list[str],
    indentation: str = "",
) -> list[str]:
    """Insert comment lines at the specified index."""
    # Add indentation to comments
    indented_comments = [indentation + line for line in comment_lines]

    # Insert comments
    result = lines[:insert_idx] + indented_comments + lines[insert_idx:]
    return result


def _get_indentation(line: str) -> str:
    """Extract leading whitespace from a line."""
    return line[:len(line) - len(line.lstrip())]


# Large token limit for flag operations (not constrained by MCP context)
_NO_TOKEN_LIMIT = 1_000_000


def _collect_hotspot_items(
    graph: "nx.DiGraph",
    entities: dict[str, "CodeEntity"],
    pagerank: dict[str, float],
    risk_threshold: float,
    exclude_tests: bool,
) -> list[dict]:
    """Collect flaggable items from hotspot analysis."""
    items = []
    hotspot_result = compute_coupling_hotspots(
        graph=graph,
        entities=entities,
        pagerank=pagerank,
        limit=FLAG_ANALYSIS_LIMIT,
        min_coupling_score=0,
        exclude_tests=exclude_tests,
        exclude_private=False,
        token_limit=_NO_TOKEN_LIMIT,
    )

    for item in hotspot_result.get("hotspots", []):
        risk = item.get("risk_score", 0)
        if risk >= risk_threshold:
            items.append({
                "type": "hotspot",
                "file": item.get("file"),
                "line": item.get("line"),
                "name": item.get("name"),
                "risk_score": risk,
                "fan_in": item.get("fan_in", 0),
                "fan_out": item.get("fan_out", 0),
                "complexity": item.get("complexity", 0),
            })
    return items


def _collect_propagation_items(
    graph: "nx.DiGraph",
    entities: dict[str, "CodeEntity"],
    propagation_threshold: float,
    exclude_tests: bool,
) -> list[dict]:
    """Collect flaggable items from propagation cost analysis."""
    from .propagation_cost import compute_propagation_cost

    items = []
    prop_result = compute_propagation_cost(
        graph=graph,
        entities=entities,
        include_breakdown=True,
        show_critical_paths=False,
        exclude_tests=exclude_tests,
        token_limit=_NO_TOKEN_LIMIT,
    )

    # Build a file-to-entities index for efficient lookup
    file_to_entities: dict[str, list[tuple[str, "CodeEntity"]]] = defaultdict(list)
    for entity_id, entity in entities.items():
        if entity.file_path:
            normalized_path = os.path.normpath(entity.file_path)
            file_to_entities[normalized_path].append((entity_id, entity))

    # Flag modules with high propagation cost
    for module in prop_result.get("module_breakdown", []):
        cost_pct = module.get("propagation_cost_pct", 0)
        if cost_pct >= propagation_threshold * 100:
            module_path = module.get("module", "")
            if not module_path:
                continue

            normalized_module = os.path.normpath(module_path)
            module_entities = file_to_entities.get(normalized_module, [])

            # Find the first function/method in this module to flag
            for entity_id, entity in module_entities:
                if entity.entity_type in ("function", "method"):
                    items.append({
                        "type": "propagation",
                        "file": entity.file_path,
                        "line": entity.line_start,
                        "name": entity.name,
                        "propagation_cost": cost_pct,
                        "direct_deps": module.get("internal_coupling", 0),
                        "transitive_deps": module.get("external_coupling", 0),
                    })
                    break
    return items


def _collect_clone_items(
    entities: dict[str, "CodeEntity"],
    clone_threshold: float,
    clone_mode: str,
    line_threshold: float,
    func_threshold: float,
    exclude_tests: bool,
) -> list[dict]:
    """Collect flaggable items from clone detection."""
    from ..clone import detect_clones_combined, detect_clones_semantic, detect_clones_syntactic
    from ..config_loader import get_semantic_model_path

    items = []
    model_path = get_semantic_model_path()

    if clone_mode == "syntactic":
        clone_result = detect_clones_syntactic(
            entities=entities,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=FLAG_ANALYSIS_LIMIT,
            exclude_tests=exclude_tests,
            min_lines=FLAG_MIN_LINES,
            token_limit=_NO_TOKEN_LIMIT,
        )
    elif clone_mode == "semantic":
        clone_result = detect_clones_semantic(
            entities=entities,
            model_path=model_path,
            threshold=clone_threshold,
            limit=FLAG_ANALYSIS_LIMIT,
            exclude_tests=exclude_tests,
            min_lines=FLAG_MIN_LINES,
            token_limit=_NO_TOKEN_LIMIT,
        )
    else:  # combined (default)
        clone_result = detect_clones_combined(
            entities=entities,
            model_path=model_path,
            semantic_threshold=clone_threshold,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=FLAG_ANALYSIS_LIMIT,
            exclude_tests=exclude_tests,
            min_lines=FLAG_MIN_LINES,
            token_limit=_NO_TOKEN_LIMIT,
        )

    for clone in clone_result.get("clones", []):
        entity1 = clone.get("entity1", {})
        entity2 = clone.get("entity2", {})
        base_item = {
            "similarity": clone.get("similarity", 0),
            "semantic_sim": clone.get("semantic_sim"),
            "syntactic_pct": clone.get("syntactic_pct"),
            "category": clone.get("category", ""),
            "blocks": clone.get("blocks", []),
            "max_block_size": clone.get("max_block_size", 0),
        }
        # Flag both sides of the clone
        items.append({
            "type": "clone",
            "file": entity1.get("file"),
            "line": entity1.get("line"),
            "name": entity1.get("name"),
            "clone_file": entity2.get("file"),
            "clone_function": entity2.get("name"),
            "clone_line": entity2.get("line"),
            **base_item,
        })
        items.append({
            "type": "clone",
            "file": entity2.get("file"),
            "line": entity2.get("line"),
            "name": entity2.get("name"),
            "clone_file": entity1.get("file"),
            "clone_function": entity1.get("name"),
            "clone_line": entity1.get("line"),
            **base_item,
        })
    return items


def _collect_echo_items(
    entities: dict[str, "CodeEntity"],
    echo_threshold: float,
    exclude_tests: bool,
    remove_comments: bool,
) -> list[dict]:
    """Collect flaggable items from echo comment detection."""
    from .echo_comments import compute_echo_comments

    items = []
    echo_result = compute_echo_comments(
        entities=entities,
        echo_threshold=echo_threshold,
        token_limit=_NO_TOKEN_LIMIT,
        include_tests=not exclude_tests,
        include_private=False,
    )

    for echo in echo_result.get("echo_comments", []):
        items.append({
            "type": "echo" if not remove_comments else "echo_remove",
            "file": echo.get("file_path"),
            "line": echo.get("line"),
            "name": echo.get("context_identifier"),
            "similarity_score": echo.get("similarity_score", 0),
            "comment_text": echo.get("comment_text", ""),
        })
    return items


def _collect_dead_code_items(
    entities: dict[str, "CodeEntity"],
    graph: "nx.DiGraph",
    dead_code_threshold: float,
    exclude_tests: bool,
    remove_dead_code: bool,
) -> list[dict]:
    """Collect flaggable items from dead code detection."""
    from ..dead_code.detector import detect_unused_functions

    items = []
    dead_result = detect_unused_functions(
        entities=entities,
        graph=graph,
        confidence_threshold=dead_code_threshold,
        exclude_tests=exclude_tests,
        include_private=False,
        min_lines=FLAG_MIN_LINES,
        limit=FLAG_ANALYSIS_LIMIT,
    )

    for item in dead_result.get("dead_code", []):
        conf = item.get("confidence", 0)
        if conf >= dead_code_threshold:
            items.append({
                "type": "dead_code" if not remove_dead_code else "dead_code_remove",
                "file": item.get("file"),
                "line": item.get("line"),
                "end_line": item.get("end_line"),
                "name": item.get("name"),
                "confidence": conf,
                "reason": item.get("reason"),
            })
    return items


def _apply_flag_to_item(
    modified_lines: list[str],
    item: dict,
    target_line: int,
    comment_prefix: str,
    language: str,
) -> tuple[list[str], bool]:
    """Apply a single flag to modified_lines. Returns (new_lines, was_flagged)."""
    item_type: str | None = item.get("type")

    # Handle removal types first
    if item_type == "echo_remove":
        new_lines = _remove_echo_comment_at_line(modified_lines, target_line, comment_prefix)
        return new_lines, True
    if item_type == "dead_code_remove":
        end_line = item.get("end_line", target_line)
        new_lines = _remove_dead_code_function(modified_lines, target_line, end_line, language)
        return new_lines, True

    # Generate comment based on type
    if not item_type:
        return modified_lines, False

    comment_generators = {
        "hotspot": _generate_hotspot_comment,
        "propagation": _generate_propagation_comment,
        "clone": _generate_clone_comment,
        "echo": _generate_echo_comment,
        "dead_code": _generate_dead_code_comment,
    }
    generator = comment_generators.get(item_type)
    if not generator:
        return modified_lines, False

    comment_lines = generator(item, comment_prefix)
    insert_idx = _find_insertion_line(modified_lines, target_line, language)

    if insert_idx < len(modified_lines):
        indentation = _get_indentation(modified_lines[insert_idx])
    else:
        indentation = ""

    comment_lines_with_newlines = [line + "\n" for line in comment_lines]
    new_lines = (
        modified_lines[:insert_idx] +
        [indentation + line for line in comment_lines_with_newlines] +
        modified_lines[insert_idx:]
    )
    return new_lines, True


def _process_flagged_file(
    file_path: str,
    items: list[dict],
    dry_run: bool,
    backup: bool,
) -> tuple[int, int]:
    """Process a single file for flag insertion. Returns (files_modified, items_flagged)."""
    if not os.path.isfile(file_path):
        return 0, 0

    language = _get_language_from_file(file_path)
    if not language:
        return 0, 0

    comment_prefix = _get_comment_prefix(language)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return 0, 0

    lines = content.splitlines(keepends=True)

    # Build mapping of original line numbers to cleaned line numbers
    original_to_clean_map: dict[int, int] = {}
    clean_line_idx = 0
    for original_idx, line in enumerate(lines):
        original_to_clean_map[original_idx + 1] = clean_line_idx + 1
        if not _is_coden_comment(line, comment_prefix):
            clean_line_idx += 1

    clean_lines = _remove_existing_coden_comments(lines, comment_prefix)

    # Sort items by MAPPED line number in reverse order
    items.sort(key=lambda x: original_to_clean_map.get(x.get("line", 0), 0), reverse=True)

    modified_lines = clean_lines.copy()
    items_flagged = 0

    for item in items:
        original_line = item.get("line", 0)
        target_line = original_to_clean_map.get(original_line, 0)
        if target_line <= 0 or target_line > len(modified_lines):
            continue

        modified_lines, was_flagged = _apply_flag_to_item(
            modified_lines, item, target_line, comment_prefix, language
        )
        if was_flagged:
            items_flagged += 1

    if items_flagged > 0:
        if not dry_run:
            _write_flagged_file(file_path, modified_lines, backup)
        return 1, items_flagged
    return 0, 0


def _write_flagged_file(file_path: str, modified_lines: list[str], backup: bool) -> None:
    """Write modified file atomically with optional backup."""
    if backup:
        backup_path = file_path + ".coden-backup"
        shutil.copy2(file_path, backup_path)

    try:
        dir_name = os.path.dirname(file_path) or "."
        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".coden_tmp_", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(modified_lines)
            os.replace(temp_path, file_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError as cleanup_err:
                logger.warning(f"Could not remove temp file {temp_path}: {cleanup_err}")
            raise
    except IOError as e:
        logger.error(f"Could not write {file_path}: {e}")


def _group_items_by_file(items: list[dict]) -> dict[str, list[dict]]:
    """Group flaggable items by file path."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        file_path = item.get("file")
        if file_path:
            by_file[file_path].append(item)
    return by_file


def _build_flag_summary(items: list[dict]) -> dict[str, int]:
    """Build summary counts by item type."""
    type_counts: dict[str, int] = defaultdict(int)
    for item in items:
        type_counts[item.get("type", "unknown")] += 1
    return {
        "hotspots": type_counts.get("hotspot", 0),
        "propagation": type_counts.get("propagation", 0),
        "clones": type_counts.get("clone", 0),
        "dead_code": type_counts.get("dead_code", 0) + type_counts.get("dead_code_remove", 0),
        "echo_comments": type_counts.get("echo", 0) + type_counts.get("echo_remove", 0),
    }


def flag_code(
    entities: dict[str, "CodeEntity"],
    graph: "nx.DiGraph",
    pagerank: dict[str, float],
    source_dir: str,
    hotspots: bool = False,
    propagation: bool = False,
    clones: bool = False,
    echo_comments: bool = False,
    dead_code: bool = False,
    risk_threshold: float = 50.0,
    propagation_threshold: float = 0.25,
    clone_threshold: float = 0.95,
    echo_threshold: float = 0.85,
    dead_code_threshold: float = 0.5,
    clone_mode: str = "combined",
    line_threshold: float = 0.70,
    func_threshold: float = 0.50,
    dry_run: bool = False,
    backup: bool = False,
    verbose: bool = False,
    exclude_tests: bool = True,
    remove_comments: bool = False,
    remove_dead_code: bool = False,
) -> dict[str, Any]:
    """Flag code objects with [CODEN] comments based on analysis.

    Args:
        entities: Dict of entity_id -> CodeEntity
        graph: Call graph (nx.DiGraph)
        pagerank: PageRank scores for entities
        source_dir: Root directory of the codebase
        hotspots: Flag hotspots from coupling analysis
        propagation: Flag high propagation cost functions
        clones: Flag detected code clones
        echo_comments: Flag echo comments (redundant comments)
        dead_code: Flag dead code (uncalled functions)
        risk_threshold: Min risk score for hotspot flagging (raw score, typically 50-200+)
        propagation_threshold: Min propagation cost % for flagging
        clone_threshold: Min similarity for clone flagging (0-1)
        echo_threshold: Min similarity for echo detection (0-1)
        dead_code_threshold: Min confidence for dead code flagging (0-1)
        dry_run: Preview changes without modifying files
        backup: Create .coden-backup files before modifying
        verbose: Show detailed output
        exclude_tests: Exclude test files from flagging
        remove_comments: Remove echo comments directly (no markers)
        remove_dead_code: Remove dead code functions directly (destructive)

    Returns:
        Dict with flagged_count, files_modified, items (list of flagged items)
    """
    if not (hotspots or propagation or clones or echo_comments or dead_code):
        return {
            "error": "At least one analysis flag (-H, -P, -C, -E, or -D) is required",
            "flagged_count": 0,
            "files_modified": 0,
        }

    flaggable_items: list[dict] = []

    # Collect items from each analysis type using helper functions
    if hotspots:
        flaggable_items.extend(_collect_hotspot_items(
            graph, entities, pagerank, risk_threshold, exclude_tests
        ))

    if propagation:
        flaggable_items.extend(_collect_propagation_items(
            graph, entities, propagation_threshold, exclude_tests
        ))

    if clones:
        flaggable_items.extend(_collect_clone_items(
            entities, clone_threshold, clone_mode, line_threshold, func_threshold, exclude_tests
        ))

    if echo_comments:
        flaggable_items.extend(_collect_echo_items(
            entities, echo_threshold, exclude_tests, remove_comments
        ))

    if dead_code:
        flaggable_items.extend(_collect_dead_code_items(
            entities, graph, dead_code_threshold, exclude_tests, remove_dead_code
        ))

    # Group items by file and apply flags
    by_file = _group_items_by_file(flaggable_items)
    files_modified = 0
    total_flagged = 0

    for file_path, items in by_file.items():
        modified, flagged = _process_flagged_file(file_path, items, dry_run, backup)
        files_modified += modified
        total_flagged += flagged

    return {
        "flagged_count": total_flagged,
        "files_modified": files_modified,
        "dry_run": dry_run,
        "items": flaggable_items,
        "summary": _build_flag_summary(flaggable_items),
    }


def flag_clear(
    source_dir: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Remove all [CODEN] comments from source files.

    Args:
        source_dir: Root directory to scan
        dry_run: Preview changes without modifying files
        verbose: Show detailed output

    Returns:
        Dict with files_cleaned, comments_removed
    """
    if not os.path.isdir(source_dir):
        return {"error": f"Directory not found: {source_dir}"}

    files_cleaned = 0
    comments_removed = 0
    files_with_flags: list[str] = []

    # Walk through all files
    for root, dirs, files in os.walk(source_dir):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {
            "node_modules", "__pycache__", "venv", ".venv", "dist", "build",
            ".git", ".hg", ".svn", "target", "vendor",
        }]

        for filename in files:
            file_path = os.path.join(root, filename)

            # Check if it's a supported source file
            language = _get_language_from_file(file_path)
            if not language:
                continue

            comment_prefix = _get_comment_prefix(language)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (IOError, UnicodeDecodeError):
                continue

            # (This is just an optimization; the real check is below)
            if CODEN_MARKER not in content:
                continue

            lines = content.splitlines(keepends=True)
            original_count = len(lines)

            # Remove CODEN comment lines (properly check comment syntax)
            clean_lines = _remove_existing_coden_comments(lines, comment_prefix)
            removed_count = original_count - len(clean_lines)

            if removed_count > 0:
                files_with_flags.append(file_path)
                comments_removed += removed_count

                if not dry_run:
                    try:
                        # Write atomically using temp file
                        dir_name = os.path.dirname(file_path) or "."
                        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".coden_tmp_", text=True)
                        try:
                            with os.fdopen(fd, "w", encoding="utf-8") as f:
                                f.writelines(clean_lines)
                            os.replace(temp_path, file_path)
                            files_cleaned += 1
                        except Exception:
                            try:
                                os.unlink(temp_path)
                            except OSError as cleanup_err:
                                logger.warning(f"Could not remove temp file {temp_path}: {cleanup_err}")
                            raise
                    except IOError as e:
                        logger.error(f"Could not write {file_path}: {e}")
                else:
                    files_cleaned += 1

    return {
        "files_cleaned": files_cleaned,
        "comments_removed": comments_removed,
        "dry_run": dry_run,
        "files": files_with_flags if verbose else [],
    }


def register_flag_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register flag tools with the MCP server.

    Args:
        mcp: The MCP server instance
        disabled_tools: Set of tool names to skip registration
    """
    disabled = disabled_tools or set()

    if "flag_code" not in disabled:
        @mcp.tool()
        async def flag_code_tool(
            root_directory: Annotated[str, Field(description="Root directory of the codebase to analyze")],
            hotspots: Annotated[bool, Field(default=False, description="Flag coupling hotspots")] = False,
            propagation: Annotated[bool, Field(default=False, description="Flag high propagation cost functions")] = False,
            clones: Annotated[bool, Field(default=False, description="Flag code clones")] = False,
            echo_comments: Annotated[bool, Field(default=False, description="Flag echo comments (redundant comments that restate the code)")] = False,
            risk_threshold: Annotated[float, Field(default=50.0, description="Min risk score for hotspot flagging (raw score, typically 50-200+)")] = 50.0,
            propagation_threshold: Annotated[float, Field(default=0.25, description="Min propagation cost for flagging (0-1)")] = 0.25,
            clone_threshold: Annotated[float, Field(default=0.95, description="Min similarity for clone flagging (0-1)")] = 0.95,
            echo_threshold: Annotated[float, Field(default=0.85, description="Min similarity for echo comment detection (0-1)")] = 0.85,
            remove_comments: Annotated[bool, Field(default=False, description="Remove echo comments directly instead of flagging them")] = False,
            dry_run: Annotated[bool, Field(default=True, description="Preview changes without modifying files")] = True,
        ) -> dict[str, Any]:
            """Insert [CODEN] comments in source code based on analysis results.

            Flags code objects (functions, methods) with inline comments based on:
            - Coupling hotspots: High fan-in × fan-out functions
            - Propagation cost: Functions with high change propagation risk
            - Code clones: Semantically similar functions
            - Echo comments: Redundant comments that merely restate the code

            The comments use [CODEN] marker for easy identification and removal.
            When remove_comments=True with echo_comments=True, echo comments are
            directly removed instead of being flagged.
            """
            from ..daemon.client import try_daemon_flag
            from ..daemon.protocol import FlagParams
            from ..cache import CacheManager
            from pathlib import Path
            import asyncio

            if not (hotspots or propagation or clones or echo_comments):
                return {"error": "At least one analysis type (hotspots, propagation, clones, or echo_comments) is required"}

            params = FlagParams(
                source_dir=root_directory,
                hotspots=hotspots,
                propagation=propagation,
                clones=clones,
                echo_comments=echo_comments,
                risk_threshold=risk_threshold,
                propagation_threshold=propagation_threshold,
                clone_threshold=clone_threshold,
                echo_threshold=echo_threshold,
                dry_run=dry_run,
                backup=False,
                verbose=False,
                exclude_tests=True,
                remove_comments=remove_comments,
            )

            daemon_result = try_daemon_flag(params)
            if daemon_result is not None:
                return daemon_result

            # Fallback to direct execution
            def _run_direct():
                cache = CacheManager(Path(root_directory))
                indices = cache.load_or_rebuild()
                return flag_code(
                    entities=indices.entities,
                    graph=indices.graph,
                    pagerank=indices.pagerank,
                    source_dir=root_directory,
                    hotspots=hotspots,
                    propagation=propagation,
                    clones=clones,
                    echo_comments=echo_comments,
                    risk_threshold=risk_threshold,
                    propagation_threshold=propagation_threshold,
                    clone_threshold=clone_threshold,
                    echo_threshold=echo_threshold,
                    dry_run=dry_run,
                    backup=False,
                    verbose=False,
                    exclude_tests=True,
                    remove_comments=remove_comments,
                )

            return await asyncio.to_thread(_run_direct)

    if "flag_clear" not in disabled:
        @mcp.tool()
        async def flag_clear_tool(
            root_directory: Annotated[str, Field(description="Root directory to scan for [CODEN] comments")],
            dry_run: Annotated[bool, Field(default=True, description="Preview changes without modifying files")] = True,
        ) -> dict[str, Any]:
            """Remove all [CODEN] comments from source files.

            Scans all source files in the directory and removes any lines
            containing the [CODEN] marker, cleaning up flags from previous runs.
            """
            from ..daemon.client import try_daemon_flag_clear
            from ..daemon.protocol import FlagClearParams
            import asyncio

            params = FlagClearParams(
                source_dir=root_directory,
                dry_run=dry_run,
                verbose=False,
            )

            daemon_result = try_daemon_flag_clear(params)
            if daemon_result is not None:
                return daemon_result

            # Fallback to direct execution
            def _run_direct():
                return flag_clear(
                    source_dir=root_directory,
                    dry_run=dry_run,
                    verbose=False,
                )

            return await asyncio.to_thread(_run_direct)
