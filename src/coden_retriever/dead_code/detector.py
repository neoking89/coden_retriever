"""Core dead code detection algorithms.

Identifies functions with no incoming calls and calculates confidence scores
using LANGUAGE-AGNOSTIC detection via tree-sitter AST attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..constants import (
    DEAD_CODE_CONFIDENCE_HIGH,
    DEAD_CODE_CONFIDENCE_MEDIUM,
)
from .confidence import calculate_confidence, get_reason

if TYPE_CHECKING:
    import networkx as nx

    from ..models import CodeEntity


# Runtime-called function names that should never be flagged as dead code
# These are called by language runtimes, not through direct calls
RUNTIME_CALLED_NAMES: frozenset[str] = frozenset({
    "init",        # Go: called automatically at package init
    "constructor", # JS/TS: called via 'new ClassName()'
})


def _is_runtime_called(name: str) -> bool:
    """Check if name is a runtime-called function.

    Includes Python dunder methods (__init__, etc.), Go init(), JS constructor.
    """
    if name.startswith("__") and name.endswith("__"):
        return True
    return name in RUNTIME_CALLED_NAMES


def detect_unused_functions(
    entities: dict[str, "CodeEntity"],
    graph: "nx.DiGraph",
    confidence_threshold: float = 0.5,
    exclude_tests: bool = True,
    include_private: bool = False,
    min_lines: int = 3,
    limit: int | None = 50,
    used_names: set[str] | None = None,
) -> dict[str, Any]:
    """Find functions with no incoming calls in the call graph.

    Args:
        entities: Dict of node_id -> CodeEntity.
        graph: NetworkX DiGraph with call edges.
        confidence_threshold: Minimum confidence to include (0.0-1.0).
        exclude_tests: Exclude test functions.
        include_private: Include private functions (_name).
        min_lines: Minimum line count to consider.
        limit: Maximum results to return.
        used_names: Identifier names referenced in codebase (Vulture-style).

    Returns:
        Dict with dead_code list and summary statistics.
    """
    results = _find_dead_code_candidates(
        entities, graph, confidence_threshold,
        exclude_tests, include_private, min_lines, used_names
    )
    results.sort(key=lambda x: x["confidence"], reverse=True)

    if limit is not None and limit > 0:
        results = results[:limit]

    return {
        "dead_code": results,
        "summary": _build_summary(results, len(entities)),
    }


def _find_dead_code_candidates(
    entities: dict[str, "CodeEntity"],
    graph: "nx.DiGraph",
    confidence_threshold: float,
    exclude_tests: bool,
    include_private: bool,
    min_lines: int,
    used_names: set[str] | None,
) -> list[dict[str, Any]]:
    """Find all dead code candidates above confidence threshold."""
    results = []
    for node_id, entity in entities.items():
        if not _should_analyze_entity(entity, exclude_tests, include_private, min_lines):
            continue
        if used_names and entity.name in used_names:
            continue
        in_degree = graph.in_degree(node_id) if node_id in graph else 0
        if in_degree > 0:
            continue
        confidence = calculate_confidence(entity, graph)
        if confidence >= confidence_threshold:
            results.append(_build_result_item(entity, confidence))
    return results


def _should_analyze_entity(
    entity: "CodeEntity",
    exclude_tests: bool,
    include_private: bool,
    min_lines: int,
) -> bool:
    """Check if entity should be analyzed for dead code."""
    if entity.entity_type == "class":
        return False
    if _is_runtime_called(entity.name):
        return False
    if entity.is_test and exclude_tests:
        return False
    if entity.is_private and not include_private:
        return False
    if entity.line_count < min_lines:
        return False
    return True


def _build_result_item(entity: "CodeEntity", confidence: float) -> dict[str, Any]:
    """Build result dictionary for a dead code item."""
    return {
        "node_id": entity.node_id,
        "name": entity.name,
        "file": entity.file_path,
        "line": entity.line_start,
        "end_line": entity.line_end,
        "type": entity.entity_type,
        "lines": entity.line_count,
        "confidence": confidence,
        "reason": get_reason(confidence),
    }


def _build_summary(results: list[dict], total_entities: int) -> dict[str, Any]:
    """Build summary statistics."""
    distribution = {"high": 0, "medium": 0, "low": 0}
    for r in results:
        conf = r["confidence"]
        if conf >= DEAD_CODE_CONFIDENCE_HIGH:
            distribution["high"] += 1
        elif conf >= DEAD_CODE_CONFIDENCE_MEDIUM:
            distribution["medium"] += 1
        else:
            distribution["low"] += 1

    return {
        "total_functions_analyzed": total_entities,
        "dead_code_found": len(results),
        "distribution": distribution,
    }
