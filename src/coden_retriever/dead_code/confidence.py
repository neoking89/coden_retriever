"""Confidence scoring for dead code detection.

Calculates likelihood that a flagged function is truly dead code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..constants import (
    DEAD_CODE_BASE_CONFIDENCE,
    DEAD_CODE_CONFIDENCE_HIGH,
    DEAD_CODE_CONFIDENCE_MEDIUM,
    DEAD_CODE_DECORATOR_PENALTY,
    DEAD_CODE_ENTRY_POINT_PENALTY,
    DEAD_CODE_METHOD_PENALTY,
    DEAD_CODE_PRIVATE_BOOST,
    DEAD_CODE_PUBLIC_MODULE_PENALTY,
)

if TYPE_CHECKING:
    import networkx as nx

    from ..models import CodeEntity


def calculate_confidence(entity: "CodeEntity", graph: "nx.DiGraph") -> float:
    """Calculate confidence that a function is truly dead code.

    Uses language-agnostic detection via entity attributes.
    Returns 0.0-1.0 where higher = more confident it's dead.
    """
    confidence = DEAD_CODE_BASE_CONFIDENCE
    confidence = _apply_decorator_penalty(confidence, entity)
    confidence = _apply_structural_adjustments(confidence, entity, graph)
    return max(0.0, min(1.0, confidence))


def _apply_decorator_penalty(confidence: float, entity: "CodeEntity") -> float:
    """Apply penalty for decorated functions (called by frameworks)."""
    if entity.is_decorated:
        confidence -= DEAD_CODE_DECORATOR_PENALTY
    return confidence


def _apply_structural_adjustments(
    confidence: float,
    entity: "CodeEntity",
    graph: "nx.DiGraph",
) -> float:
    """Apply adjustments based on code structure."""
    if entity.is_private:
        confidence += DEAD_CODE_PRIVATE_BOOST

    out_degree, in_degree = _get_graph_degrees(entity.node_id, graph)

    if _is_entry_point_pattern(entity, out_degree, in_degree):
        confidence -= DEAD_CODE_ENTRY_POINT_PENALTY

    if entity.parent_class:
        confidence -= DEAD_CODE_METHOD_PENALTY

    if _is_public_module_function(entity):
        confidence -= DEAD_CODE_PUBLIC_MODULE_PENALTY

    return confidence


def _get_graph_degrees(node_id: str, graph: "nx.DiGraph") -> tuple[int, int]:
    """Get out-degree and in-degree for a node."""
    if node_id not in graph:
        return 0, 0
    return graph.out_degree(node_id), graph.in_degree(node_id)


def _is_entry_point_pattern(
    entity: "CodeEntity",
    out_degree: int,
    in_degree: int,
) -> bool:
    """Check if entity matches entry point pattern."""
    is_module_level = not entity.parent_class
    is_public = not entity.is_private
    has_callees = out_degree > 0
    has_no_callers = in_degree == 0
    return is_module_level and is_public and has_callees and has_no_callers


def _is_public_module_function(entity: "CodeEntity") -> bool:
    """Check if entity is a public module-level function."""
    is_module_level = not entity.parent_class
    is_public = not entity.is_private
    return entity.entity_type == "function" and is_module_level and is_public


def get_reason(confidence: float) -> str:
    """Generate human-readable reason for flagging."""
    if confidence >= DEAD_CODE_CONFIDENCE_HIGH:
        return "No callers found, high confidence"
    elif confidence >= DEAD_CODE_CONFIDENCE_MEDIUM:
        return "No callers found, may be entry point or framework hook"
    return "No callers found, likely false positive (entry point pattern)"
