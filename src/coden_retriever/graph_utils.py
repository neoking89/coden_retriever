"""Shared utility functions for graph-based code analysis.

This module contains common helpers used by both the daemon server and MCP tools
for graph analysis operations like symbol lookup, caller traversal, and hotspot detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .constants import DEFAULT_TOKEN_BUDGET
from .token_estimator import count_tokens

if TYPE_CHECKING:
    import networkx as nx
    from .models import CodeEntity

# High-connectivity protection thresholds
HIGH_FANIN_THRESHOLD = 100
AUTO_MIN_IMPORTANCE = 0.001

# Token budget overhead estimates (in tokens)
_TOKEN_OVERHEAD_CHANGE_IMPACT = 200
_TOKEN_OVERHEAD_HOTSPOTS = 150
_TOKEN_OVERHEAD_BOTTLENECKS = 150
_TOKEN_PER_CALLER = 10
_TOKEN_PER_HOTSPOT = 15
_TOKEN_PER_BOTTLENECK = 20

# Thresholds for risk categorization
COUPLING_HIGH_THRESHOLD = 50  # Fan-in + Fan-out above this is "high coupling"
COMPLEXITY_HIGH_THRESHOLD = 10  # Cyclomatic complexity above this is "high complexity"


def calculate_risk_score(fan_in: int, fan_out: int, complexity: int | None) -> float:
    """Calculate composite risk score combining coupling and complexity.

    Formula: Risk Score = (Fan_In + Fan_Out) * log(Cyclomatic_Complexity + 1)

    Args:
        fan_in: Number of callers
        fan_out: Number of callees
        complexity: Cyclomatic complexity (None defaults to 1)

    Returns:
        Composite risk score
    """
    coupling = fan_in + fan_out
    cc = complexity if complexity is not None else 1
    return coupling * math.log1p(cc)


def calculate_dispatcher_score(fan_out: int, complexity: int | None) -> float:
    """Calculate dispatcher score: downstream-only complexity weight.

    Formula: Dispatcher Score = Fan_Out * log(Cyclomatic_Complexity + 1)

    Drops fan_in deliberately: PageRank and Betweenness already reward fan_in,
    so a third fan_in-aware signal would be redundant. This rewards functions
    that orchestrate many callees with branching logic — the textbook dispatcher
    shape (e.g. processCommand, beginWork) — which the fan_in-correlated signals
    bury under leaf utilities.
    """
    cc = complexity if complexity is not None else 1
    return fan_out * math.log1p(cc)


def categorize_risk(fan_in: int, fan_out: int, complexity: int | None) -> str:
    """Categorize a function's risk based on coupling and complexity.

    Categories:
    - "Danger Zone": High Coupling AND High Complexity (highest risk)
    - "Traffic Jam": High Coupling, Low Complexity (architectural hub)
    - "Local Mess": Low Coupling, High Complexity (hard to test)
    - "Low Risk": Low on both dimensions

    Args:
        fan_in: Number of callers
        fan_out: Number of callees
        complexity: Cyclomatic complexity (None defaults to 1)

    Returns:
        Risk category string
    """
    coupling = fan_in + fan_out
    cc = complexity if complexity is not None else 1

    high_coupling = coupling >= COUPLING_HIGH_THRESHOLD
    high_complexity = cc >= COMPLEXITY_HIGH_THRESHOLD

    if high_coupling and high_complexity:
        return "Danger Zone"
    elif high_coupling and not high_complexity:
        return "Traffic Jam"
    elif not high_coupling and high_complexity:
        return "Local Mess"
    else:
        return "Low Risk"


def get_entity_display_name(entity: "CodeEntity") -> str:
    """Get a human-readable display name for an entity.

    Args:
        entity: The code entity to get a name for.

    Returns:
        Display name in format "filename::ClassName.method" or "filename::function".
    """
    rel_path = Path(entity.file_path).name
    if entity.parent_class:
        return f"{rel_path}::{entity.parent_class}.{entity.name}"
    return f"{rel_path}::{entity.name}"


def extract_module_from_path(file_path: str, root_path: Path) -> str | None:
    """Extract the top-level module name from a file path.

    Args:
        file_path: Absolute path to the file.
        root_path: Root directory path.

    Returns:
        Top-level module name or None if extraction fails.
    """
    try:
        rel_path = Path(file_path).relative_to(root_path)
        if len(rel_path.parts) > 0:
            return rel_path.parts[0]
    except ValueError:
        pass
    return None


def find_symbol_nodes(
    symbol_name: str,
    name_to_nodes: dict[str, list[str]],
) -> list[str]:
    """Find graph nodes matching a symbol name.

    First tries exact match, then falls back to case-insensitive partial match.

    Args:
        symbol_name: Name of the symbol to find.
        name_to_nodes: Mapping from symbol names to node IDs.

    Returns:
        List of matching node IDs.
    """
    matching_nodes = list(name_to_nodes.get(symbol_name, []))

    if not matching_nodes:
        for name, nodes in name_to_nodes.items():
            if symbol_name.lower() in name.lower():
                matching_nodes.extend(nodes)

    return matching_nodes


def build_caller_info(
    entity: "CodeEntity",
    importance: float,
) -> dict[str, Any]:
    """Build a standardized caller info dictionary.

    Args:
        entity: The caller entity.
        importance: The PageRank importance score.

    Returns:
        Dictionary with caller information.
    """
    return {
        "name": get_entity_display_name(entity),
        "file": entity.file_path,
        "line": entity.line_start,
        "type": entity.entity_type,
        "importance": round(importance, 6),
    }


def build_hotspot_info(
    entity: "CodeEntity",
    fan_in: int,
    fan_out: int,
    pagerank_score: float,
    complexity: int | None = None,
) -> dict[str, Any]:
    """Build a standardized coupling hotspot info dictionary.

    Args:
        entity: The hotspot entity.
        fan_in: Number of incoming edges.
        fan_out: Number of outgoing edges.
        pagerank_score: The PageRank importance score.
        complexity: Cyclomatic complexity (None defaults to 1).

    Returns:
        Dictionary with hotspot information including risk scoring.
    """
    cc = complexity if complexity is not None else 1
    risk_score = calculate_risk_score(fan_in, fan_out, cc)
    category = categorize_risk(fan_in, fan_out, cc)

    return {
        "name": get_entity_display_name(entity),
        "file": entity.file_path,
        "line": entity.line_start,
        "type": entity.entity_type,
        "fan_in": fan_in,
        "fan_out": fan_out,
        "coupling_score": fan_in * fan_out,
        "complexity": cc,
        "risk_score": round(risk_score, 2),
        "category": category,
        "importance": round(pagerank_score, 6),
        "lines": entity.line_count,
    }


def build_bottleneck_info(
    entity: "CodeEntity",
    bt_score: float,
    fan_in: int,
    fan_out: int,
    pagerank_score: float,
    connected_modules: set[str],
) -> dict[str, Any]:
    """Build a standardized bottleneck info dictionary.

    Args:
        entity: The bottleneck entity.
        bt_score: Betweenness centrality score.
        fan_in: Number of incoming edges.
        fan_out: Number of outgoing edges.
        pagerank_score: The PageRank importance score.
        connected_modules: Set of connected module names.

    Returns:
        Dictionary with bottleneck information.
    """
    return {
        "name": get_entity_display_name(entity),
        "file": entity.file_path,
        "line": entity.line_start,
        "type": entity.entity_type,
        "betweenness": round(bt_score, 6),
        "fan_in": fan_in,
        "fan_out": fan_out,
        "importance": round(pagerank_score, 6),
        "connected_modules": sorted(connected_modules),
        "lines": entity.line_count,
    }


def get_connected_modules(
    node_id: str,
    graph: "nx.DiGraph",
    entities: dict[str, "CodeEntity"],
    root_path: Path,
) -> set[str]:
    """Get the set of modules connected to a node (via predecessors and successors).

    Args:
        node_id: The node to check connectivity for.
        graph: The call graph.
        entities: Mapping from node IDs to entities.
        root_path: Root directory path for module extraction.

    Returns:
        Set of connected module names.
    """
    connected_modules: set[str] = set()

    for pred in graph.predecessors(node_id):
        pred_entity = entities.get(pred)
        if pred_entity:
            module = extract_module_from_path(pred_entity.file_path, root_path)
            if module:
                connected_modules.add(module)

    for succ in graph.successors(node_id):
        succ_entity = entities.get(succ)
        if succ_entity:
            module = extract_module_from_path(succ_entity.file_path, root_path)
            if module:
                connected_modules.add(module)

    return connected_modules


def detect_high_connectivity(
    entity: "CodeEntity",
    fan_in: int,
) -> tuple[bool, dict[str, Any] | None]:
    """Detect if a target is high-connectivity and may need filtering.

    Args:
        entity: The target entity.
        fan_in: Number of incoming edges.

    Returns:
        Tuple of (is_high_connectivity, warning_dict or None).
    """
    is_high_connectivity = entity.is_utility or fan_in > HIGH_FANIN_THRESHOLD

    if not is_high_connectivity:
        return False, None

    return True, {
        "type": "high_connectivity_target",
        "reason": "utility_name" if entity.is_utility else "high_fan_in",
        "fan_in": fan_in,
    }


def apply_token_budget_filter(
    items: list[dict[str, Any]],
    token_limit: int | None,
    used_tokens: int,
    tokens_per_item: int,
    text_fields: list[str],
    text_builder: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Filter a list of items to fit within a token budget.

    When ``text_builder`` is provided, it produces the per-item text used for
    token estimation. Otherwise the named ``text_fields`` are space-joined.

    Returns:
        Tuple of (filtered_items, used_tokens, token_budget_exceeded).
    """
    if token_limit is None:
        return items, used_tokens, False

    filtered = []
    budget_exceeded = False

    for item in items:
        if text_builder is not None:
            text = text_builder(item)
        else:
            text = " ".join(str(item.get(field, "")) for field in text_fields)
        tokens = count_tokens(text, is_code=False) + tokens_per_item
        if used_tokens + tokens > token_limit:
            budget_exceeded = True
            break
        used_tokens += tokens
        filtered.append(item)

    return filtered, used_tokens, budget_exceeded


def compute_percentile_scores(
    scores: list[float | int],
) -> tuple[float, float, float, float]:
    """Compute summary statistics for a list of scores.

    Args:
        scores: List of numeric scores.

    Returns:
        Tuple of (average, p90, p99, highest) or (0, 0, 0, 0) if empty.
    """
    if not scores:
        return 0.0, 0, 0, 0

    sorted_scores = sorted(scores, reverse=True)
    avg = sum(sorted_scores) / len(sorted_scores)
    p90 = sorted_scores[len(sorted_scores) // 10] if len(sorted_scores) >= 10 else sorted_scores[0]
    p99 = sorted_scores[len(sorted_scores) // 100] if len(sorted_scores) >= 100 else sorted_scores[0]

    return avg, p90, p99, sorted_scores[0]


def compute_coupling_hotspots(
    graph: "nx.DiGraph",
    entities: dict[str, "CodeEntity"],
    pagerank: dict[str, float],
    limit: int = 20,
    min_coupling_score: int = 10,
    exclude_tests: bool = True,
    exclude_private: bool = False,
    token_limit: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Compute coupling hotspots from a code dependency graph.

    This is the core computation shared by both daemon handlers and MCP tools.
    Functions with high fan-in x fan-out are candidates for refactoring.

    Args:
        graph: NetworkX DiGraph of code dependencies.
        entities: Mapping from node IDs to CodeEntity objects.
        pagerank: Mapping from node IDs to PageRank scores.
        limit: Maximum number of hotspots to return.
        min_coupling_score: Minimum fan_in * fan_out to include.
        exclude_tests: Whether to exclude test functions.
        exclude_private: Whether to exclude private/internal functions.
        token_limit: Soft limit on return size in tokens.

    Returns:
        Dictionary with 'hotspots' list and 'summary' statistics.
    """
    hotspots = []
    all_coupling_scores = []
    all_risk_scores = []
    all_complexities = []
    category_counts = {"Traffic Jam": 0, "Local Mess": 0, "Danger Zone": 0, "Low Risk": 0}

    for node_id in graph.nodes():
        entity = entities.get(node_id)
        if not entity:
            continue

        if exclude_tests and entity.is_test:
            continue
        if exclude_private and entity.is_private:
            continue

        fan_in = graph.in_degree(node_id)
        fan_out = graph.out_degree(node_id)
        coupling_score = fan_in * fan_out

        # Get cyclomatic complexity from entity (may be None for classes)
        complexity = getattr(entity, 'cyclomatic_complexity', None)
        if complexity is not None:
            all_complexities.append(complexity)

        # Calculate composite risk score
        risk_score = calculate_risk_score(fan_in, fan_out, complexity)
        category = categorize_risk(fan_in, fan_out, complexity)

        all_coupling_scores.append(coupling_score)
        all_risk_scores.append(risk_score)
        category_counts[category] += 1

        if coupling_score >= min_coupling_score:
            hotspots.append(build_hotspot_info(
                entity, fan_in, fan_out, pagerank.get(node_id, 0.0), complexity
            ))

    # Sort by risk_score (composite) instead of just coupling_score
    hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
    hotspots = hotspots[:limit]

    # Apply token budget filtering
    filtered_hotspots, _, token_budget_exceeded = apply_token_budget_filter(
        hotspots,
        token_limit,
        _TOKEN_OVERHEAD_HOTSPOTS,
        _TOKEN_PER_HOTSPOT,
        ["name", "file", "type"],
    )

    # Coupling score statistics
    avg_coupling, p90_coupling, p99_coupling, highest_coupling = compute_percentile_scores(
        all_coupling_scores
    )

    # Risk score statistics
    if all_risk_scores:
        all_risk_scores.sort(reverse=True)
        avg_risk = sum(all_risk_scores) / len(all_risk_scores)
        max_risk = all_risk_scores[0]
    else:
        avg_risk = max_risk = 0

    # Complexity statistics
    if all_complexities:
        avg_complexity = sum(all_complexities) / len(all_complexities)
        max_complexity = max(all_complexities)
    else:
        avg_complexity = max_complexity = 1

    return {
        "hotspots": filtered_hotspots,
        "summary": {
            "total_functions_analyzed": len(all_coupling_scores),
            "functions_above_threshold": len([s for s in all_coupling_scores if s >= min_coupling_score]),
            "average_coupling_score": round(avg_coupling, 2),
            "p90_coupling_score": p90_coupling,
            "p99_coupling_score": p99_coupling,
            "highest_coupling_score": highest_coupling,
            "average_complexity": round(avg_complexity, 2),
            "max_complexity": max_complexity,
            "average_risk_score": round(avg_risk, 2),
            "max_risk_score": round(max_risk, 2),
            "category_distribution": category_counts,
            "token_budget_exceeded": token_budget_exceeded,
        },
    }


@dataclass
class BfsContext:
    """Read-only inputs for an upstream-caller BFS over the call graph.

    Carried unchanged through every level of the traversal; the per-traversal
    mutable state lives in `BfsAccumulator`.
    """

    graph: Any
    entities: dict[str, Any]
    pagerank: dict[str, float]
    min_importance: float


@dataclass
class BfsAccumulator:
    """Mutable BFS state that grows as the upstream traversal proceeds.

    `expand_callers_one_level` adds to `visited`, `affected_files`, and
    `root_callers`. `trace_impact_bfs` additionally tracks the per-depth
    callers map and token-budget bookkeeping.
    """

    visited: set[str] = field(default_factory=set)
    affected_files: set[str] = field(default_factory=set)
    root_callers: list[dict[str, Any]] = field(default_factory=list)
    callers_by_depth: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    used_tokens: int = 0
    token_budget_exceeded: bool = False


def expand_callers_one_level(
    current_level: set[str],
    ctx: BfsContext,
    acc: BfsAccumulator,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Walk one BFS step upstream. Returns (next_level, callers_at_this_depth).

    Mutates `acc` (visited, affected_files, root_callers) as the traversal
    accumulates across depth levels.
    """
    next_level: set[str] = set()
    depth_callers: list[dict[str, Any]] = []

    for node in current_level:
        if node not in ctx.graph:
            continue
        for caller_node in ctx.graph.predecessors(node):
            if caller_node in acc.visited:
                continue
            acc.visited.add(caller_node)
            next_level.add(caller_node)
            caller_entity = ctx.entities.get(caller_node)
            if not caller_entity:
                continue
            importance = ctx.pagerank.get(caller_node, 0.0)
            if importance < ctx.min_importance:
                continue
            caller_info = build_caller_info(caller_entity, importance)
            depth_callers.append(caller_info)
            acc.affected_files.add(caller_entity.file_path)
            if ctx.graph.in_degree(caller_node) == 0:
                acc.root_callers.append(caller_info)
    return next_level, depth_callers


def trace_impact_bfs(
    target_node: str,
    ctx: BfsContext,
    max_depth: int,
    token_limit: int,
) -> BfsAccumulator:
    """BFS upstream from `target_node`, respecting token budget."""
    acc = BfsAccumulator(
        visited={target_node},
        used_tokens=_TOKEN_OVERHEAD_CHANGE_IMPACT,
    )
    current_level: set[str] = {target_node}

    for depth in range(1, max_depth + 1):
        next_level, depth_callers = expand_callers_one_level(current_level, ctx, acc)
        if depth_callers:
            depth_callers.sort(key=lambda x: x["importance"], reverse=True)
            kept, acc.used_tokens, acc.token_budget_exceeded = apply_token_budget_filter(
                depth_callers,
                token_limit,
                acc.used_tokens,
                _TOKEN_PER_CALLER,
                ["name", "file", "type"],
            )
            acc.callers_by_depth[depth] = kept
        if not next_level or acc.token_budget_exceeded:
            break
        current_level = next_level

    return acc
