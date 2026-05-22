"""Propagation cost analysis for architecture health.

Language-agnostic approach:
- Uses call graph from tree-sitter parsing
- Computes transitive closure for reachability
- No language-specific logic - pure graph analysis

Research basis: MacCormack et al. (2006) - "Exploring the Structure of
Complex Software Designs: An Empirical Study of Open Source and Proprietary Code"
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import networkx as nx
from pydantic import Field

from ..cache import CacheManager
from ..constants import (
    MAX_TOKEN_LIMIT,
    MIN_TOKEN_LIMIT,
    PC_APPROXIMATE_STATUS,
    PC_DENSE_GRAPH_DENSITY_THRESHOLD,
    PC_DENSE_GRAPH_NODE_THRESHOLD,
    PC_MAX_NODES_FOR_CLOSURE,
    PC_THRESHOLD_CRITICAL,
    PC_THRESHOLD_GOOD,
    PC_THRESHOLD_WARNING,
)
from ._defaults import RESOLVED_DEFAULT_TOKEN_BUDGET
from .tool_timeout import worker_safe
from .validation import validate_root_directory
from ..config_loader import daemon_enabled
from ..daemon.client import try_daemon_propagation_cost as _daemon_propagation_cost
from ..daemon.protocol import PropagationCostParams
from ..token_estimator import count_tokens

if TYPE_CHECKING:
    from ..cache.models import CachedIndices

logger = logging.getLogger(__name__)

# Token budget constants
_TOKEN_OVERHEAD_PROPAGATION = 200
_TOKEN_PER_MODULE = 50
_TOKEN_PER_PATH = 60

# Re-exported from constants.py so tests and the formatter can monkeypatch /
# reference these via the propagation_cost module without importing constants.
MAX_NODES_FOR_CLOSURE = PC_MAX_NODES_FOR_CLOSURE
DENSE_GRAPH_NODE_THRESHOLD = PC_DENSE_GRAPH_NODE_THRESHOLD
DENSE_GRAPH_DENSITY_THRESHOLD = PC_DENSE_GRAPH_DENSITY_THRESHOLD
APPROXIMATE_STATUS = PC_APPROXIMATE_STATUS


async def _load_cached_indices(root_directory: str) -> "CachedIndices":
    def _load_sync() -> "CachedIndices":
        cache = CacheManager(Path(root_directory))
        return cache.load_or_rebuild()
    return await asyncio.to_thread(_load_sync)


def _extract_module_at_depth(parts: tuple[str, ...], src_idx: int, depth: int) -> str:
    """Extract module name at specified depth after src/ directory."""
    if src_idx >= 0 and src_idx + depth < len(parts):
        return "/".join(parts[src_idx + 1:src_idx + 1 + depth])
    elif len(parts) > depth:
        return "/".join(parts[-1 - depth:-1])
    return "root"


def _group_nodes_by_module(
    graph: nx.DiGraph,
    entities: dict[str, Any],
    depth: int,
) -> dict[str, set[str]]:
    """Group graph nodes by module at specified depth after src/ directory."""
    module_nodes: dict[str, set[str]] = defaultdict(set)
    for node_id in graph.nodes():
        entity = entities.get(node_id)
        if entity and hasattr(entity, 'file_path') and entity.file_path:
            parts = Path(entity.file_path).parts
            src_idx = next((i for i, p in enumerate(parts) if p == "src"), -1)
            module = _extract_module_at_depth(parts, src_idx, depth)
            module_nodes[module].add(node_id)
    return module_nodes


def _compute_module_coupling(
    module_nodes: dict[str, set[str]],
    closure: nx.DiGraph,
    token_limit: int | None,
) -> list[dict[str, Any]]:
    """Calculate coupling metrics for each module."""
    breakdown: list[dict[str, Any]] = []
    used_tokens = 0

    for module, nodes in sorted(module_nodes.items(), key=lambda x: -len(x[1])):
        internal_edges = sum(1 for u, v in closure.edges() if u in nodes and v in nodes)
        external_edges = sum(1 for u, v in closure.edges() if (u in nodes) != (v in nodes))
        internal_possible = len(nodes) * (len(nodes) - 1) if len(nodes) > 1 else 1
        internal_pc = internal_edges / internal_possible if internal_possible > 0 else 0

        entry = {
            "module": module,
            "functions": len(nodes),
            "internal_coupling": round(internal_pc, 4),
            "external_edges": external_edges,
        }

        if token_limit is not None:
            entry_tokens = count_tokens(str(entry), is_code=False) + _TOKEN_PER_MODULE
            if used_tokens + entry_tokens > token_limit:
                break
            used_tokens += entry_tokens

        breakdown.append(entry)

    breakdown.sort(key=lambda x: x["internal_coupling"] + x["external_edges"], reverse=True)
    return breakdown


# Maximum modules to return in breakdown
_MAX_MODULE_BREAKDOWN = 10


def _compute_module_breakdown(
    graph: nx.DiGraph,
    entities: dict[str, Any],
    closure: nx.DiGraph,
    token_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Compute per-module coupling contribution.

    Language-agnostic: Extracts module from file path, works for any language.
    Automatically uses subdirectories if only one top-level module exists.
    """
    # First pass: group by first level after src/
    module_nodes = _group_nodes_by_module(graph, entities, depth=1)

    # If only 1 module, try deeper grouping for more useful breakdown
    if len(module_nodes) == 1:
        module_nodes = _group_nodes_by_module(graph, entities, depth=2)

    breakdown = _compute_module_coupling(module_nodes, closure, token_limit)
    return breakdown[:_MAX_MODULE_BREAKDOWN]


def _find_critical_paths(
    graph: nx.DiGraph,
    entities: dict[str, Any],
    limit: int = 5,
    token_limit: int | None = None,
) -> list[dict]:
    """Find most connected paths (highest downstream impact).

    Language-agnostic: Uses graph structure only.
    """
    paths = []
    used_tokens = 0

    for node in graph.nodes():
        descendants = nx.descendants(graph, node)
        if len(descendants) > 0:
            entity = entities.get(node)
            name = getattr(entity, 'name', None) or node.split("::")[-1]
            file_path = getattr(entity, 'file_path', "") or ""
            line = getattr(entity, 'line_start', 1) or 1

            entry = {
                "start": name,
                "file": file_path,
                "line": line,
                "downstream_count": len(descendants),
                "downstream_sample": [
                    getattr(entities.get(d), 'name', None) or d.split("::")[-1]
                    for d in list(descendants)[:5]
                ],
            }

            # Apply token budget if set
            if token_limit is not None:
                entry_tokens = count_tokens(str(entry), is_code=False) + _TOKEN_PER_PATH
                if used_tokens + entry_tokens > token_limit:
                    continue
                used_tokens += entry_tokens

            paths.append(entry)

    # Sort by downstream count descending
    paths.sort(key=lambda x: x["downstream_count"], reverse=True)

    return paths[:limit]


def _summarize_max_module_coupling(
    module_breakdown: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Headline figure that doesn't shrink with total ``n``.

    The overall propagation cost has an n² denominator, so larger codebases trivially
    score lower. The max per-module internal coupling uses a module-scoped denominator
    and stays comparable across codebase sizes.
    """
    if not module_breakdown:
        return None
    top = max(module_breakdown, key=lambda m: m.get("internal_coupling", 0))
    return {
        "module": top.get("module", "?"),
        "value": top.get("internal_coupling", 0),
        "functions": top.get("functions", 0),
    }


def _generate_recommendations(
    pc: float,
    module_breakdown: list,
    approximation_used: bool = False,
) -> list[str]:
    """Generate actionable recommendations based on analysis.

    Language-agnostic: Based on metrics, not code content.
    """
    recommendations = []

    if approximation_used:
        recommendations.append(
            "[APPROXIMATION] Overall PC is a direct-edges lower bound; "
            "use the per-module breakdown below for actionable signal."
        )
    elif pc <= PC_THRESHOLD_GOOD:
        recommendations.append(f"[OK] Propagation cost is healthy ({pc*100:.1f}% < 10%)")
    elif pc <= PC_THRESHOLD_WARNING:
        recommendations.append(f"[WARNING] Moderate coupling ({pc*100:.1f}%) - monitor for increases")
    elif pc <= PC_THRESHOLD_CRITICAL:
        recommendations.append(f"[WARNING] High coupling ({pc*100:.1f}%) - consider interface extraction")
    else:
        recommendations.append(f"[CRITICAL] Architectural decay ({pc*100:.1f}% > 43%) - immediate action needed")

    # Module-specific recommendations
    if module_breakdown:
        high_coupling_modules = [
            m for m in module_breakdown
            if m["internal_coupling"] > 0.3
        ]
        for mod in high_coupling_modules[:3]:
            recommendations.append(
                f"[INFO] {mod['module']}/ has high internal coupling "
                f"({mod['internal_coupling']*100:.1f}%) - consider splitting"
            )

    return recommendations


def compute_propagation_cost(
    graph: nx.DiGraph,
    entities: dict[str, Any],
    include_breakdown: bool = True,
    show_critical_paths: bool = True,
    exclude_tests: bool = True,
    token_limit: int | None = None,  # None = no limit (CLI), int = limit (MCP)
    approximate: bool = False,
) -> dict[str, Any]:
    """Compute propagation cost metric.

    Language-agnostic: Uses only graph structure, no language-specific logic.

    Args:
        graph: Call graph (directed)
        entities: Code entities for metadata
        include_breakdown: Include per-module breakdown
        show_critical_paths: Include most connected paths
        exclude_tests: Exclude test nodes from analysis
        token_limit: Token budget (None = no limit for CLI, int = limit for MCP)
        approximate: When True, allow the direct-edges fallback for graphs that exceed
            the closure cutoff. The PASS/WARNING/CRITICAL verdict is suppressed in this
            mode because reachable_pairs is a lower bound, not the true reachability.

    Returns:
        Dict with propagation_cost, status, and detailed breakdown. If the graph exceeds
        the closure cutoff and ``approximate`` is False, returns an error dict instead.
    """
    # Filter out test nodes if requested
    if exclude_tests:
        test_nodes = {
            node_id for node_id, entity in entities.items()
            if getattr(entity, 'is_test', False)
        }
        graph = graph.subgraph([n for n in graph.nodes() if n not in test_nodes]).copy()

    n = graph.number_of_nodes()
    if n < 2:
        return {
            "propagation_cost": 0.0,
            "propagation_cost_percent": "0.00%",
            "status": "N/A",
            "threshold": PC_THRESHOLD_CRITICAL,
            "interpretation": "Too few nodes for meaningful analysis",
            "graph_stats": {
                "nodes": n,
                "edges": graph.number_of_edges(),
                "reachable_pairs": 0,
                "possible_pairs": 0,
            },
            "recommendations": ["[INFO] Add more functions to enable propagation cost analysis"],
        }

    # Compute transitive closure (reachability).
    # Beyond MAX_NODES_FOR_CLOSURE the closure is intractable; the caller must opt
    # in to the direct-edges lower bound via approximate=True.
    edge_count = graph.number_of_edges()
    density = edge_count / (n * (n - 1)) if n > 1 else 0

    cutoff_reason: str | None = None
    if n > MAX_NODES_FOR_CLOSURE:
        cutoff_reason = (
            f"graph has {n:,} nodes (> {MAX_NODES_FOR_CLOSURE:,} closure limit)"
        )
    elif density > DENSE_GRAPH_DENSITY_THRESHOLD and n > DENSE_GRAPH_NODE_THRESHOLD:
        cutoff_reason = (
            f"graph is dense ({density*100:.1f}% edge density at {n:,} nodes, "
            f"> {DENSE_GRAPH_DENSITY_THRESHOLD*100:.0f}% above {DENSE_GRAPH_NODE_THRESHOLD:,})"
        )

    if cutoff_reason and not approximate:
        return {
            "error": (
                f"Cannot compute exact propagation cost: {cutoff_reason}. "
                f"Pass --approximate to compute a direct-edges lower bound; the "
                f"PASS/WARNING/CRITICAL verdict is suppressed in approximation mode."
            ),
            "graph_stats": {
                "nodes": n,
                "edges": edge_count,
                "density": round(density, 4),
            },
        }

    use_approximation = cutoff_reason is not None
    if use_approximation:
        logger.warning(
            f"Propagation cost: {cutoff_reason}. "
            f"Using direct-edges approximation (--approximate enabled)."
        )

    if use_approximation:
        closure = graph
        reachable_pairs = edge_count
    else:
        try:
            closure = nx.transitive_closure(graph, reflexive=False)
            reachable_pairs = closure.number_of_edges()
        except MemoryError:
            logger.error(
                f"Memory error computing transitive closure ({n} nodes, {edge_count} edges). "
                f"Falling back to direct edges - propagation cost will be underestimated."
            )
            closure = graph
            reachable_pairs = edge_count
        except Exception as e:
            logger.warning(
                f"Transitive closure failed: {e}. "
                f"Falling back to direct edges - propagation cost will be underestimated."
            )
            closure = graph
            reachable_pairs = edge_count

    # Calculate propagation cost
    possible_pairs = n * (n - 1)
    pc = reachable_pairs / possible_pairs if possible_pairs > 0 else 0

    # In approximation mode the MacCormack thresholds do not apply: reachable_pairs
    # is a direct-edges lower bound, so the verdict is suppressed.
    if use_approximation:
        status = APPROXIMATE_STATUS
        interpretation = (
            f"Direct-edges lower bound ({pc*100:.2f}%). PASS/WARNING/CRITICAL verdict "
            f"suppressed - the approximation systematically under-counts reachable pairs."
        )
    elif pc <= PC_THRESHOLD_GOOD:
        status = "PASS"
        interpretation = f"Excellent decoupling - changes affect only {pc*100:.1f}% of codebase"
    elif pc <= PC_THRESHOLD_WARNING:
        status = "WARNING"
        interpretation = f"Moderate coupling - {pc*100:.1f}% affected per change. Consider refactoring."
    elif pc <= PC_THRESHOLD_CRITICAL:
        status = "WARNING"
        interpretation = f"High coupling - {pc*100:.1f}% affected. Architectural review recommended."
    else:
        status = "CRITICAL"
        interpretation = f"Architectural decay - {pc*100:.1f}% affected. Urgent refactoring needed."

    result: dict[str, Any] = {
        "propagation_cost": round(pc, 4),
        "propagation_cost_percent": f"{pc*100:.2f}%",
        "status": status,
        "threshold": PC_THRESHOLD_CRITICAL,
        "interpretation": interpretation,
        "approximation_used": use_approximation,
        "graph_stats": {
            "nodes": n,
            "edges": edge_count,
            "reachable_pairs": reachable_pairs,
            "possible_pairs": possible_pairs,
            "used_approximation": use_approximation,
            "closure_node_limit": MAX_NODES_FOR_CLOSURE,
        },
    }

    # Calculate remaining token budget for breakdown and paths
    breakdown_budget = None
    paths_budget = None
    if token_limit is not None:
        base_tokens = count_tokens(str(result), is_code=False) + _TOKEN_OVERHEAD_PROPAGATION
        remaining = token_limit - base_tokens
        if remaining > 0:
            # Split remaining budget between breakdown and paths
            breakdown_budget = remaining // 2
            paths_budget = remaining - breakdown_budget

    # Add per-module breakdown
    if include_breakdown:
        result["module_breakdown"] = _compute_module_breakdown(
            graph, entities, closure, breakdown_budget
        )
        result["max_module_coupling"] = _summarize_max_module_coupling(
            result["module_breakdown"]
        )

    # Add critical paths
    if show_critical_paths:
        result["critical_paths"] = _find_critical_paths(
            graph, entities, limit=5, token_limit=paths_budget
        )

    # Add recommendations
    result["recommendations"] = _generate_recommendations(
        pc, result.get("module_breakdown", []), use_approximation
    )

    return result


@worker_safe
async def propagation_cost(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    include_breakdown: Annotated[
        bool,
        Field(description="Include per-module breakdown of coupling"),
    ] = True,
    show_critical_paths: Annotated[
        bool,
        Field(description="Show most connected paths (bottlenecks)"),
    ] = True,
    exclude_tests: Annotated[
        bool,
        Field(description="Exclude test functions from analysis"),
    ] = True,
    token_limit: Annotated[
        int | None,
        Field(description="Soft limit on return size in tokens (None=no limit)", ge=MIN_TOKEN_LIMIT, le=MAX_TOKEN_LIMIT),
    ] = RESOLVED_DEFAULT_TOKEN_BUDGET,
    approximate: Annotated[
        bool,
        Field(description=(
            "Allow direct-edges approximation when the call graph exceeds the closure "
            "limit. When True, the PASS/WARNING/CRITICAL verdict is suppressed because "
            "reachable_pairs becomes a lower bound."
        )),
    ] = False,
) -> dict[str, Any]:
    """Measure propagation cost - how changes ripple through the codebase.

    Based on MacCormack et al. (2006) research showing that PC > 43%
    indicates architectural decay and predicts higher defect rates.

    WHEN TO USE:
    - To assess overall architecture health
    - Before major refactoring to establish baseline
    - To track coupling trends over time
    - To justify technical debt paydown

    WHEN NOT TO USE:
    - For day-to-day code changes (use change_impact_radius instead)
    - To find specific bad functions (use coupling_hotspots instead)

    OUTPUT:
    - propagation_cost: Overall PC percentage
    - status: PASS/WARNING/CRITICAL based on thresholds
    - module_breakdown: Per-module coupling contribution
    - critical_paths: Most connected function chains
    - recommendations: Actionable suggestions to reduce coupling
    """
    validation_error = validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    daemon_params = PropagationCostParams(
        source_dir=str(Path(root_directory).resolve()),
        include_breakdown=include_breakdown,
        show_critical_paths=show_critical_paths,
        exclude_tests=exclude_tests,
        token_limit=token_limit,
        approximate=approximate,
    )
    daemon_result = None
    if daemon_enabled():
        daemon_result = _daemon_propagation_cost(daemon_params, auto_start=False)
    if daemon_result is not None:
        return daemon_result

    try:
        indices = await _load_cached_indices(root_directory)
    except Exception as e:
        logger.exception("Failed to load cache for propagation cost")
        return {"error": f"Failed to load cache: {e}"}

    return compute_propagation_cost(
        graph=indices.graph,
        entities=indices.entities,
        include_breakdown=include_breakdown,
        show_critical_paths=show_critical_paths,
        exclude_tests=exclude_tests,
        token_limit=token_limit,
        approximate=approximate,
    )


def register_propagation_cost_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register propagation cost tools with the MCP server."""
    disabled = disabled_tools or set()
    tools = [("propagation_cost", propagation_cost)]
    for tool_name, tool_func in tools:
        if tool_name not in disabled:
            mcp.tool()(tool_func)
