"""Graph-based code analysis tools.

This module provides MCP tools that leverage the call graph for analysis
that standard linters cannot do:
- change_impact_radius: Trace blast radius of changing a function
- coupling_hotspots: Find over-connected functions (high fan-in x fan-out)
- architectural_bottlenecks: Find chokepoint functions (high betweenness)

Performance: These tools use the daemon when available for fast in-memory
access to indices. Falls back to disk-based cache when daemon is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from ..cache import CacheManager
from ..daemon.client import DaemonClient, DaemonConnectionError, DaemonRequestError
from ..daemon.protocol import GraphAnalysisParams
from ..token_estimator import count_tokens
from ..graph_utils import (
    AUTO_MIN_IMPORTANCE,
    _TOKEN_OVERHEAD_CHANGE_IMPACT,
    _TOKEN_OVERHEAD_BOTTLENECKS,
    _TOKEN_PER_CALLER,
    _TOKEN_PER_BOTTLENECK,
    build_caller_info,
    compute_coupling_hotspots,
    detect_high_connectivity,
    extract_module_from_path,
    find_symbol_nodes,
    get_connected_modules,
    get_entity_display_name,
)

if TYPE_CHECKING:
    from ..cache.models import CachedIndices

logger = logging.getLogger(__name__)


def _validate_root_directory(root_directory: str) -> dict[str, Any] | None:
    """Validate that root_directory exists and is a directory.

    Args:
        root_directory: Path to validate.

    Returns:
        Error dict if invalid, None if valid.
    """
    if not root_directory:
        return {"error": "root_directory is required"}
    if not os.path.isdir(root_directory):
        return {"error": f"Root directory not found: {root_directory}"}
    return None


def _try_daemon_graph_analysis(
    method: str,
    params: GraphAnalysisParams,
) -> dict[str, Any] | None:
    """Try to perform graph analysis via daemon.

    Args:
        method: The daemon method to call ('architectural_bottlenecks', etc.)
        params: GraphAnalysisParams with parameters for the analysis.

    Returns:
        Result dict if daemon available and succeeded, None otherwise.
    """
    try:
        client = DaemonClient(timeout=5.0)
        if method == "architectural_bottlenecks":
            return client.architectural_bottlenecks(params)
        elif method == "coupling_hotspots":
            return client.coupling_hotspots(params)
        elif method == "change_impact_radius":
            return client.change_impact_radius(params)
        else:
            return None
    except (DaemonConnectionError, DaemonRequestError) as e:
        logger.debug(f"Daemon unavailable for {method}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Daemon error for {method}: {e}")
        return None


async def _load_cached_indices(root_directory: str) -> "CachedIndices":
    """Load cached indices for graph analysis.

    Uses CacheManager for disk-based caching. For faster repeated access,
    use the daemon which keeps indices in memory.

    Args:
        root_directory: Absolute path to the project root.

    Returns:
        CachedIndices containing graph, entities, and centrality metrics.

    Raises:
        Exception: If cache loading fails.
    """
    def _load_sync() -> "CachedIndices":
        cache = CacheManager(Path(root_directory), enable_semantic=False)
        return cache.load_or_rebuild()

    return await asyncio.to_thread(_load_sync)



async def change_impact_radius(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    symbol_name: Annotated[
        str,
        Field(description="Name of the function/class/method to analyze"),
    ],
    max_depth: Annotated[
        int,
        Field(description="Maximum levels of callers to trace upstream", ge=1, le=20),
    ] = 5,
    min_importance: Annotated[
        float,
        Field(description="Filter callers with importance score above threshold", ge=0.0, le=1.0),
    ] = 0.0,
    token_limit: Annotated[
        int,
        Field(description="Soft limit on the return size in tokens", ge=100, le=100000),
    ] = 4000,
) -> dict[str, Any]:
    """Analyze the blast radius of changing a function - trace all upstream callers.

    Shows what code will be affected if you refactor or change a function.
    Each caller is scored by importance (pagerank) to help prioritize testing.

    WHEN TO USE:
    - Before refactoring a function to understand what will be affected
    - To identify which tests should be run after a change
    - To find entry points (CLI commands, API handlers) that use a function

    WHEN NOT TO USE:
    - For finding callers without graph context (use find_identifier instead)
    - For downstream dependencies (what does this function call)

    OUTPUT:
    - symbol: The function being analyzed
    - direct_callers: Count of immediate callers
    - total_affected: All functions in the blast radius
    - affected_files: Number of files impacted
    - affected_modules: Distinct modules (top-level directories)
    - root_callers: Callers with no incoming edges (true entry points)
    - callers_by_depth: Detailed caller information grouped by distance
    """
    # Validate root directory exists
    validation_error = _validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    # Validate symbol_name is not empty
    if not symbol_name or not symbol_name.strip():
        return {"error": "symbol_name is required"}
    symbol_name = symbol_name.strip()

    # Try daemon first for fast in-memory access
    daemon_params = GraphAnalysisParams(
        source_dir=str(Path(root_directory).resolve()),
        symbol_name=symbol_name,
        max_depth=max_depth,
        min_importance=min_importance,
        token_limit=token_limit,
    )
    daemon_result = _try_daemon_graph_analysis("change_impact_radius", daemon_params)
    if daemon_result is not None:
        return daemon_result

    # Fallback to disk-based cache
    try:
        indices = await _load_cached_indices(root_directory)
    except Exception as e:
        logger.exception("Failed to load cache for change_impact_radius")
        return {"error": f"Failed to load cache: {e}"}

    graph = indices.graph
    entities = indices.entities
    pagerank = indices.pagerank
    name_to_nodes = indices.name_to_nodes

    # Use shared helper for symbol lookup
    matching_nodes = find_symbol_nodes(symbol_name, name_to_nodes)

    if not matching_nodes:
        return {
            "error": f"Symbol '{symbol_name}' not found in codebase",
            "suggestion": "Try a partial name or check spelling",
        }

    target_node = matching_nodes[0]
    target_entity = entities.get(target_node)

    if not target_entity:
        return {"error": f"Entity data not found for {target_node}"}

    # Use shared helper for high-connectivity detection
    target_fan_in = graph.in_degree(target_node)
    is_high_connectivity, warning_info = detect_high_connectivity(
        target_entity, target_fan_in
    )

    high_connectivity_warning = None
    if is_high_connectivity and warning_info:
        high_connectivity_warning = {
            **warning_info,
            "auto_filtered": min_importance < AUTO_MIN_IMPORTANCE,
        }
        # Auto-raise min_importance to prevent explosion
        if min_importance < AUTO_MIN_IMPORTANCE:
            min_importance = AUTO_MIN_IMPORTANCE

    # Token budget tracking
    used_tokens = _TOKEN_OVERHEAD_CHANGE_IMPACT
    token_budget_exceeded = False

    # BFS to find all upstream callers
    callers_by_depth: dict[int, list[dict]] = {}
    visited = {target_node}
    current_level = {target_node}
    all_affected_files = set()
    root_callers = []  # Callers with fan_in=0 (true entry points)

    for depth in range(1, max_depth + 1):
        next_level = set()
        depth_callers = []

        for node in current_level:
            if node in graph:
                for caller_node in graph.predecessors(node):
                    if caller_node not in visited:
                        visited.add(caller_node)
                        next_level.add(caller_node)

                        caller_entity = entities.get(caller_node)
                        if caller_entity:
                            importance = pagerank.get(caller_node, 0.0)

                            if importance < min_importance:
                                continue

                            # Use shared helper for building caller info
                            caller_info = build_caller_info(caller_entity, importance)
                            depth_callers.append(caller_info)
                            all_affected_files.add(caller_entity.file_path)

                            # Root callers have no incoming edges
                            if graph.in_degree(caller_node) == 0:
                                root_callers.append(caller_info)

        if depth_callers:
            depth_callers.sort(key=lambda x: x["importance"], reverse=True)

            # Apply token budget: keep callers until we exceed the limit
            filtered_callers = []
            for caller in depth_callers:
                # Estimate tokens for this caller entry
                caller_text = f"{caller['name']} {caller['file']} {caller['type']}"
                caller_tokens = count_tokens(caller_text, is_code=False) + _TOKEN_PER_CALLER
                if used_tokens + caller_tokens > token_limit:
                    token_budget_exceeded = True
                    break
                used_tokens += caller_tokens
                filtered_callers.append(caller)

            callers_by_depth[depth] = filtered_callers

        if not next_level or token_budget_exceeded:
            break
        current_level = next_level

    # Calculate affected modules using helper function
    root_path = Path(root_directory)
    affected_modules: set[str] = set()
    for file_path in all_affected_files:
        module = extract_module_from_path(file_path, root_path)
        if module:
            affected_modules.add(module)

    total_callers = sum(len(callers) for callers in callers_by_depth.values())
    direct_count = len(callers_by_depth.get(1, []))

    result = {
        "symbol": build_caller_info(target_entity, pagerank.get(target_node, 0.0)),
        "impact_summary": {
            "direct_callers": direct_count,
            "total_affected": total_callers,
            "affected_files": len(all_affected_files),
            "affected_modules": sorted(affected_modules),  # Sort for deterministic output
            "max_depth_reached": max(callers_by_depth.keys()) if callers_by_depth else 0,
            "token_budget_exceeded": token_budget_exceeded,
        },
        "root_callers": root_callers[:10],
        "callers_by_depth": callers_by_depth,
    }

    # Include warning for high-connectivity targets
    if high_connectivity_warning:
        result["high_connectivity_warning"] = high_connectivity_warning

    return result


async def coupling_hotspots(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    limit: Annotated[
        int,
        Field(description="Maximum number of hotspots to return", ge=1, le=100),
    ] = 20,
    min_coupling_score: Annotated[
        int,
        Field(description="Minimum fan_in x fan_out score to include", ge=0),
    ] = 10,
    exclude_tests: Annotated[
        bool,
        Field(description="Exclude test functions from results"),
    ] = True,
    exclude_private: Annotated[
        bool,
        Field(description="Exclude private/internal functions (starting with _)"),
    ] = False,
    token_limit: Annotated[
        int,
        Field(description="Soft limit on the return size in tokens", ge=100, le=100000),
    ] = 4000,
) -> dict[str, Any]:
    """Find functions with high coupling - many callers AND many dependencies.

    High fan-in x fan-out indicates functions that:
    - Are called by many places (high fan-in = change affects many callers)
    - Call many other functions (high fan-out = complex, hard to understand)
    - Are prime candidates for refactoring into smaller pieces

    WHEN TO USE:
    - To find functions that are too complex and need refactoring
    - To identify code that is hard to maintain due to high connectivity
    - To prioritize code review efforts on high-risk code

    WHEN NOT TO USE:
    - To find git churn hotspots (use find_hotspots instead)
    - To understand call paths (use trace_dependency_path instead)

    OUTPUT:
    - hotspots: List of functions ranked by coupling score
    - Each hotspot includes: name, file, fan_in, fan_out, coupling_score
    - summary: Statistics about coupling distribution
    """
    # Validate root directory exists
    validation_error = _validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    # Try daemon first for fast in-memory access
    daemon_params = GraphAnalysisParams(
        source_dir=str(Path(root_directory).resolve()),
        limit=limit,
        min_coupling_score=min_coupling_score,
        exclude_tests=exclude_tests,
        exclude_private=exclude_private,
        token_limit=token_limit,
    )
    daemon_result = _try_daemon_graph_analysis("coupling_hotspots", daemon_params)
    if daemon_result is not None:
        return daemon_result

    # Fallback to disk-based cache
    try:
        indices = await _load_cached_indices(root_directory)
    except Exception as e:
        logger.exception("Failed to load cache for coupling_hotspots")
        return {"error": f"Failed to load cache: {e}"}

    return compute_coupling_hotspots(
        graph=indices.graph,
        entities=indices.entities,
        pagerank=indices.pagerank,
        limit=limit,
        min_coupling_score=min_coupling_score,
        exclude_tests=exclude_tests,
        exclude_private=exclude_private,
        token_limit=token_limit,
    )


async def architectural_bottlenecks(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    limit: Annotated[
        int,
        Field(description="Maximum number of bottlenecks to return", ge=1, le=100),
    ] = 20,
    min_betweenness: Annotated[
        float,
        Field(description="Minimum betweenness centrality score (0.0-1.0)", ge=0.0, le=1.0),
    ] = 0.001,
    exclude_tests: Annotated[
        bool,
        Field(description="Exclude test functions from results"),
    ] = True,
    token_limit: Annotated[
        int,
        Field(description="Soft limit on the return size in tokens", ge=100, le=100000),
    ] = 4000,
) -> dict[str, Any]:
    """Find architectural bottlenecks - functions on many paths between other functions.

    High betweenness centrality means:
    - The function is a "bridge" connecting different parts of the codebase
    - Many call paths flow through this function
    - Changes here can have cascading effects across modules

    WHEN TO USE:
    - To identify critical code paths that need extra testing
    - To find natural boundaries for microservice extraction
    - To understand which functions act as bridges between modules

    WHEN NOT TO USE:
    - To find frequently changed code (use find_hotspots instead)
    - To find high fan-in/fan-out code (use coupling_hotspots instead)

    OUTPUT:
    - bottlenecks: Functions ranked by betweenness centrality
    - Each includes: name, betweenness score, fan_in, fan_out, connected_modules
    """
    # Validate root directory exists
    validation_error = _validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    # Try daemon first for fast in-memory access
    daemon_params = GraphAnalysisParams(
        source_dir=str(Path(root_directory).resolve()),
        limit=limit,
        min_betweenness=min_betweenness,
        exclude_tests=exclude_tests,
        token_limit=token_limit,
    )
    daemon_result = _try_daemon_graph_analysis("architectural_bottlenecks", daemon_params)
    if daemon_result is not None:
        return daemon_result

    # Fallback to disk-based cache
    try:
        indices = await _load_cached_indices(root_directory)
    except Exception as e:
        logger.exception("Failed to load cache for architectural_bottlenecks")
        return {"error": f"Failed to load cache: {e}"}

    graph = indices.graph
    entities = indices.entities
    betweenness = indices.betweenness
    pagerank = indices.pagerank

    root_path = Path(root_directory)
    bottlenecks = []

    for node_id, bt_score in betweenness.items():
        if bt_score < min_betweenness:
            continue

        entity = entities.get(node_id)
        if not entity:
            continue

        if exclude_tests and entity.is_test:
            continue

        # Check if node exists in graph (betweenness dict may include stale entries)
        if node_id not in graph:
            continue

        fan_in = graph.in_degree(node_id)
        fan_out = graph.out_degree(node_id)
        connected_modules = get_connected_modules(node_id, graph, entities, root_path)

        bottlenecks.append({
            "name": get_entity_display_name(entity),
            "file": entity.file_path,
            "line": entity.line_start,
            "type": entity.entity_type,
            "betweenness": round(bt_score, 6),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "importance": round(pagerank.get(node_id, 0.0), 6),
            "connected_modules": sorted(connected_modules),  # Sort for deterministic output
            "lines": entity.line_count,
        })

    bottlenecks.sort(key=lambda x: x["betweenness"], reverse=True)
    bottlenecks = bottlenecks[:limit]

    # Apply token budget filtering
    used_tokens = _TOKEN_OVERHEAD_BOTTLENECKS
    token_budget_exceeded = False

    filtered_bottlenecks = []
    for bottleneck in bottlenecks:
        # Include connected_modules in token estimate
        modules_text = " ".join(bottleneck["connected_modules"])
        bottleneck_text = f"{bottleneck['name']} {bottleneck['file']} {bottleneck['type']} {modules_text}"
        bottleneck_tokens = count_tokens(bottleneck_text, is_code=False) + _TOKEN_PER_BOTTLENECK
        if used_tokens + bottleneck_tokens > token_limit:
            token_budget_exceeded = True
            break
        used_tokens += bottleneck_tokens
        filtered_bottlenecks.append(bottleneck)
    bottlenecks = filtered_bottlenecks

    all_bt = list(betweenness.values())
    if all_bt:
        avg_bt = sum(all_bt) / len(all_bt)
        max_bt = max(all_bt)
    else:
        avg_bt = max_bt = 0

    return {
        "bottlenecks": bottlenecks,
        "summary": {
            "total_functions_analyzed": len(betweenness),
            "functions_above_threshold": len([b for b in all_bt if b >= min_betweenness]),
            "average_betweenness": round(avg_bt, 6),
            "max_betweenness": round(max_bt, 6),
            "token_budget_exceeded": token_budget_exceeded,
        },
    }


def register_graph_analysis_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register all graph analysis tools with the MCP server."""
    disabled = disabled_tools or set()

    tools = [
        ("change_impact_radius", change_impact_radius),
        ("coupling_hotspots", coupling_hotspots),
        ("architectural_bottlenecks", architectural_bottlenecks),
    ]

    for tool_name, tool_func in tools:
        if tool_name not in disabled:
            mcp.tool()(tool_func)
