"""Shared utilities for building and analyzing code graphs.

This module contains common helpers for graph construction and centrality
computation, used by both CacheManager and IncrementalUpdater.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from ..config import Config
from ..constants import AMBIGUOUS_METHOD_NAMES
from ..models.scores import CentralityCache
from .models import EntityLookups

if TYPE_CHECKING:
    import networkx as nx
    from ..models import CodeEntity


def compute_centrality(
    graph: "nx.DiGraph",
    type_graph: "nx.DiGraph | None" = None,
) -> CentralityCache:
    """
    Compute graph centrality metrics for ranking code entities.

    Returns three complementary metrics packaged in a `CentralityCache`:

    1. PageRank — structural importance via the "random surfer" model on the
       call graph.
    2. Betweenness — architectural bridging on the call graph.
    3. Type-PageRank — per-relation PR over the type-annotation subgraph,
       personalized by call-graph PR so types referenced from architecturally
       important code outweigh types referenced from leaf utility/config code.

    Args:
        graph: Directed call graph of code entity references with edge weights.
        type_graph: Optional dedicated type-annotation subgraph. When supplied
            and non-empty, drives the `type_pagerank` field.
    """
    if len(graph) == 0 and (type_graph is None or len(type_graph) == 0):
        return CentralityCache()

    import networkx as nx  # lazy import: 140ms startup cost

    if len(graph) > 0:
        try:
            pagerank = nx.pagerank(graph, weight="weight")

            # Detect degenerate uniform distribution (indicates disconnected
            # graph or equal-weight edges that don't differentiate nodes)
            values = list(pagerank.values())
            if len(values) >= 2 and (max(values) - min(values)) <= 1e-12:
                raise ValueError("Uniform distribution")
        except Exception:
            # Fallback: use weighted in-degree as a simpler importance proxy.
            in_degrees = {n: float(d) for n, d in graph.in_degree(weight="weight")}
            total = sum(in_degrees.values())
            if total > 0:
                pagerank = {n: v / total for n, v in in_degrees.items()}
            else:
                pagerank = {n: 0.0 for n in graph.nodes()}

        try:
            k = min(200, len(graph))
            betweenness = nx.betweenness_centrality(graph, k=k, weight="weight", seed=42)
        except Exception:
            betweenness = {n: 0.0 for n in graph.nodes()}
    else:
        pagerank = {}
        betweenness = {}

    type_pagerank: dict[str, float] = {}
    if type_graph is not None and len(type_graph) > 0:
        try:
            # Personalize by call-graph PR so types reached FROM important code
            # outrank types reached only from leaf utility/config code.
            personalization = {
                n: max(pagerank.get(n, 0.0), 1e-9) for n in type_graph.nodes()
            }
            type_pagerank = nx.pagerank(
                type_graph, weight="weight", personalization=personalization
            )
        except Exception:
            type_pagerank = {}

    return CentralityCache(
        pagerank=pagerank,
        betweenness=betweenness,
        type_pagerank=type_pagerank,
    )


def build_file_scopes(
    entities: dict[str, "CodeEntity"],
) -> dict[str, list[tuple[int, int, str]]]:
    """Build a mapping from file paths to sorted entity scopes.

    Args:
        entities: Mapping from node IDs to CodeEntity objects.

    Returns:
        Mapping from file path to list of (start_line, end_line, node_id) tuples,
        sorted by scope size (smallest first) to enable proper nesting detection.
    """
    file_scopes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for node_id, entity in entities.items():
        file_scopes[entity.file_path].append(
            (entity.line_start, entity.line_end, node_id)
        )

    # Sort scopes by size (smallest first) for proper nesting detection
    for file_path in file_scopes:
        file_scopes[file_path].sort(key=lambda x: x[1] - x[0])

    return dict(file_scopes)


def find_containing_scope(
    file_scopes: dict[str, list[tuple[int, int, str]]],
    file_path: str,
    line: int,
) -> str | None:
    """Find the innermost entity scope containing a given line.

    Args:
        file_scopes: Mapping from file path to sorted scope tuples.
        file_path: The file to search in.
        line: The line number to find.

    Returns:
        Node ID of the containing scope, or None if not found.
    """
    scopes = file_scopes.get(file_path, [])
    for start, end, node_id in scopes:
        if start <= line <= end:
            return node_id
    return None


def calculate_edge_weight(
    ref_type: str,
    target: "CodeEntity",
    dilution: float,
) -> float:
    """Calculate the weight for an edge based on reference type and target properties.

    Args:
        ref_type: The type of reference (call, import, etc.).
        target: The target entity being referenced.
        dilution: Dilution factor for ambiguous targets (1/sqrt(num_targets)).

    Returns:
        The calculated edge weight.
    """
    weight = Config.EDGE_WEIGHTS.get(ref_type, 1.0) * dilution

    if target.is_utility:
        weight *= Config.PENALTY_UTILITY
    if target.is_tiny:
        weight *= Config.PENALTY_TINY_FUNC
    if target.is_test:
        weight *= Config.PENALTY_TEST

    return weight


def add_or_update_edge(
    graph: "nx.DiGraph",
    source_id: str,
    target_id: str,
    weight: float,
    ref_type: str,
) -> None:
    """Add a new edge or update an existing edge with additional weight.

    Args:
        graph: The graph to modify.
        source_id: Source node ID.
        target_id: Target node ID.
        weight: Weight to add.
        ref_type: Reference type to record.
    """
    if graph.has_edge(source_id, target_id):
        graph[source_id][target_id]["weight"] += weight
        graph[source_id][target_id]["types"].add(ref_type)
    else:
        graph.add_edge(source_id, target_id, weight=weight, types={ref_type})


def build_edges_from_references(
    graph: "nx.DiGraph",
    entities: dict[str, "CodeEntity"],
    references: list[tuple[str, int, str, str, str | None]],
    name_to_nodes: dict[str, list[str]],
    qualified_name_to_nodes: dict[str, list[str]] | None = None,
    type_graph: "nx.DiGraph | None" = None,
) -> None:
    """Build graph edges from code references.

    Args:
        graph: The main call graph; receives every resolved reference.
        entities: Mapping from node IDs to CodeEntity objects.
        references: List of (file_path, line, target_name, ref_type, receiver) tuples.
            receiver is the object name for method calls (e.g., 'cache' in 'cache.get()').
        name_to_nodes: Mapping from symbol names to node IDs.
        qualified_name_to_nodes: Mapping from "ClassName.method" to node IDs.
            Used for precise resolution of method calls.
        type_graph: Dedicated subgraph that receives only `ref_type == "type"`
            edges, in parallel with the main graph. Drives the type_ref signal.
    """
    file_scopes = build_file_scopes(entities)
    if qualified_name_to_nodes is None:
        qualified_name_to_nodes = {}

    for file_path, line, target_name, ref_type, receiver in references:
        source_id = find_containing_scope(file_scopes, file_path, line)
        if not source_id:
            continue

        # Try qualified lookup first if receiver is available
        targets = []
        if receiver and qualified_name_to_nodes:
            qualified_name = f"{receiver}.{target_name}"
            targets = qualified_name_to_nodes.get(qualified_name, [])

        # Fall back to simple name lookup, but skip ambiguous method names which give too many false positives
        if not targets:
            if receiver and target_name in AMBIGUOUS_METHOD_NAMES:
                # Skip: e.g., cache.get() where we don't know what class cache is
                continue
            targets = name_to_nodes.get(target_name, [])

        if not targets:
            continue

        dilution = 1.0 / math.sqrt(len(targets))

        for target_id in targets:
            if source_id == target_id:
                continue

            target = entities.get(target_id)
            if not target:
                continue

            weight = calculate_edge_weight(ref_type, target, dilution)
            add_or_update_edge(graph, source_id, target_id, weight, ref_type)
            if ref_type == "type" and type_graph is not None:
                add_or_update_edge(type_graph, source_id, target_id, weight, ref_type)


def build_lookup_structures(
    entities: dict[str, "CodeEntity"],
) -> EntityLookups:
    """Build per-entity lookup structures.

    `qualified_name_to_nodes` maps "ClassName.method" to node IDs so callers
    can resolve method references when the receiver type is known.
    """
    name_to_nodes: dict[str, list[str]] = defaultdict(list)
    qualified_name_to_nodes: dict[str, list[str]] = defaultdict(list)
    file_scopes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    file_to_entities: dict[str, list[str]] = defaultdict(list)

    for node_id, entity in entities.items():
        name_to_nodes[entity.name].append(node_id)
        file_scopes[entity.file_path].append(
            (entity.line_start, entity.line_end, node_id)
        )
        file_to_entities[entity.file_path].append(node_id)

        if entity.parent_class and entity.entity_type in ("method", "function"):
            qualified_name = f"{entity.parent_class}.{entity.name}"
            qualified_name_to_nodes[qualified_name].append(node_id)

    # Smallest scope first so containing-scope lookups find the innermost match.
    for file_path in file_scopes:
        file_scopes[file_path].sort(key=lambda x: x[1] - x[0])

    return EntityLookups(
        name_to_nodes=dict(name_to_nodes),
        file_scopes=dict(file_scopes),
        file_to_entities=dict(file_to_entities),
        qualified_name_to_nodes=dict(qualified_name_to_nodes),
    )
