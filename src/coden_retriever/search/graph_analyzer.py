"""
Graph analysis for coden-retriever.

Handles construction and analysis of the code call graph, including:
- Graph building from parsed references
- Centrality computation (PageRank, Betweenness)
- Path tracing between symbols
- Score aggregation for classes

This module is extracted from SearchEngine to follow SRP.
"""
import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

from ..config import Config
from ..constants import AMBIGUOUS_METHOD_NAMES
from ..models import PathTraceResult

if TYPE_CHECKING:
    from networkx import DiGraph
    from ..models.entities import CodeEntity

logger = logging.getLogger(__name__)

# Module-level reference for tests to mock (populated on first use)
_nx_module = None


def _get_nx():
    """Lazy import networkx to avoid 140ms startup cost."""
    global _nx_module
    import networkx as nx
    _nx_module = nx
    return nx



class GraphAnalyzer:
    """
    Analyzes code structure through graph algorithms.

    Responsible for:
    - Building call graphs from parsed code references
    - Computing centrality metrics (PageRank, Betweenness)
    - Tracing paths between code entities
    - Aggregating scores to parent classes

    This class encapsulates all graph-related logic that was previously
    embedded in SearchEngine.
    """

    def __init__(
        self,
        entities: dict[str, "CodeEntity"],
        name_to_nodes: dict[str, list[str]],
        file_scopes: dict[str, list[tuple[int, int, str]]],
        verbose: bool = False,
    ):
        """
        Initialize the graph analyzer.

        Args:
            entities: Mapping of node_id to CodeEntity
            name_to_nodes: Mapping of entity name to list of node_ids
            file_scopes: Mapping of file_path to list of (start, end, node_id)
            verbose: Enable verbose logging
        """
        self._entities = entities
        self._name_to_nodes = name_to_nodes
        self._file_scopes = file_scopes
        self._verbose = verbose

        nx = _get_nx()
        self._graph: "DiGraph" = nx.DiGraph()
        self._pagerank_cache: dict[str, float] = {}
        self._betweenness_cache: dict[str, float] = {}

    @property
    def graph(self) -> "DiGraph":
        """Get the underlying graph."""
        return self._graph

    @property
    def pagerank_cache(self) -> dict[str, float]:
        """Get cached PageRank scores."""
        return self._pagerank_cache

    @property
    def betweenness_cache(self) -> dict[str, float]:
        """Get cached Betweenness centrality scores."""
        return self._betweenness_cache

    def build_graph(self, references: list[tuple[str, int, str, str, str | None]]) -> None:
        """
        Build the code graph from parsed references.

        Args:
            references: List of (file_path, line, target_name, ref_type, receiver) tuples.
                receiver is the object name for method calls (e.g., 'cache' in 'cache.get()').
        """
        # Build qualified name index for methods (ClassName.method)
        qualified_name_to_nodes: dict[str, list[str]] = defaultdict(list)
        for node_id, entity in self._entities.items():
            if entity.parent_class and entity.entity_type in ("method", "function"):
                qualified_name = f"{entity.parent_class}.{entity.name}"
                qualified_name_to_nodes[qualified_name].append(node_id)

        for file_path, line, target_name, ref_type, receiver in references:
            source_id = self._find_containing_scope(file_path, line)
            if not source_id:
                continue

            # Try qualified lookup first if receiver is available
            targets = []
            if receiver and qualified_name_to_nodes:
                qualified_name = f"{receiver}.{target_name}"
                targets = qualified_name_to_nodes.get(qualified_name, [])

            # Fall back to simple name lookup, but skip ambiguous method names
            # when we have a receiver but couldn't resolve it (too many false positives)
            if not targets:
                if receiver and target_name in AMBIGUOUS_METHOD_NAMES:
                    # Skip: e.g., cache.get() where we don't know what class cache is
                    continue
                targets = self._name_to_nodes.get(target_name, [])

            if not targets:
                continue

            dilution = 1.0 / math.sqrt(len(targets))

            for target_id in targets:
                if source_id == target_id:
                    continue

                target = self._entities[target_id]
                weight = Config.EDGE_WEIGHTS.get(ref_type, 1.0) * dilution

                if target.is_utility:
                    weight *= Config.PENALTY_UTILITY
                if target.is_tiny:
                    weight *= Config.PENALTY_TINY_FUNC
                if target.is_test:
                    weight *= Config.PENALTY_TEST

                if self._graph.has_edge(source_id, target_id):
                    self._graph[source_id][target_id]["weight"] += weight
                    self._graph[source_id][target_id]["types"].add(ref_type)
                else:
                    self._graph.add_edge(source_id, target_id, weight=weight, types={ref_type})

    def _find_containing_scope(self, file_path: str, line: int) -> str | None:
        """Find the entity containing a given line."""
        scopes = self._file_scopes.get(file_path, [])
        for start, end, node_id in scopes:
            if start <= line <= end:
                return node_id
        return None

    def compute_centrality(self) -> None:
        """
        Pre-compute graph centrality metrics for ranking code entities.

        Computes two complementary centrality measures:

        1. PageRank - Measures "importance" based on incoming references.
           Entities that are referenced by many important entities get high scores.

        2. Betweenness Centrality - Measures "architectural bridging".
           Entities that lie on many shortest paths between other entities
           are architectural bridges/connectors in the codebase.

        These metrics are pre-computed during indexing for O(1) lookup during search.
        """
        if len(self._graph) == 0:
            self._pagerank_cache = {}
            self._betweenness_cache = {}
            return

        # PageRank Computation
        nx = _get_nx()
        try:
            pagerank_scores = nx.pagerank(self._graph, weight="weight")
            if self._scores_are_uniform(pagerank_scores):
                raise ValueError("PageRank returned a uniform distribution")
            self._pagerank_cache = pagerank_scores
        except Exception as e:
            logger.warning(f"PageRank failed or was uniform ({e}); falling back to weighted in-degree")
            self._pagerank_cache = self._fallback_pagerank_scores()

        # Betweenness Centrality Computation
        try:
            k = min(200, len(self._graph))
            self._betweenness_cache = nx.betweenness_centrality(
                self._graph, k=k, weight="weight", seed=42
            )
        except Exception:
            self._betweenness_cache = {n: 0.0 for n in self._graph.nodes()}

    def _scores_are_uniform(self, scores: dict[str, float], abs_tol: float = 1e-12) -> bool:
        """Return True when all scores are effectively identical."""
        if not scores:
            return True
        values = list(scores.values())
        if len(values) < 2:
            return True
        return (max(values) - min(values)) <= abs_tol

    def _fallback_pagerank_scores(self) -> dict[str, float]:
        """Fallback structural importance signal when PageRank is unusable."""
        if len(self._graph) == 0:
            return {}

        in_degrees = {n: float(d) for n, d in self._graph.in_degree(weight="weight")}
        total = sum(in_degrees.values())

        if total > 0:
            return {n: v / total for n, v in in_degrees.items()}

        return {n: 0.0 for n in self._graph.nodes()}

    def get_pagerank(self, scores_bm25: dict[str, float] | None = None) -> dict[str, float]:
        """
        Get PageRank scores, personalized if BM25 scores are provided.

        Args:
            scores_bm25: Optional BM25 scores for personalization

        Returns:
            Dictionary mapping node_id to PageRank score
        """
        if not self._graph.nodes():
            return {}

        personalization: dict[str, float] = {}

        if scores_bm25:
            top_seeds = sorted(scores_bm25.keys(), key=lambda k: scores_bm25[k], reverse=True)[:30]
            for node_id in top_seeds:
                if node_id in self._graph:
                    personalization[node_id] = personalization.get(node_id, 0) + scores_bm25[node_id]

        if personalization:
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v / total for k, v in personalization.items()}
            else:
                personalization = {}

            if personalization:
                nx = _get_nx()
                try:
                    scores = nx.pagerank(self._graph, personalization=personalization, weight="weight")
                    if not self._scores_are_uniform(scores):
                        return scores
                except Exception:
                    pass

                return self._fallback_pagerank_scores()

        return self._pagerank_cache or {}

    def aggregate_scores_to_classes(
        self,
        scores_pr: dict[str, float],
        scores_bt: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Aggregate method/function scores to their parent classes.

        For PageRank: Use sqrt-normalized sum to prevent runaway scores
        For Betweenness: Use max + sum to boost classes with important methods

        Args:
            scores_pr: PageRank scores
            scores_bt: Betweenness scores

        Returns:
            Tuple of (aggregated_pr, aggregated_bt) dictionaries
        """
        # Build fast lookup: (parent_class_name, file_path) -> class_node_id
        class_lookup: dict[tuple, str] = {}
        for node_id, entity in self._entities.items():
            if entity.entity_type == "class":
                class_lookup[(entity.name, entity.file_path)] = node_id

        # Track aggregated scores per class
        class_pr_contributions: dict[str, list] = defaultdict(list)
        class_bt_contributions: dict[str, list] = defaultdict(list)

        # Collect method scores for their parent classes
        for node_id, entity in self._entities.items():
            if entity.entity_type in ("method", "function") and entity.parent_class:
                parent_id = class_lookup.get((entity.parent_class, entity.file_path))
                if parent_id:
                    if node_id in scores_pr:
                        class_pr_contributions[parent_id].append(scores_pr[node_id])
                    if node_id in scores_bt:
                        class_bt_contributions[parent_id].append(scores_bt[node_id])

        # Create copies for aggregated scores
        agg_pr = dict(scores_pr)
        agg_bt = dict(scores_bt)

        # Aggregate to classes with dampening to balance class vs function visibility
        dampening = Config.MAP_AGGREGATION_DAMPENING
        for class_id, pr_values in class_pr_contributions.items():
            if pr_values:
                agg_pr[class_id] = agg_pr.get(class_id, 0.0) + dampening * math.sqrt(sum(pr_values))

        for class_id, bt_values in class_bt_contributions.items():
            if bt_values:
                agg_bt[class_id] = agg_bt.get(class_id, 0.0) + dampening * (max(bt_values) + sum(bt_values))

        return agg_pr, agg_bt

    def trace_call_path(
        self,
        start_identifier: str,
        end_identifier: str | None = None,
        direction: str = "downstream",
        max_depth: int = 5,
        limit_paths: int = 10,
        min_weight: float = 0.1
    ) -> PathTraceResult:
        """
        Trace execution or dependency paths between symbols in the call graph.

        Args:
            start_identifier: The name of the function/class to start from.
            end_identifier: Optional target symbol. If None, returns all reachable nodes.
            direction: "upstream" (who calls me), "downstream" (what do I call), or "both".
            max_depth: Maximum depth to traverse.
            limit_paths: Maximum number of paths to return.
            min_weight: Minimum edge weight to consider.

        Returns:
            PathTraceResult containing paths and reachable nodes.
        """
        # Find start node(s)
        start_nodes = self._name_to_nodes.get(start_identifier, [])
        if not start_nodes:
            return PathTraceResult(
                source=start_identifier,
                target=end_identifier,
                direction=direction
            )

        start_node = start_nodes[0]

        # Find end node(s) if specified
        end_nodes = []
        if end_identifier:
            end_nodes = self._name_to_nodes.get(end_identifier, [])
            if not end_nodes:
                return PathTraceResult(
                    source=start_identifier,
                    target=end_identifier,
                    direction=direction
                )

        paths: list[list[tuple[str, str, str]]] = []
        reachable = set()
        max_depth_reached = 0

        nx = _get_nx()

        # Filter graph by edge weight
        filtered_graph = self._graph.copy()
        edges_to_remove = [
            (u, v) for u, v, data in filtered_graph.edges(data=True)
            if data.get("weight", 0) < min_weight
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        # Create directional view
        if direction == "upstream":
            graph_view = filtered_graph.reverse()
        elif direction == "both":
            graph_view = filtered_graph.to_undirected()
        else:
            graph_view = filtered_graph

        try:
            if end_identifier and end_nodes:
                end_node = end_nodes[0]
                try:
                    all_paths = nx.all_simple_paths(
                        graph_view, start_node, end_node, cutoff=max_depth
                    )
                    for path in all_paths:
                        if len(paths) >= limit_paths:
                            break
                        path_info = []
                        for node_id in path:
                            entity = self._entities.get(node_id)
                            if entity:
                                path_info.append((node_id, entity.name, entity.entity_type))
                        paths.append(path_info)
                        max_depth_reached = max(max_depth_reached, len(path) - 1)
                        reachable.update(path)
                except nx.NetworkXNoPath:
                    pass
            else:
                if direction == "upstream":
                    # Use descendants on reversed graph to find callers
                    # (ancestors would incorrectly return callees)
                    reachable_nodes = nx.descendants(graph_view, start_node)
                elif direction == "both":
                    visited = set([start_node])
                    queue = [(start_node, 0)]
                    while queue:
                        node, depth = queue.pop(0)
                        if depth < max_depth:
                            for neighbor in graph_view.neighbors(node):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, depth + 1))
                                    max_depth_reached = max(max_depth_reached, depth + 1)
                    reachable_nodes = visited - {start_node}
                else:
                    reachable_nodes = nx.descendants(graph_view, start_node)

                reachable.update(reachable_nodes)

                sample_targets = list(reachable_nodes)[:limit_paths]
                for target_node in sample_targets:
                    try:
                        path = nx.shortest_path(graph_view, start_node, target_node)
                        path_info = []
                        for node_id in path:
                            entity = self._entities.get(node_id)
                            if entity:
                                path_info.append((node_id, entity.name, entity.entity_type))
                        paths.append(path_info)
                        max_depth_reached = max(max_depth_reached, len(path) - 1)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

        except Exception as e:
            logger.error(f"Error tracing paths: {e}")

        # Convert reachable nodes to info tuples
        reachable_info = []
        for node_id in reachable:
            entity = self._entities.get(node_id)
            if entity:
                reachable_info.append((node_id, entity.name, entity.entity_type))

        return PathTraceResult(
            source=start_identifier,
            target=end_identifier,
            direction=direction,
            paths=paths,
            reachable_nodes=reachable_info,
            max_depth_reached=max_depth_reached,
            total_affected=len(reachable),
            paths_found=len(paths),
            requested_max_depth=max_depth,
            requested_path_limit=limit_paths
        )

    def get_edge_info(self, source_id: str, target_id: str) -> dict | None:
        """Get information about an edge between two nodes."""
        if self._graph.has_edge(source_id, target_id):
            return dict(self._graph[source_id][target_id])
        return None

    def get_node_neighbors(
        self,
        node_id: str,
        direction: str = "both"
    ) -> list[tuple[str, float, set]]:
        """
        Get neighbors of a node with edge info.

        Args:
            node_id: The node to get neighbors for
            direction: "in" for predecessors, "out" for successors, "both" for all

        Returns:
            List of (neighbor_id, weight, types) tuples
        """
        result: list[tuple[str, float, set[str]]] = []

        if node_id not in self._graph:
            return result

        if direction in ("in", "both"):
            for pred in self._graph.predecessors(node_id):
                edge = self._graph[pred][node_id]
                result.append((pred, edge.get("weight", 0), edge.get("types", set())))

        if direction in ("out", "both"):
            for succ in self._graph.successors(node_id):
                edge = self._graph[node_id][succ]
                result.append((succ, edge.get("weight", 0), edge.get("types", set())))

        return result
