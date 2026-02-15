"""
Request handlers for the daemon server.

Implements the Strategy pattern for handling different request types.
Each handler is a callable that processes request parameters and returns a result.
"""
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..cache import CachedIndices
from ..config import OutputFormat
from ..constants import DEFAULT_SEARCH_RESULT_LIMIT
from ..formatters import get_formatter
from ..formatters.base import OutputFormatter
from ..graph_utils import (
    _TOKEN_OVERHEAD_BOTTLENECKS,
    _TOKEN_OVERHEAD_CHANGE_IMPACT,
    _TOKEN_PER_BOTTLENECK,
    _TOKEN_PER_CALLER,
    AUTO_MIN_IMPORTANCE,
    apply_token_budget_filter,
    build_bottleneck_info,
    build_caller_info,
    compute_coupling_hotspots,
    detect_high_connectivity,
    extract_module_from_path,
    find_symbol_nodes,
    get_connected_modules,
)
from ..search import SearchEngine
from ..token_estimator import count_tokens
from .protocol import (
    CloneDetectionParams,
    DeadCodeParams,
    FlagClearParams,
    FlagParams,
    PropagationCostParams,
    GraphAnalysisParams,
    SearchParams,
    SensitiveValueParams,
    StacktraceParams,
    TraceDependencyParams,
    TrampDataParams,
)
if TYPE_CHECKING:
    from .server import DaemonServer

logger = logging.getLogger(__name__)


class ProjectLoader(Protocol):
    """Protocol for loading projects from cache or disk."""

    def get_or_load_project(
        self,
        source_dir: str,
        enable_semantic: bool = False,
        model_path: str | None = None,
    ) -> tuple[CachedIndices, SearchEngine]:
        """Get project from cache or load it."""
        ...


class BaseHandler(ABC):
    """Abstract base class for request handlers.

    Provides common functionality and defines the interface for all handlers.
    """

    def __init__(self, server: "DaemonServer"):
        """Initialize handler with server reference.

        Args:
            server: The daemon server instance for accessing shared state.
        """
        self._server = server

    @abstractmethod
    def handle(self, params: dict) -> dict:
        """Handle a request and return the result.

        Args:
            params: Request parameters dictionary.

        Returns:
            Result dictionary.

        Raises:
            ValueError: If required parameters are missing.
            Exception: If processing fails.
        """
        pass


class PingHandler(BaseHandler):
    """Handler for ping requests."""

    def handle(self, params: dict) -> dict:
        return {"pong": True, "timestamp": time.time()}


class StatusHandler(BaseHandler):
    """Handler for status requests."""

    def handle(self, params: dict) -> dict:
        server = self._server
        current_time = time.time()
        with server._activity_lock:
            idle_seconds = current_time - server._last_activity
        return {
            "running": True,
            "host": server.host,
            "port": server.port,
            "uptime_seconds": current_time - server._start_time,
            "idle_seconds": idle_seconds,
            "cache": server._project_cache.status(),
        }


class InvalidateHandler(BaseHandler):
    """Handler for cache invalidation requests."""

    def handle(self, params: dict) -> dict:
        source_dir = params.get("source_dir")
        invalidate_all = params.get("all", False)

        if invalidate_all:
            self._server._project_cache.invalidate()
            return {"invalidated": "all"}
        elif source_dir:
            source_path = str(Path(source_dir).resolve())
            self._server._project_cache.invalidate(source_path)
            return {"invalidated": source_path}
        else:
            return {"invalidated": None, "message": "No source_dir or --all specified"}


class ShutdownHandler(BaseHandler):
    """Handler for shutdown requests."""

    def handle(self, params: dict) -> dict:
        logger.info("Shutdown requested via RPC")
        self._server.stop()
        return {"shutdown": True}


class SearchHandlerBase(BaseHandler):
    """Base class for search-related handlers with common functionality."""

    def _apply_token_budget(
        self, results: list, search_params: SearchParams
    ) -> tuple[list, int]:
        """Apply token budget to results and return included results with token count."""
        included_results = OutputFormatter.filter_by_token_budget(
            results,
            search_params.tokens,
            search_params.show_deps,
        )

        used_tokens = 100  # Base overhead
        for result in included_results:
            code = result.entity.get_context_snippet()
            used_tokens += count_tokens(code) + 30
            if (
                search_params.show_deps
                and result.dependency_context
                and not result.dependency_context.is_empty()
            ):
                used_tokens += count_tokens(result.dependency_context.format_compact())

        return included_results, used_tokens

    def _format_output(
        self,
        results: list,
        engine: SearchEngine,
        search_params: SearchParams,
    ) -> str:
        """Format search results for output."""
        root_path = Path(search_params.source_dir).resolve()
        output_format = OutputFormat(search_params.output_format)
        formatter = get_formatter(output_format)

        output_parts = []

        formatted_output = formatter.format_results(
            results,
            root_path,
            search_params.tokens,
            search_params.show_deps,
        )

        tree_output = None
        if search_params.dir_tree:
            tree_output = engine.generate_directory_tree(results)

        if search_params.reverse:
            # Reversed mode: results first, then tree
            output_parts.append(formatted_output)
            if tree_output:
                output_parts.append("\n" + "=" * 60 + "\n")
                output_parts.append(tree_output)
        else:
            # Normal mode: tree first, then results
            if tree_output:
                output_parts.append(tree_output)
                output_parts.append("\n" + "=" * 60 + "\n")
            output_parts.append(formatted_output)

        return "\n".join(output_parts)

    def _execute_search_operation(
        self,
        params: dict,
        search_func,
    ) -> dict:
        """Common execution logic for search and find operations.

        Args:
            params: Raw request parameters.
            search_func: Function that performs the actual search.

        Returns:
            Result dictionary with output, counts, timing, etc.
        """
        search_params = SearchParams.from_dict(params)

        indices, engine = self._server._get_or_load_project(
            search_params.source_dir,
            search_params.enable_semantic,
            search_params.model_path,
        )

        start_time = time.time()
        results = search_func(search_params, engine)
        included_results, used_tokens = self._apply_token_budget(results, search_params)

        # Get stats BEFORE reversing (stats always show normal order 1, 2, 3...)
        stats_output = None
        if search_params.stats:
            stats_output = engine.format_stats(
                included_results, limit=len(included_results)
            )

        # Reverse results if requested (highest score last) - only affects display
        display_results = included_results
        if search_params.reverse:
            display_results = list(reversed(included_results))

        search_time_ms = (time.time() - start_time) * 1000
        output = self._format_output(display_results, engine, search_params)

        result = {
            "output": output,
            "result_count": len(included_results),
            "total_matched": len(results),
            "search_time_ms": round(search_time_ms, 2),
            "tokens_used": used_tokens,
        }

        # Add warning when token budget filtered out all results
        if len(results) > 0 and len(included_results) == 0:
            result["warning"] = (
                f"All {len(results)} matching results were filtered out due to token_limit={search_params.tokens}. "
                f"Increase token_limit (e.g., 4000 or higher) to get meaningful results."
            )

        if stats_output:
            result["stats_output"] = stats_output

        return result


class SearchHandler(SearchHandlerBase):
    """Handler for search requests."""

    def handle(self, params: dict) -> dict:
        def do_search(search_params: SearchParams, engine: SearchEngine) -> list:
            limit = search_params.limit if search_params.limit is not None else DEFAULT_SEARCH_RESULT_LIMIT
            if search_params.map_mode or not search_params.query:
                return engine.search(
                    query="",
                    use_architecture=True,
                    include_deps=search_params.show_deps,
                    limit=limit,
                )
            else:
                return engine.search(
                    query=search_params.query,
                    include_deps=search_params.show_deps,
                    limit=limit,
                )

        return self._execute_search_operation(params, do_search)


class FindHandler(SearchHandlerBase):
    """Handler for find identifier requests."""

    def handle(self, params: dict) -> dict:
        search_params = SearchParams.from_dict(params)
        if not search_params.find_identifier:
            raise ValueError("find_identifier is required")

        identifier = search_params.find_identifier

        def do_find(search_params: SearchParams, engine: SearchEngine) -> list:
            limit = search_params.limit if search_params.limit is not None else DEFAULT_SEARCH_RESULT_LIMIT
            return engine.find_identifiers(
                identifier,
                limit=limit,
                include_deps=search_params.show_deps,
            )

        return self._execute_search_operation(params, do_find)


class ArchitecturalBottlenecksHandler(BaseHandler):
    """Handler for architectural_bottlenecks requests."""

    def handle(self, params: dict) -> dict:
        graph_params = GraphAnalysisParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(graph_params.source_dir)

        graph = indices.graph
        entities = indices.entities
        betweenness = indices.betweenness
        pagerank = indices.pagerank

        root_path = Path(graph_params.source_dir).resolve()
        bottlenecks = []

        for node_id, bt_score in betweenness.items():
            if bt_score < graph_params.min_betweenness:
                continue

            entity = entities.get(node_id)
            if not entity:
                continue

            if graph_params.exclude_tests and entity.is_test:
                continue

            if node_id not in graph:
                continue

            fan_in = graph.in_degree(node_id)
            fan_out = graph.out_degree(node_id)
            connected_modules = get_connected_modules(
                node_id, graph, entities, root_path
            )

            bottlenecks.append(
                build_bottleneck_info(
                    entity,
                    bt_score,
                    fan_in,
                    fan_out,
                    pagerank.get(node_id, 0.0),
                    connected_modules,
                )
            )

        bottlenecks.sort(key=lambda x: x["betweenness"], reverse=True)
        bottlenecks = bottlenecks[: graph_params.limit]

        # Apply token budget using shared function
        filtered_bottlenecks, token_budget_exceeded = apply_token_budget_filter(
            bottlenecks,
            graph_params.token_limit,
            _TOKEN_OVERHEAD_BOTTLENECKS,
            _TOKEN_PER_BOTTLENECK,
            ["name", "file", "type", "connected_modules"],
        )

        all_bt = list(betweenness.values())
        avg_bt = sum(all_bt) / len(all_bt) if all_bt else 0
        max_bt = max(all_bt) if all_bt else 0

        return {
            "bottlenecks": filtered_bottlenecks,
            "summary": {
                "total_functions_analyzed": len(betweenness),
                "functions_above_threshold": len(
                    [b for b in all_bt if b >= graph_params.min_betweenness]
                ),
                "average_betweenness": round(avg_bt, 6),
                "max_betweenness": round(max_bt, 6),
                "token_budget_exceeded": token_budget_exceeded,
            },
            "source": "daemon",
        }


class CouplingHotspotsHandler(BaseHandler):
    """Handler for coupling_hotspots requests."""

    def handle(self, params: dict) -> dict:
        graph_params = GraphAnalysisParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(graph_params.source_dir)

        result = compute_coupling_hotspots(
            graph=indices.graph,
            entities=indices.entities,
            pagerank=indices.pagerank,
            limit=graph_params.limit,
            min_coupling_score=graph_params.min_coupling_score,
            exclude_tests=graph_params.exclude_tests,
            exclude_private=graph_params.exclude_private,
            token_limit=graph_params.token_limit,
        )
        result["source"] = "daemon"
        return result


class ChangeImpactRadiusHandler(BaseHandler):
    """Handler for change_impact_radius requests."""

    def handle(self, params: dict) -> dict:
        graph_params = GraphAnalysisParams.from_dict(params)

        if not graph_params.symbol_name:
            return {"error": "symbol_name is required"}

        indices, _ = self._server._get_or_load_project(graph_params.source_dir)

        graph = indices.graph
        entities = indices.entities
        pagerank = indices.pagerank
        name_to_nodes = indices.name_to_nodes

        # Use shared helper for symbol lookup
        symbol_name = graph_params.symbol_name.strip()
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

        # Token budget tracking using shared constants
        used_tokens = _TOKEN_OVERHEAD_CHANGE_IMPACT
        token_budget_exceeded = False

        min_importance = graph_params.min_importance

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
            if min_importance < AUTO_MIN_IMPORTANCE:
                min_importance = AUTO_MIN_IMPORTANCE

        # BFS to find all upstream callers
        callers_by_depth: dict[int, list[dict]] = {}
        visited = {target_node}
        current_level = {target_node}
        all_affected_files: set[str] = set()
        root_callers: list[dict] = []

        root_path = Path(graph_params.source_dir).resolve()

        for depth in range(1, graph_params.max_depth + 1):
            next_level: set[str] = set()
            depth_callers: list[dict] = []

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

                                if graph.in_degree(caller_node) == 0:
                                    root_callers.append(caller_info)

            if depth_callers:
                depth_callers.sort(key=lambda x: x["importance"], reverse=True)

                filtered_callers = []
                for caller in depth_callers:
                    text = f"{caller['name']} {caller['file']} {caller['type']}"
                    tokens = count_tokens(text, is_code=False) + _TOKEN_PER_CALLER
                    if used_tokens + tokens > graph_params.token_limit:
                        token_budget_exceeded = True
                        break
                    used_tokens += tokens
                    filtered_callers.append(caller)

                callers_by_depth[depth] = filtered_callers

            if not next_level or token_budget_exceeded:
                break
            current_level = next_level

        # Calculate affected modules
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
                "affected_modules": sorted(affected_modules),
                "max_depth_reached": max(callers_by_depth.keys())
                if callers_by_depth
                else 0,
                "token_budget_exceeded": token_budget_exceeded,
            },
            "root_callers": root_callers[:10],
            "callers_by_depth": callers_by_depth,
            "source": "daemon",
        }

        if high_connectivity_warning:
            result["high_connectivity_warning"] = high_connectivity_warning

        return result


class TraceDependencyHandler(BaseHandler):
    """Handler for trace_dependency_path requests."""

    def handle(self, params: dict) -> dict:
        trace_params = TraceDependencyParams.from_dict(params)
        indices, engine = self._server._get_or_load_project(trace_params.source_dir)

        result = engine.trace_call_path(
            start_identifier=trace_params.start_identifier,
            end_identifier=trace_params.end_identifier,
            direction=trace_params.direction,
            max_depth=trace_params.max_depth,
            limit_paths=trace_params.limit_paths,
        )

        formatted = result.format_enhanced_output(root_directory=trace_params.source_dir)
        formatted["source"] = "daemon"
        return formatted


class DebugStacktraceHandler(BaseHandler):
    """Handler for debug_stacktrace requests."""

    def handle(self, params: dict) -> dict:
        from ..mcp.stacktrace import StacktraceParser

        st_params = StacktraceParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(st_params.source_dir)

        parser = StacktraceParser()
        frames = parser.parse(st_params.stacktrace)

        if not frames:
            return {
                "error": "Could not parse stacktrace",
                "suggestion": (
                    "Please provide a raw stacktrace from Python, "
                    "JavaScript, Go, Rust, Java, or Ruby"
                ),
            }

        # Get indexed file paths
        local_files = set(
            indices.entities[nid].file_path for nid in indices.entities
        )

        # Resolve frames
        resolved_frames = []
        for frame in frames:
            resolved_path = parser.resolve_path(frame.file_path, local_files)
            frame.resolved_path = resolved_path
            frame.is_external = resolved_path is None

            frame_dict: dict = {
                "file": frame.file_path,
                "line": frame.line_number,
                "function": frame.function_name,
                "resolved_path": resolved_path,
                "is_external": frame.is_external,
            }

            if resolved_path and st_params.show_dependencies:
                # Find entities at this line
                file_entities = indices.file_to_entities.get(resolved_path, [])
                for entity_id in file_entities:
                    entity = indices.entities.get(entity_id)
                    if (
                        entity
                        and entity.line_start <= frame.line_number <= entity.line_end
                    ):
                        # Get context snippet
                        frame_dict["context"] = entity.get_context_snippet(
                            max_lines=st_params.context_lines
                        )
                        frame_dict["entity_name"] = entity.name
                        frame_dict["entity_type"] = entity.entity_type
                        break

            resolved_frames.append(frame_dict)

        return {
            "frames": resolved_frames,
            "total_frames": len(frames),
            "local_frames": len([f for f in resolved_frames if not f["is_external"]]),
            "external_frames": len([f for f in resolved_frames if f["is_external"]]),
            "source": "daemon",
        }


class CloneDetectionHandler(BaseHandler):
    """Handler for clone detection requests.

    Supports three modes:
    - combined (default): Both semantic + syntactic analysis
    - semantic: Model2Vec embedding similarity only
    - syntactic: Line-by-line Jaccard similarity only
    """

    def handle(self, params: dict) -> dict:
        from ..clone import (
            detect_clones_combined,
            detect_clones_semantic,
            detect_clones_syntactic,
        )
        from ..config_loader import get_semantic_model_path

        clone_params = CloneDetectionParams.from_dict(params)
        mode = clone_params.mode

        # Load project with semantic support if needed
        enable_semantic = mode in ("combined", "semantic")
        indices, _ = self._server._get_or_load_project(
            clone_params.source_dir,
            enable_semantic=enable_semantic,
        )

        model_path = get_semantic_model_path()

        if mode == "syntactic":
            result = detect_clones_syntactic(
                entities=indices.entities,
                line_threshold=clone_params.line_threshold,
                func_threshold=clone_params.func_threshold,
                min_shared_lines=clone_params.min_shared_lines,
                limit=clone_params.limit,
                exclude_tests=clone_params.exclude_tests,
                min_lines=clone_params.min_lines,
                token_limit=clone_params.token_limit,
            )
        elif mode == "semantic":
            result = detect_clones_semantic(
                entities=indices.entities,
                model_path=model_path,
                threshold=clone_params.similarity_threshold,
                limit=clone_params.limit,
                exclude_tests=clone_params.exclude_tests,
                min_lines=clone_params.min_lines,
                token_limit=clone_params.token_limit,
            )
        else:  # combined (default)
            result = detect_clones_combined(
                entities=indices.entities,
                model_path=model_path,
                semantic_threshold=clone_params.similarity_threshold,
                line_threshold=clone_params.line_threshold,
                func_threshold=clone_params.func_threshold,
                min_shared_lines=clone_params.min_shared_lines,
                limit=clone_params.limit,
                exclude_tests=clone_params.exclude_tests,
                min_lines=clone_params.min_lines,
                token_limit=clone_params.token_limit,
                semantic_weight=clone_params.semantic_weight,
                syntactic_weight=clone_params.syntactic_weight,
            )

        result["source"] = "daemon"
        return result


class PropagationCostHandler(BaseHandler):
    """Handler for propagation cost requests."""

    def handle(self, params: dict) -> dict:
        from ..mcp.propagation_cost import compute_propagation_cost

        pc_params = PropagationCostParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(pc_params.source_dir)

        result = compute_propagation_cost(
            graph=indices.graph,
            entities=indices.entities,
            include_breakdown=pc_params.include_breakdown,
            show_critical_paths=pc_params.show_critical_paths,
            exclude_tests=pc_params.exclude_tests,
            token_limit=pc_params.token_limit,
        )
        result["source"] = "daemon"
        return result


class DeadCodeHandler(BaseHandler):
    """Handler for dead code detection requests."""

    def handle(self, params: dict) -> dict:
        from ..dead_code.detector import detect_unused_functions

        dc_params = DeadCodeParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(dc_params.source_dir)

        result = detect_unused_functions(
            entities=indices.entities,
            graph=indices.graph,
            confidence_threshold=dc_params.confidence_threshold,
            exclude_tests=dc_params.exclude_tests,
            include_private=dc_params.include_private,
            min_lines=dc_params.min_lines,
            limit=dc_params.limit,
        )
        result["source"] = "daemon"
        return result


class TrampDataHandler(BaseHandler):
    """Handler for tramp data detection requests."""

    def handle(self, params: dict) -> dict:
        from ..tramp_data.detector import detect_tramp_data

        td_params = TrampDataParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(td_params.source_dir)

        result = detect_tramp_data(
            entities=indices.entities,
            min_occurrences=td_params.min_occurrences,
            exclude_tests=td_params.exclude_tests,
            limit=td_params.limit,
            min_group_size=td_params.min_group_size,
        )
        result["source"] = "daemon"
        return result


class SensitiveValueHandler(BaseHandler):
    """Handler for sensitive value detection requests."""

    def handle(self, params: dict) -> dict:
        from ..sensitive_values.detector import detect_sensitive_values

        sv_params = SensitiveValueParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(sv_params.source_dir)

        result = detect_sensitive_values(
            entities=indices.entities,
            confidence_threshold=sv_params.confidence_threshold,
            exclude_tests=sv_params.exclude_tests,
            limit=sv_params.limit,
            replace_value=sv_params.replace_value,
            whitelist=sv_params.whitelist,
            root_dir=sv_params.source_dir,
        )
        result["source"] = "daemon"
        return result


class FlagHandler(BaseHandler):
    """Handler for code flagging requests."""

    def handle(self, params: dict) -> dict:
        from ..mcp.flag_insertion import flag_code

        flag_params = FlagParams.from_dict(params)
        indices, _ = self._server._get_or_load_project(flag_params.source_dir)

        result = flag_code(
            entities=indices.entities,
            graph=indices.graph,
            pagerank=indices.pagerank,
            params=flag_params,
        )
        result["source"] = "daemon"
        return result


class FlagClearHandler(BaseHandler):
    """Handler for clearing [CODEN] flags from code."""

    def handle(self, params: dict) -> dict:
        from ..mcp.flag_insertion import flag_clear

        clear_params = FlagClearParams.from_dict(params)

        result = flag_clear(
            source_dir=clear_params.source_dir,
            dry_run=clear_params.dry_run,
            verbose=clear_params.verbose,
        )
        result["source"] = "daemon"
        return result


def create_handler_registry(server: "DaemonServer") -> dict[str, BaseHandler]:
    """Create the handler registry mapping method names to handler instances.

    Args:
        server: The daemon server instance.

    Returns:
        Dictionary mapping method names to handler instances.
    """
    return {
        "ping": PingHandler(server),
        "status": StatusHandler(server),
        "invalidate": InvalidateHandler(server),
        "shutdown": ShutdownHandler(server),
        "search": SearchHandler(server),
        "find": FindHandler(server),
        "architectural_bottlenecks": ArchitecturalBottlenecksHandler(server),
        "coupling_hotspots": CouplingHotspotsHandler(server),
        "change_impact_radius": ChangeImpactRadiusHandler(server),
        "trace_dependency": TraceDependencyHandler(server),
        "debug_stacktrace": DebugStacktraceHandler(server),
        "detect_clones": CloneDetectionHandler(server),
        "propagation_cost": PropagationCostHandler(server),
        "detect_dead_code": DeadCodeHandler(server),
        "detect_tramp_data": TrampDataHandler(server),
        "detect_sensitive_values": SensitiveValueHandler(server),
        "flag_code": FlagHandler(server),
        "flag_clear": FlagClearHandler(server),
    }
