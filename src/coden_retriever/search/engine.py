"""Search engine module.

Main orchestrator for code search, combining multiple ranking signals.
"""
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from ..cache.layout import LITE_LAYOUT, STATIC_LAYOUT, CacheLayout
from ..config import Config, MapMode
from ..config_loader import get_semantic_model_path
from ..constants import (
    MILLISECONDS_PER_SECOND,
    SEMANTIC_INDEX_PROGRESS_LABEL,
    SIMPLE_MAP_LINE_TIEBREAK_DIVISOR,
)
from ..git.process_metrics import (
    harvest_change_count,
    harvest_line_blame_commits,
    history_is_locally_available,
    to_repo_relative_posix,
)
from ..semantic_config import SemanticConfig

if TYPE_CHECKING:
    import networkx as nx
    from ..cache import CachedIndices, LiteCachedIndices
    from ..cache.manager import CacheManager
    from .graph_analyzer import GraphAnalyzer
    from .semantic import SemanticIndex

from ..formatters.directory_tree_formatter import DirectoryTreeFormatter
from ..formatters.terminal_style import get_terminal_style
from ..graph_utils import calculate_dispatcher_score
from ..models import (
    CentralityCache,
    CodeEntity,
    DependencyContext,
    IndexStats,
    PathTraceResult,
    RankingSignals,
    SearchResult,
)
from ..parsers import RepoParser
from ..utils.progress import encoding_progress
from ..utils.source_walker import iter_source_files
from .bm25 import BM25Index
from .signals import SIGNALS, Mode, Signal, signals_for_mode, sqrt_sum

logger = logging.getLogger(__name__)


def _create_digraph() -> "nx.DiGraph":
    """Create a new DiGraph with lazy import."""
    import networkx as nx
    return nx.DiGraph()


class SearchEngine:
    """
    Hybrid code search engine combining lexical and structural signals.
    Optimized for LLM context generation.
    """

    @staticmethod
    def _init_default_state(engine: "SearchEngine") -> None:
        """Set every internally-managed attribute to its default.

        Called by every constructor — `__init__` plus the three classmethod
        builders — so the default-attribute list lives in one place.
        Constructors override specific fields after this call (e.g. swap in
        a cached graph, populate entities from a parse pass).

        The four caller-supplied attrs (`root`, `verbose`, `enable_semantic`,
        `model_path`) stay caller-managed because their values vary per
        constructor.
        """
        engine._graph = _create_digraph()
        engine._entities = {}
        engine._bm25 = BM25Index()
        engine._parser = RepoParser()
        engine._semantic_index = None
        engine._name_to_nodes = defaultdict(list)
        engine._file_scopes = defaultdict(list)
        engine._file_to_entities = defaultdict(list)
        engine._stats = IndexStats()
        engine._indexed = False
        engine._centrality = CentralityCache()
        engine._graph_analyzer = None
        engine._entry_scores = None
        # Set only by from_lite_cache: pre-harvested change_count to consume
        # in _simple_map_search instead of re-running git log every call.
        engine._lite_change_count = None

    def __init__(
        self,
        root: str,
        verbose: bool = False,
        semantic: SemanticConfig = SemanticConfig(),
    ):
        self.root = Path(root).resolve()
        self.verbose = verbose
        self.enable_semantic = semantic.enabled
        self.model_path = semantic.model_path
        self._init_default_state(self)

        # Lazy load semantic index only if enabled (saves memory/startup time).
        # Runs after _init_default_state so it can flip _semantic_index from None.
        if self.enable_semantic:
            self._init_semantic_index()

    @classmethod
    def from_cached_indices(cls, cached: "CachedIndices", verbose: bool = False) -> "SearchEngine":
        """Create a SearchEngine from cached indices."""
        engine = cls.__new__(cls)
        engine.root = cached.source_dir
        engine.verbose = verbose
        engine.enable_semantic = cached.has_semantic
        engine.model_path = cached.model_path
        cls._init_default_state(engine)

        engine._graph = cached.graph
        engine._entities = cached.entities
        engine._bm25 = cached.bm25_index

        if cached.has_semantic and cached.embeddings is not None:
            try:
                from .semantic import SemanticIndex
                # Restore semantic index from cached embeddings.
                # The ONNX model loads lazily on first query via onnx_encode().
                engine._semantic_index = SemanticIndex(cached.model_path)
                engine._semantic_index._embeddings = cached.embeddings
                engine._semantic_index._node_ids = cached.node_ids
            except ImportError:
                engine.enable_semantic = False

        engine._name_to_nodes = defaultdict(list, cached.lookups.name_to_nodes)
        engine._file_scopes = defaultdict(list, cached.lookups.file_scopes)
        engine._file_to_entities = defaultdict(list, cached.lookups.file_to_entities)
        engine._centrality = cached.centrality

        from .graph_analyzer import GraphAnalyzer
        engine._graph_analyzer = GraphAnalyzer(
            entities=engine._entities,
            name_to_nodes=engine._name_to_nodes,
            file_scopes=engine._file_scopes,
            verbose=verbose,
        )
        engine._graph_analyzer._graph = cached.graph
        engine._graph_analyzer._centrality = engine._centrality

        engine._stats.total_entities = len(cached.entities)
        engine._stats.total_edges = cached.graph.number_of_edges()
        engine._stats.total_files = cached.manifest.get("file_count", 0)
        engine._stats.parsed_files = cached.manifest.get("file_count", 0)
        engine._indexed = True
        return engine

    def _init_semantic_index(self) -> None:
        """Initialize semantic search index with graceful fallback."""
        try:
            from .semantic import SemanticIndex

            model_path = self.model_path or get_semantic_model_path()

            if not Path(model_path).exists():
                logger.warning(
                    "Semantic model not found at %s. "
                    "Falling back to BM25-only search. "
                    "Configure via `coden config set search.semantic_model_path <path>` "
                    "or the CODEN_RETRIEVER_MODEL_PATH env var.",
                    model_path,
                )
                self.enable_semantic = False
                return

            self.model_path = model_path
            self._semantic_index = SemanticIndex(model_path)
            logger.info(f"Semantic search enabled with model at: {model_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize semantic search: {e}. Falling back to BM25-only.")
            self.enable_semantic = False

    @classmethod
    def lite_from_root(cls, root: str | Path, verbose: bool = False) -> "SearchEngine":
        """Construct a parsing-only engine for `--map-mode simple`.

        Skips call graph, BM25, semantic, and centrality construction — none
        of which `_simple_map_search` consults. Cuts cold-start time on a
        ~1k-file repo from minutes (full centrality) to seconds (parse only).
        """
        engine = cls.__new__(cls)
        engine.root = Path(root).resolve()
        engine.verbose = verbose
        engine.enable_semantic = False
        engine.model_path = None
        cls._init_default_state(engine)

        # Parse files; references are discarded (no graph in lite mode).
        documents: dict[str, str] = {}
        discarded_refs: list[tuple[str, int, str, str, str | None]] = []
        files = list(engine._collect_files())
        engine._stats.total_files = len(files)
        for file_path in files:
            if engine._parse_file(file_path, documents, discarded_refs):
                engine._stats.parsed_files += 1
            else:
                engine._stats.failed_files += 1

        engine._stats.total_entities = len(engine._entities)
        engine._indexed = True
        return engine

    @classmethod
    def from_lite_cache(
        cls, cached: "LiteCachedIndices", verbose: bool = False
    ) -> "SearchEngine":
        """Construct a `--map-mode simple` engine from a warm lite cache.

        Skips parsing entirely — entities come straight from the pickle.
        Pre-harvested `change_count` is stashed on the engine so
        `_simple_map_search` reuses it instead of re-running `git log`.
        Per-file blame stays lazy (and uncached on disk) — the cost is
        bounded by `SIMPLE_MAP_BLAME_TIMEOUT_SECONDS` per file and only
        paid for top-ranked files.
        """
        engine = cls.__new__(cls)
        engine.root = Path(cached.source_dir).resolve()
        engine.verbose = verbose
        engine.enable_semantic = False
        engine.model_path = None
        cls._init_default_state(engine)

        engine._entities = dict(cached.entities)
        for nid, entity in engine._entities.items():
            engine._name_to_nodes[entity.name].append(nid)
            engine._file_scopes[entity.file_path].append(
                (entity.line_start, entity.line_end, nid)
            )
            engine._file_to_entities[entity.file_path].append(nid)
        for scopes in engine._file_scopes.values():
            scopes.sort(key=lambda x: x[1] - x[0])

        engine._stats.total_files = cached.manifest.get("file_count", 0)
        engine._stats.parsed_files = cached.manifest.get("file_count", 0)
        engine._stats.total_entities = len(engine._entities)
        engine._lite_change_count = cached.change_count
        engine._indexed = True
        return engine

    def index(self) -> IndexStats:
        """Index the repository."""
        # Reset all state at the beginning of each index operation
        self._stats = IndexStats()
        self._entities = {}
        self._graph = _create_digraph()
        self._name_to_nodes = defaultdict(list)
        self._file_scopes = defaultdict(list)
        self._file_to_entities = defaultdict(list)
        self._centrality = CentralityCache()
        self._graph_analyzer = None
        self._entry_scores = None

        start_time = time.time()
        logger.info(f"Indexing repository: {self.root}")

        documents: dict[str, str] = {}
        all_references: list[tuple[str, int, str, str, str | None]] = []

        files = self._collect_files()
        self._stats.total_files = len(files)

        logger.info(f"Parsing {len(files)} source files...")
        for file_path in files:
            success = self._parse_file(file_path, documents, all_references)
            if success:
                self._stats.parsed_files += 1
            else:
                self._stats.failed_files += 1

        # Create graph analyzer and build graph
        from .graph_analyzer import GraphAnalyzer
        self._graph_analyzer = GraphAnalyzer(
            entities=self._entities,
            name_to_nodes=self._name_to_nodes,
            file_scopes=self._file_scopes,
            verbose=self.verbose,
        )

        logger.info(f"Building call graph from {len(all_references)} references...")
        self._graph_analyzer.build_graph(all_references)
        self._graph = self._graph_analyzer.graph  # Keep reference for backwards compatibility

        logger.info("Building BM25 index...")
        self._bm25.index(documents)

        # Build semantic index if enabled
        if self.enable_semantic and self._semantic_index:
            logger.info("Building semantic index...")
            try:
                with encoding_progress(SEMANTIC_INDEX_PROGRESS_LABEL, len(self._entities)) as advance:
                    self._semantic_index.index(self._entities, on_batch_done=advance)
            except Exception as e:
                logger.warning(f"Semantic indexing failed: {e}. Continuing with BM25-only.")
                self.enable_semantic = False

        logger.info("Computing centrality metrics...")
        self._graph_analyzer.compute_centrality()
        self._centrality = self._graph_analyzer.centrality

        self._stats.total_entities = len(self._entities)
        self._stats.total_edges = self._graph.number_of_edges()
        self._stats.index_time_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND

        self._indexed = True

        logger.info(f"Indexing complete: {self._stats.total_entities} entities, "
                    f"{self._stats.total_edges} edges in {self._stats.index_time_ms:.0f}ms")

        return self._stats

    def _collect_files(self) -> list[Path]:
        """Collect all source files to index."""
        return [path for path, _ in iter_source_files(self.root)]

    def _parse_file(
        self,
        file_path: Path,
        documents: dict[str, str],
        all_refs: list[tuple[str, int, str, str, str | None]]
    ) -> bool:
        """Parse a single file.
        
        Returns:
            True if file was successfully parsed (even if no entities found),
            False only if an actual parsing error occurred.
        """
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="ignore")
            entities, references, _ = self._parser.parse_file(str(file_path), source_code)

            # Process entities if any were found
            # Note: Empty entity list is valid (e.g., __init__.py with only imports)
            for entity in entities:
                node_id = entity.node_id
                self._entities[node_id] = entity
                self._graph.add_node(node_id)
                documents[node_id] = entity.searchable_text

                self._name_to_nodes[entity.name].append(node_id)
                self._file_scopes[str(file_path)].append(
                    (entity.line_start, entity.line_end, node_id)
                )
                self._file_to_entities[str(file_path)].append(node_id)

                self._stats.entities_by_type[entity.entity_type] = \
                    self._stats.entities_by_type.get(entity.entity_type, 0) + 1
                self._stats.entities_by_language[entity.language] = \
                    self._stats.entities_by_language.get(entity.language, 0) + 1

            if entities:  # Only sort if we have entities
                self._file_scopes[str(file_path)].sort(key=lambda x: x[1] - x[0])

            # Process references (parser returns 4-tuples: line, name, ref_type, receiver)
            for line, name, ref_type, receiver in references:
                all_refs.append((str(file_path), line, name, ref_type, receiver))

            # File was successfully parsed (even if no entities/references found)
            return True

        except Exception as e:
            if self.verbose:
                logger.debug(f"Failed to parse {file_path}: {e}")
            return False

    def get_dependency_context(
        self,
        node_id: str,
        max_callers: int = Config.DEPENDENCY_MAX_CALLERS,
        max_callees: int = Config.DEPENDENCY_MAX_CALLEES,
        min_weight: float = Config.DEPENDENCY_MIN_WEIGHT
    ) -> DependencyContext:
        """Extract dependency context for an entity."""
        ctx = DependencyContext()

        if node_id not in self._graph:
            return ctx

        # Get callers (predecessors) sorted by edge weight
        callers = []
        for pred in self._graph.predecessors(node_id):
            edge_data = self._graph[pred][node_id]
            weight = edge_data.get("weight", 0)
            if weight >= min_weight:
                entity = self._entities.get(pred)
                if entity and not entity.is_utility:
                    callers.append((pred, entity.name, entity.entity_type, weight))
        callers.sort(key=lambda x: x[3], reverse=True)
        ctx.callers = callers[:max_callers]

        # Get callees (successors) sorted by edge weight
        callees = []
        for succ in self._graph.successors(node_id):
            edge_data = self._graph[node_id][succ]
            weight = edge_data.get("weight", 0)
            if weight >= min_weight:
                entity = self._entities.get(succ)
                if entity and not entity.is_utility:
                    callees.append((succ, entity.name, entity.entity_type, weight))
        callees.sort(key=lambda x: x[3], reverse=True)
        ctx.callees = callees[:max_callees]

        return ctx

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

        Useful for understanding how code flows through the system, identifying
        impact of changes, and discovering hidden dependencies.

        Args:
            start_identifier: The name of the function/class to start from.
            end_identifier: Optional target symbol. If None, returns all reachable nodes.
            direction: "upstream" (who calls me), "downstream" (what do I call), or "both".
            max_depth: Maximum depth to traverse (prevents infinite loops).
            limit_paths: Maximum number of paths to return.
            min_weight: Minimum edge weight to consider (filters weak references).

        Returns:
            PathTraceResult containing paths and reachable nodes.

        Example:
            >>> engine = SearchEngine("/path/to/repo")
            >>> engine.index()
            >>>
            >>> # Find what functions call "validate_user"
            >>> result = engine.trace_call_path("validate_user", direction="upstream")
            >>> print(f"Found {result.total_affected} callers")
            Found 12 callers
            >>> for path in result.paths[:3]:
            ...     print(" -> ".join(name for _, name, _ in path))
            handle_login -> authenticate -> validate_user
            api_handler -> check_auth -> validate_user
            middleware -> verify_token -> validate_user
            >>>
            >>> # Find path between two specific functions
            >>> result = engine.trace_call_path("main", "save_to_db")
            >>> if result.paths:
            ...     print("Connection found!")
            ...     print(" -> ".join(name for _, name, _ in result.paths[0]))
            Connection found!
            main -> process_request -> handle_data -> save_to_db
        """
        if not self._indexed:
            self.index()

        if self._graph_analyzer is not None:
            return self._graph_analyzer.trace_call_path(
                start_identifier=start_identifier,
                end_identifier=end_identifier,
                direction=direction,
                max_depth=max_depth,
                limit_paths=limit_paths,
                min_weight=min_weight,
            )

        # Fallback for when analyzer is not available
        return PathTraceResult(
            source=start_identifier,
            target=end_identifier,
            direction=direction,
        )

    def search(
        self,
        query: str = "",
        use_architecture: bool = True,
        include_deps: bool = False,
        limit: int = 100,
        map_mode: MapMode = MapMode.STATIC,
    ) -> list[SearchResult]:
        """
        Search the codebase with hybrid ranking.

        Uses multi-signal ranking combining lexical (BM25), structural (PageRank),
        and architectural (Betweenness) signals via Reciprocal Rank Fusion.

        Args:
            query: Search query (empty for map mode which returns architectural overview)
            use_architecture: Include betweenness centrality in ranking
            include_deps: Include caller/callee dependency context in results
            limit: Maximum results to return
            map_mode: When `MapMode.SIMPLE` and the call resolves to map mode,
                bypass the combined-signal pipeline and rank by per-file git
                commit count (line-count fallback when not in a git repository).
                Validated +0.120 mean P@10 vs LOC on the n=5 click-GT corpus.
                Ignored for non-map calls.

        Returns:
            List of SearchResult objects sorted by relevance score.

        Example:
            >>> engine = SearchEngine("/path/to/repo")
            >>> engine.index()
            >>> # Search for authentication-related code
            >>> results = engine.search("user authentication login")
            >>> for r in results[:5]:
            ...     print(f"{r.rank}. {r.entity.qualified_name} (score: {r.score:.3f})")
            1. auth.UserAuthenticator (score: 0.142)
            2. auth.LoginHandler (score: 0.128)
            3. models.User (score: 0.095)

            >>> # Map mode: get architectural overview (no query)
            >>> overview = engine.search("", limit=20)
            >>> for r in overview[:3]:
            ...     print(f"{r.entity.qualified_name} - {r.entity.entity_type}")
            core.Application - class
            database.Repository - class
            api.Router - class
        """
        if not self._indexed:
            self.index()

        # Mode picks which signals run and which weights each contributes.
        # See search/signals.py for the registry that maps modes -> active signals.
        mode = self._select_mode(query)

        return strategy_for(map_mode).rank(
            self,
            query,
            mode=mode,
            use_architecture=use_architecture,
            include_deps=include_deps,
            limit=limit,
        )

    def _full_signal_search(
        self,
        query: str,
        *,
        mode: Mode,
        use_architecture: bool,
        include_deps: bool,
        limit: int,
    ) -> list[SearchResult]:
        """Combined-signal RRF ranking — the default ranking path.

        Used by `StaticModeStrategy.rank` and by `SimpleModeStrategy.rank` for
        non-map calls (where lexical/structural signals still matter).
        """
        active = signals_for_mode(mode)
        by_signal: dict[str, dict[str, float]] = {}
        for sig in active:
            # Architectural signal (betweenness) is a user-toggleable channel.
            if sig.name == "bt" and not use_architecture:
                continue
            by_signal[sig.name] = sig.compute(self, query)

        # Query-time semantic failure → fall back to BM25 mode. _semantic returns {}
        # on exception; without this, BM25 has no query_semantic weight and the fused
        # ranking would lose its lexical signal entirely.
        if mode == "query_semantic" and not by_signal.get("semantic"):
            mode = "query_bm25"
            active = signals_for_mode(mode)
            for sig in active:
                if sig.name == "bt" and not use_architecture:
                    continue
                if sig.name not in by_signal:
                    by_signal[sig.name] = sig.compute(self, query)

        signals = RankingSignals(by_signal=by_signal)

        # In map mode, aggregate method/function scores into their parent classes
        # so the architectural overview surfaces classes, not individual methods.
        if mode == "map":
            signals = signals.aggregate_to_classes(
                active, self._entities, Config.MAP_AGGREGATION_DAMPENING
            )

        final_scores = self._fuse_rankings(signals, mode)

        results = []
        ranked_nodes = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)

        for i, node_id in enumerate(ranked_nodes[:limit]):
            entity = self._entities[node_id]
            dep_context = self.get_dependency_context(node_id) if include_deps else None
            results.append(SearchResult(
                rank=i + 1,
                entity=entity,
                score=final_scores[node_id],
                components=signals.components_for(node_id),
                dependency_context=dep_context,
            ))

        return results

    def _simple_map_search(
        self, include_deps: bool, limit: int
    ) -> list[SearchResult]:
        """Map-mode ranker for `--simple`: per-object git commits + line-count tiebreak.

        See `research/improving-code-map/process_metrics/report.md` for the
        empirical justification (+0.120 mean P@10 vs LOC, n=5 click-GT corpus).

        Falls back to pure line-count when the project isn't a git repository
        (or git history is unavailable locally, or a bounded blame probe fails).
        Output is structurally identical to regular map mode — only the ranking
        signal differs.
        """
        entity_lines: dict[str, float] = {
            nid: float(entity.line_count) for nid, entity in self._entities.items()
        }

        root_str = str(self.root)
        use_git = history_is_locally_available(root_str)
        if self._lite_change_count is not None:
            file_counts = self._lite_change_count
        else:
            file_counts = harvest_change_count(root_str) if use_git else {}
        if not use_git:
            logger.info(
                "--simple: git history is incomplete locally (shallow/promisor clone); "
                "falling back to line-count ranking"
            )
        elif not file_counts:
            logger.info(
                "--simple: no git history available, falling back to line-count ranking"
            )
            use_git = False

        entity_scores: dict[str, float] = {}
        entity_commits: dict[str, float] = {}

        if use_git:
            entities_by_file: dict[str, list[tuple[str, CodeEntity]]] = defaultdict(list)
            for nid, entity in self._entities.items():
                rel = to_repo_relative_posix(entity.file_path, root_str)
                if rel is not None:
                    entities_by_file[rel].append((nid, entity))

            ranked_files = sorted(
                entities_by_file,
                key=lambda rel: file_counts.get(rel, 0),
                reverse=True,
            )

            # Process ranked_files in batches of 8 with a thread pool. The
            # kth-score early-termination gate is checked BEFORE each batch
            # (ranked_files is sorted by file_counts desc, so files[idx] holds
            # the max remaining upper_bound — if even that is below kth, no
            # later file can enter the top-N). Within a batch, blame I/O runs
            # in parallel; results are scored in submission order, equivalent
            # to the sequential semantics modulo up-to-7 extra files per batch.
            BATCH_SIZE = 8
            idx = 0
            blame_failed = False
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as ex:
                while idx < len(ranked_files) and not blame_failed:
                    if len(entity_scores) >= limit:
                        kth_score = sorted(entity_scores.values(), reverse=True)[limit - 1]
                        if file_counts.get(ranked_files[idx], 0) < int(kth_score):
                            break

                    batch = ranked_files[idx:idx + BATCH_SIZE]
                    idx += BATCH_SIZE

                    zero_files = [rel for rel in batch if file_counts.get(rel, 0) <= 0]
                    blame_files = [rel for rel in batch if file_counts.get(rel, 0) > 0]

                    for rel in zero_files:
                        for nid, _ in entities_by_file[rel]:
                            entity_commits.setdefault(nid, 0.0)
                            entity_scores.setdefault(
                                nid,
                                entity_lines[nid] / SIMPLE_MAP_LINE_TIEBREAK_DIVISOR,
                            )

                    if blame_files:
                        blame_results = list(
                            ex.map(lambda rel: (rel, harvest_line_blame_commits(root_str, rel)),
                                   blame_files)
                        )
                        for rel, line_commits in blame_results:
                            if line_commits is None:
                                logger.info(
                                    "--simple: git blame failed or timed out for %s; falling back to "
                                    "line-count ranking",
                                    rel,
                                )
                                blame_failed = True
                                break

                            for nid, entity in entities_by_file[rel]:
                                commits = len({
                                    line_commits[line_no]
                                    for line_no in range(entity.line_start, entity.line_end + 1)
                                    if line_no in line_commits
                                })
                                entity_commits[nid] = float(commits)
                                # Integer commit count dominates; line_count breaks ties within
                                # one commit bucket. The divisor (1e6) keeps even huge entities
                                # below 1.0 so the tiebreak can never overpower one extra commit.
                                entity_scores[nid] = (
                                    commits + entity.line_count / SIMPLE_MAP_LINE_TIEBREAK_DIVISOR
                                )

            if blame_failed:
                use_git = False
                entity_scores.clear()
                entity_commits.clear()

            if use_git:
                for nid, lines in entity_lines.items():
                    entity_commits.setdefault(nid, 0.0)
                    entity_scores.setdefault(
                        nid,
                        lines / SIMPLE_MAP_LINE_TIEBREAK_DIVISOR,
                    )

        if not use_git:
            entity_scores = dict(entity_lines)

        # Synthetic single-signal so map's class aggregation runs unchanged.
        # `compute` is unused here (we passed scores in directly via by_signal).
        simple_signal = Signal(
            name="simple",
            compute=lambda e, q: {},
            weights={"map": 1.0},
            aggregate=sqrt_sum,
        )
        signals = RankingSignals(by_signal={
            "simple": dict(entity_scores),
            "simple_commits": entity_commits,
            "simple_lines": entity_lines,
        })
        signals = signals.aggregate_to_classes(
            [simple_signal], self._entities, Config.MAP_AGGREGATION_DAMPENING
        )

        final_scores = signals.by_signal["simple"]
        ranked = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)

        results: list[SearchResult] = []
        for i, node_id in enumerate(ranked[:limit]):
            entity = self._entities[node_id]
            dep_context = self.get_dependency_context(node_id) if include_deps else None
            results.append(SearchResult(
                rank=i + 1,
                entity=entity,
                score=final_scores[node_id],
                components=signals.components_for(node_id),
                dependency_context=dep_context,
            ))
        return results

    def _get_pagerank(
        self,
        scores_bm25: dict[str, float]
    ) -> dict[str, float]:
        """Get PageRank scores, personalized if possible."""
        if self._graph_analyzer is not None:
            return self._graph_analyzer.get_pagerank(scores_bm25)
        return self._centrality.pagerank or {}

    def _get_dispatcher_scores(self) -> dict[str, float]:
        """Per-function dispatcher score: fan_out * log1p(CC).

        Drops fan_in deliberately — PageRank and Betweenness already reward fan_in,
        so a fan_in-aware Disp would be redundant and would let leaf utilities
        (sdslen, zfree) outscore real dispatchers (processCommand, beginWork).
        Skips entities without cyclomatic_complexity (classes, modules) — they pick
        up dispatcher signal via class aggregation."""
        scores: dict[str, float] = {}
        for node_id in self._graph.nodes():
            entity = self._entities.get(node_id)
            if entity is None or entity.cyclomatic_complexity is None:
                continue
            fan_out = self._graph.out_degree(node_id)
            scores[node_id] = calculate_dispatcher_score(fan_out, entity.cyclomatic_complexity)
        return scores

    def _get_entry_scores(self) -> dict[str, float]:
        """Reverse-PageRank: entry-point roots (no callers, broad downstream reach)
        become sinks in the reversed graph and accumulate score there. Surfaces
        CLI/server entry points (e.g. main, serverCron) that PR/BT/Disp systematically
        bury because they have fan_in≈0 and are not on shortest paths.
        Lazy, cached per SearchEngine instance."""
        if self._entry_scores is not None:
            return self._entry_scores
        import networkx as nx
        try:
            self._entry_scores = nx.pagerank(self._graph.reverse(copy=False), weight="weight")
        except Exception as e:
            logger.warning(f"Entry-signal (reverse-PageRank) failed: {e}.")
            self._entry_scores = {}
        return self._entry_scores

    def _select_mode(self, query: str) -> Mode:
        """Pick which signal set runs for this call.

        - empty query → "map" (architectural overview using static signals)
        - query + semantic available → "query_semantic" (let semantic dominate)
        - query, no semantic → "query_bm25" (lexical-leaning fallback)
        """
        if not query:
            return "map"
        if self.enable_semantic and self._semantic_index:
            return "query_semantic"
        return "query_bm25"

    def _fuse_rankings(self, signals: RankingSignals, mode: Mode) -> dict[str, float]:
        """
        Fuse multiple ranking signals using Reciprocal Rank Fusion (RRF).

        RRF is a rank aggregation technique that combines multiple ranked lists
        into a single ranking. It's robust to outliers and doesn't require score
        normalization across different ranking signals.

        Algorithm:
            RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

        Where:
            - d is a document (code entity)
            - k is a constant (default 60) that dampens the impact of high ranks
            - rank_i(d) is the rank of document d in ranking list i
            - weight_i is the importance weight of ranking signal i

        The k parameter controls rank sensitivity:
            - Lower k: Top ranks dominate (more aggressive)
            - Higher k: Ranks are more evenly weighted (more conservative)

        Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        "Reciprocal rank fusion outperforms condorcet and individual rank
        learning methods." SIGIR '09.
        """
        final_scores: dict[str, float] = defaultdict(float)

        # k=60 is the standard RRF constant from the original paper.
        # It provides a good balance between emphasizing top results
        # while still giving credit to lower-ranked but consistently appearing items.
        k = Config.RRF_K

        # Method penalty differs by mode: map mode shows methods almost equally,
        # query mode lightly favors classes.
        method_penalty = (
            Config.MAP_PENALTY_METHOD if mode == "map" else Config.QUERY_PENALTY_METHOD
        )

        def add_rrf(scores: dict[str, float], weight: float) -> None:
            """
            Add RRF contribution from one ranking signal.

            Each signal contributes: weight / (k + effective_rank)
            This ensures top-ranked items get higher scores while
            avoiding division by zero (since k > 0).
            """
            if not scores:
                return

            # Sort entities by their score in this ranking signal (descending)
            ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

            # Track rank with tie handling - entities with same score share rank
            current_rank = 0
            last_score: float | None = None

            for i, node_id in enumerate(ranked):
                score = scores.get(node_id, 0.0)

                # Tie-aware ranking: identical raw scores share the same rank.
                # This prevents arbitrary tie-breaking from affecting results.
                # Example: if items at positions 0,1,2 all have score 0.8,
                # they all get rank 0, and position 3 gets rank 3.
                if last_score is None or score != last_score:
                    current_rank = i
                    last_score = score

                entity = self._entities[node_id]

                # Apply rank penalties based on entity characteristics.
                # These shift the entity down in the ranking, reducing its RRF score.
                # Penalties are additive to the rank position.
                effective_rank = current_rank

                # Private entities (e.g., _helper, __internal) are less relevant
                # for external consumers, so demote them in search results
                if entity.is_private:
                    effective_rank += Config.PENALTY_PRIVATE

                # Test files are typically not what users search for in production code
                if entity.is_test:
                    effective_rank += 10

                # Penalize methods/functions to favor class-level entities.
                # Classes provide better architectural context in search results.
                if entity.entity_type in ("method", "function"):
                    effective_rank += method_penalty

                # RRF formula: contribution = weight / (k + rank)
                # Higher weight = more important signal
                # Lower rank = higher contribution (rank 0 contributes most)
                final_scores[node_id] += weight * (1.0 / (k + effective_rank))

        # Iterate the registry: each Signal carries its per-mode weight, so the
        # mode dispatch reduces to "skip signals that don't apply to this mode".
        for sig in signals_for_mode(mode):
            add_rrf(signals.by_signal.get(sig.name, {}), sig.weights[mode])

        if mode == "query_semantic":
            # Penalize entities with no semantic relevance (below threshold).
            # This prevents high-PageRank entities from dominating when they're
            # semantically irrelevant to the query.
            sem = signals.by_signal.get("semantic", {})
            penalty = Config.SEMANTIC_IRRELEVANT_PENALTY
            for node_id in list(final_scores):
                if node_id not in sem:
                    final_scores[node_id] *= penalty

        return dict(final_scores)

    def find_identifiers(
        self,
        query: str,
        limit: int = 50,
        include_deps: bool = False
    ) -> list[SearchResult]:
        """
        Find specific identifiers (function/class/method names) in the codebase.

        Unlike search(), this method performs exact and prefix matching on
        entity names rather than full-text search. Use this when you know
        the exact or partial name of what you're looking for.

        Matching priority (score):
            - Exact match: 100.0
            - Prefix match: 75.0
            - Substring match: 50.0
            - Content match: 25.0

        Args:
            query: Identifier name to search for (case-insensitive)
            limit: Maximum results to return
            include_deps: Include caller/callee dependency context

        Returns:
            List of SearchResult objects sorted by match quality.

        Example:
            >>> engine = SearchEngine("/path/to/repo")
            >>> engine.index()
            >>> # Find all entities named "parse" or starting with "parse"
            >>> results = engine.find_identifiers("parse")
            >>> for r in results[:5]:
            ...     print(f"{r.entity.name} ({r.entity.entity_type}): {r.score}")
            parse (function): 100.0
            parse_file (method): 75.0
            parse_config (function): 75.0
            XMLParser (class): 50.0
        """
        if not self._indexed:
            self.index()

        query_lower = query.lower()
        matches = []

        for node_id, entity in self._entities.items():
            score = 0.0

            if entity.name.lower() == query_lower:
                score = 100.0
            elif entity.name.lower().startswith(query_lower):
                score = 75.0
            elif query_lower in entity.name.lower():
                score = 50.0
            elif query_lower in entity.searchable_text.lower():
                score = 25.0

            if score > 0:
                matches.append((node_id, score))

        matches.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, (node_id, score) in enumerate(matches[:limit]):
            entity = self._entities[node_id]

            dep_context = None
            if include_deps:
                dep_context = self.get_dependency_context(node_id)

            results.append(SearchResult(
                rank=i + 1,
                entity=entity,
                score=score,
                components={"match": score},
                dependency_context=dep_context,
            ))

        return results

    def generate_directory_tree(self, results: list[SearchResult]) -> str:
        """
        Generate a recursive directory tree showing ONLY the provided results.
        This is used to give a structural overview of the top-scoring entities.
        """
        formatter = DirectoryTreeFormatter(
            root=self.root,
            entities=self._entities,
            file_to_entities=self._file_to_entities,
        )
        return formatter.format_tree(results)

    def format_stats(self, results: list[SearchResult], limit: int = 20) -> str:
        """Format ranking statistics with Rich colors and clickable links."""
        if not results:
            return ""

        style = get_terminal_style()
        max_score = max(r.score for r in results)

        active_names = set(results[0].components)
        is_simple = "simple" in active_names

        if is_simple:
            has_commits = any(r.components.get("simple_commits", 0.0) > 0.0 for r in results)
            if has_commits:
                header = f"{'Rank':<4} | {'Score':<8} | {'Commits':<7} | {'Lines':<5} | Entity"
            else:
                header = f"{'Rank':<4} | {'Score':<8} | {'Lines':<5} | Entity"
            sep_width = 60
        else:
            columns = [s for s in SIGNALS if s.column and s.name in active_names]
            widths = [len(s.column[1].format(0.0)) for s in columns]
            header_cells = [f"{s.column[0]:<{w}}" for s, w in zip(columns, widths)]
            header = (
                f"{'Rank':<4} | {'Score':<8} | "
                + " | ".join(header_cells)
                + f" | {'Lines':<5} | Entity"
            )
            sep_width = 60 + sum(widths) + 3 * len(columns)

        lines = ["", header, "-" * sep_width]

        for r in results[:limit]:
            name = r.entity.qualified_name
            if len(name) > 30:
                name = "..." + name[-27:]

            flags = [n for n, on in [
                ("priv", r.entity.is_private),
                ("tiny", r.entity.is_tiny),
                ("util", r.entity.is_utility),
                ("test", r.entity.is_test),
            ] if on]
            flag_str = f" [{','.join(flags)}]" if flags else ""

            colored_score = style.format_rank(r.score, max_score, text=f"{r.score:<8.4f}")
            colored_entity = style.format_stats_entity(
                name=name,
                file_path=r.entity.file_path,
                line=r.entity.line_start,
                score=r.score,
                max_score=max_score,
                flags=flag_str,
            )

            if is_simple:
                if has_commits:
                    commits = int(r.components.get("simple_commits", 0.0))
                    middle = f"{commits:<7} | {r.entity.line_count:<5}"
                else:
                    middle = f"{r.entity.line_count:<5}"
                lines.append(f"{r.rank:<4} | {colored_score} | {middle} | {colored_entity}")
            else:
                cells = [s.column[1].format(r.components.get(s.name, 0.0)) for s in columns]
                lines.append(
                    f"{r.rank:<4} | {colored_score} | "
                    + " | ".join(cells)
                    + f" | {r.entity.line_count:<5} | {colored_entity}"
                )

        lines.append("-" * sep_width)
        lines.append("")
        return "\n".join(lines)

    def get_stats(self) -> IndexStats:
        """Get indexing statistics."""
        return self._stats


class ModeStrategy(ABC):
    """Per-MapMode dispatch for engine construction, ranking, and daemon bypass.

    Each MapMode value maps to one ModeStrategy instance in `_MODE_DISPATCH`,
    so adding a mode is one enum member plus one strategy class registered in
    one place — call sites never branch on MapMode directly.
    """

    layout: CacheLayout
    semantic_override: SemanticConfig | None  # None = honor caller's semantic
    bypasses_daemon: bool

    @abstractmethod
    def build_engine(self, cache: "CacheManager", verbose: bool) -> SearchEngine:
        """Load the cache for this mode's layout and construct a SearchEngine."""

    @abstractmethod
    def rank(
        self,
        engine: SearchEngine,
        query: str,
        *,
        mode: Mode,
        use_architecture: bool,
        include_deps: bool,
        limit: int,
    ) -> list[SearchResult]:
        """Produce ranked SearchResults for this mode's ranking strategy."""


class StaticModeStrategy(ModeStrategy):
    """Default `--map-mode static` strategy: full graph + combined-signal RRF."""

    layout = STATIC_LAYOUT
    semantic_override = None
    bypasses_daemon = False

    def build_engine(self, cache: "CacheManager", verbose: bool) -> SearchEngine:
        cached = cache.load_or_rebuild()
        return SearchEngine.from_cached_indices(cached, verbose=verbose)

    def rank(
        self,
        engine: SearchEngine,
        query: str,
        *,
        mode: Mode,
        use_architecture: bool,
        include_deps: bool,
        limit: int,
    ) -> list[SearchResult]:
        return engine._full_signal_search(
            query,
            mode=mode,
            use_architecture=use_architecture,
            include_deps=include_deps,
            limit=limit,
        )


class SimpleModeStrategy(ModeStrategy):
    """`--map-mode simple` strategy: lite cache + per-object git ranking.

    Validated +0.120 mean P@10 vs LOC on the n=5 click-GT corpus; see
    `_simple_map_search` for the ranking signal. The CLI also runs this mode
    daemon-free (`bypasses_daemon = True`) because the lite cache plus a
    bounded git-blame probe is faster than a daemon cold-start.
    """

    layout = LITE_LAYOUT
    semantic_override = SemanticConfig(enabled=False)
    bypasses_daemon = True

    def build_engine(self, cache: "CacheManager", verbose: bool) -> SearchEngine:
        cached = cache.load_or_rebuild_lite()
        return SearchEngine.from_lite_cache(cached, verbose=verbose)

    def rank(
        self,
        engine: SearchEngine,
        query: str,
        *,
        mode: Mode,
        use_architecture: bool,
        include_deps: bool,
        limit: int,
    ) -> list[SearchResult]:
        # SIMPLE only short-circuits in map mode; for a real query we still
        # need the lexical/structural signals (matches `--map` semantics).
        if mode == "map":
            return engine._simple_map_search(include_deps, limit)
        return engine._full_signal_search(
            query,
            mode=mode,
            use_architecture=use_architecture,
            include_deps=include_deps,
            limit=limit,
        )


_MODE_DISPATCH: dict[MapMode, ModeStrategy] = {
    MapMode.STATIC: StaticModeStrategy(),
    MapMode.SIMPLE: SimpleModeStrategy(),
}


def strategy_for(mode: MapMode) -> ModeStrategy:
    """Return the ModeStrategy registered for `mode`.

    Raises ValueError naming the missing mode and the registry, so adding a
    new MapMode without an _MODE_DISPATCH entry fails loudly instead of
    surfacing a bare KeyError from deep inside the pipeline.
    """
    try:
        return _MODE_DISPATCH[mode]
    except KeyError as exc:
        raise ValueError(
            f"No ModeStrategy registered for MapMode {mode!r}. "
            f"Add an entry to _MODE_DISPATCH in {__name__}."
        ) from exc
