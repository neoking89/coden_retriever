"""Search engine module.

Main orchestrator for code search, combining multiple ranking signals.
"""
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config

if TYPE_CHECKING:
    import networkx as nx
    from ..cache import CachedIndices
    from .graph_analyzer import GraphAnalyzer

from ..formatters.directory_tree_formatter import DirectoryTreeFormatter
from ..formatters.terminal_style import get_terminal_style
from ..language import LANGUAGE_MAP
from ..models import CodeEntity, DependencyContext, IndexStats, PathTraceResult, SearchResult
from ..parsers import RepoParser
from .bm25 import BM25Index

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

    def __init__(
        self,
        root: str,
        verbose: bool = False,
        enable_semantic: bool = False,
        model_path: str | None = None
    ):
        self.root = Path(root).resolve()
        self.verbose = verbose
        self.enable_semantic = enable_semantic
        self.model_path = model_path

        self._graph = _create_digraph()
        self._entities: dict[str, CodeEntity] = {}
        self._bm25 = BM25Index()
        self._parser = RepoParser()

        # Lazy load semantic index only if enabled (saves memory/startup time)
        self._semantic_index = None
        if self.enable_semantic:
            self._init_semantic_index()

        self._name_to_nodes: dict[str, list[str]] = defaultdict(list)
        self._file_scopes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        self._file_to_entities: dict[str, list[str]] = defaultdict(list)

        self._stats = IndexStats()
        self._indexed = False

        self._pagerank_cache: dict[str, float] | None = None
        self._betweenness_cache: dict[str, float] | None = None
        self._graph_analyzer: "GraphAnalyzer | None" = None

    @classmethod
    def from_cached_indices(cls, cached: "CachedIndices", verbose: bool = False) -> "SearchEngine":
        """Create a SearchEngine from cached indices.

        Args:
            cached: CachedIndices containing all pre-computed data.
            verbose: Enable verbose logging.

        Returns:
            SearchEngine instance ready for searching.
        """
        engine = cls.__new__(cls)
        engine.root = cached.source_dir
        engine.verbose = verbose
        engine.enable_semantic = cached.has_semantic
        engine.model_path = None

        # Set cached data directly
        engine._graph = cached.graph
        engine._entities = cached.entities
        engine._bm25 = cached.bm25_index
        engine._parser = RepoParser()

        # Set semantic index if available
        engine._semantic_index = None
        if cached.has_semantic and cached.embeddings is not None:
            try:
                from .semantic import SemanticIndex
                # Create a semantic index with pre-computed embeddings
                model_path = Config.get_semantic_model_path(None)
                if model_path:
                    engine._semantic_index = SemanticIndex(model_path)
                    # Load the model (needed for query encoding)
                    engine._semantic_index._load_model()
                    # Set cached embeddings and node IDs
                    engine._semantic_index._embeddings = cached.embeddings
                    engine._semantic_index._node_ids = cached.node_ids
            except ImportError:
                engine.enable_semantic = False

        # Set lookup structures
        engine._name_to_nodes = defaultdict(list, cached.name_to_nodes)
        engine._file_scopes = defaultdict(list, cached.file_scopes)
        engine._file_to_entities = defaultdict(list, cached.file_to_entities)

        # Set centrality caches
        engine._pagerank_cache = cached.pagerank
        engine._betweenness_cache = cached.betweenness

        # Create graph analyzer with cached data
        from .graph_analyzer import GraphAnalyzer
        engine._graph_analyzer = GraphAnalyzer(
            entities=engine._entities,
            name_to_nodes=engine._name_to_nodes,
            file_scopes=engine._file_scopes,
            verbose=verbose,
        )
        # Set the graph and caches on the analyzer
        engine._graph_analyzer._graph = cached.graph
        engine._graph_analyzer._pagerank_cache = cached.pagerank or {}
        engine._graph_analyzer._betweenness_cache = cached.betweenness or {}

        # Create stats
        engine._stats = IndexStats()
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

            # Get model path with priority: CLI flag > env var > package model
            model_path = Config.get_semantic_model_path(self.model_path)

            # Check if model exists
            if model_path is None or not Path(model_path).exists():
                logger.warning(
                    "Semantic model not found. "
                    "Falling back to BM25-only search. "
                    "To use semantic search, ensure the model is available at:\n"
                    "  1. Custom path: --model-path <path>\n"
                    "  2. Environment: CODEN_RETRIEVER_MODEL_PATH=<path>\n"
                    "  3. Package model: models/embeddings/model2vec_embed_distill"
                )
                self.enable_semantic = False
                return

            self._semantic_index = SemanticIndex(model_path)
            logger.info(f"Semantic search enabled with model at: {model_path}")

        except ImportError as e:
            logger.warning(
                f"Cannot enable semantic search: {e}. "
                "Install model2vec with: pip install model2vec"
            )
            self.enable_semantic = False
        except Exception as e:
            logger.warning(f"Failed to initialize semantic search: {e}. Falling back to BM25-only.")
            self.enable_semantic = False

    def index(self) -> IndexStats:
        """Index the repository."""
        # Reset all state at the beginning of each index operation
        self._stats = IndexStats()
        self._entities = {}
        self._graph = _create_digraph()
        self._name_to_nodes = defaultdict(list)
        self._file_scopes = defaultdict(list)
        self._file_to_entities = defaultdict(list)
        self._pagerank_cache = None
        self._betweenness_cache = None
        self._graph_analyzer = None

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
                self._semantic_index.index(self._entities)
            except Exception as e:
                logger.warning(f"Semantic indexing failed: {e}. Continuing with BM25-only.")
                self.enable_semantic = False

        logger.info("Computing centrality metrics...")
        self._graph_analyzer.compute_centrality()
        # Keep caches for backwards compatibility
        self._pagerank_cache = self._graph_analyzer.pagerank_cache
        self._betweenness_cache = self._graph_analyzer.betweenness_cache

        self._stats.total_entities = len(self._entities)
        self._stats.total_edges = self._graph.number_of_edges()
        self._stats.index_time_ms = (time.time() - start_time) * 1000

        self._indexed = True

        logger.info(f"Indexing complete: {self._stats.total_entities} entities, "
                    f"{self._stats.total_edges} edges in {self._stats.index_time_ms:.0f}ms")

        return self._stats

    def _collect_files(self) -> list[Path]:
        """Collect all source files to index."""
        files = []
        # Convert SKIP_DIRS to a set for O(1) lookups
        skip_dirs = Config.SKIP_DIRS
        skip_files = Config.SKIP_FILES
        
        for root, dirs, filenames in os.walk(self.root):
            # Prune directories in-place to stop os.walk from entering them
            # This is the critical performance fix
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            
            for name in filenames:
                if name in skip_files:
                    continue
                
                path = Path(root) / name
                if path.suffix.lower() in LANGUAGE_MAP:
                    try:
                        if path.stat().st_size > 1_000_000:
                            continue
                    except Exception:
                        continue
                    
                    files.append(path)
        return files

    def _parse_file(
        self,
        file_path: Path,
        documents: dict[str, str],
        all_refs: list
    ) -> bool:
        """Parse a single file.
        
        Returns:
            True if file was successfully parsed (even if no entities found),
            False only if an actual parsing error occurred.
        """
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="ignore")
            entities, references = self._parser.parse_file(str(file_path), source_code)

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
        limit: int = 100
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

        # Lexical/Semantic scores (mutually exclusive - toggle between them)
        scores_bm25 = {}
        scores_semantic = {}

        if query:
            if self.enable_semantic and self._semantic_index:
                # Use semantic search (replaces BM25)
                try:
                    scores_semantic = self._semantic_index.score_all(query)
                except Exception as e:
                    logger.warning(f"Semantic scoring failed: {e}. Falling back to BM25.")
                    scores_bm25 = self._bm25.score_all(query)
            else:
                # Use BM25 keyword search (default)
                scores_bm25 = self._bm25.score_all(query)

        # PageRank (personalized if we have seeds)
        scores_pr = self._get_pagerank(scores_bm25)

        # Betweenness centrality
        scores_bt = self._betweenness_cache if use_architecture else {}

        # In map mode (no query), aggregate method scores to parent classes
        # This boosts class-level entities for architectural overview
        if not query:
            scores_pr, scores_bt = self._aggregate_scores_to_classes(scores_pr, scores_bt or {})

        # Fuse rankings
        final_scores = self._fuse_rankings(
            scores_bm25, scores_pr, scores_bt or {}, scores_semantic,
            has_query=bool(query)
        )

        # Build results
        results = []
        ranked_nodes = sorted(final_scores.keys(), key=final_scores.get, reverse=True)

        for i, node_id in enumerate(ranked_nodes[:limit]):
            entity = self._entities[node_id]

            dep_context = None
            if include_deps:
                dep_context = self.get_dependency_context(node_id)

            components_dict = {
                "bm25": scores_bm25.get(node_id, 0.0),
                "pr": scores_pr.get(node_id, 0.0),
                "bt": scores_bt.get(node_id, 0.0),
            }

            # Add semantic score if enabled
            if scores_semantic:
                components_dict["semantic"] = scores_semantic.get(node_id, 0.0)

            results.append(SearchResult(
                rank=i + 1,
                entity=entity,
                score=final_scores[node_id],
                components=components_dict,
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
        return self._pagerank_cache or {}

    def _aggregate_scores_to_classes(
        self,
        scores_pr: dict[str, float],
        scores_bt: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Aggregate method/function scores to their parent classes.

        For PageRank: Use sqrt-normalized sum to prevent runaway scores
        For Betweenness: Use max + sum to boost classes with important methods

        Returns:
            Tuple of (aggregated_pr, aggregated_bt) dictionaries
        """
        if self._graph_analyzer is not None:
            return self._graph_analyzer.aggregate_scores_to_classes(scores_pr, scores_bt)
        return scores_pr, scores_bt

    def _fuse_rankings(
        self,
        scores_bm25: dict[str, float],
        scores_pr: dict[str, float],
        scores_bt: dict[str, float],
        scores_semantic: dict[str, float],
        has_query: bool
    ) -> dict[str, float]:
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

        # Use different method penalties based on mode:
        # - Map mode: use MAP_PENALTY_METHOD (very low, methods shown equally)
        # - Query mode: use QUERY_PENALTY_METHOD (low, slight class preference)
        # - No query + not map: use PENALTY_METHOD (high, strong class preference)
        method_penalty = Config.MAP_PENALTY_METHOD if not has_query else Config.QUERY_PENALTY_METHOD

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
            ranked = sorted(scores.keys(), key=scores.get, reverse=True)

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

        # Query mode: combine all available signals with their configured weights
        if has_query:
            if scores_semantic:
                # Semantic mode: use reduced structural weights to let semantic dominate
                add_rrf(scores_pr, Config.WEIGHT_PAGERANK_SEMANTIC)
                add_rrf(scores_bt, Config.WEIGHT_BETWEENNESS_SEMANTIC)
                add_rrf(scores_semantic, Config.WEIGHT_SEMANTIC)

                # Penalize entities with no semantic relevance (below threshold)
                # This prevents high-PageRank entities from dominating when they're
                # semantically irrelevant to the query.
                penalty = Config.SEMANTIC_IRRELEVANT_PENALTY
                for node_id in final_scores:
                    if node_id not in scores_semantic:
                        final_scores[node_id] *= penalty
            else:
                # BM25 mode: use full structural weights for keyword search
                add_rrf(scores_bm25, Config.WEIGHT_BM25)
                add_rrf(scores_pr, Config.WEIGHT_PAGERANK_BM25)
                add_rrf(scores_bt, Config.WEIGHT_BETWEENNESS_BM25)
        else:
            # Map mode (no query): only use structural signals
            # BM25/semantic require a query, so we rely on graph-based metrics
            add_rrf(scores_pr, Config.MAP_WEIGHT_PAGERANK)
            add_rrf(scores_bt, Config.MAP_WEIGHT_BETWEENNESS)

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

    def print_stats(self, results: list[SearchResult], limit: int = 20) -> None:
        """Print ranking statistics to stderr."""
        print(file=sys.stderr)

        # Check if semantic scores are present
        has_semantic = results and "semantic" in results[0].components

        if has_semantic:
            header = f"{'Rank':<4} │ {'Score':<8} │ {'BM25':<6} │ {'Sem':<6} │ {'PR':<8} │ {'BT':<8} │ {'Lines':<5} │ {'Entity'}"
        else:
            header = f"{'Rank':<4} │ {'Score':<8} │ {'BM25':<6} │ {'PR':<8} │ {'BT':<8} │ {'Lines':<5} │ {'Entity'}"

        print(header, file=sys.stderr)
        print("─" * 120 if has_semantic else "─" * 100, file=sys.stderr)

        for r in results[:limit]:
            name = r.entity.qualified_name
            if len(name) > 30:
                name = "..." + name[-27:]

            flags = []
            if r.entity.is_private:
                flags.append("priv")
            if r.entity.is_tiny:
                flags.append("tiny")
            if r.entity.is_utility:
                flags.append("util")
            if r.entity.is_test:
                flags.append("test")
            flag_str = f" [{','.join(flags)}]" if flags else ""

            if has_semantic:
                print(
                    f"{r.rank:<4} │ {r.score:<8.4f} │ {r.components.get('bm25', 0):<6.2f} │ "
                    f"{r.components.get('semantic', 0):<6.3f} │ {r.components.get('pr', 0):<8.5f} │ "
                    f"{r.components.get('bt', 0):<8.5f} │ {r.entity.line_count:<5} │ {name}{flag_str}",
                    file=sys.stderr
                )
            else:
                print(
                    f"{r.rank:<4} │ {r.score:<8.4f} │ {r.components.get('bm25', 0):<6.2f} │ "
                    f"{r.components.get('pr', 0):<8.5f} │ {r.components.get('bt', 0):<8.5f} │ "
                    f"{r.entity.line_count:<5} │ {name}{flag_str}",
                    file=sys.stderr
                )

        print("─" * 120 if has_semantic else "─" * 100, file=sys.stderr)
        print(file=sys.stderr)

    def format_stats(self, results: list[SearchResult], limit: int = 20) -> str:
        """Format ranking statistics with Rich colors and clickable links."""
        if not results:
            return ""

        # Get Rich styling
        style = get_terminal_style()
        max_score = max(r.score for r in results)

        lines = []
        lines.append("")

        # Check if semantic scores are present
        has_semantic = results and "semantic" in results[0].components

        if has_semantic:
            header = f"{'Rank':<4} │ {'Score':<8} │ {'BM25':<6} │ {'Sem':<6} │ {'PR':<8} │ {'BT':<8} │ {'Lines':<5} │ {'Entity'}"
        else:
            header = f"{'Rank':<4} │ {'Score':<8} │ {'BM25':<6} │ {'PR':<8} │ {'BT':<8} │ {'Lines':<5} │ {'Entity'}"

        lines.append(header)
        lines.append("─" * 120 if has_semantic else "─" * 100)

        for r in results[:limit]:
            name = r.entity.qualified_name
            if len(name) > 30:
                name = "..." + name[-27:]

            flags = []
            if r.entity.is_private:
                flags.append("priv")
            if r.entity.is_tiny:
                flags.append("tiny")
            if r.entity.is_utility:
                flags.append("util")
            if r.entity.is_test:
                flags.append("test")
            flag_str = f" [{','.join(flags)}]" if flags else ""

            # Color the score using Rich
            colored_score = style.format_rank(r.score, max_score)

            # Make entity name clickable and colored using Rich
            colored_entity = style.format_stats_entity(
                name=name,
                file_path=r.entity.file_path,
                line=r.entity.line_start,
                score=r.score,
                max_score=max_score,
                flags=flag_str,
            )

            if has_semantic:
                lines.append(
                    f"{r.rank:<4} │ {colored_score:<8} │ {r.components.get('bm25', 0):<6.2f} │ "
                    f"{r.components.get('semantic', 0):<6.3f} │ {r.components.get('pr', 0):<8.5f} │ "
                    f"{r.components.get('bt', 0):<8.5f} │ {r.entity.line_count:<5} │ {colored_entity}"
                )
            else:
                lines.append(
                    f"{r.rank:<4} │ {colored_score:<8} │ {r.components.get('bm25', 0):<6.2f} │ "
                    f"{r.components.get('pr', 0):<8.5f} │ {r.components.get('bt', 0):<8.5f} │ "
                    f"{r.entity.line_count:<5} │ {colored_entity}"
                )

        lines.append("─" * 120 if has_semantic else "─" * 100)
        lines.append("")
        return "\n".join(lines)

    def get_stats(self) -> IndexStats:
        """Get indexing statistics."""
        return self._stats
