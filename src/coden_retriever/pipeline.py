"""
Search pipeline for coden-retriever.

Consolidates the common search workflow used across CLI, daemon, and MCP interfaces.
This eliminates duplication of cache initialization, engine creation, search execution,
token budgeting, and output formatting logic.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .cache.manager import CacheManager
from .config import MapMode, OutputFormat
from .constants import DEFAULT_SEARCH_RESULT_LIMIT
from .formatters import get_formatter
from .formatters.base import OutputFormatter
from .search.engine import SearchEngine, strategy_for
from .semantic_config import SemanticConfig
from .token_estimator import count_tokens


@dataclass
class SearchConfig:
    """Configuration for a search pipeline execution."""

    root_path: Path
    query: str = ""
    token_limit: Optional[int] = None
    output_format: OutputFormat = OutputFormat.XML
    enable_semantic: bool = False
    model_path: Optional[str] = None
    show_deps: bool = False
    dir_tree: bool = False
    show_map: bool = False
    find_mode: Optional[str] = None
    limit: int | None = 20
    verbose: bool = False
    show_stats: bool = False
    reverse: bool = False
    map_mode: MapMode = MapMode.STATIC

    @classmethod
    def from_cli_args(cls, args, root_path: Path) -> "SearchConfig":
        """Create config from CLI argument namespace."""
        return cls(
            root_path=root_path,
            query=getattr(args, "query", "") or "",
            token_limit=getattr(args, "tokens", None),
            output_format=OutputFormat(getattr(args, "format", "xml")),
            enable_semantic=getattr(args, "enable_semantic", False),
            model_path=getattr(args, "model_path", None),
            show_deps=getattr(args, "show_deps", False),
            dir_tree=getattr(args, "dir_tree", False),
            show_map=getattr(args, "map", False),
            find_mode=getattr(args, "find", None),
            limit=getattr(args, "limit", 20),
            verbose=getattr(args, "verbose", False),
            show_stats=getattr(args, "stats", False),
            reverse=getattr(args, "reverse", False),
            map_mode=MapMode(getattr(args, "map_mode", MapMode.STATIC.value)),
        )


def check_semantic_used(results: list) -> bool:
    """Check whether semantic scoring actually contributed to search results.

    Prevents misleading 'SEMANTIC SEARCH MODE' banners when semantic
    was requested but silently fell back to BM25.
    """
    return bool(
        results
        and hasattr(results[0], "components")
        and "semantic" in results[0].components
    )


@dataclass
class SearchResult:
    """Result of a pipeline execution."""

    results: list  # List of search result objects
    filtered_results: list  # Results after token budget filtering
    used_tokens: int
    formatted_output: str
    tree_output: Optional[str] = None
    stats: Optional[str] = None
    semantic_used: bool = False


class SearchPipeline:
    """
    Unified search pipeline that encapsulates the common workflow:

    1. Initialize CacheManager
    2. Load/rebuild indices
    3. Create SearchEngine
    4. Execute search (based on mode)
    5. Apply token budget filtering
    6. Format output

    This class is used by CLI, daemon, and MCP interfaces to avoid code duplication.
    """

    def __init__(
        self,
        config: SearchConfig,
        cache: Optional[CacheManager] = None,
        engine: Optional[SearchEngine] = None,
    ):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Search configuration
            cache: Optional pre-initialized CacheManager
            engine: Optional pre-initialized SearchEngine
        """
        self.config = config
        self._cache: Optional[CacheManager] = cache
        self._engine: Optional[SearchEngine] = engine

    def create_engine(self) -> SearchEngine:
        """
        Initialize cache and create search engine.

        Returns:
            Configured SearchEngine instance
        """
        if self._engine is not None:
            return self._engine

        # Map context is the only place `--map-mode simple` is honored — a real
        # query or find call needs the full graph regardless of the user's
        # map_mode preference, so we pin to STATIC outside the map context.
        in_map_context = (
            (self.config.show_map or not self.config.query)
            and not self.config.find_mode
        )
        strategy = strategy_for(
            self.config.map_mode if in_map_context else MapMode.STATIC
        )

        if self._cache is None or self._cache._layout is not strategy.layout:
            semantic = strategy.semantic_override or SemanticConfig(
                enabled=self.config.enable_semantic,
                model_path=self.config.model_path,
            )
            self._cache = CacheManager(
                self.config.root_path,
                semantic=semantic,
                verbose=self.config.verbose,
                layout=strategy.layout,
            )

        self._engine = strategy.build_engine(self._cache, self.config.verbose)
        return self._engine

    def run_search(self, engine: Optional[SearchEngine] = None) -> list:
        """
        Execute search based on configured mode.

        Args:
            engine: Optional engine to use (creates one if not provided)

        Returns:
            List of raw search results
        """
        if engine is None:
            engine = self.create_engine()

        config = self.config

        limit = config.limit if config.limit is not None else DEFAULT_SEARCH_RESULT_LIMIT

        if config.find_mode:
            # Find identifier mode
            return engine.find_identifiers(
                config.find_mode,
                limit=limit,
                include_deps=config.show_deps,
            )
        elif config.show_map or not config.query:
            # Map mode (architecture overview)
            return engine.search(
                query="",
                use_architecture=True,
                include_deps=config.show_deps,
                limit=limit,
                map_mode=config.map_mode,
            )
        else:
            # Regular search mode
            return engine.search(
                query=config.query,
                include_deps=config.show_deps,
                limit=limit,
            )

    def filter_results(self, results: list) -> tuple[list, int]:
        """
        Apply token budget filtering to results.

        Args:
            results: Raw search results

        Returns:
            Tuple of (filtered_results, used_tokens)
        """
        filtered = OutputFormatter.filter_by_token_budget(
            results,
            self.config.token_limit,
            self.config.show_deps,
        )

        used_tokens = 100  # Base overhead
        for result in filtered:
            code = result.entity.get_context_snippet()
            used_tokens += count_tokens(code) + 30
            if self.config.show_deps and result.dependency_context and not result.dependency_context.is_empty():
                used_tokens += count_tokens(result.dependency_context.format_compact())

        return filtered, used_tokens

    def format_output(
        self,
        results: list,
        engine: Optional[SearchEngine] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Format search results for output.

        Args:
            results: Filtered search results
            engine: SearchEngine instance (needed for directory tree)

        Returns:
            Tuple of (formatted_output, tree_output)
        """
        formatter = get_formatter(self.config.output_format)
        formatted_output = formatter.format_results(
            results,
            self.config.root_path,
            self.config.token_limit,
            self.config.show_deps,
        )

        tree_output = None
        if self.config.dir_tree and engine is not None:
            tree_output = engine.generate_directory_tree(results)

        return formatted_output, tree_output

    def execute(self) -> SearchResult:
        """
        Execute the full search pipeline.

        Returns:
            SearchResult containing all outputs
        """
        engine = self.create_engine()

        # Run search
        raw_results = self.run_search(engine)

        # Apply token budget
        filtered_results, used_tokens = self.filter_results(raw_results)

        semantic_used = check_semantic_used(raw_results)

        # Get stats BEFORE reversing (stats always show normal order 1, 2, 3...)
        # Show stats for all filtered results (matching the actual output)
        stats = None
        if self.config.show_stats:
            stats = engine.format_stats(filtered_results, limit=len(filtered_results))

        # Reverse results if requested (highest score last) - only affects display
        display_results = filtered_results
        if self.config.reverse:
            display_results = list(reversed(filtered_results))

        # Format output with potentially reversed results
        formatted_output, tree_output = self.format_output(display_results, engine)

        return SearchResult(
            results=raw_results,
            filtered_results=filtered_results,
            used_tokens=used_tokens,
            formatted_output=formatted_output,
            tree_output=tree_output,
            stats=stats,
            semantic_used=semantic_used,
        )


def execute_search_pipeline(
    root_path: Path,
    query: str = "",
    token_limit: int = 8000,
    output_format: str = "xml",
    enable_semantic: bool = False,
    model_path: Optional[str] = None,
    show_deps: bool = False,
    dir_tree: bool = False,
    show_map: bool = False,
    find_mode: Optional[str] = None,
    limit: int = 20,
    verbose: bool = False,
    reverse: bool = False,
) -> dict:
    """
    Convenience function to execute search pipeline with simple parameters.

    This provides a simple functional interface for common use cases.

    Args:
        root_path: Path to the source code root
        query: Search query string
        token_limit: Maximum tokens for output
        output_format: Output format (xml, markdown, tree, json)
        enable_semantic: Enable semantic search
        model_path: Path to semantic model
        show_deps: Show dependency context
        dir_tree: Generate directory tree
        show_map: Force map output even with a query (auto-on when no query)
        find_mode: Identifier to find (enables find mode)
        limit: Maximum results to return
        verbose: Enable verbose output
        reverse: Reverse result order (highest score last)

    Returns:
        Dictionary with search results and metadata
    """
    config = SearchConfig(
        root_path=root_path,
        query=query,
        token_limit=token_limit,
        output_format=OutputFormat(output_format),
        enable_semantic=enable_semantic,
        model_path=model_path,
        show_deps=show_deps,
        dir_tree=dir_tree,
        show_map=show_map,
        find_mode=find_mode,
        limit=limit,
        verbose=verbose,
        reverse=reverse,
    )

    pipeline = SearchPipeline(config)
    result = pipeline.execute()

    return {
        "output": result.formatted_output,
        "tree": result.tree_output,
        "stats": result.stats,
        "result_count": len(result.filtered_results),
        "total_count": len(result.results),
        "used_tokens": result.used_tokens,
        "semantic_used": result.semantic_used,
    }
