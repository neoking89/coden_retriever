"""
Semantic tool filtering module.

Implements semantic filtering of MCP tools using Model2Vec embeddings to reduce
context window usage by only including relevant tools for a given query.

The filtering works purely on semantic similarity between the user's query
and tool descriptions - no keyword-based rules.

Users can adjust the threshold via:
- Environment variable: CODEN_RETRIEVER_TOOL_FILTER_THRESHOLD
- Config setting: /config set tool_filter_threshold <value>

Requires the 'semantic' extra:
    pip install 'coden-retriever[semantic]'
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING, cast

from rich.console import Console

from ..search.semantic import get_cached_model
from ..utils.optional_deps import get_numpy

if TYPE_CHECKING:
    import numpy as np
    from model2vec import StaticModel

logger = logging.getLogger(__name__)

# Default model path (same as used for code search)
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "embeddings" / "model2vec_embed_distill"

# Environment variable names for configuration
ENV_TOOL_FILTER_ENABLED = "CODEN_RETRIEVER_TOOL_FILTER_ENABLED"
ENV_TOOL_FILTER_THRESHOLD = "CODEN_RETRIEVER_TOOL_FILTER_THRESHOLD"
ENV_TOOL_FILTER_TOP_K = "CODEN_RETRIEVER_TOOL_FILTER_TOP_K"
ENV_TOOL_FILTER_MIN_TOOLS = "CODEN_RETRIEVER_TOOL_FILTER_MIN_TOOLS"

DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 50  # High limit - let threshold do the filtering
DEFAULT_MIN_TOOLS = 1  # Minimal safety net - threshold is the primary filter

# CORE tools - always shown, never filtered.
# These are essential tools that the LLM needs for basic code operations.
CORE_TOOLS = frozenset({
    # Code Discovery - essential for exploring codebases
    "code_search",
    "code_map",
    # Symbol Lookup - essential for finding definitions
    "find_identifier",
    # Code Inspection - essential for reading code
    "read_source_range",
    "read_source_ranges",
    # File Editing - essential for modifying code
    "write_file",
    "edit_file",
    "delete_file",
    "undo_file_change",
})



def is_tool_filter_enabled() -> bool:
    """Check if semantic tool filtering is enabled via environment variable."""
    return os.environ.get(ENV_TOOL_FILTER_ENABLED, "").lower() in ("1", "true", "yes")


def get_filter_threshold() -> float:
    """Get the similarity threshold from environment variable or default.

    Returns:
        Threshold value between 0.0 and 1.0.
    """
    try:
        threshold = float(os.environ.get(ENV_TOOL_FILTER_THRESHOLD, DEFAULT_THRESHOLD))
        # Clamp to valid range
        if threshold < 0.0:
            logger.warning(f"Threshold {threshold} is below 0.0, clamping to 0.0")
            return 0.0
        if threshold > 1.0:
            logger.warning(f"Threshold {threshold} is above 1.0, clamping to 1.0")
            return 1.0
        return threshold
    except ValueError:
        return DEFAULT_THRESHOLD


def get_filter_top_k() -> int:
    """Get the top_k limit from environment variable or default."""
    try:
        return int(os.environ.get(ENV_TOOL_FILTER_TOP_K, DEFAULT_TOP_K))
    except ValueError:
        return DEFAULT_TOP_K


def get_filter_min_tools() -> int:
    """Get the minimum number of tools to return from environment variable or default."""
    try:
        return int(os.environ.get(ENV_TOOL_FILTER_MIN_TOOLS, DEFAULT_MIN_TOOLS))
    except ValueError:
        return DEFAULT_MIN_TOOLS


@dataclass
class ToolMetadata:
    """
    Metadata about a tool for semantic filtering.

    Attributes:
        name: The tool function name.
        description: The tool's docstring/description.
        category: Optional category for grouping (e.g., "Code Discovery").
        parameters_schema: Optional JSON schema of parameters for enhanced embedding.
        example_code: Optional code example for code-aware embeddings.
    """
    name: str
    description: str
    category: str = ""
    parameters_schema: str = ""
    example_code: str = ""

    @property
    def embedding_text(self) -> str:
        """
        Generate the text to embed for this tool.

        Uses Name + Description format (Option B from analysis) as the default.
        This provides a good balance of keyword matching and semantic meaning.
        """
        return f"{self.name}: {self.description}"

    @property
    def enhanced_embedding_text(self) -> str:
        """
        Generate enhanced text including code context for embedding.

        Uses Name + Description + Code Example format (Option C from analysis).
        Better for tools with clear usage patterns, but may dilute signal for others.
        """
        parts = [f"Tool: {self.name}", f"Description: {self.description}"]
        if self.parameters_schema:
            parts.append(f"Parameters: {self.parameters_schema}")
        if self.example_code:
            parts.append(f"Example:\n{self.example_code}")
        return "\n".join(parts)


@dataclass
class FilteredTool:
    """Result of tool filtering with score information."""
    metadata: ToolMetadata
    score: float
    is_core: bool = False  # True if this is a CORE tool (always shown)


@dataclass
class FilterResult:
    """Result containing both core and filtered domain tools."""
    core_tools: list[FilteredTool]  # CORE tools (always shown)
    domain_tools: list[FilteredTool]  # Domain-specific tools (filtered by query)

    @property
    def all_tools(self) -> list[FilteredTool]:
        """Get all tools (core + domain) for backwards compatibility."""
        return self.core_tools + self.domain_tools

    def __len__(self) -> int:
        return len(self.core_tools) + len(self.domain_tools)


class ToolFilter:
    """
    Semantic tool filter using Model2Vec embeddings.

    Separates tools into two categories:
    - CORE tools: Always shown (code_search, code_map, read/write, etc.)
    - Domain tools: Filtered by semantic similarity to the query

    Only domain-specific tools go through semantic filtering.

    Example:
        >>> tools = [ToolMetadata("code_search", "Search code..."), ...]
        >>> filter = ToolFilter(tools)
        >>> result = filter.filter("find graph dependencies")
        >>> print(f"Core: {len(result.core_tools)}, Domain: {len(result.domain_tools)}")
    """

    def __init__(
        self,
        tools: list[ToolMetadata],
        model_path: str | Path | None = None,
        core_tools: frozenset[str] | None = None,
        use_enhanced_embedding: bool = False,
    ):
        """
        Initialize the tool filter.

        Args:
            tools: List of ToolMetadata objects describing available tools.
            model_path: Path to the Model2Vec model. Uses default if not specified.
            core_tools: Set of core tool names (always shown). Uses CORE_TOOLS if None.
            use_enhanced_embedding: If True, use enhanced embedding with code examples.
        """
        self.all_tools = {t.name: t for t in tools}
        self.model_path = str(model_path or _DEFAULT_MODEL_PATH)
        self.core_tool_names = core_tools if core_tools is not None else CORE_TOOLS
        self.use_enhanced_embedding = use_enhanced_embedding

        # Separate core and domain tools
        self.core_tools: dict[str, ToolMetadata] = {}
        self.domain_tools: dict[str, ToolMetadata] = {}

        for name, tool in self.all_tools.items():
            if name in self.core_tool_names:
                self.core_tools[name] = tool
            else:
                self.domain_tools[name] = tool

        self._model: StaticModel | None = None
        self._embeddings: np.ndarray | None = None
        self._domain_tool_names: list[str] = []

        # Pre-compute embeddings for domain tools only
        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        """Pre-compute normalized embeddings for domain tools only."""
        if not self.domain_tools:
            logger.info("No domain tools to filter - all tools are core")
            return

        # Load model using cached loader
        self._model = get_cached_model(self.model_path)

        # Extract domain tool names and texts in consistent order
        self._domain_tool_names = list(self.domain_tools.keys())

        if self.use_enhanced_embedding:
            texts = [self.domain_tools[name].enhanced_embedding_text for name in self._domain_tool_names]
        else:
            texts = [self.domain_tools[name].embedding_text for name in self._domain_tool_names]

        logger.info(f"Computing embeddings for {len(texts)} domain tools (skipping {len(self.core_tools)} core tools)...")

        # Generate embeddings (model is set on line 233 via get_cached_model)
        model = cast("StaticModel", self._model)
        self._embeddings = model.encode(texts)

        # Normalize for cosine similarity (dot product of normalized vectors)
        np = get_numpy()
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        self._embeddings = self._embeddings / norms

        logger.info(f"Domain tool embeddings computed ({self._embeddings.shape})")

    def filter(
        self,
        query: str,
        threshold: float | None = None,
        top_k: int | None = None,
        min_tools: int | None = None,
    ) -> FilterResult:
        """
        Filter domain tools by semantic similarity to query.

        CORE tools are always returned (not filtered).
        Only domain-specific tools go through semantic filtering.

        Args:
            query: Natural language query describing the task.
            threshold: Similarity threshold (0-1) for domain tools. Values outside
                       this range will be clamped.
            top_k: Maximum number of domain tools to return (must be >= 0).
            min_tools: Minimum number of domain tools to return (must be >= 0).

        Returns:
            FilterResult with core_tools and domain_tools lists.
        """
        # Use provided values or fall back to config/defaults
        threshold = threshold if threshold is not None else get_filter_threshold()
        top_k = top_k if top_k is not None else get_filter_top_k()
        min_tools = min_tools if min_tools is not None else get_filter_min_tools()

        # Validate and clamp threshold to [0.0, 1.0]
        threshold = max(0.0, min(1.0, threshold))

        # Validate top_k and min_tools are non-negative
        top_k = max(0, top_k)
        min_tools = max(0, min_tools)

        # Build core tools list (always included, no scores needed)
        core_results = [
            FilteredTool(metadata=tool, score=1.0, is_core=True)
            for tool in self.core_tools.values()
        ]

        # If no domain tools or no embeddings, return only core tools
        if not self.domain_tools or self._embeddings is None or self._model is None:
            return FilterResult(core_tools=core_results, domain_tools=[])

        # Handle empty query - return only core tools
        if not query.strip():
            return FilterResult(core_tools=core_results, domain_tools=[])

        # Encode and normalize query
        np = get_numpy()
        query_vec = self._model.encode([query])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            logger.warning("Query embedding is zero vector")
            return FilterResult(core_tools=core_results, domain_tools=[])
        query_vec = query_vec / query_norm

        # Compute cosine similarities for domain tools
        scores = np.dot(self._embeddings, query_vec)

        # Get indices sorted by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Build domain results - tools above threshold (up to top_k)
        domain_results: list[FilteredTool] = []
        for idx in sorted_indices:
            if len(domain_results) >= top_k:
                break

            name = self._domain_tool_names[idx]
            score = float(scores[idx])

            if score >= threshold:
                domain_results.append(FilteredTool(
                    metadata=self.domain_tools[name],
                    score=score,
                    is_core=False,
                ))

        # Ensure minimum number of domain tools
        if len(domain_results) < min_tools:
            for idx in sorted_indices:
                if len(domain_results) >= min_tools:
                    break

                name = self._domain_tool_names[idx]
                if not any(r.metadata.name == name for r in domain_results):
                    domain_results.append(FilteredTool(
                        metadata=self.domain_tools[name],
                        score=float(scores[idx]),
                        is_core=False,
                    ))

        logger.debug(
            f"Filter result: {len(core_results)} core + {len(domain_results)} domain tools "
            f"(threshold={threshold}, query='{query[:50]}...')"
        )

        return FilterResult(core_tools=core_results, domain_tools=domain_results)

    def get_domain_scores(self, query: str) -> dict[str, float]:
        """
        Get similarity scores for domain tools only.

        Useful for debugging and threshold tuning.
        Core tools are not scored (they're always included).

        Args:
            query: Natural language query.

        Returns:
            Dictionary mapping domain tool name to similarity score.
        """
        if self._embeddings is None or self._model is None:
            return {}

        if not query.strip():
            return {name: 0.0 for name in self._domain_tool_names}

        # Encode and normalize query
        np = get_numpy()
        query_vec = self._model.encode([query])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return {name: 0.0 for name in self._domain_tool_names}
        query_vec = query_vec / query_norm

        # Compute similarities
        scores = np.dot(self._embeddings, query_vec)

        return {
            name: float(score)
            for name, score in zip(self._domain_tool_names, scores)
        }


def extract_tool_metadata(func: Callable[..., Any]) -> ToolMetadata:
    """
    Extract ToolMetadata from a tool function.

    Extracts name from __name__ and description from __doc__.

    Args:
        func: The tool function (typically decorated with @mcp.tool()).

    Returns:
        ToolMetadata with name and description populated.
    """
    name = func.__name__
    description = func.__doc__ or ""

    # Clean up docstring - take first paragraph or full text
    description = description.strip()

    return ToolMetadata(name=name, description=description)


def create_tool_filter_from_functions(
    funcs: list[Callable[..., Any]],
    model_path: str | Path | None = None,
    core_tools: frozenset[str] | None = None,
) -> ToolFilter:
    """
    Create a ToolFilter from a list of tool functions.

    Convenience function that extracts metadata from functions and creates the filter.

    Args:
        funcs: List of tool functions.
        model_path: Path to Model2Vec model.
        core_tools: Set of core tool names (always shown).

    Returns:
        Configured ToolFilter instance.
    """
    tools = [extract_tool_metadata(func) for func in funcs]
    return ToolFilter(tools, model_path=model_path, core_tools=core_tools)


def display_filtered_tools(
    result: FilterResult,
    console: Console | None = None,
) -> None:
    """
    Display the filtered tools to the console in agent mode.

    Shows CORE tools (always available) and domain-specific tools
    (filtered by query) separately.

    Args:
        result: FilterResult from ToolFilter.filter().
        console: Rich Console instance. If None, creates a default one.
    """
    if console is None:
        console = Console()

    # Build core tools display (no scores, always shown)
    core_names = [f"[cyan]{t.metadata.name}[/cyan]" for t in result.core_tools]

    # Build domain tools display with scores
    domain_parts = []
    for t in result.domain_tools:
        name = t.metadata.name
        score_str = f"{t.score:.2f}"
        if t.score >= 0.7:
            domain_parts.append(f"[green]{name}[/green][dim]({score_str})[/dim]")
        elif t.score >= 0.4:
            domain_parts.append(f"[yellow]{name}[/yellow][dim]({score_str})[/dim]")
        else:
            domain_parts.append(f"[dim]{name}({score_str})[/dim]")

    # Display both sections
    console.print(f"[dim]>> Core tools ({len(result.core_tools)}):[/dim] " + ", ".join(core_names))
    if domain_parts:
        console.print(f"[dim]>> Domain tools ({len(result.domain_tools)}):[/dim] " + ", ".join(domain_parts))
    else:
        console.print("[dim]>> Domain tools: (none matched)[/dim]")
