"""Filtering wrapper for pydantic-ai toolsets.

Uses pydantic-ai's built-in FilteredToolset to filter which tools are visible
to the LLM based on semantic similarity to the current query.

This is used with dynamic_tool_filtering to reduce context window usage
and focus the LLM on relevant tools.
"""

from typing import Any, Optional

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FilteredToolset, AbstractToolset
from pydantic_ai.tools import ToolDefinition

from ..config_loader import get_config
from ..mcp.tool_filter import ToolFilter


class SemanticToolFilter:
    """Manages semantic filtering state for tool selection.

    This class holds the filter state (current query, allowed tools) and provides
    the filter function that pydantic-ai's FilteredToolset uses.

    The filter is updated per-query via set_filter_for_query() before agent.run().
    """

    def __init__(
        self,
        tool_filter: Optional[ToolFilter] = None,
        threshold: float = 0.5,
    ):
        """Initialize the semantic tool filter.

        Args:
            tool_filter: ToolFilter instance for semantic filtering.
            threshold: Similarity threshold for filtering (0-1).
        """
        self.tool_filter = tool_filter
        self.threshold = threshold
        self._allowed_tools: Optional[set[str]] = None

    def set_filter_for_query(self, query: str) -> set[str]:
        """Update the allowed tools based on the query.

        Call this before agent.run() to filter tools for that query.
        Reads threshold from current config for immediate updates.

        Args:
            query: The user's query text.

        Returns:
            Set of allowed tool names.
        """
        if self.tool_filter is None:
            self._allowed_tools = None
            return set()

        # Read threshold from config for immediate updates via /config set
        config = get_config()
        threshold = config.agent.tool_filter_threshold if config else self.threshold

        filter_result = self.tool_filter.filter(query, threshold=threshold)

        allowed = set()
        for tool in filter_result.core_tools:
            allowed.add(tool.metadata.name)
        for tool in filter_result.domain_tools:
            allowed.add(tool.metadata.name)

        self._allowed_tools = allowed
        return allowed

    def clear_filter(self) -> None:
        """Clear the filter, allowing all tools."""
        self._allowed_tools = None

    def filter_func(
        self,
        ctx: RunContext[Any],
        tool_def: ToolDefinition,
    ) -> bool:
        """Filter function for pydantic-ai's FilteredToolset.

        Returns True if the tool should be included, False otherwise.
        """
        if self._allowed_tools is None:
            return True
        return tool_def.name in self._allowed_tools


def create_filtered_toolset(
    toolset: AbstractToolset,
    tool_filter: Optional[ToolFilter] = None,
    threshold: float = 0.5,
) -> tuple[FilteredToolset, SemanticToolFilter]:
    """Create a filtered toolset using pydantic-ai's built-in FilteredToolset.

    Args:
        toolset: The base toolset to wrap.
        tool_filter: ToolFilter instance for semantic filtering.
        threshold: Similarity threshold for filtering.

    Returns:
        Tuple of (FilteredToolset, SemanticToolFilter) - the filter object
        is returned so callers can call set_filter_for_query() per request.
    """
    semantic_filter = SemanticToolFilter(
        tool_filter=tool_filter,
        threshold=threshold,
    )

    filtered_toolset = FilteredToolset(
        wrapped=toolset,
        filter_func=semantic_filter.filter_func,
    )

    return filtered_toolset, semantic_filter
