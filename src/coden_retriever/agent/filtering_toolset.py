"""Filtering wrapper for pydantic-ai toolsets.

Uses pydantic-ai's built-in FilteredToolset to filter which tools are visible
to the LLM based on LLM-routed relevance to the current query.

This is used with dynamic_tool_filtering to reduce context window usage
and focus the LLM on relevant tools.
"""

from typing import Any, Callable, Optional

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FilteredToolset, AbstractToolset
from pydantic_ai.tools import ToolDefinition

from ..mcp.llm_tool_router import LLMToolRouter
from ..mcp.tool_filter import FilterResult


class LLMToolFilter:
    """Manages LLM-based filtering state for tool selection.

    This class holds the filter state (current query, allowed tools) and provides
    the filter function that pydantic-ai's FilteredToolset uses.

    The filter is updated per-query via set_filter_for_query() before agent.run().
    """

    def __init__(self, tool_router: Optional[LLMToolRouter] = None):
        """Initialize the LLM tool filter.

        Args:
            tool_router: LLMToolRouter instance for LLM-based filtering.
        """
        self.tool_router = tool_router
        self._allowed_tools: Optional[set[str]] = None

    async def set_filter_for_query(
        self,
        query: str,
        event_stream_handler: Optional[Callable[..., Any]] = None,
    ) -> FilterResult | None:
        """Update the allowed tools based on the query.

        Call this before agent.run() to filter tools for that query.

        Args:
            query: The user's query text.
            event_stream_handler: Optional pydantic-ai event stream handler
                for streaming the router's LLM output to the console.

        Returns:
            FilterResult from the router, or None if no router is configured.
        """
        if self.tool_router is None:
            self._allowed_tools = None
            return None

        filter_result = await self.tool_router.filter(
            query, event_stream_handler=event_stream_handler
        )
        allowed = {tool.metadata.name for tool in filter_result.all_tools}
        self._allowed_tools = allowed
        return filter_result

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
    tool_router: Optional[LLMToolRouter] = None,
) -> tuple[FilteredToolset, LLMToolFilter]:
    """Create a filtered toolset using pydantic-ai's built-in FilteredToolset.

    Args:
        toolset: The base toolset to wrap.
        tool_router: LLMToolRouter instance for LLM-based filtering.

    Returns:
        Tuple of (FilteredToolset, LLMToolFilter) - the filter object
        is returned so callers can call set_filter_for_query() per request.
    """
    llm_filter = LLMToolFilter(tool_router=tool_router)

    filtered_toolset = FilteredToolset(
        wrapped=toolset,
        filter_func=llm_filter.filter_func,
    )

    return filtered_toolset, llm_filter
