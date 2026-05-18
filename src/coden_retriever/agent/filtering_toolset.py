"""Filtering wrapper for pydantic-ai toolsets.

Uses pydantic-ai's built-in `FilteredToolset` to expose only an allow-listed
subset of tools to the model per query. The library is agnostic about how
the allow-list is computed: callers pass a `ToolFilterFn` callable that
returns the set of tool names allowed for a given query string.

Plug in an LLM-based router, a rule-based filter, or a no-op as needed.
"""

from typing import Any, Optional

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset

from .protocols import ToolFilterFn


class ToolFilter:
    """Per-query state holder for `FilteredToolset`.

    `set_filter_for_query()` calls the injected `filter_fn` to refresh the
    allow-list; `filter_func` is the sync predicate pydantic-ai invokes
    per tool definition during the run.
    """

    def __init__(self, filter_fn: Optional[ToolFilterFn] = None) -> None:
        self.filter_fn = filter_fn
        self._allowed_tools: Optional[set[str]] = None

    async def set_filter_for_query(self, query: str, **_unused: Any) -> Optional[set[str]]:
        """Resolve and store the allow-list for this query.

        `**_unused` absorbs extra kwargs from older call sites so the hook
        signature can grow without breaking existing wiring.

        Returns the resolved set (or None if no filter is configured).
        """
        if self.filter_fn is None:
            self._allowed_tools = None
            return None
        self._allowed_tools = await self.filter_fn(query)
        return self._allowed_tools

    def apply_allowlist(self, names: Optional[set[str]]) -> None:
        """Set the allow-list directly when the caller already computed it.

        Use this when the same router decision feeds both this gate and a
        separate display path — avoids running the underlying router twice.
        """
        self._allowed_tools = names

    def clear_filter(self) -> None:
        """Drop the per-query allow-list (subsequent runs see all tools)."""
        self._allowed_tools = None

    def filter_func(self, ctx: RunContext[Any], tool_def: ToolDefinition) -> bool:
        """Predicate pydantic-ai's `FilteredToolset` invokes per tool."""
        if self._allowed_tools is None:
            return True
        return tool_def.name in self._allowed_tools


def create_filtered_toolset(
    toolset: AbstractToolset,
    filter_fn: Optional[ToolFilterFn] = None,
) -> tuple[FilteredToolset, ToolFilter]:
    """Wrap a toolset with a per-query filter.

    Returns the wrapped toolset plus the `ToolFilter` so callers can refresh
    the allow-list via `await tf.set_filter_for_query(query)` between turns.
    """
    tf = ToolFilter(filter_fn=filter_fn)
    return (
        FilteredToolset(wrapped=toolset, filter_func=tf.filter_func),
        tf,
    )
