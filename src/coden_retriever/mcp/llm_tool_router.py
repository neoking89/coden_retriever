"""LLM-based tool routing for dynamic MCP tool filtering.

Replaces embedding-based filtering with a lightweight LLM subagent that
selects relevant tools based on the user's query. Uses the same model
the user already configured for the main agent.

The router prioritizes recall over precision — it's better to include
an extra tool than to miss a needed one.
"""
import logging
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ..config_loader import get_config, resolve_or_default
from ..constants import DEFAULT_TOOL_ROUTER_PROMPT_TEMPLATE
from .tool_filter import (
    CORE_TOOLS,
    TOOL_GROUPS,
    TOOL_QUERY_DESCRIPTIONS,
    FilterResult,
    FilteredTool,
    ToolMetadata,
)

logger = logging.getLogger(__name__)

# Router response cap — tool names only, no prose needed
MAX_ROUTER_TOKENS = 512

# Deterministic routing — no creativity needed for tool selection
ROUTER_TEMPERATURE = 0.0

# Fail fast — router should be lightweight, not a bottleneck
ROUTER_TIMEOUT = 30.0

# Fixed score for LLM-selected tools (binary selection, no ranking)
SELECTED_TOOL_SCORE = 1.0


class ToolSelectionResult(BaseModel):
    """Structured output from the LLM tool router."""

    selected_tools: list[str] = Field(
        description="List of relevant tool names from the catalog"
    )


class LLMToolRouter:
    """LLM-based tool router for dynamic MCP tool filtering.

    Uses the user's configured LLM as a lightweight router to decide
    which domain tools are relevant for a given query. CORE tools
    are always included. TOOL_GROUPS are expanded (if one member
    is selected, all members are included).
    """

    def __init__(
        self,
        tools: list[ToolMetadata],
        model: Any,
        core_tools: frozenset[str] | None = None,
    ):
        self.all_tools = {t.name: t for t in tools}
        self._model = model
        self.core_tool_names = core_tools if core_tools is not None else CORE_TOOLS

        self.core_tools: dict[str, ToolMetadata] = {
            name: tool for name, tool in self.all_tools.items()
            if name in self.core_tool_names
        }
        self.domain_tools: dict[str, ToolMetadata] = {
            name: tool for name, tool in self.all_tools.items()
            if name not in self.core_tool_names
        }

        # Reverse lookup: tool name -> groups it belongs to
        self._tool_groups: dict[str, list[frozenset[str]]] = {}
        for group in TOOL_GROUPS.values():
            for tool_name in group:
                if tool_name in self.domain_tools:
                    self._tool_groups.setdefault(tool_name, []).append(group)

        self._tool_catalog = self._build_tool_catalog()

    def _build_tool_catalog(self) -> str:
        """Build compact tool catalog for the router prompt.

        Uses short query-style descriptions from TOOL_QUERY_DESCRIPTIONS
        when available, falling back to the tool's full description.
        """
        lines: list[str] = []
        for name, tool in self.domain_tools.items():
            desc = TOOL_QUERY_DESCRIPTIONS.get(name, "")
            if not desc:
                desc = tool.query_description or tool.description
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _expand_groups(self, selected: set[str]) -> set[str]:
        """Expand tool groups: if any member is selected, include all members.

        Ensures logically related tools (e.g. all debug tools) appear
        together — missing one from a group is worse than an extra tool.
        """
        expanded: set[str] = set(selected)
        for name in selected:
            for group in self._tool_groups.get(name, []):
                expanded |= {t for t in group if t in self.domain_tools}
        return expanded

    def _build_filter_result(self, selected_names: set[str]) -> FilterResult:
        """Construct FilterResult from the set of selected tool names."""
        core_results = [
            FilteredTool(metadata=tool, score=SELECTED_TOOL_SCORE, is_core=True)
            for tool in self.core_tools.values()
        ]
        domain_results = [
            FilteredTool(
                metadata=self.domain_tools[name],
                score=SELECTED_TOOL_SCORE,
                is_core=False,
            )
            for name in selected_names
            if name in self.domain_tools
        ]
        return FilterResult(core_tools=core_results, domain_tools=domain_results)

    def update_model(self, model: Any) -> None:
        """Update the model used for routing.

        Called when the user switches models via /model so the router
        doesn't keep using a stale (possibly unloaded) model.
        """
        self._model = model

    def _all_domain_result(self) -> FilterResult:
        """Fallback: return all domain tools (safe when router fails)."""
        return self._build_filter_result(set(self.domain_tools.keys()))

    async def filter(
        self,
        query: str,
        event_stream_handler: Optional[Callable[..., Any]] = None,
    ) -> FilterResult:
        """Filter domain tools by LLM-based relevance routing.

        Sends the query and tool catalog to the LLM, parses the
        structured response, expands tool groups, and returns a
        FilterResult. On any failure, falls back to including all
        domain tools.

        Args:
            query: Natural language query describing the user's task.
            event_stream_handler: Optional pydantic-ai event stream handler
                for streaming the router's LLM output to the console.

        Returns:
            FilterResult with core_tools (always) and domain_tools (selected).
        """
        if not self.domain_tools or not query.strip():
            return self._build_filter_result(set())

        try:
            return await self._run_router(query, event_stream_handler)
        except Exception as e:
            logger.warning(f"LLM tool router failed, including all tools: {e}")
            return self._all_domain_result()

    async def _run_router(
        self,
        query: str,
        event_stream_handler: Optional[Callable[..., Any]] = None,
    ) -> FilterResult:
        """Execute the LLM router call and build the result."""
        system_prompt = resolve_or_default(
            get_config().agent.tool_router_prompt_template,
            DEFAULT_TOOL_ROUTER_PROMPT_TEMPLATE,
            "tool_router_prompt_template",
        )
        router_agent: Agent[None, ToolSelectionResult] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=ToolSelectionResult,
            model_settings=ModelSettings(
                temperature=ROUTER_TEMPERATURE,
                max_tokens=MAX_ROUTER_TOKENS,
                timeout=ROUTER_TIMEOUT,
            ),
        )

        user_message = f"Task: {query}\n\nAvailable tools:\n{self._tool_catalog}\n\nSelect the relevant tool names."
        run_kwargs: dict[str, Any] = {}
        if event_stream_handler is not None:
            run_kwargs["event_stream_handler"] = event_stream_handler
        result = await router_agent.run(user_message, **run_kwargs)
        raw_names = set(result.output.selected_tools)

        # Filter out names not in domain_tools (LLM hallucination guard)
        valid_names = raw_names & set(self.domain_tools.keys())

        # Empty selection means the model failed to select tools (poor
        # structured output support) or returned only core tool names
        # which got filtered out. Fall back to all domain tools — same
        # rationale as the error path: false negatives are costly.
        if not valid_names:
            logger.warning(
                "Router returned no valid domain tools, including all"
            )
            return self._all_domain_result()

        expanded = self._expand_groups(valid_names)

        query_preview = query[:50] + ("..." if len(query) > 50 else "")
        logger.debug(
            f"Router selected {len(valid_names)} tools "
            f"(expanded to {len(expanded)} with groups, "
            f"query='{query_preview}')"
        )

        return self._build_filter_result(expanded)
