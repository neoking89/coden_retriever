"""Fallback parsing for text-based tool calls from smaller LLMs.

Some smaller models (e.g., qwen2.5-coder:7b) output tool calls as JSON text
instead of using the proper OpenAI function calling format. This module
provides utilities to detect and parse these text-based tool calls.

Supported formats:
- {"name": "tool_name", "arguments": {...}}
- {"tool": "tool_name", "args": {...}}
- Partial JSON at the end of text responses
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.messages import ModelMessage

    from .debug_logger import DebugLogger
    from .models import FallbackIterationResult, ReActStep

    # Type alias for permission checker callback
    # Returns: True (allowed), False (denied), None (always allow for session)
    PermissionChecker = Callable[[str, dict[str, Any]], Awaitable[bool | None]]

logger = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    """A tool call parsed from text output."""

    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str
    raw_json: str  # Original JSON string for debugging


# Regex patterns to match JSON tool call objects
# Matches: {"name": "...", "arguments": {...}}
TOOL_CALL_PATTERN_NAME = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
    re.DOTALL
)

# Matches: {"tool": "...", "args": {...}}
TOOL_CALL_PATTERN_TOOL = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
    re.DOTALL
)

# Matches: {"function": "...", "parameters": {...}}
TOOL_CALL_PATTERN_FUNCTION = re.compile(
    r'\{\s*"function"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
    re.DOTALL
)


def parse_text_tool_calls(text: str) -> list[ParsedToolCall]:
    """Parse tool calls from text output.

    Attempts to find JSON-formatted tool calls in the text using multiple
    patterns. Returns all found tool calls.

    Args:
        text: The text response from the model.

    Returns:
        List of ParsedToolCall objects found in the text.
    """
    tool_calls: list[ParsedToolCall] = []

    # Try each pattern
    for pattern in [TOOL_CALL_PATTERN_NAME, TOOL_CALL_PATTERN_TOOL, TOOL_CALL_PATTERN_FUNCTION]:
        for match in pattern.finditer(text):
            tool_name = match.group(1)
            args_json = match.group(2)

            try:
                arguments = json.loads(args_json)
                if not isinstance(arguments, dict):
                    continue

                tool_calls.append(ParsedToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_call_id=f"fallback_{uuid.uuid4().hex[:8]}",
                    raw_json=match.group(0),
                ))
            except json.JSONDecodeError:
                continue

    # Also try to find standalone JSON objects that look like tool calls
    if not tool_calls:
        tool_calls.extend(_parse_standalone_json(text))

    return tool_calls


def _parse_standalone_json(text: str) -> list[ParsedToolCall]:
    """Try to parse standalone JSON objects that look like tool calls.

    This handles cases where the model outputs the JSON with slight formatting
    variations not caught by the regex patterns.
    """
    tool_calls: list[ParsedToolCall] = []

    # Find all JSON-like objects in the text
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

    for match in json_pattern.finditer(text):
        json_str = match.group(0)
        try:
            obj = json.loads(json_str)
            if not isinstance(obj, dict):
                continue

            # Check for various tool call formats
            tool_name = None
            arguments = None

            if "name" in obj and "arguments" in obj:
                tool_name = obj["name"]
                arguments = obj["arguments"]
            elif "tool" in obj and "args" in obj:
                tool_name = obj["tool"]
                arguments = obj["args"]
            elif "function" in obj and "parameters" in obj:
                tool_name = obj["function"]
                arguments = obj["parameters"]

            if tool_name and isinstance(arguments, dict):
                tool_calls.append(ParsedToolCall(
                    tool_name=str(tool_name),
                    arguments=arguments,
                    tool_call_id=f"fallback_{uuid.uuid4().hex[:8]}",
                    raw_json=json_str,
                ))
        except json.JSONDecodeError:
            continue

    return tool_calls


def contains_tool_call(text: str) -> bool:
    """Quick check if text might contain a tool call.

    Faster than full parsing for filtering.
    """
    # Quick heuristic checks
    if '"name"' not in text and '"tool"' not in text and '"function"' not in text:
        return False

    if '"arguments"' not in text and '"args"' not in text and '"parameters"' not in text:
        return False

    return True


def extract_thinking_and_tool_call(text: str) -> tuple[str | None, list[ParsedToolCall]]:
    """Extract any thinking/reasoning text and tool calls from response.

    Some models output thinking before the tool call JSON.
    This separates them.

    Returns:
        Tuple of (thinking_text, tool_calls)
    """
    tool_calls = parse_text_tool_calls(text)

    if not tool_calls:
        return None, []

    # Find where the first tool call JSON starts
    first_json = tool_calls[0].raw_json
    json_start = text.find(first_json)

    thinking = None
    if json_start > 0:
        thinking = text[:json_start].strip()
        if not thinking:
            thinking = None

    return thinking, tool_calls


@dataclass
class FallbackExecutionResult:
    """Result of executing fallback tool calls."""

    tool_results: list[str]  # Formatted results for continuation prompt
    steps: list[ReActStep]  # ReAct steps for display
    tool_call_count: int  # Number of tool calls executed


async def execute_fallback_tool_calls(
    server: MCPServerStdio,
    text: str,
    step_number_start: int = 0,
    debug_logger: DebugLogger | None = None,
    ask_permission: PermissionChecker | None = None,
) -> FallbackExecutionResult | None:
    """Execute text-based tool calls and return results.

    This is the shared implementation used by both run_with_react_display()
    and _run_query() to avoid code duplication.

    Args:
        server: MCP server with direct_call_tool method.
        text: Text containing potential tool calls.
        step_number_start: Starting step number for ReAct steps.
        debug_logger: Optional debug logger for logging tool calls.
        ask_permission: Optional async callback to check permission before
            executing each tool. Returns True (allowed), False (denied),
            or None (always allow for session). If not provided, tools
            execute without permission checks.

    Returns:
        FallbackExecutionResult if tool calls were found and executed,
        None if no tool calls were found or parsing failed.
    """
    # Import here to avoid circular imports
    from .models import Action, Observation, ReActStep, Thought

    if not contains_tool_call(text):
        return None

    thinking, tool_calls = extract_thinking_and_tool_call(text)

    if not tool_calls:
        return None

    tool_results: list[str] = []
    steps: list[ReActStep] = []
    step_number = step_number_start

    for tc in tool_calls:
        step_number += 1

        # Create ReAct step for this fallback tool call
        step = ReActStep(
            step_number=step_number,
            thought=Thought(
                reasoning=thinking or f"Calling {tc.tool_name} (fallback)",
                next_action=f"Execute {tc.tool_name}",
            ),
            action=Action(tool_name=tc.tool_name, tool_input=tc.arguments),
        )

        # Log the tool call if logger provided
        if debug_logger:
            debug_logger.log_tool_call(tc.tool_name, tc.arguments, tc.tool_call_id)

        # Check permission before executing
        if ask_permission is not None:
            if not await ask_permission(tc.tool_name, tc.arguments):
                # Permission denied
                result_str = (
                    f"[TOOL DENIED] The user denied permission to execute tool "
                    f"'{tc.tool_name}'. Please acknowledge this and continue "
                    "without using this tool, or try a different approach."
                )
                logger.info("Fallback tool call denied by user: %s", tc.tool_name)
                if debug_logger:
                    debug_logger.log_tool_result(tc.tool_name, result_str, success=False)

                step.observation = Observation(
                    tool_name=tc.tool_name,
                    result=None,
                    success=False,
                    error=result_str,
                )
                steps.append(step)
                tool_results.append(f"Tool {tc.tool_name} failed:\n{result_str}")
                continue

        # Execute the tool
        try:
            result_content = await server.direct_call_tool(
                tc.tool_name, tc.arguments
            )
            success = True
            result_str = str(result_content)
            if debug_logger:
                debug_logger.log_tool_result(tc.tool_name, result_content, success=True)
        except Exception as e:
            success = False
            result_str = f"Error: {e}"
            logger.warning("Fallback tool call failed: %s - %s", tc.tool_name, e)
            if debug_logger:
                debug_logger.log_tool_result(tc.tool_name, result_str, success=False)

        step.observation = Observation(
            tool_name=tc.tool_name,
            result=result_str if success else None,
            success=success,
            error=result_str if not success else None,
        )
        steps.append(step)

        # Format result for the model
        if success:
            tool_results.append(f"Tool {tc.tool_name} returned:\n{result_str}")
        else:
            tool_results.append(f"Tool {tc.tool_name} failed:\n{result_str}")

    return FallbackExecutionResult(
        tool_results=tool_results,
        steps=steps,
        tool_call_count=len(tool_calls),
    )


def build_continuation_prompt(tool_results: list[str]) -> str:
    """Build continuation prompt from tool results.

    Args:
        tool_results: List of formatted tool result strings.

    Returns:
        Continuation prompt string for the model.
    """
    continuation_msg = "\n\n".join(tool_results)
    continuation_msg += "\n\nPlease continue based on these results."
    return continuation_msg


async def handle_fallback_iteration(
    server: MCPServerStdio | None,
    answer_text: str,
    total_tool_calls: int,
    all_messages: list[ModelMessage],
    step_number_start: int = 0,
    debug_logger: DebugLogger | None = None,
    ask_permission: PermissionChecker | None = None,
) -> FallbackIterationResult:
    """Check for text-based tool calls and execute fallback if needed.

    This handles the common pattern where models output tool calls as text
    instead of using proper function calling format.

    Args:
        server: MCP server instance for executing tools.
        answer_text: The model's response text to check for tool calls.
        total_tool_calls: Number of proper tool calls already made.
        all_messages: Current message history.
        step_number_start: Starting step number for fallback steps.
        debug_logger: Optional logger for debugging.
        ask_permission: Optional permission checker for tool execution.

    Returns:
        FallbackIterationResult indicating whether to continue and updated state.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from .models import FallbackIterationResult

    if total_tool_calls > 0 or not answer_text or not contains_tool_call(answer_text):
        return FallbackIterationResult(should_continue=False)

    if server is None:
        return FallbackIterationResult(should_continue=False)

    fallback_result = await execute_fallback_tool_calls(
        server=server,
        text=answer_text,
        step_number_start=step_number_start,
        debug_logger=debug_logger,
        ask_permission=ask_permission,
    )

    if fallback_result is None:
        return FallbackIterationResult(should_continue=False)

    continuation_msg = build_continuation_prompt(fallback_result.tool_results)
    updated_history = all_messages + [
        ModelRequest(parts=[UserPromptPart(content=continuation_msg)])
    ]

    return FallbackIterationResult(
        should_continue=True,
        updated_history=updated_history,
        continuation_prompt=continuation_msg,
        steps=fallback_result.steps,
        tool_call_count=fallback_result.tool_call_count,
    )
