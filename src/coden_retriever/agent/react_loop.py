"""ReAct loop utilities for the coding agent.

The actual ReAct loop is owned by pydantic-ai's run loop. This module just
extracts structured `ReActStep`s from the message history pydantic-ai
emits, so callers can render the reasoning chain.
"""

import json
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from .models import Action, Observation, ReActStep, Thought


def extract_tool_calls(message: ModelResponse) -> list[tuple[str, dict[str, Any], str]]:
    """Extract tool calls from a model response."""
    tool_calls: list[tuple[str, dict[str, Any], str]] = []
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            raw_args = part.args
            if isinstance(raw_args, str):
                try:
                    parsed_args: dict[str, Any] = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {"raw": raw_args}
            else:
                parsed_args = raw_args if isinstance(raw_args, dict) else {}
            tool_calls.append((part.tool_name, parsed_args, part.tool_call_id))
    return tool_calls


def extract_tool_results(message: ModelRequest) -> dict[str, tuple[Any, bool]]:
    """Extract tool results from a model request (which contains tool returns)."""
    results = {}
    for part in message.parts:
        if isinstance(part, ToolReturnPart):
            content = part.content
            is_error = False
            if isinstance(content, dict) and "error" in content:
                is_error = True
            elif isinstance(content, str) and content.startswith("Error:"):
                is_error = True
            results[part.tool_call_id] = (content, not is_error)
    return results


def parse_messages_to_steps(messages: list[ModelMessage]) -> list[ReActStep]:
    """Parse pydantic-ai message history into ReAct steps.

    The message history alternates between:
    - ModelRequest (user prompt or tool results)
    - ModelResponse (model reasoning + tool calls or final answer)
    """
    steps = []
    step_number = 0
    pending_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

    for msg in messages:
        if isinstance(msg, ModelResponse):
            tool_calls = extract_tool_calls(msg)

            if tool_calls:
                for tool_name, tool_args, tool_call_id in tool_calls:
                    step_number += 1
                    pending_tool_calls[tool_call_id] = (tool_name, tool_args)

                    step = ReActStep(
                        step_number=step_number,
                        thought=Thought(
                            reasoning=f"Calling {tool_name} to gather information",
                            next_action=f"Execute {tool_name}",
                        ),
                        action=Action(tool_name=tool_name, tool_input=tool_args),
                    )
                    steps.append(step)

        elif isinstance(msg, ModelRequest):
            tool_results = extract_tool_results(msg)

            for tool_call_id, (result, success) in tool_results.items():
                if tool_call_id in pending_tool_calls:
                    tool_name, _ = pending_tool_calls[tool_call_id]

                    for step in reversed(steps):
                        if (
                            step.action
                            and step.action.tool_name == tool_name
                            and step.observation is None
                        ):
                            result_str = str(result)
                            step.observation = Observation(
                                tool_name=tool_name,
                                result=result_str if success else None,
                                success=success,
                                error=result_str if not success else None,
                            )
                            break

                    del pending_tool_calls[tool_call_id]

    return steps
