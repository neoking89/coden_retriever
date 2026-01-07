"""ReAct loop utilities for the coding agent.

Extracts and formats reasoning steps from pydantic-ai message history.
The actual ReAct loop is handled automatically by agent.run().

Includes fallback support for models that output tool calls as text
instead of using the proper function calling format.
"""

import json
from typing import Any

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from .models import Action, AgentResponse, Observation, ReActStep, Thought
from .text_tool_fallback import contains_tool_call, handle_fallback_iteration


def extract_tool_calls(message: ModelResponse) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from a model response."""
    tool_calls = []
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            # Parse arguments - could be string or dict
            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append((part.tool_name, args, part.tool_call_id))
    return tool_calls


def extract_tool_results(message: ModelRequest) -> dict[str, tuple[Any, bool]]:
    """Extract tool results from a model request (which contains tool returns)."""
    results = {}
    for part in message.parts:
        if isinstance(part, ToolReturnPart):
            # Check if result indicates an error
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
    pending_tool_calls: dict[str, tuple[str, dict]] = {}  # tool_call_id -> (name, args)

    for msg in messages:
        if isinstance(msg, ModelResponse):
            # Model made a decision - extract any tool calls
            tool_calls = extract_tool_calls(msg)

            if tool_calls:
                # Model is calling tools - this is an action step
                for tool_name, tool_args, tool_call_id in tool_calls:
                    step_number += 1
                    pending_tool_calls[tool_call_id] = (tool_name, tool_args)

                    # Create step with action (observation will be filled later)
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
            # This may contain tool results
            tool_results = extract_tool_results(msg)

            for tool_call_id, (result, success) in tool_results.items():
                if tool_call_id in pending_tool_calls:
                    tool_name, _ = pending_tool_calls[tool_call_id]

                    # Find the corresponding step and add observation
                    for step in reversed(steps):
                        if (
                            step.action
                            and step.action.tool_name == tool_name
                            and step.observation is None
                        ):
                            # Show full results (no truncation)
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


def _get_text_from_response(message: ModelResponse) -> str | None:
    """Extract text content from a model response."""
    for part in message.parts:
        if isinstance(part, TextPart):
            return part.content
    return None


def _response_has_tool_calls(message: ModelResponse) -> bool:
    """Check if response has proper API-level tool calls."""
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            return True
    return False


async def run_with_react_display(
    agent: Agent,
    prompt: str,
    message_history: list[ModelMessage] | None = None,
    max_steps: int = 10,
    server=None,
) -> AgentResponse:
    """Run agent and extract ReAct steps from message history.

    Includes fallback support for models that output tool calls as text
    instead of using proper function calling. When a text-based tool call
    is detected, it's executed manually and the loop continues.

    Args:
        agent: The pydantic-ai agent to run.
        prompt: User prompt to send.
        message_history: Optional conversation history for multi-turn.
        max_steps: Maximum tool calls allowed (for safety).
        server: Optional MCP server for fallback tool execution.
                Required for text-based tool call fallback.

    Returns:
        AgentResponse with answer, steps, and message history.
    """
    current_history = message_history
    fallback_steps: list[ReActStep] = []
    fallback_tool_calls = 0
    all_messages: list[ModelMessage] = []

    for iteration in range(max_steps):
        # Run the agent
        result = await agent.run(
            prompt if iteration == 0 else "",  # Only send prompt on first iteration
            message_history=current_history,
            usage_limits=UsageLimits(request_limit=max_steps * 2),
        )

        all_messages = result.all_messages()

        # Check the last response for text-based tool calls
        last_response = None
        for msg in reversed(all_messages):
            if isinstance(msg, ModelResponse):
                last_response = msg
                break

        if last_response is None:
            break

        # If the response has proper tool calls, no fallback needed
        if _response_has_tool_calls(last_response):
            # Normal flow - agent handled tool calls properly
            steps = parse_messages_to_steps(all_messages)
            total_tool_calls = sum(1 for step in steps if step.action is not None)

            return AgentResponse(
                answer=str(result.output),
                steps=fallback_steps + steps,
                total_tool_calls=fallback_tool_calls + total_tool_calls,
                reached_max_steps=(fallback_tool_calls + total_tool_calls) >= max_steps,
                messages=all_messages,
            )

        # Check for text-based tool calls using the shared helper
        text_content = _get_text_from_response(last_response) or ""

        # Special handling when server is not available for fallback
        if server is None and contains_tool_call(text_content):
            steps = parse_messages_to_steps(all_messages)
            return AgentResponse(
                answer=f"[Model attempted tool call but fallback not available]\n{text_content}",
                steps=fallback_steps + steps,
                total_tool_calls=fallback_tool_calls,
                reached_max_steps=False,
                messages=all_messages,
            )

        # Use shared helper to handle fallback iteration
        fallback_iter_result = await handle_fallback_iteration(
            server=server,
            answer_text=text_content,
            total_tool_calls=0,  # We know there are no proper tool calls
            all_messages=all_messages,
            step_number_start=len(fallback_steps),
        )

        if fallback_iter_result.should_continue:
            # Continue fallback loop
            fallback_steps.extend(fallback_iter_result.steps)
            fallback_tool_calls += fallback_iter_result.tool_call_count
            current_history = fallback_iter_result.updated_history
            prompt = fallback_iter_result.continuation_prompt
            continue

        # No fallback needed - return final answer
        steps = parse_messages_to_steps(all_messages)
        total_tool_calls = sum(1 for step in steps if step.action is not None)

        return AgentResponse(
            answer=str(result.output),
            steps=fallback_steps + steps,
            total_tool_calls=fallback_tool_calls + total_tool_calls,
            reached_max_steps=(fallback_tool_calls + total_tool_calls) >= max_steps,
            messages=all_messages,
        )

    # Hit max iterations
    steps = parse_messages_to_steps(all_messages) if all_messages else []
    return AgentResponse(
        answer="[Max iterations reached]",
        steps=fallback_steps + steps,
        total_tool_calls=fallback_tool_calls,
        reached_max_steps=True,
        messages=current_history or [],
    )


def format_step_for_display(step: ReActStep) -> str:
    """Format a single ReAct step for console display."""
    lines = [f"\n[Step {step.step_number}]"]

    if step.thought:
        lines.append(f"  Thought: {step.thought.reasoning}")

    if step.action:
        # Format tool input nicely (no truncation - show full args)
        args_str = ", ".join(f"{k}={v!r}" for k, v in step.action.tool_input.items())
        lines.append(f"  Action: {step.action.tool_name}({args_str})")

    if step.observation:
        status = "✓" if step.observation.success else "✗"
        content = step.observation.result or step.observation.error
        lines.append(f"  Observation: {status} {content}")

    return "\n".join(lines)


def print_steps(steps: list[ReActStep]) -> None:
    """Print all ReAct steps to console with Rich formatting."""
    from .rich_console import print_steps_rich

    print_steps_rich(steps)
