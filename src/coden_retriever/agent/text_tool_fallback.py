"""Fallback parsing for text-based tool calls from smaller LLMs.

Some smaller models (e.g., qwen2.5-coder:7b) output tool calls as JSON text
instead of using the proper OpenAI function calling format. This module
provides the parser; `TextToolFallbackCapability` in `capabilities.py`
applies it inside pydantic-ai's run loop.

Supported formats:
- {"name": "tool_name", "arguments": {...}}
- {"tool": "tool_name", "args": {...}}
- {"function": "tool_name", "parameters": {...}}
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


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
    re.DOTALL,
)

# Matches: {"tool": "...", "args": {...}}
TOOL_CALL_PATTERN_TOOL = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
    re.DOTALL,
)

# Matches: {"function": "...", "parameters": {...}}
TOOL_CALL_PATTERN_FUNCTION = re.compile(
    r'\{\s*"function"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}',
    re.DOTALL,
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

    for pattern in (TOOL_CALL_PATTERN_NAME, TOOL_CALL_PATTERN_TOOL, TOOL_CALL_PATTERN_FUNCTION):
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

    if not tool_calls:
        tool_calls.extend(_parse_standalone_json(text))

    return tool_calls


def _parse_standalone_json(text: str) -> list[ParsedToolCall]:
    """Try to parse standalone JSON objects that look like tool calls.

    Handles formatting variations not caught by the regex patterns.
    """
    tool_calls: list[ParsedToolCall] = []

    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

    for match in json_pattern.finditer(text):
        json_str = match.group(0)
        try:
            obj = json.loads(json_str)
            if not isinstance(obj, dict):
                continue

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
    if '"name"' not in text and '"tool"' not in text and '"function"' not in text:
        return False

    if '"arguments"' not in text and '"args"' not in text and '"parameters"' not in text:
        return False

    return True


def extract_thinking_and_tool_call(text: str) -> tuple[str | None, list[ParsedToolCall]]:
    """Extract any thinking/reasoning text and tool calls from response.

    Some models output thinking before the tool call JSON. This separates them.

    Returns:
        Tuple of (thinking_text, tool_calls).
    """
    tool_calls = parse_text_tool_calls(text)

    if not tool_calls:
        return None, []

    first_json = tool_calls[0].raw_json
    json_start = text.find(first_json)

    thinking = None
    if json_start > 0:
        thinking = text[:json_start].strip()
        if not thinking:
            thinking = None

    return thinking, tool_calls
