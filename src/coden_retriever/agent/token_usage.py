"""Active-context size estimation for agent conversations.

Derives the retained active-context size from the latest model response when
available, falling back to a local estimate over the serialized history.
"""

import json
from dataclasses import dataclass
from typing import Sequence

from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ..token_estimator import count_tokens


_SEGMENT_SEPARATOR = "\n\n"


@dataclass
class ActiveContextUsage:
    """Retained active-context size after the current turn."""

    tokens: int = 0
    estimated: bool = True


def get_active_context_usage(messages: Sequence[ModelMessage]) -> ActiveContextUsage:
    """Get retained active-context tokens from the latest response if available."""
    last_usage = _get_last_response_usage(messages)
    if last_usage and (last_usage.input_tokens > 0 or last_usage.output_tokens > 0):
        return ActiveContextUsage(
            tokens=last_usage.input_tokens + last_usage.output_tokens,
            estimated=False,
        )

    history_tokens = estimate_history_tokens(messages)
    return ActiveContextUsage(tokens=history_tokens, estimated=True)


def estimate_history_tokens(messages: Sequence[ModelMessage]) -> int:
    """Estimate retained history tokens from the serialized message content."""
    segments: list[str] = []
    for message in messages:
        segments.extend(_message_segments(message))

    if not segments:
        return 0

    return count_tokens(_SEGMENT_SEPARATOR.join(segments), is_code=False)


def _get_last_response_usage(messages: Sequence[ModelMessage]) -> RequestUsage | None:
    """Return usage for the last model response in the message history."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return message.usage
    return None


def _message_segments(message: ModelMessage) -> list[str]:
    """Extract text segments from one model message."""
    segments: list[str] = []
    if isinstance(message, ModelRequest):
        has_system_prompt = any(
            isinstance(part, SystemPromptPart) for part in message.parts
        )
        if message.instructions and not has_system_prompt:
            segments.append(message.instructions)

    for part in message.parts:
        text = _part_text(part)
        if text:
            segments.append(text)

    return segments


def _part_text(part: object) -> str:
    """Serialize a message part into the text that contributes to context."""
    if isinstance(part, (SystemPromptPart, TextPart, ThinkingPart)):
        return part.content
    if isinstance(part, UserPromptPart):
        return _stringify_value(part.content)
    if isinstance(part, RetryPromptPart):
        return part.model_response()
    if isinstance(part, BaseToolCallPart):
        return _join_non_empty((part.tool_name, part.args_as_json_str()))
    if isinstance(part, BaseToolReturnPart):
        return _join_non_empty((part.tool_name, part.model_response_str()))
    return ""


def _stringify_value(value: object) -> str:
    """Convert mixed content payloads to plain text for token estimation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=True)
    return str(value)


def _join_non_empty(parts: Sequence[str]) -> str:
    """Join non-empty text fragments with the shared separator."""
    return _SEGMENT_SEPARATOR.join(part for part in parts if part)


def derive_baseline_from_first_turn(messages: Sequence[ModelMessage]) -> int | None:
    """Compute system+tools token count from the first API call's reported usage.

    Provider-reported input_tokens on the first response equals
    system_prompt + tool_definitions + first_user_prompt. Subtracting a local
    count of the user prompt yields a measured baseline that reflects what the
    provider actually tokenized (including any normalization it applied).

    Returns None if usage wasn't reported by the provider.
    """
    if not messages:
        return None

    first_request = messages[0]
    if not isinstance(first_request, ModelRequest):
        return None

    first_usage: RequestUsage | None = None
    for message in messages:
        if isinstance(message, ModelResponse) and message.usage and message.usage.input_tokens > 0:
            first_usage = message.usage
            break
    if first_usage is None:
        return None

    user_prompt_text = ""
    for part in first_request.parts:
        if isinstance(part, UserPromptPart):
            user_prompt_text = _stringify_value(part.content)
            break

    user_tokens = count_tokens(user_prompt_text, is_code=False) if user_prompt_text else 0
    return max(first_usage.input_tokens - user_tokens, 0)
