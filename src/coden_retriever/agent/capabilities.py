"""Capabilities that fold the wrapper's custom run-loop into pydantic-ai's own loop.

- `TextToolFallbackCapability` rewrites a model response whose only payload is
  a JSON tool-call emitted as text (the qwen/llama bug) into a proper
  `ToolCallPart`, so pydantic-ai's `CallToolsNode` dispatches the tool natively.
- `Sequential400RetryCapability` translates a provider-side malformed-tool-call
  HTTP 400 into a `ModelRetry(SEQUENTIAL_TOOL_HINT)`, so pydantic-ai's own
  retry-with-`RetryPromptPart` machinery handles the recovery — bounded by
  `Agent(retries=...)`.

Both hooks fire inside the framework's run loop. The custom outer
`for iteration in range(max_steps)` and `tool_error_retries` accumulator in
`coding_agent.py` / `query_executor.py` go away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext

from ._constants import (
    HTTP_BAD_REQUEST,
    INVALID_REQUEST_ERROR_TYPE,
    SEQUENTIAL_TOOL_HINT,
    TOOL_CALL_ERROR_KEYWORDS,
)
from .text_tool_fallback import extract_thinking_and_tool_call


@dataclass
class TextToolFallbackCapability(AbstractCapability[Any]):
    """Rewrite text-as-tool-call responses into proper ToolCallParts.

    Some smaller models (qwen2.5-coder:7b et al.) emit tool calls as JSON
    inside a `TextPart` instead of using OpenAI's function-call API. When
    that happens, pydantic-ai's `CallToolsNode` sees no `ToolCallPart` and
    treats the response as final text — the tool is never dispatched.

    By rewriting the response post-model and pre-tool-dispatch, the
    framework's normal loop picks the synthesized `ToolCallPart`, dispatches
    against the registered toolset, and feeds the result back to the model.
    No continuation prompt, no `handle_fallback_iteration` loop.
    """

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if any(isinstance(p, ToolCallPart) for p in response.parts):
            return response

        text_part = next((p for p in response.parts if isinstance(p, TextPart)), None)
        if text_part is None:
            return response

        thinking, tool_calls = extract_thinking_and_tool_call(text_part.content)
        if not tool_calls:
            return response

        # Preserve prose reasoning (if any) for cross-tool-call continuity per
        # Anthropic's thinking-block guidance. Always drop the raw JSON portion
        # to avoid re-feeding the text-as-tool-call anti-pattern.
        synthesized_parts: list[Any] = []
        if thinking:
            synthesized_parts.append(TextPart(content=thinking))
        synthesized_parts.extend(
            ToolCallPart(
                tool_name=tc.tool_name,
                args=tc.arguments,
                tool_call_id=tc.tool_call_id,
            )
            for tc in tool_calls
        )
        return ModelResponse(
            parts=synthesized_parts,
            usage=response.usage,
            model_name=response.model_name,
            timestamp=response.timestamp,
            provider_name=response.provider_name,
            provider_details=response.provider_details,
            finish_reason=response.finish_reason,
        )


@dataclass
class Sequential400RetryCapability(AbstractCapability[Any]):
    """Retry malformed-tool-call HTTP 400s with a single-tool steering hint.

    Some models (notably via Ollama's OpenAI shim) concatenate multiple tool
    names into a single call, which the provider rejects with HTTP 400.
    Raising `ModelRetry(SEQUENTIAL_TOOL_HINT)` from `on_model_request_error`
    triggers pydantic-ai's native `RetryPromptPart` flow — bounded by
    `Agent(retries=...)`.

    The body-keyword check prevents retrying unrelated 400s (e.g. invalid
    model name, context-window overflow, auth errors that also return 400).
    """

    async def on_model_request_error(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        if isinstance(error, ModelHTTPError) and is_tool_call_error(error):
            raise ModelRetry(SEQUENTIAL_TOOL_HINT)
        raise error


def is_tool_call_error(e: ModelHTTPError) -> bool:
    if e.status_code != HTTP_BAD_REQUEST:
        return False
    body = e.body if isinstance(e.body, dict) else {}
    if body.get("type") != INVALID_REQUEST_ERROR_TYPE:
        return False
    message = str(body.get("message", "")).lower()
    return any(kw in message for kw in TOOL_CALL_ERROR_KEYWORDS)
