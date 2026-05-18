"""Public `CodingAgent` wrapper around `pydantic_ai.Agent`.

Builds one long-lived `Agent` at construction time, wired with capabilities
that fold the text-tool fallback and HTTP-400 retry into pydantic-ai's own
run loop. There is no outer iteration loop: `agent.run()` / `agent.run_stream()`
each fire exactly once per call.

Supported model formats:
- `"ollama:model_name"`   - Ollama server
- `"llamacpp:model_name"` - llama-cpp-server
- `"openai:model_name"`   - Official OpenAI API
- `"model_name"` with `base_url=` - Any OpenAI-compatible endpoint
"""
from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, Callable, Optional, Sequence

from pydantic_ai import Agent, AgentStreamEvent, RunContext, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, TextPartDelta
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset

from ._constants import DEFAULT_MAX_RETRIES, DEFAULT_MAX_STEPS
from .capabilities import Sequential400RetryCapability, TextToolFallbackCapability
from .model_factory import ModelFactory
from .models import AgentResponse
from .protocols import EventCallbacks
from .react_loop import parse_messages_to_steps
from .settings import GenerationSettings

# Multiplier for request limit relative to max_steps. Gives headroom for
# `Agent(retries=...)` to chase malformed-tool-call recovery without
# tripping the framework's per-run round-trip cap.
_REQUEST_LIMIT_MULTIPLIER = 2


def _default_settings_provider() -> GenerationSettings:
    return GenerationSettings()


class CodingAgent:
    """Reusable pydantic-ai agent wrapper.

    The underlying `Agent` is constructed once and reused for every call —
    consistent with pydantic-ai's "Agent is a stateless module-level singleton"
    mental model. Live `temperature` / `timeout` / `max_tokens` updates flow
    through `settings_provider`, which is invoked per model request.

    `settings_provider().api_key` is read once at construction (baked into
    the underlying provider). If the API key needs to change mid-session,
    rebuild the `CodingAgent` instance.
    """

    def __init__(
        self,
        *,
        model: str = "ollama:",
        base_url: Optional[str] = None,
        system_prompt: str = "",
        settings_provider: Optional[Callable[[], GenerationSettings]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        toolsets: Optional[Sequence[AbstractToolset]] = None,
        event_callbacks: EventCallbacks = EventCallbacks(),
        capabilities: Optional[Sequence[AbstractCapability]] = None,
    ) -> None:
        self.model_str = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.toolsets: list[AbstractToolset] = list(toolsets) if toolsets else []
        self.event_callbacks = event_callbacks
        self.capabilities: list[AbstractCapability] = (
            list(capabilities) if capabilities else []
        )
        self._settings_provider: Callable[[], GenerationSettings] = (
            settings_provider or _default_settings_provider
        )
        self.rebuild_pydantic_agent()

    def rebuild_pydantic_agent(self) -> None:
        """Re-construct the underlying `Agent` from the current state.

        Call after mutating `model_str`, `base_url`, `system_prompt`,
        `toolsets`, or `capabilities`. Live temperature/timeout/max_tokens
        updates do NOT require a rebuild — they flow through
        `settings_provider` via `_build_model_settings`.
        """
        initial = self._settings_provider()
        self._model = ModelFactory(
            self.model_str, self.base_url, api_key=initial.api_key,
        ).create_model()
        self._agent: Agent = Agent(
            self._model,
            instructions=self.system_prompt,
            toolsets=self.toolsets,
            retries=DEFAULT_MAX_RETRIES,
            capabilities=[
                TextToolFallbackCapability(),
                Sequential400RetryCapability(),
                *self.capabilities,
            ],
        )

    def _build_model_settings(self, _ctx: RunContext[Any]) -> ModelSettings:
        """Resolve live generation settings into pydantic-ai's ModelSettings.

        Invoked by pydantic-ai before each model request inside a run, so
        `settings_provider()` updates take effect mid-session for
        temperature/timeout/max_tokens.
        """
        gen = self._settings_provider()
        settings: ModelSettings = {
            "temperature": gen.temperature,
            "timeout": gen.timeout,
            "parallel_tool_calls": False,
        }
        if gen.max_tokens is not None:
            settings["max_tokens"] = gen.max_tokens
        return settings

    @property
    def pydantic_agent(self) -> Agent:
        """The long-lived underlying `pydantic_ai.Agent`."""
        return self._agent

    @property
    def model(self) -> OpenAIChatModel:
        """The current LLM model instance. Replaced on `rebuild_pydantic_agent`."""
        return self._model

    async def run(
        self,
        prompt: str,
        message_history: Optional[list[ModelMessage]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
    ) -> AgentResponse:
        """Run a query and return the structured response.

        Pass `on_text_chunk` to receive each text delta as the model
        generates. The full answer is also returned in
        `AgentResponse.answer`.
        """
        chunks: list[str] = []

        async def stream_handler(
            _ctx: RunContext[Any],
            events: AsyncIterable[AgentStreamEvent],
        ) -> None:
            async for event in events:
                delta = getattr(event, "delta", None)
                if isinstance(delta, TextPartDelta):
                    content = getattr(delta, "content_delta", "")
                    if content:
                        chunks.append(content)
                        if on_text_chunk:
                            on_text_chunk(content)

        result = await self._agent.run(
            prompt,
            message_history=message_history,
            model_settings=self._build_model_settings,
            event_stream_handler=stream_handler,
            usage_limits=UsageLimits(
                request_limit=self.max_steps * _REQUEST_LIMIT_MULTIPLIER,
            ),
        )

        all_messages = result.all_messages()
        steps = parse_messages_to_steps(all_messages)
        total_tool_calls = sum(1 for step in steps if step.action is not None)
        answer = str(result.output) if result.output else "".join(chunks)

        return AgentResponse(
            answer=answer,
            steps=steps,
            total_tool_calls=total_tool_calls,
            reached_max_steps=total_tool_calls >= self.max_steps,
            messages=all_messages,
        )

    def run_stream(
        self,
        prompt: str,
        message_history: Optional[list[ModelMessage]] = None,
    ) -> Any:
        """Open a streamed run; returns pydantic-ai's `StreamedRunResult`
        async context manager. Use for full event/tool-call streaming;
        use `stream_text` for plain text deltas.
        """
        return self._agent.run_stream(
            prompt,
            message_history=message_history,
            model_settings=self._build_model_settings,
            usage_limits=UsageLimits(
                request_limit=self.max_steps * _REQUEST_LIMIT_MULTIPLIER,
            ),
        )

    async def stream_text(
        self,
        prompt: str,
        message_history: Optional[list[ModelMessage]] = None,
    ) -> AsyncIterator[str]:
        """Stream raw text chunks from the model."""
        async with self._agent.run_stream(
            prompt,
            message_history=message_history,
            model_settings=self._build_model_settings,
            usage_limits=UsageLimits(
                request_limit=self.max_steps * _REQUEST_LIMIT_MULTIPLIER,
            ),
        ) as result:
            async for text in result.stream_text(delta=True):
                yield text
