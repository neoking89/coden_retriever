"""Thin executor that runs one turn against an externally-owned `Agent`.

All retry / fallback logic lives in `capabilities.py`, registered on the
`Agent` at its construction. This class is the seam for callers that
manage the `Agent` lifecycle themselves (e.g. coden's interactive session)
and want a one-shot `(prompt, history) -> AgentResponse` entry point with
streaming + debug-log wiring already attached.
"""

from typing import Optional

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import ModelMessage

from .models import AgentResponse
from .protocols import (
    NULL_DEBUG_LOGGER,
    DebugLoggerProtocol,
    EventCallbacks,
)
from .react_loop import parse_messages_to_steps
from .stream_handler import StreamEventHandler

# Multiplier for request limit relative to max_steps. Gives headroom for
# pydantic-ai's `retries=` budget to chase malformed-tool-call recovery
# without tripping the per-run round-trip cap.
REQUEST_LIMIT_MULTIPLIER = 2


class QueryExecutor:
    """Run a single user turn through a pre-built pydantic-ai `Agent`.

    The `Agent` passed to `execute()` owns its own retries (via
    `Agent(retries=...)`) and live settings (via the `model_settings=`
    kwarg on its own `run()` call, or via a capability). This executor
    just attaches the streaming + debug-log fan-out and returns the
    parsed `AgentResponse`.
    """

    def __init__(
        self,
        *,
        max_steps: int,
        event_callbacks: EventCallbacks = EventCallbacks(),
        debug_logger: DebugLoggerProtocol = NULL_DEBUG_LOGGER,
    ) -> None:
        self.max_steps = max_steps
        self.event_callbacks = event_callbacks
        self.debug_logger = debug_logger

    async def execute(
        self,
        agent: Agent,
        prompt: str,
        *,
        message_history: Optional[list[ModelMessage]] = None,
    ) -> AgentResponse:
        """Run one turn and return its parsed response."""
        self.debug_logger.log_user_prompt(prompt)
        stream_handler = StreamEventHandler(
            debug_logger=self.debug_logger,
            event_callbacks=self.event_callbacks,
        )

        result = None
        try:
            result = await agent.run(
                prompt,
                message_history=message_history,
                event_stream_handler=stream_handler.handle_events,
                usage_limits=UsageLimits(
                    request_limit=self.max_steps * REQUEST_LIMIT_MULTIPLIER,
                ),
            )
        finally:
            stream_handler.flush_remaining(error_occurred=result is None)

        if result is None:
            raise RuntimeError("Agent run failed without exception")

        all_messages = result.all_messages()
        steps = parse_messages_to_steps(all_messages)
        total_tool_calls = sum(1 for step in steps if step.action is not None)
        answer_text = (
            str(result.output) if result.output else stream_handler.get_streamed_text()
        )

        if self.event_callbacks.on_step:
            for step in steps:
                self.event_callbacks.on_step(step)

        self.debug_logger.log_message_history(all_messages)
        self.debug_logger.log_model_response(answer_text, is_final=True)

        return AgentResponse(
            answer=answer_text,
            steps=steps,
            total_tool_calls=total_tool_calls,
            reached_max_steps=total_tool_calls >= self.max_steps,
            messages=all_messages,
        )
