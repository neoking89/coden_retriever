"""Stream event handler for pydantic-ai agent responses.

Emits raw model text via `EventCallbacks.on_text` and structured tool
events via `on_tool_call` / `on_tool_result`. No display formatting —
library callers attach their own renderer (rich, ANSI, plain print) via
the callbacks.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterable, Protocol

from pydantic_ai import AgentStreamEvent
from pydantic_ai.messages import (
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from .protocols import (
    DebugLoggerProtocol,
    EventCallbacks,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent as ToolCallEventDC,
    ToolResultEvent as ToolResultEventDC,
)


class ToolCallEvent(Protocol):
    """Protocol for tool call events."""

    event_kind: str

    @property
    def part(self) -> Any: ...


class ToolResultEvent(Protocol):
    """Protocol for tool result events."""

    event_kind: str

    @property
    def result(self) -> Any: ...


@dataclass
class StreamState:
    """Accumulated state during streaming."""

    streamed_text: str = ""
    accumulated_text: str = ""
    accumulated_thinking: str = ""


class StreamEventHandler:
    """Handles pydantic-ai stream events.

    Fires structured callbacks from the injected `EventCallbacks`:
    - `on_text(cumulative_text)` — refreshed on every text delta
    - `on_thinking(delta)` — each thinking-trace delta as it arrives
    - `on_tool_call(name, args_dict)` — when the model invokes a tool
    - `on_tool_result(name, raw_result)` — when the tool returns

    `get_streamed_text()` returns the plain-text answer accumulator and is
    the documented fallback when the final `result.output` is empty.
    """

    def __init__(
        self,
        debug_logger: DebugLoggerProtocol,
        event_callbacks: EventCallbacks = EventCallbacks(),
    ) -> None:
        self.debug_logger = debug_logger
        self.event_callbacks = event_callbacks
        self.state = StreamState()

    def reset(self) -> None:
        """Reset state for a new query."""
        self.state = StreamState()

    def get_streamed_text(self) -> str:
        """Get the accumulated streamed answer text."""
        return self.state.streamed_text

    def _notify_text_update(self) -> None:
        if self.event_callbacks.on_text:
            self.event_callbacks.on_text(TextEvent(cumulative=self.state.streamed_text))

    def _handle_thinking_delta(self, thinking_delta: str) -> None:
        self.state.accumulated_thinking += thinking_delta
        if self.event_callbacks.on_thinking:
            self.event_callbacks.on_thinking(ThinkingEvent(delta=thinking_delta))

    def _handle_text_delta(self, content: str) -> None:
        self.state.streamed_text += content
        self.state.accumulated_text += content
        self._notify_text_update()

    def _handle_part_start(self, part: Any) -> None:
        """Capture the opening chunk of a streamed text/thinking part.

        pydantic-ai delivers the first segment of a part in the
        ``PartStartEvent``, then only increments via ``PartDeltaEvent``.
        Without this the first token (e.g. the leading word of the answer)
        is dropped from the accumulator — masked in interactive mode, which
        reprints the final ``result.output``, but exposed by ``-p`` mode,
        which streams the accumulator verbatim with no reprint.
        """
        content = getattr(part, "content", None)
        if not isinstance(content, str) or not content:
            return
        if isinstance(part, ThinkingPart):
            self._handle_thinking_delta(content)
        elif isinstance(part, TextPart):
            self._handle_text_delta(content)

    def _handle_tool_call(self, event: ToolCallEvent) -> None:
        tool_name = event.part.tool_name
        tool_call_id = getattr(event.part, "tool_call_id", None)

        # Flush partial content so debug logs stay in chronological order
        self._flush_thinking()
        self._flush_text(is_final=False)

        args_dict = event.part.args_as_dict()
        self.debug_logger.log_tool_call(tool_name, args_dict, tool_call_id)

        if self.event_callbacks.on_tool_call:
            self.event_callbacks.on_tool_call(
                ToolCallEventDC(name=tool_name, args=args_dict, tool_call_id=tool_call_id),
            )

    def _handle_tool_result(self, event: ToolResultEvent) -> None:
        result_content = getattr(event.result, "content", None)
        tool_name = getattr(event.result, "tool_name", "unknown")
        tool_call_id = getattr(event.result, "tool_call_id", None)

        # Prefer structured checks over a fragile string prefix, because tool
        # results can legitimately begin with "Error" in normal output. Two
        # dict shapes reach here: the debug-tool standard ({status: "error",
        # error_type, ...}) and the raw wire shape ({error: "..."}) from
        # adapter methods that surface errors untransformed.
        if isinstance(result_content, dict):
            is_error = (
                result_content.get("status") == "error"
                or bool(result_content.get("error"))
            )
        else:
            result_str = str(result_content) if result_content else ""
            is_error = bool(result_str) and result_str.lower().startswith("error:")

        self.debug_logger.log_tool_result(
            tool_name,
            result_content,
            success=not is_error,
            tool_call_id=tool_call_id,
        )

        if self.event_callbacks.on_tool_result:
            self.event_callbacks.on_tool_result(
                ToolResultEventDC(
                    name=tool_name,
                    content=result_content,
                    tool_call_id=tool_call_id,
                    is_error=is_error,
                ),
            )

    def _flush_thinking(self) -> None:
        if self.state.accumulated_thinking.strip():
            self.debug_logger.log_thinking_trace(self.state.accumulated_thinking.strip())
            self.state.accumulated_thinking = ""

    def _flush_text(self, is_final: bool = False) -> None:
        if self.state.accumulated_text.strip():
            self.debug_logger.log_model_response(
                self.state.accumulated_text.strip(),
                is_final=is_final,
            )
            self.state.accumulated_text = ""

    def flush_remaining(self, error_occurred: bool = False) -> None:
        """Flush any remaining accumulated content after stream end or error."""
        self._flush_thinking()
        self._flush_text(is_final=error_occurred)

    async def handle_events(self, ctx: Any, events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            self._process_event(event)

    def _process_event(self, event: AgentStreamEvent) -> None:
        """Dispatch a single stream event by delta or event_kind."""
        event_kind = getattr(event, "event_kind", None)
        delta = getattr(event, "delta", None)

        if isinstance(event, PartStartEvent):
            self._handle_part_start(event.part)
            return

        if isinstance(delta, ThinkingPartDelta):
            thinking_delta = getattr(delta, "content_delta", None)
            if thinking_delta:
                self._handle_thinking_delta(thinking_delta)

        elif isinstance(delta, TextPartDelta):
            content = getattr(delta, "content_delta", "")
            if content:
                self._handle_text_delta(content)

        elif delta is not None and hasattr(delta, "content_delta"):
            content = delta.content_delta
            if content:
                self._handle_text_delta(content)

        elif event_kind == "function_tool_call":
            self._handle_tool_call(event)  # type: ignore[arg-type]

        elif event_kind == "function_tool_result":
            self._handle_tool_result(event)  # type: ignore[arg-type]
