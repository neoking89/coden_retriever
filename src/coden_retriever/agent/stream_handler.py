"""Stream event handler for pydantic-ai agent responses.

Handles real-time streaming of:
- Text content deltas
- Thinking/reasoning traces
- Tool call events and results

Follows Single Responsibility Principle - only handles stream events.
"""

import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, Callable, Optional, Protocol

from pydantic_ai import AgentStreamEvent
from pydantic_ai.messages import ThinkingPartDelta, TextPartDelta
from rich.markup import escape

from ..token_estimator import count_tokens
from .debug_logger import DebugLogger

# WHY 50: individual arg values beyond this length are noise in the inline tool-call display
_MAX_ARG_VALUE_DISPLAY_LEN = 50
# WHY 200: total args string cap for non-dict tool arguments in the inline display
_MAX_ARGS_TOTAL_DISPLAY_LEN = 200
# WHY 300: tool result preview — enough to diagnose errors without swamping the stream
_MAX_RESULT_PREVIEW_LEN = 300
# Visible placeholder while a tool executes. Written in one place, stripped in
# another via `.replace(...)` — the literals must match exactly, so the two
# sites share a single constant to prevent drift.
_RUNNING_SENTINEL_MARKUP = "[dim italic]     running...[/dim italic]"


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
    tool_start_time: float | None = None  # Tracks tool execution duration


class StreamEventHandler:
    """Handles pydantic-ai stream events with logging and display.

    Encapsulates the event handling logic that was previously nested
    inside run_interactive. Supports both display and debug logging.
    """

    def __init__(
        self,
        debug_logger: DebugLogger,
        on_update: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize the stream event handler.

        Args:
            debug_logger: Logger for debug output.
            on_update: Optional callback when display text changes.
        """
        self.debug_logger = debug_logger
        self.on_update = on_update
        self.state = StreamState()

    def reset(self) -> None:
        """Reset state for a new query."""
        self.state = StreamState()

    def get_streamed_text(self) -> str:
        """Get the accumulated streamed text for display."""
        return self.state.streamed_text

    def _notify_update(self) -> None:
        """Notify listener of display update."""
        if self.on_update:
            self.on_update(self.state.streamed_text)

    def _handle_thinking_delta(self, thinking_delta: str) -> None:
        """Handle thinking/reasoning content delta."""
        self.state.accumulated_thinking += thinking_delta
        self.state.streamed_text += f"[dim italic]{escape(thinking_delta)}[/dim italic]"
        self._notify_update()

    def _handle_text_delta(self, content: str) -> None:
        """Handle regular text content delta."""
        self.state.streamed_text += escape(content)
        self.state.accumulated_text += content
        self._notify_update()

    def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool call start event."""
        tool_name = event.part.tool_name
        tool_args = event.part.args
        tool_call_id = getattr(event.part, 'tool_call_id', None)

        # Flush partial content so debug logs stay in chronological order
        self._flush_thinking()
        self._flush_text(is_final=False)

        args_dict = tool_args if isinstance(tool_args, dict) else {"raw": str(tool_args)}
        self.debug_logger.log_tool_call(tool_name, args_dict, tool_call_id)

        self.state.tool_start_time = time.monotonic()

        args_str = self._format_tool_args(tool_args)
        self.state.streamed_text += (
            f"\n[bold cyan]  🔧 {escape(tool_name)}[/bold cyan]"
            f"([dim]{escape(args_str)}[/dim])\n"
        )
        self.state.streamed_text += _RUNNING_SENTINEL_MARKUP
        self._notify_update()

    def _handle_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool call result event."""
        result_content = getattr(event.result, 'content', None)
        tool_name = getattr(event.result, 'tool_name', 'unknown')
        tool_call_id = getattr(event.result, 'tool_call_id', None)

        duration: float | None = None
        if self.state.tool_start_time is not None:
            duration = time.monotonic() - self.state.tool_start_time
            self.state.tool_start_time = None

        # Prefer structured checks over fragile string prefix, because tool
        # results can legitimately start with "Error" in normal output. Two
        # dict shapes reach here: the debug-tool standard ({status: "error",
        # error_type, ...}) and the raw DAP-wire shape ({error: "..."}) from
        # dap_client methods that surface adapter errors untransformed.
        result_str = str(result_content) if result_content else ""
        stats = self._format_tool_stats(duration, result_str)
        if isinstance(result_content, dict):
            is_error = (
                result_content.get("status") == "error"
                or bool(result_content.get("error"))
            )
        else:
            is_error = bool(result_str) and result_str.lower().startswith("error:")

        self.debug_logger.log_tool_result(
            tool_name,
            result_content,
            success=not is_error,
            tool_call_id=tool_call_id
        )

        self.state.streamed_text = self.state.streamed_text.replace(
            _RUNNING_SENTINEL_MARKUP, "",
        )
        if result_content:
            result_preview = self._truncate_result(result_str)
            self.state.streamed_text += f"[dim]     {escape(result_preview)}[/dim]\n"

        if is_error and result_content:
            self.state.streamed_text += f"[bold red]     x Failed{stats}[/bold red]\n"
        else:
            self.state.streamed_text += f"[bold green]     v Done{stats}[/bold green]\n"

        self._notify_update()

    def _format_tool_stats(self, duration: float | None, result_str: str) -> str:
        """Build the inline ' (0.6s, ~1.2k tok)' suffix for the Done/Failed line."""
        parts: list[str] = []
        if duration is not None:
            parts.append(f"{duration:.1f}s")
        if result_str:
            parts.append(f"~{count_tokens(result_str)} tok")
        if not parts:
            return ""
        return f" [dim]({', '.join(parts)})[/dim]"

    def _format_tool_args(self, tool_args: Any) -> str:
        """Format tool arguments for display."""
        if isinstance(tool_args, dict):
            return ", ".join(
                f"{k}={repr(v)[:_MAX_ARG_VALUE_DISPLAY_LEN]}"
                for k, v in tool_args.items()
            )
        return str(tool_args)[:_MAX_ARGS_TOTAL_DISPLAY_LEN]

    def _truncate_result(self, result: str) -> str:
        """Truncate result for display, collapsing newlines."""
        if len(result) > _MAX_RESULT_PREVIEW_LEN:
            return result[:_MAX_RESULT_PREVIEW_LEN].replace('\n', ' ') + "..."
        return result.replace('\n', ' ')

    def _flush_thinking(self) -> None:
        """Log accumulated thinking and reset."""
        if self.state.accumulated_thinking.strip():
            self.debug_logger.log_thinking_trace(self.state.accumulated_thinking.strip())
            self.state.accumulated_thinking = ""

    def _flush_text(self, is_final: bool = False) -> None:
        """Log accumulated text and reset."""
        if self.state.accumulated_text.strip():
            self.debug_logger.log_model_response(
                self.state.accumulated_text.strip(),
                is_final=is_final
            )
            self.state.accumulated_text = ""

    def flush_remaining(self, error_occurred: bool = False) -> None:
        """Flush any remaining accumulated content.

        Call this after streaming completes or on error.
        """
        self._flush_thinking()
        self._flush_text(is_final=error_occurred)

    async def handle_events(self, ctx: Any, events: AsyncIterable[AgentStreamEvent]) -> None:
        """Handle stream events from pydantic-ai agent."""
        async for event in events:
            self._process_event(event)

    def _process_event(self, event: AgentStreamEvent) -> None:
        """Process a single stream event.

        Uses isinstance checks instead of string comparison for type safety.
        """
        event_kind = getattr(event, 'event_kind', None)
        delta = getattr(event, 'delta', None)

        if isinstance(delta, ThinkingPartDelta):
            thinking_delta = getattr(delta, 'content_delta', None)
            if thinking_delta:
                self._handle_thinking_delta(thinking_delta)

        elif isinstance(delta, TextPartDelta):
            content = getattr(delta, 'content_delta', '')
            if content:
                self._handle_text_delta(content)

        elif delta is not None and hasattr(delta, 'content_delta'):
            content = delta.content_delta
            if content:
                self._handle_text_delta(content)

        elif event_kind == 'function_tool_call':
            self._handle_tool_call(event)  # type: ignore[arg-type]

        elif event_kind == 'function_tool_result':
            self._handle_tool_result(event)  # type: ignore[arg-type]
