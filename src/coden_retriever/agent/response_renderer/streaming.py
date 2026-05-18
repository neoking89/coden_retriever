"""Live, in-place rendering of the agent's streaming output.

Wraps Rich's :class:`~rich.live.Live` panel and translates structured
callbacks from the agent core (:class:`TextEvent`, :class:`ThinkingEvent`,
:class:`ToolCallEvent`, :class:`ToolResultEvent`) into a single
chronologically-interleaved Rich markup buffer. The result is one
continuously-updating region that mixes thinking italics, tool-call
lines, tool-result previews, and final answer text in the order they
happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

from rich.live import Live
from rich.markup import escape
from rich.text import Text

from ..protocols import (
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ..rich_console import console, set_active_live

# Refresh budget for Rich's Live region. 4 fps stays smooth without
# saturating slow Windows terminals.
DEFAULT_REFRESH_PER_SECOND = 4


@dataclass(frozen=True)
class StreamRendererStyle:
    """Visible labels, indents, markup templates, and truncation thresholds
    for the streamed Live panel. Construct a custom instance to theme the
    panel; the default matches the historical ``coden -a`` look.

    Truncation thresholds:
        max_arg_value_len — per-arg-value cap. Defaults to 50 because
            longer values are noise on the single-line tool-call display.
        max_args_total_len — cap on non-dict args strings. Defaults to
            200 so the args clause fits one terminal row at typical widths.
        max_result_preview_len — tool-result preview cap. Defaults to 300:
            enough to surface errors, not enough to swamp the live panel.
    """

    tool_call_icon: str = "🔧"
    tool_success_label: str = "v Done"
    tool_error_label: str = "x Failed"
    running_label: str = "running..."

    tool_call_indent: str = "  "      # 2 spaces
    tool_result_indent: str = "     "  # 5 spaces

    thinking_markup: str = "[dim italic]{delta}[/dim italic]"
    tool_call_markup: str = (
        "\n[bold cyan]{indent}{icon} {name}[/bold cyan]"
        "([dim]{args}[/dim])\n"
    )
    result_preview_markup: str = "[dim]{indent}{preview}[/dim]\n"
    result_stats_markup: str = " [dim]({duration:.1f}s)[/dim]"
    result_error_markup: str = "[bold red]{indent}{label}{stats}[/bold red]\n"
    result_success_markup: str = "[bold green]{indent}{label}{stats}[/bold green]\n"

    max_arg_value_len: int = 50
    max_args_total_len: int = 200
    max_result_preview_len: int = 300

    @property
    def running_sentinel(self) -> str:
        return f"[dim italic]{self.tool_result_indent}{self.running_label}[/dim italic]"


DEFAULT_STREAM_STYLE = StreamRendererStyle()


# ── Argument & result formatting ─────────────────────────────────────────


def _format_tool_args(tool_args: object, style: StreamRendererStyle) -> str:
    if isinstance(tool_args, dict):
        return ", ".join(
            f"{k}={repr(v)[:style.max_arg_value_len]}"
            for k, v in tool_args.items()
        )
    return str(tool_args)[:style.max_args_total_len]


def _truncate_result_preview(result: str, style: StreamRendererStyle) -> str:
    if len(result) > style.max_result_preview_len:
        return result[:style.max_result_preview_len].replace("\n", " ") + "..."
    return result.replace("\n", " ")


# ── StreamRenderer ───────────────────────────────────────────────────────


class StreamRenderer:
    """Refreshes a Rich ``Live`` panel from typed agent stream events.

    Uses ``vertical_overflow="visible"`` so the panel scrolls naturally
    instead of showing an ellipsis when content exceeds terminal height.
    """

    def __init__(
        self,
        refresh_per_second: int = DEFAULT_REFRESH_PER_SECOND,
        max_lines: Optional[int] = None,
        style: StreamRendererStyle = DEFAULT_STREAM_STYLE,
    ) -> None:
        self.refresh_per_second = refresh_per_second
        self.max_lines = max_lines
        self._style = style
        self._live: Optional[Live] = None
        self._buffer: str = ""
        # Length of the cumulative answer text last observed via on_text;
        # subtracting gives us the delta to append.
        self._answer_len: int = 0
        self._tool_start: Optional[float] = None

    # ── context-manager surface ─────────────────────────────────────────

    def __enter__(self) -> "StreamRenderer":
        self._reset_buffer()
        self._live = Live(
            Text(""),
            console=console,
            refresh_per_second=self.refresh_per_second,
            transient=True,
            vertical_overflow="visible",
        )
        set_active_live(self._live)
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._live is None:
            return
        set_active_live(None)
        self._live.__exit__(exc_type, exc_val, exc_tb)
        self._live = None

    # ── event handlers ──────────────────────────────────────────────────

    def on_text(self, event: TextEvent) -> None:
        """Append the newly-streamed answer-text delta to the live buffer.

        ``event.cumulative`` is the full accumulated answer text; we diff
        against the last-seen length to recover the per-chunk delta.
        """
        delta = event.cumulative[self._answer_len:]
        self._answer_len = len(event.cumulative)
        if not delta:
            return
        self._buffer += escape(delta)
        self._refresh()

    def on_thinking(self, event: ThinkingEvent) -> None:
        if not event.delta:
            return
        self._buffer += self._style.thinking_markup.format(delta=escape(event.delta))
        self._refresh()

    def on_tool_call(self, event: ToolCallEvent) -> None:
        self._tool_start = time.monotonic()
        self._buffer += self._style.tool_call_markup.format(
            indent=self._style.tool_call_indent,
            icon=self._style.tool_call_icon,
            name=escape(event.name),
            args=escape(_format_tool_args(event.args, self._style)),
        )
        self._buffer += self._style.running_sentinel
        self._refresh()

    def on_tool_result(self, event: ToolResultEvent) -> None:
        duration = self._pop_tool_duration()
        self._buffer = self._buffer.replace(self._style.running_sentinel, "")
        self._append_result_preview(event)
        self._append_result_status(event, duration)
        self._refresh()

    # ── internal helpers ────────────────────────────────────────────────

    def _reset_buffer(self) -> None:
        self._buffer = ""
        self._answer_len = 0
        self._tool_start = None

    def _pop_tool_duration(self) -> Optional[float]:
        if self._tool_start is None:
            return None
        duration = time.monotonic() - self._tool_start
        self._tool_start = None
        return duration

    def _append_result_preview(self, event: ToolResultEvent) -> None:
        if not event.content:
            return
        preview = _truncate_result_preview(str(event.content), self._style)
        self._buffer += self._style.result_preview_markup.format(
            indent=self._style.tool_result_indent,
            preview=escape(preview),
        )

    def _append_result_status(
        self,
        event: ToolResultEvent,
        duration: Optional[float],
    ) -> None:
        stats = (
            self._style.result_stats_markup.format(duration=duration)
            if duration is not None
            else ""
        )
        is_error = event.is_error and event.content
        template = (
            self._style.result_error_markup
            if is_error
            else self._style.result_success_markup
        )
        label = (
            self._style.tool_error_label
            if is_error
            else self._style.tool_success_label
        )
        self._buffer += template.format(
            indent=self._style.tool_result_indent,
            label=label,
            stats=stats,
        )

    def _refresh(self) -> None:
        if self._live is None:
            return
        display = self._buffer
        if self.max_lines is not None:
            lines = display.split("\n")
            if len(lines) > self.max_lines:
                display = "\n".join(lines[-self.max_lines:])
        self._live.update(Text.from_markup(display))
