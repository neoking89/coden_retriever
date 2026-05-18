"""Dependency-injection protocols and event-callback container.

The agent core depends on these abstractions instead of concrete types
(MCP server, tool routers, console pickers, debug loggers). Callers adapt
their concrete implementations to these Protocols at wiring time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Protocol

from pydantic_ai.messages import ModelMessage

if TYPE_CHECKING:
    from .models import ReActStep


class DebugLoggerProtocol(Protocol):
    """Structured event sink for the library's tracing surface.

    The methods listed here are the union of what `stream_handler` and
    `query_executor` actually call. Implement any concrete logger that
    matches this shape, or pass `NULL_DEBUG_LOGGER` (the default) to
    disable structured logging entirely.
    """

    def log_user_prompt(self, prompt: str) -> None: ...
    def log_message_history(self, messages: list[ModelMessage]) -> None: ...
    def log_model_response(self, response_text: str, is_final: bool = ...) -> None: ...
    def log_thinking_trace(self, thinking: str) -> None: ...
    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: Optional[str] = ...,
    ) -> None: ...
    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_call_id: Optional[str] = ...,
    ) -> None: ...


class _NullDebugLogger:
    """No-op sink. Used as the default so library callers and the
    handler can call methods unconditionally without a None guard."""

    def log_user_prompt(self, prompt: str) -> None: pass
    def log_message_history(self, messages: list[ModelMessage]) -> None: pass
    def log_model_response(self, response_text: str, is_final: bool = False) -> None: pass
    def log_thinking_trace(self, thinking: str) -> None: pass
    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> None: pass
    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_call_id: Optional[str] = None,
    ) -> None: pass


NULL_DEBUG_LOGGER: DebugLoggerProtocol = _NullDebugLogger()


class PermissionChoice(Enum):
    """User's response to a tool-permission prompt."""

    ALLOW = "allow"
    DENY = "deny"
    ALWAYS_ALLOW = "always_allow"


PickerCallback = Callable[[str, dict[str, Any]], Awaitable[Optional[PermissionChoice]]]
"""Per-tool permission prompt. Returns None when the user cancels."""

ToolFilterFn = Callable[[str], Awaitable[set[str]]]
"""Per-query tool allowlist filter. Receives the user query and returns
the names of tools to expose to the model for that turn."""


@dataclass(frozen=True)
class TextEvent:
    """A streamed answer-text update. `cumulative` is the full accumulated
    answer text at this point in the turn (not a per-chunk delta)."""

    cumulative: str


@dataclass(frozen=True)
class ThinkingEvent:
    """A streamed thinking-trace delta — only the new chunk since the
    previous fire, not the full accumulated thinking."""

    delta: str


@dataclass(frozen=True)
class ToolCallEvent:
    """The model invoked a tool. `tool_call_id` correlates with the matching
    `ToolResultEvent` and is None only on providers that omit it."""

    name: str
    args: dict[str, Any]
    tool_call_id: Optional[str]


@dataclass(frozen=True)
class ToolResultEvent:
    """A tool returned. `content` is the raw return value untouched; the
    library pre-classifies it via `is_error` so consumers don't re-implement
    the dict-shape + ``error:`` prefix heuristics."""

    name: str
    content: Any
    tool_call_id: Optional[str]
    is_error: bool


@dataclass(frozen=True)
class EventCallbacks:
    """Structured hooks the agent fires during a run. All optional.

    Callers attach handlers selectively — the agent core invokes only the
    ones present and ignores the rest, so rich-console renderers, token
    counters, and debug loggers can all attach through this single seam
    without being imported inside the core.
    """

    on_text: Optional[Callable[[TextEvent], None]] = None
    on_thinking: Optional[Callable[[ThinkingEvent], None]] = None
    on_tool_call: Optional[Callable[[ToolCallEvent], None]] = None
    on_tool_result: Optional[Callable[[ToolResultEvent], None]] = None
    on_step: Optional[Callable[["ReActStep"], None]] = None
    on_warning: Optional[Callable[[str], None]] = None
    on_error: Optional[Callable[[str, Exception], None]] = None
    on_retry: Optional[Callable[[int, Exception], None]] = None
