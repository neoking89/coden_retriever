"""Permission-gating capability for pydantic-ai toolsets.

`PermissionCapability` wraps the agent's toolset with the native
`ApprovalRequiredToolset` (which raises `ApprovalRequired` when a tool is
unapproved), and implements `handle_deferred_tool_calls` so the framework
resumes the run automatically — no manual deferral loop in callers.

The gate, the picker, and the user notification are all injected via
callables. Wire whatever picker UI you prefer (stdin prompt, GUI, deny-all,
auto-approve, ...) by satisfying the `PickerCallback` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)

from .protocols import PermissionChoice, PickerCallback


@dataclass
class _SessionPermissionState:
    """Mutable container for session-level permission state.

    Lives on the capability instance and is shared across runs — the agent
    that owns this capability is long-lived, so the always-allow flag
    persists across turns naturally.
    """

    always_allow: bool = False


def _deny_picker(_tool_name: str, _tool_args: dict[str, Any]) -> Any:
    """Safe default picker: deny every call."""

    async def _coro() -> Optional[PermissionChoice]:
        return PermissionChoice.DENY

    return _coro()


def _always_enabled() -> bool:
    """Safe default `is_enabled`: permission gate is on."""
    return True


@dataclass
class PermissionCapability(AbstractCapability[Any]):
    """Permission-gating capability built on `ApprovalRequiredToolset`.

    On each tool call, `ApprovalRequiredToolset.approval_required_func` runs.
    If it returns True, the toolset raises `ApprovalRequired` and the
    framework defers the call. `handle_deferred_tool_calls` then fires the
    picker (3-way ALLOW / ALWAYS_ALLOW / DENY) and resolves the deferral —
    the run continues automatically.

    The `is_enabled()` callable is consulted dynamically so consumers can
    flip the gate at runtime (e.g. via a config-toggle slash command)
    without rebuilding the capability or the agent. `on_message`, if set,
    fires for state-change notifications.
    """

    is_enabled: Callable[[], bool] = field(default=_always_enabled)
    picker: PickerCallback = field(default=_deny_picker)
    on_message: Optional[Callable[[str], None]] = field(default=None)
    _state: _SessionPermissionState = field(default_factory=_SessionPermissionState)

    @property
    def session_always_allow(self) -> bool:
        """Whether the user has selected 'Always Allow' for this session."""
        return self._state.always_allow

    @session_always_allow.setter
    def session_always_allow(self, value: bool) -> None:
        self._state.always_allow = value

    def _notify(self, message: str) -> None:
        if self.on_message is not None:
            self.on_message(message)

    def get_wrapper_toolset(
        self, toolset: AbstractToolset[Any]
    ) -> AbstractToolset[Any] | None:
        return ApprovalRequiredToolset(
            wrapped=toolset,
            approval_required_func=self._approval_required,
        )

    def _approval_required(
        self,
        _ctx: RunContext[Any],
        _tool_def: ToolDefinition,
        _tool_args: dict[str, Any],
    ) -> bool:
        if not self.is_enabled():
            return False
        if self._state.always_allow:
            return False
        return True

    async def handle_deferred_tool_calls(
        self,
        _ctx: RunContext[Any],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        if not requests.approvals:
            return None

        resolved: dict[str, Any] = {}
        for call in requests.approvals:
            args = call.args_as_dict()
            choice = await self.picker(call.tool_name, args)

            if choice is PermissionChoice.ALLOW:
                resolved[call.tool_call_id] = ToolApproved()
            elif choice is PermissionChoice.ALWAYS_ALLOW:
                self._state.always_allow = True
                self._notify("Auto-allowing tools for this session")
                resolved[call.tool_call_id] = ToolApproved()
            else:
                if choice is None:
                    self._notify(f"Tool '{call.tool_name}' execution cancelled")
                else:
                    self._notify(f"Tool '{call.tool_name}' execution denied")
                resolved[call.tool_call_id] = ToolDenied(
                    message=(
                        f"[TOOL DENIED] The user denied permission to execute tool "
                        f"'{call.tool_name}'. Please acknowledge this and continue "
                        "without using this tool, or try a different approach."
                    )
                )
        return DeferredToolResults(approvals=resolved)
