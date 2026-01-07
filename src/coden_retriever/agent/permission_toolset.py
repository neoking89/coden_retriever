"""Permission-gating wrapper for pydantic-ai toolsets.

Provides a wrapper that asks for user permission before executing tools,
using an arrow-key based selection UI.

This extends pydantic-ai's WrapperToolset to maintain inline approval behavior
while leveraging the framework's patterns.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.toolsets import WrapperToolset, AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from .tool_permission_picker import (
    PermissionChoice,
    ToolPermissionRequest,
    run_tool_permission_picker,
)
from .rich_console import console, get_active_live

# Generic type var for deps - we don't require specific deps type
DepsT = TypeVar("DepsT")


def _get_ask_permission_setting() -> bool:
    """Get the current ask_tool_permission setting from config.

    Uses get_config() which returns the cached config. Changes via
    /config set command update the cache, so this is always up-to-date
    without needing to read from disk on every tool call.
    """
    from ..config_loader import get_config
    return get_config().agent.ask_tool_permission


@dataclass
class PermissionToolsetWrapper(WrapperToolset[DepsT]):
    """A toolset wrapper that asks for permission before executing tools.

    This wrapper intercepts call_tool and presents a permission dialog.
    When permission is denied, it returns an error message to the agent
    instead of executing the tool.

    The permission setting is checked dynamically from config, so changes
    via /config set take effect immediately without restart.

    Supports a session-based "Always Allow" mode that bypasses permission
    prompts for the duration of the session.

    Extends pydantic-ai's WrapperToolset for proper framework integration.
    """

    session_always_allow: bool = field(default=False)

    async def check_permission(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if permission is granted to execute a tool.

        This is the central permission checking method used by both:
        - call_tool() for normal pydantic-ai tool execution
        - Fallback tool execution via ask_permission_for_fallback callback

        Permission checking is bypassed if:
        - ask_tool_permission config setting is False
        - session_always_allow is True (user selected "Always Allow" for session)

        Args:
            tool_name: Name of the tool to execute.
            args: Tool arguments.

        Returns:
            True if permission granted (including when permission not required).
            False if permission denied.
        """
        if not _get_ask_permission_setting():
            return True

        if self.session_always_allow:
            return True

        result = await self._prompt_user_permission(tool_name, args)

        if result is None:
            self.session_always_allow = True
            console.print("[green]Auto-allowing tools for this session[/green]")
            return True

        return result

    async def _prompt_user_permission(self, tool_name: str, args: dict) -> bool | None:
        """Prompt user for permission using the arrow-key picker UI.

        Runs in a thread pool to avoid blocking the async event loop.
        Pauses Rich Live display to prevent rendering conflicts with prompt_toolkit.

        Returns:
            True if user allows this tool.
            False if user denies or cancels.
            None if user selected "Always Allow" for session.
        """
        request = ToolPermissionRequest(
            tool_name=tool_name,
            tool_args=args,
        )

        # Pause Rich Live display if active to prevent rendering conflicts
        # (Rich runs in a refresh thread, prompt_toolkit takes over stdout)
        live = get_active_live()
        live_was_stopped = False
        if live is not None:
            try:
                live.stop()
                live_was_stopped = True
            except Exception:
                # If stop fails, don't try to restart later
                live = None

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                run_tool_permission_picker,
                request,
            )
        finally:
            # Resume Live display after picker closes
            if live is not None and live_was_stopped:
                try:
                    live.start()
                except Exception:
                    # Best effort restart - if it fails, the display is gone
                    pass

        if result is None:
            console.print("[yellow]Tool execution cancelled[/yellow]")
            return False

        if result == PermissionChoice.ALLOW:
            return True
        elif result == PermissionChoice.ALWAYS_ALLOW:
            return None
        else:
            console.print(f"[yellow]Tool '{tool_name}' execution denied[/yellow]")
            return False

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[DepsT],
        tool: ToolsetTool[DepsT],
    ) -> Any:
        """Call a tool with permission checking.

        If permission is denied, returns an error message to the agent.
        """
        if not await self.check_permission(name, tool_args):
            return (
                f"[TOOL DENIED] The user denied permission to execute tool '{name}'. "
                "Please acknowledge this and continue without using this tool, "
                "or try a different approach."
            )

        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def ask_permission_for_fallback(
        self, tool_name: str, args: dict[str, Any]
    ) -> bool:
        """Check permission for fallback tool calls.

        This method can be passed as a callback to execute_fallback_tool_calls()
        to ensure fallback tool execution respects permission settings.

        Uses the same check_permission logic as call_tool for consistency.

        Returns:
            True if allowed, False if denied.
        """
        return await self.check_permission(tool_name, args)


def wrap_toolset_with_permission(
    toolset: AbstractToolset[DepsT],
) -> PermissionToolsetWrapper[DepsT]:
    """Wrap a toolset with permission checking.

    Creates a wrapper that asks for user permission before each tool call.
    The permission setting is checked dynamically from config via
    _get_ask_permission_setting().

    Args:
        toolset: The toolset to wrap.

    Returns:
        A wrapped toolset that checks permission before tool calls.
    """
    return PermissionToolsetWrapper(wrapped=toolset)
