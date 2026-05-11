"""Interactive REPL loop for the coding agent.

Handles:
- Command dispatch and execution
- Tool wizard integration
- Directory navigation
- Model switching
- Settings management

Follows Single Responsibility Principle - only handles REPL control flow.
"""

import asyncio
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from rich import box
from rich.table import Table

from ..config_loader import assign_config_value, load_config, reload_config, save_config
from ..constants import DEFAULT_MAX_RETRIES, SHELL_COMMAND_TIMEOUT
from .commands import execute_command, _REBUILD_REQUIRED_SETTINGS, _RESTART_REQUIRED_SETTINGS
from .config_picker import run_config_picker_async
from .directory_browser import run_directory_browser_async, print_directory_changed
from .input_prompt import create_prompt_session, get_user_input_async
from .rich_console import console, print_shell_result
from .shell_exec import detect_shell, execute_shell, format_shell_message
from .shell_parse import split_command_and_query
from .tool_picker import run_tool_picker_async
from .tool_wizard import inject_manual_tool_result, run_tool_wizard
from .undo import ConversationTree, run_undo_picker_async
from .undo.picker_state import ACTION_FORK, ACTION_SWITCH

if TYPE_CHECKING:
    from ..mcp.llm_tool_router import LLMToolRouter
    from .filtering_toolset import LLMToolFilter
    from pydantic_ai.toolsets import FilteredToolset


@dataclass
class CommandContext:
    """Shared state for command execution."""

    model: str
    base_url: Optional[str]
    max_steps: int
    max_retries: int = DEFAULT_MAX_RETRIES
    debug: bool = False
    debug_logger: Any = None
    available_tools: list = field(default_factory=list)
    disabled_tools: set = field(default_factory=set)
    settings: Any = None
    root_directory: str = ""
    previous_directory: Optional[str] = None
    server: Any = None
    toolset: Any = None
    # Study mode
    study_mode: bool = False
    study_topic: Optional[str] = None
    # Permission settings
    ask_tool_permission: bool = True
    # Tool filtering
    dynamic_tool_filtering: bool = False
    tool_filter_model: Optional[str] = None
    tool_router: Optional["LLMToolRouter"] = None
    filtering_toolset: Optional["FilteredToolset"] = None
    llm_filter: Optional["LLMToolFilter"] = None
    # Stashed by /config for the interactive config picker
    config_runtime_values: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for legacy command interface."""
        return asdict(self)

@dataclass
class CommandResult:
    """Result of command execution."""

    should_exit: bool = False
    should_continue: bool = False  # Skip query execution
    history_cleared: bool = False
    directory_changed: bool = False
    model_switched: bool = False
    new_model: Optional[str] = None
    wizard_result: Any = None
    study_mode_changed: bool = False
    config_changed: bool = False
    # When set, the main loop feeds this synthetic prompt to the LLM in place
    # of whatever the user typed. Populated by `! cmd @@ ...` shell commands.
    shell_to_llm: Optional[str] = None


def _print_config_change_summary(
    current_values: dict[str, str], changes: dict[str, Any],
) -> None:
    """Render a Rich table showing each changed setting's old → new value."""
    summary = Table(
        box=box.ROUNDED,
        show_header=True,
        title="Settings saved",
        title_style="bold green",
    )
    summary.add_column("Setting", style="cyan", no_wrap=True)
    summary.add_column("Old", style="dim")
    summary.add_column("New", style="bold green")
    for key, new_val in changes.items():
        old_val = current_values.get(key, "?")
        display_new = "***" if key == "api_key" and new_val else str(new_val)
        display_old = "***" if key == "api_key" and old_val else str(old_val)
        summary.add_row(key, display_old, display_new)

    console.print()
    console.print(summary)


class InteractiveLoop:
    """Manages the interactive REPL loop."""

    def __init__(
        self,
        context: CommandContext,
        on_model_switch: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the interactive loop.

        Args:
            context: Shared command context.
            on_model_switch: Callback when model is switched.
        """
        self.context = context
        self.on_model_switch = on_model_switch
        self.tree: ConversationTree = ConversationTree.empty()
        self.baseline_shown: bool = False
        self._pending_prompt: Optional[str] = None
        self._prompt_session = create_prompt_session(
            get_current_dir=lambda: self.context.root_directory
        )

    @property
    def history(self) -> list:
        """Live message list of the current branch (backwards-compat view)."""
        return self.tree.current.messages

    @history.setter
    def history(self, messages: list) -> None:
        self.tree.update_current(messages)

    async def get_input(self) -> Optional[str]:
        """Get user input asynchronously, or replay a one-shot directive."""
        if self._pending_prompt is not None:
            pending = self._pending_prompt
            self._pending_prompt = None
            console.print(f"[dim]↻ directive:[/dim] [cyan]{pending}[/cyan]")
            return pending
        return await get_user_input_async(self._prompt_session)

    def update_history(self, messages: list) -> None:
        """Update conversation history on the current branch."""
        self.tree.update_current(messages)

    def clear_history(self) -> None:
        """Drop every branch and start fresh with a single empty root."""
        self.tree.clear()

    async def process_command(self, user_input: str) -> CommandResult:
        """Process a slash or shell command and return result.

        Args:
            user_input: User input string.

        Returns:
            CommandResult with action flags.
        """
        result = CommandResult()

        # Shell passthrough: `! <cmd>` runs locally, `! <cmd> @@ <query>` also
        # sends the output to the LLM. Must be checked before slash-command
        # dispatch because `!` is not a slash prefix.
        stripped = user_input.lstrip()
        if stripped.startswith("!"):
            await self._handle_shell_command(stripped[1:], result)
            return result

        # Commands now modify self.context directly - no sync needed
        was_command, action = execute_command(user_input, self.context)

        if not was_command:
            return result

        result.should_continue = True

        if action == "exit":
            result.should_exit = True
            return result

        # Use dispatch table for action handling
        handler = self._ACTION_HANDLERS.get(action)
        if handler:
            await handler(self, result)

        return result

    async def _handle_shell_command(
        self, raw: str, result: CommandResult,
    ) -> None:
        """Run a `!`-prefixed shell command and optionally pipe output to LLM."""
        shell_kind, shell_path = detect_shell()
        cmd, query = split_command_and_query(raw, shell_kind)
        if not cmd:
            console.print("[yellow]Empty shell command — nothing to run.[/yellow]")
            result.should_continue = True
            return

        try:
            shell_result = await execute_shell(
                cmd=cmd,
                cwd=self.context.root_directory,
                timeout=SHELL_COMMAND_TIMEOUT,
                shell=shell_kind,
                shell_path=shell_path,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Shell command interrupted.[/dim]")
            result.should_continue = True
            return

        print_shell_result(shell_result)

        if query is None:
            result.should_continue = True
        else:
            result.shell_to_llm = format_shell_message(shell_result, query)

    # Action handler methods for dispatch table
    async def _action_run_wizard(self, result: CommandResult) -> None:
        """Handle run_wizard action."""
        wizard_result = await self._run_wizard()
        if wizard_result and wizard_result.success:
            result.wizard_result = wizard_result
            self.history = inject_manual_tool_result(self.history, wizard_result)

    async def _action_history_cleared(self, result: CommandResult) -> None:
        """Handle history_cleared action."""
        result.history_cleared = True
        self.clear_history()

    async def _action_open_browser(self, result: CommandResult) -> None:
        """Handle open_browser action."""
        await self._handle_directory_browser()
        result.directory_changed = True

    async def _action_cd_success(self, result: CommandResult) -> None:
        """Handle cd_success action."""
        result.directory_changed = True

    async def _action_model_switch(self, result: CommandResult) -> None:
        """Handle model_switch_requested action."""
        result.model_switched = True
        result.new_model = self.context.model
        if self.on_model_switch:
            self.on_model_switch(self.context.model)
        console.print("[green]Model switched successfully![/green]")
        console.print()

    async def _action_open_tool_picker(self, result: CommandResult) -> None:
        """Handle open_tool_picker action."""
        await self._handle_tool_picker()

    async def _action_study_mode_enabled(self, result: CommandResult) -> None:
        """Handle study_mode_enabled action."""
        self.context.study_mode = True
        result.study_mode_changed = True
        self.clear_history()  # Fresh start for study session

    async def _action_study_mode_disabled(self, result: CommandResult) -> None:
        """Handle study_mode_disabled action."""
        self.context.study_mode = False
        self.context.study_topic = None
        result.study_mode_changed = True
        self.clear_history()  # Fresh start after study

    async def _action_config_changed(self, result: CommandResult) -> None:
        """Handle config_changed action."""
        result.config_changed = True

    async def _action_filter_model_switch(self, result: CommandResult) -> None:
        """Handle filter_model_switch_requested — trigger agent rebuild.

        The rebuild path in run_interactive() calls _resolve_filter_model()
        which picks up the new context.tool_filter_model value.
        """
        result.config_changed = True

    async def _action_open_config_picker(self, result: CommandResult) -> None:
        """Handle open_config_picker action."""
        changed = await self._handle_config_picker()
        if changed:
            result.config_changed = True

    async def _action_open_undo_picker(self, result: CommandResult) -> None:
        """Handle open_undo_picker action — fork/switch the conversation tree."""
        await self._handle_undo_picker()

    # Action dispatch table — maps action strings to handler methods.
    # Return values not listed here (e.g. "model_invalid", "config_error",
    # "filter_model_shown") are informational no-ops: the command already
    # printed its output, so the loop just continues to the next prompt.
    _ACTION_HANDLERS: dict[str, Callable[["InteractiveLoop", CommandResult], Awaitable[None]]] = {
        "run_wizard": _action_run_wizard,
        "history_cleared": _action_history_cleared,
        "open_browser": _action_open_browser,
        "cd_success": _action_cd_success,
        "model_switch_requested": _action_model_switch,
        "open_tool_picker": _action_open_tool_picker,
        "open_config_picker": _action_open_config_picker,
        "open_undo_picker": _action_open_undo_picker,
        "study_mode_enabled": _action_study_mode_enabled,
        "study_mode_disabled": _action_study_mode_disabled,
        "config_changed": _action_config_changed,
        "filter_model_switch_requested": _action_filter_model_switch,
    }

    async def _run_wizard(self) -> Any:
        """Run the tool wizard."""
        return await run_tool_wizard(
            tools=self.context.available_tools,
            root_directory=self.context.root_directory,
            server=self.context.server,
        )

    async def _handle_directory_browser(self) -> None:
        """Handle interactive directory browser."""
        new_dir = await run_directory_browser_async(self.context.root_directory)
        if new_dir and new_dir != self.context.root_directory:
            old_dir = self.context.root_directory
            self.context.previous_directory = old_dir
            self.context.root_directory = new_dir
            print_directory_changed(old_dir, new_dir)

    async def _handle_tool_picker(self) -> None:
        """Handle interactive tool picker."""
        new_disabled = await run_tool_picker_async(
            available_tools=self.context.available_tools,
            disabled_tools=self.context.disabled_tools,
        )

        if new_disabled is None:
            return

        if new_disabled == self.context.disabled_tools:
            console.print("[dim]No changes made[/dim]")
            return

        self.context.disabled_tools = new_disabled

        # Persist to config
        config = load_config()
        config.agent.disabled_tools = list(new_disabled)
        save_config(config)

        console.print()
        console.print("[green]Tool settings saved![/green]")
        console.print("[yellow]Restart agent (/exit then run again) to apply changes[/yellow]")
        console.print()

    async def _handle_config_picker(self) -> bool:
        """Handle interactive config picker.

        Returns:
            True if any settings were changed and saved, False if cancelled.
        """
        # Get runtime values stashed by cmd_config
        current_values = self.context.config_runtime_values
        if not current_values:
            return False

        changes = await run_config_picker_async(current_values)
        if not changes:
            return False

        config = load_config()
        for key, new_value in changes.items():
            assign_config_value(config, key, new_value)
            if hasattr(self.context, key):
                setattr(self.context, key, new_value)

        if not save_config(config):
            console.print("[red]Warning: Failed to save config to disk[/red]")
        reload_config()

        _print_config_change_summary(current_values, changes)

        if _RESTART_REQUIRED_SETTINGS & changes.keys():
            console.print("[yellow]Restart agent (/exit then run again) to apply changes[/yellow]")
        console.print()

        # Only trigger an agent rebuild when a setting that's baked into the
        # agent actually changed; pure data-layer settings (default_limit, etc.)
        # don't need it. Matches the semantics of /config set via _cmd_config_set_value.
        return bool(_REBUILD_REQUIRED_SETTINGS & changes.keys())

    async def _handle_undo_picker(self) -> None:
        """Run the undo picker and apply fork/switch + directive to the tree."""
        result = await run_undo_picker_async(self.tree)
        if result is None:
            return

        if result.action == ACTION_FORK:
            new_id = self.tree.fork(result.branch_id, result.fork_message_index)
            console.print(
                f"[green]Forked new branch [bold]{new_id}[/bold] from "
                f"{result.branch_id}@{result.fork_message_index}[/green]",
            )
        elif result.action == ACTION_SWITCH:
            self.tree.switch(result.branch_id)
            console.print(f"[green]Switched to branch [bold]{result.branch_id}[/bold][/green]")

        console.print(
            "[dim]Conversation history updated. Files on disk are unchanged — "
            "use git to revert file edits.[/dim]",
        )
        console.print()

        if result.directive:
            self._pending_prompt = result.directive
