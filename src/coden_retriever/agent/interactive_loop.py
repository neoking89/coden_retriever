"""Interactive REPL loop for the coding agent.

Handles:
- Command dispatch and execution
- Tool wizard integration
- Directory navigation
- Model switching
- Settings management

Follows Single Responsibility Principle - only handles REPL control flow.
"""

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Callable, Optional

from .commands import execute_command

if TYPE_CHECKING:
    from ..mcp.tool_filter import ToolFilter
    from .filtering_toolset import SemanticToolFilter
    from pydantic_ai.toolsets import FilteredToolset
from .directory_browser import run_directory_browser_async, print_directory_changed
from .input_prompt import create_prompt_session, get_user_input_async
from .rich_console import console
from .tool_picker import run_tool_picker_async
from .tool_wizard import inject_manual_tool_result, run_tool_wizard


@dataclass
class CommandContext:
    """Shared state for command execution."""

    model: str
    base_url: Optional[str]
    max_steps: int
    max_retries: int = 3
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
    tool_filter: Optional["ToolFilter"] = None
    filtering_toolset: Optional["FilteredToolset"] = None
    semantic_filter: Optional["SemanticToolFilter"] = None
    tool_filter_threshold: float = 0.5

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
        self.history: list = []
        self._prompt_session = create_prompt_session(
            get_current_dir=lambda: self.context.root_directory
        )

    async def get_input(self) -> Optional[str]:
        """Get user input asynchronously."""
        return await get_user_input_async(self._prompt_session)

    def update_history(self, messages: list) -> None:
        """Update conversation history."""
        self.history = messages

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    async def process_command(self, user_input: str) -> CommandResult:
        """Process a slash command and return result.

        Args:
            user_input: User input string.

        Returns:
            CommandResult with action flags.
        """
        result = CommandResult()

        # Commands now modify self.context directly - no sync needed
        was_command, action = execute_command(user_input, self.context)

        if not was_command:
            return result

        result.should_continue = True

        if action == "exit":
            result.should_exit = True
            return result

        if action == "run_wizard":
            wizard_result = await self._run_wizard()
            if wizard_result and wizard_result.success:
                result.wizard_result = wizard_result
                self.history = inject_manual_tool_result(self.history, wizard_result)

        elif action == "history_cleared":
            result.history_cleared = True
            self.clear_history()

        elif action == "open_browser":
            await self._handle_directory_browser()
            result.directory_changed = True

        elif action == "cd_success":
            # Directory already updated by command
            result.directory_changed = True

        elif action == "model_switch_requested":
            # Model already updated by command on self.context
            result.model_switched = True
            result.new_model = self.context.model
            if self.on_model_switch:
                self.on_model_switch(self.context.model)
            console.print("[green]Model switched successfully![/green]")
            console.print()

        elif action == "open_tool_picker":
            await self._handle_tool_picker()

        elif action == "study_mode_enabled":
            # study_topic already set by command
            self.context.study_mode = True
            result.study_mode_changed = True
            self.clear_history()  # Fresh start for study session

        elif action == "study_mode_disabled":
            self.context.study_mode = False
            self.context.study_topic = None
            result.study_mode_changed = True
            self.clear_history()  # Fresh start after study

        elif action == "config_changed":
            result.config_changed = True

        return result

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
        from ..config_loader import load_config, save_config

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
