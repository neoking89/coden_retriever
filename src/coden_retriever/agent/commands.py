"""Slash command system for the interactive coding agent.

Provides a registry for /commands with tab auto-completion support.
Commands allow users to configure settings, switch models, and control
the agent without sending messages to the LLM.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich import box
from rich.table import Table

from ..cache import CacheManager
from ..config import get_central_cache_root, get_project_cache_dir
from ..config_loader import (
    SETTING_LOCATIONS,
    SETTING_METADATA,
    assign_config_value,
    get_config_file,
    load_config,
    parse_config_value,
    reload_config,
    reset_config,
    save_config,
    validate_config_value,
)
from .debug_logger import create_debug_logger
from .rich_console import Panel, console

if TYPE_CHECKING:
    from .interactive_loop import CommandContext


@dataclass
class Command:
    """A registered slash command."""

    name: str
    description: str
    handler: Callable[..., Any]
    usage: str = ""
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    """Registry for slash commands with fuzzy matching support."""

    def __init__(self):
        self.commands: dict[str, Command] = {}
        self._alias_map: dict[str, str] = {}  # alias -> command name

    def register(
        self,
        name: str,
        description: str,
        usage: str = "",
        aliases: list[str] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a command handler.

        Args:
            name: Command name (without leading /).
            description: Short description for /help.
            usage: Usage string showing arguments.
            aliases: Alternative names for the command.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            cmd = Command(
                name=name,
                description=description,
                handler=func,
                usage=usage,
                aliases=aliases or [],
            )
            self.commands[name] = cmd

            # Register aliases
            for alias in (aliases or []):
                self._alias_map[alias] = name

            return func
        return decorator

    def get_command(self, name: str) -> Command | None:
        """Get a command by name or alias."""
        # Direct match
        if name in self.commands:
            return self.commands[name]
        # Alias match
        if name in self._alias_map:
            return self.commands[self._alias_map[name]]
        return None

    def parse(self, user_input: str) -> tuple[Command | None, list[str]]:
        """Parse user input to extract command and arguments.

        Args:
            user_input: Raw user input string.

        Returns:
            Tuple of (Command or None, list of arguments).
        """
        if not user_input.startswith("/"):
            return None, []

        # Split into parts
        parts = user_input[1:].split()
        if not parts:
            return None, []

        cmd_name = parts[0].lower()
        args = parts[1:]

        command = self.get_command(cmd_name)
        return command, args

    def get_suggestions(self, partial: str, threshold: float = 0.6) -> list[str]:
        """Get command suggestions for a partial/misspelled command.

        Args:
            partial: The partial or misspelled command name.
            threshold: Minimum similarity ratio (0-1) to include.

        Returns:
            List of suggested command names, sorted by similarity.
        """
        suggestions = []
        all_names = list(self.commands.keys()) + list(self._alias_map.keys())

        for name in all_names:
            ratio = SequenceMatcher(None, partial.lower(), name.lower()).ratio()
            if ratio >= threshold:
                # Resolve alias to actual command name
                actual_name = self._alias_map.get(name, name)
                if actual_name not in suggestions:
                    suggestions.append((actual_name, ratio))

        # Sort by similarity, highest first
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in suggestions]

    def get_all_command_names(self) -> list[str]:
        """Get all command names (including aliases) for auto-completion."""
        names = list(self.commands.keys())
        names.extend(self._alias_map.keys())
        return sorted(set(names))

    def get_completions(self, partial: str) -> list[tuple[str, str]]:
        """Get command completions for partial input.

        Args:
            partial: Partial command name (without leading /).

        Returns:
            List of (command_name, description) tuples.
        """
        completions = []
        partial_lower = partial.lower()

        for name, cmd in self.commands.items():
            if name.startswith(partial_lower):
                completions.append((name, cmd.description))

        # Also check aliases
        for alias, cmd_name in self._alias_map.items():
            if alias.startswith(partial_lower):
                cmd = self.commands[cmd_name]
                completions.append((alias, f"(alias for {cmd_name}) {cmd.description}"))

        return sorted(completions, key=lambda x: x[0])


# Global command registry
registry = CommandRegistry()


@registry.register("help", "Show all available commands", usage="/help")
def cmd_help(args: list[str], context: "CommandContext") -> str:
    """Display help table with all commands."""
    table = Table(
        title="Available Commands",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold green",
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    for name, cmd in sorted(registry.commands.items()):
        usage = cmd.usage or f"/{name}"
        table.add_row(usage, cmd.description)

    console.print()
    console.print(table)
    console.print()
    return "help_displayed"


@registry.register(
    "config",
    "View or modify settings",
    usage="/config [set <key> <value> | reset]",
)
def cmd_config(args: list[str], context: "CommandContext") -> str:
    """Handle config commands: show, set, reset."""
    config = load_config()

    if not args:
        # Show current settings with descriptions
        console.print()
        console.print("[bold cyan]Configuration[/bold cyan]")
        console.print(f"[dim]File: {get_config_file()}[/dim]")
        console.print()

        # Get values from context (runtime) or config (persisted)
        model = context.model or config.model.default or "ollama:"
        base_url = context.base_url or config.model.base_url or "(auto)"
        max_steps = context.max_steps if context.max_steps else config.agent.max_steps
        max_retries = context.max_retries if context.max_retries else config.agent.max_retries
        debug = context.debug if context.debug is not None else config.agent.debug
        tool_instructions = config.agent.tool_instructions
        ask_tool_permission = config.agent.ask_tool_permission
        dynamic_tool_filtering = config.agent.dynamic_tool_filtering
        tool_filter_threshold = config.agent.tool_filter_threshold
        # Model generation parameters
        gen = config.model.generation
        temperature = gen.temperature
        max_tokens = gen.max_tokens
        timeout = gen.timeout
        api_key = gen.api_key

        # Build settings list from centralized metadata
        runtime_values = {
            "model": str(model),
            "base_url": str(base_url),
            "max_steps": str(max_steps),
            "max_retries": str(max_retries),
            "debug": str(debug).lower(),
            "tool_instructions": str(tool_instructions).lower(),
            "ask_tool_permission": str(ask_tool_permission).lower(),
            "dynamic_tool_filtering": str(dynamic_tool_filtering).lower(),
            "tool_filter_threshold": str(tool_filter_threshold),
            "temperature": str(temperature),
            "max_tokens": str(max_tokens) if max_tokens is not None else "(model default)",
            "timeout": str(timeout),
            "api_key": "***" if api_key else "(not set)",
            "host": config.daemon.host,
            "port": str(config.daemon.port),
            "daemon_timeout": str(config.daemon.daemon_timeout),
            "max_projects": str(config.daemon.max_projects),
            "default_tokens": str(config.search.default_tokens),
            "default_limit": str(config.search.default_limit),
            "semantic_model_path": config.search.semantic_model_path or "(default)",
        }
        settings = [
            (meta.key, runtime_values[meta.key], meta.long_desc)
            for meta in SETTING_METADATA.values()
        ]

        table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1))
        table.add_column("Setting", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="green", no_wrap=True)
        table.add_column("Description", style="dim")

        for key, value, desc in settings:
            table.add_row(key, value, desc)

        console.print(table)
        console.print()
        console.print("[dim]Usage: /config set <setting> <value>  |  /config reset[/dim]")
        console.print()
        return "config_shown"

    subcmd = args[0].lower()

    if subcmd == "set" and len(args) >= 3:
        key = args[1].lower()
        value_str = " ".join(args[2:])

        # Parse value using centralized function
        success, parsed_value, error = parse_config_value(key, value_str)
        if not success:
            console.print(f"[red]{error}[/red]")
            return "config_error"

        # Validate value using centralized function
        is_valid, error = validate_config_value(key, parsed_value)
        if not is_valid:
            console.print(f"[red]{error}[/red]")
            return "config_error"

        # Handle special case for base_url "(auto)"
        if key == "base_url" and value_str == "(auto)":
            parsed_value = None

        # Update context (for runtime settings that exist on context)
        if hasattr(context, key):
            setattr(context, key, parsed_value)

        # Persist to config using centralized function
        if key in SETTING_LOCATIONS:
            assign_config_value(config, key, parsed_value)
        else:
            # Key exists in metadata but not in locations - display-only setting
            console.print(f"[yellow]Warning: {key} is read-only[/yellow]")
            return "config_error"

        if not save_config(config):
            console.print("[red]Warning: Failed to save config to disk[/red]")

        # Update the global config cache so get_config() returns fresh values
        reload_config()

        display_value = "***" if key == "api_key" and parsed_value else parsed_value
        console.print()
        console.print(f"[green]{key}[/green] = [cyan]{display_value}[/cyan]")

        # Settings that truly require session restart (can't be changed mid-session)
        restart_required = ("dynamic_tool_filtering",)
        if key in restart_required:
            console.print("[yellow]Restart agent (/exit then run again) to apply[/yellow]")
            console.print()
            return "config_set"

        # Settings that require agent rebuild (applied immediately)
        rebuild_required = (
            "tool_instructions", "tool_filter_threshold",
            "temperature", "max_tokens", "timeout", "api_key",
            "max_retries", "max_steps"
        )

        if key in rebuild_required:
            console.print("[dim]Applied immediately[/dim]")
            console.print()
            return "config_changed"

        console.print()
        return "config_set"

    elif subcmd == "reset":
        # Reset to defaults
        reset_config()
        # Use reload_config to update the global cache
        new_config = reload_config()

        # Update context directly
        context.model = new_config.model.default
        context.base_url = new_config.model.base_url
        context.max_steps = new_config.agent.max_steps
        context.max_retries = new_config.agent.max_retries
        context.debug = new_config.agent.debug
        context.ask_tool_permission = new_config.agent.ask_tool_permission
        context.dynamic_tool_filtering = new_config.agent.dynamic_tool_filtering
        context.tool_filter_threshold = new_config.agent.tool_filter_threshold

        console.print()
        console.print("[green]Configuration reset to defaults[/green]")
        console.print()
        return "config_reset"

    else:
        console.print("[red]Usage: /config [set <key> <value> | reset][/red]")
        return "config_error"


@registry.register(
    "model",
    "Show or switch the current model",
    usage="/model [name]",
    aliases=["m"],
)
def cmd_model(args: list[str], context: "CommandContext") -> str:
    """Show current model or switch to a new one."""
    current_model = context.model or "ollama:"

    if not args:
        # Show current model and usage guide
        console.print()
        console.print(f"[bold]Current model:[/bold] [cyan]{current_model}[/cyan]")
        console.print()
        console.print("[bold]Model formats:[/bold]")
        console.print("  [cyan]ollama:model-name[/cyan]      Ollama (auto: localhost:11434)")
        console.print("  [cyan]llamacpp:model-name[/cyan]    llama-cpp-server (auto: localhost:8080)")
        console.print("  [cyan]openai:gpt-4o[/cyan]          Official OpenAI API (needs OPENAI_API_KEY)")
        console.print()
        console.print("[bold]OpenAI-compatible endpoints[/bold] (vLLM, LM Studio, etc):")
        console.print("  Use [cyan]/config set base_url http://localhost:8000/v1[/cyan]")
        console.print("  Then [cyan]/model your-model-name[/cyan] (no prefix)")
        console.print()
        return "model_shown"

    new_model = args[0].strip()

    # Validate model name - check for empty or whitespace-only
    if not new_model:
        console.print()
        console.print("[red]Error: Model name cannot be empty[/red]")
        console.print("[dim]Example: /model ollama:qwen2.5-coder:14b[/dim]")
        return "model_invalid"

    # Validate model name - check for empty model after prefix
    if new_model in ("ollama:", "llamacpp:", "openai:"):
        console.print()
        console.print(f"[red]Error: Model name required after '{new_model}'[/red]")
        console.print(f"[dim]Example: {new_model}qwen2.5-coder:14b[/dim]")
        return "model_invalid"

    # Update model directly on context
    context.model = new_model

    console.print()
    console.print(f"[yellow]Switching to {new_model}...[/yellow]")

    # Persist to config file
    config = load_config()
    config.model.default = new_model
    if save_config(config):
        console.print(f"[dim]Model saved to {get_config_file()}[/dim]")
    else:
        console.print("[red]Warning: Failed to save model to config file![/red]")

    return "model_switch_requested"


@registry.register(
    "tools",
    "Open interactive tool settings",
    usage="/tools",
    aliases=["t"],
)
def cmd_tools(args: list[str], context: "CommandContext") -> str:
    """Open interactive tool picker to enable/disable MCP tools.
    
    Opens a checkbox-based UI where you can:
    - Navigate with arrow keys
    - Toggle tools with Space
    - Enable all with 'a', disable all with 'n'
    - Apply changes with Enter
    """
    # Signal the agent loop to open the tool picker
    return "open_tool_picker"


@registry.register("run", "Open interactive tool wizard", usage="/run", aliases=["r", "execute"])
def cmd_run(args: list[str], context: "CommandContext") -> str:
    """Trigger the tool wizard - returns signal for agent loop to handle."""
    return "run_wizard"


@registry.register("clear", "Clear conversation history", usage="/clear", aliases=["c"])
def cmd_clear(args: list[str], context: "CommandContext") -> str:
    """Clear the conversation history."""
    console.print()
    console.print("[green]Conversation history cleared[/green]")
    console.print()
    return "history_cleared"


@registry.register("exit", "Exit the agent", usage="/exit", aliases=["quit", "q"])
def cmd_exit(args: list[str], context: "CommandContext") -> str:
    """Exit the agent."""
    return "exit"


@registry.register(
    "debug",
    "Toggle debug mode (logs all prompts/tool calls)",
    usage="/debug [on|off]",
    aliases=["d"],
)
def cmd_debug(args: list[str], context: "CommandContext") -> str:
    """Toggle or show debug mode status.

    When debug is enabled:
    - Complete system prompts are logged
    - All tool calls with timestamps are logged
    - Model responses and thinking traces are logged
    - Logs are streamed to ~/.coden-retriever/{project}/logs/
    """
    config = load_config()
    current_debug = context.debug if context.debug is not None else config.agent.debug
    debug_logger = context.debug_logger

    if not args:
        # Show current status
        console.print()
        status = "[bold green]ENABLED[/bold green]" if current_debug else "[bold red]DISABLED[/bold red]"
        console.print(f"[bold]Debug mode:[/bold] {status}")

        if current_debug and debug_logger and debug_logger.enabled:
            log_path = debug_logger.get_log_path()
            if log_path:
                console.print(f"[dim]Log file: {log_path}[/dim]")
        else:
            console.print()
            console.print("[dim]When enabled, debug logs include:[/dim]")
            console.print("[dim]  - Complete system prompts[/dim]")
            console.print("[dim]  - All tool calls with timestamps[/dim]")
            console.print("[dim]  - Model responses and thinking traces[/dim]")
            console.print("[dim]  - Logs saved to: ~/.coden-retriever/{project}/logs/[/dim]")

        console.print()
        console.print("[dim]Usage: /debug on  or  /debug off[/dim]")
        console.print()
        return "debug_shown"

    action = args[0].lower()
    new_debug = action in ("on", "true", "1", "yes", "enable")

    if action not in ("on", "off", "true", "false", "1", "0", "yes", "no", "enable", "disable"):
        console.print("[red]Usage: /debug [on|off][/red]")
        return "debug_error"

    # Update config (persisted for future sessions)
    config.agent.debug = new_debug
    save_config(config)
    context.debug = new_debug

    console.print()
    if new_debug:
        # Create new debug logger IMMEDIATELY so logging starts now
        root_directory = context.root_directory or "."
        new_logger = create_debug_logger(root_directory, debug=True)

        # Log session info for the new logger
        new_logger.log_session_start(
            model=context.model or "unknown",
            base_url=context.base_url,
            max_steps=context.max_steps or 10,
        )
        new_logger._write_subsection("DEBUG ENABLED MID-SESSION")

        # Close old logger if exists, replace with new one
        if debug_logger:
            debug_logger.close()
        context.debug_logger = new_logger

        console.print("[bold green]Debug mode enabled[/bold green]")
        console.print(f"[cyan]Logging started: {new_logger.get_log_path()}[/cyan]")
    else:
        console.print("[bold red]Debug mode disabled[/bold red]")
        if debug_logger and debug_logger.enabled:
            log_path = debug_logger.get_log_path()
            debug_logger.close()
            context.debug_logger = create_debug_logger(".", debug=False)  # Disabled logger
            if log_path:
                console.print(f"[dim]Log saved to: {log_path}[/dim]")
    console.print()

    return "debug_toggled"


@registry.register(
    "cd",
    "Change working directory (interactive browser)",
    usage="/cd [path]",
    aliases=["dir", "chdir"],
)
def cmd_cd(args: list[str], context: "CommandContext") -> str:
    """Change the working directory with interactive browser.

    If a path is provided, navigate directly to it.
    If no path, open interactive directory browser.
    """
    current_dir = Path(context.root_directory) if context.root_directory else Path.cwd()

    if args:
        # Direct path navigation
        target = args[0]

        # Handle special cases
        if target == "~":
            new_path = Path.home()
        elif target == "-":
            # Go to previous directory
            prev = context.previous_directory
            if prev:
                new_path = Path(prev)
            else:
                console.print("[yellow]No previous directory[/yellow]")
                return "cd_error"
        else:
            # Resolve relative or absolute path
            if Path(target).is_absolute():
                new_path = Path(target)
            else:
                new_path = current_dir / target

        new_path = new_path.resolve()

        if not new_path.exists():
            console.print(f"[red]Directory not found: {new_path}[/red]")
            return "cd_error"

        if not new_path.is_dir():
            console.print(f"[red]Not a directory: {new_path}[/red]")
            return "cd_error"

        # Store previous and update current directly on context
        context.previous_directory = str(current_dir)
        context.root_directory = str(new_path)

        console.print()
        console.print(f"[green]Changed to:[/green] [bold]{new_path}[/bold]")
        console.print()
        return "cd_success"

    # No args - open interactive browser
    return "open_browser"


@registry.register(
    "study",
    "Enter adaptive study mode to learn the codebase",
    usage="/study [topic]",
    aliases=["learn", "quiz"],
)
def cmd_study(args: list[str], context: "CommandContext") -> str:
    """Enter study mode where the agent helps you master the codebase.

    The agent will first assess your knowledge level and then guide you
    through the architecture or specific features using real code examples.
    """
    default_topic = "General Architecture"
    if args:
        topic = " ".join(args)
    else:
        # Prompt user for topic
        console.print()
        console.print("[bold cyan]What would you like to study?[/bold cyan]")
        console.print("[dim]Examples: architecture, data flow, error handling, authentication[/dim]")
        console.print(f"[dim]Press Enter for default: {default_topic}[/dim]")
        console.print()
        topic = console.input("[bold]Topic:[/bold] ").strip()
        if not topic:
            topic = default_topic

    context.study_topic = topic

    console.print()
    console.print(Panel(
        f"[bold cyan]Adaptive Study Mode[/bold cyan]\n\n"
        f"Topic: [bold white]{topic}[/bold white]\n\n"
        f"[dim]The tutor will assess your level and guide your learning.\n"
        f"Type /exit-study to return to normal mode.[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    return "study_mode_enabled"


@registry.register(
    "exit-study",
    "Exit study mode and return to normal assistant",
    usage="/exit-study",
    aliases=["stop-study"],
)
def cmd_exit_study(args: list[str], context: "CommandContext") -> str:
    """Exit study mode and return to normal assistant mode."""
    if not context.study_mode:
        console.print("[yellow]Not currently in study mode[/yellow]")
        return "not_in_study_mode"

    console.print()
    console.print("[green]Exiting study mode[/green]")
    console.print("[dim]Back to normal assistant mode.[/dim]")
    console.print()

    return "study_mode_disabled"


@registry.register(
    "cache",
    "Manage project caches",
    usage="/cache [list|clear|status]",
)
def cmd_cache(args: list[str], context: "CommandContext") -> str:
    """Manage caches. Sub-commands: list, clear, status."""
    root = context.root_directory
    if not root:
        console.print("[red]No project directory set[/red]")
        return "no_project"

    root_path = Path(root)
    sub_cmd = args[0] if args else "list"

    console.print()

    if sub_cmd == "list":
        # List all cached projects
        caches = CacheManager.list_all_caches()
        if not caches:
            console.print("[yellow]No cached projects found[/yellow]")
            console.print(f"[dim]Cache directory: {get_central_cache_root()}[/dim]")
            return "cache_empty"

        console.print(f"[bold]Cached projects ({len(caches)}):[/bold]")
        console.print(f"[dim]Location: {get_central_cache_root()}[/dim]")
        console.print()

        total_size = 0
        for cache in caches:
            total_size += cache["size_mb"]
            source = cache["source_dir"]
            if len(source) > 50:
                source = "..." + source[-47:]
            console.print(f"  [cyan]{source}[/cyan]")
            console.print(f"    [dim]Entities: {cache['entity_count']:,} | Files: {cache['file_count']:,} | Size: {cache['size_mb']:.1f} MB[/dim]")

        console.print()
        console.print(f"[bold]Total: {total_size:.1f} MB[/bold]")
        return "cache_listed"

    elif sub_cmd == "clear":
        # Check for --all flag
        clear_all = "--all" in args or "-a" in args

        if clear_all:
            count, errors = CacheManager.clear_all_caches()
            if count > 0:
                console.print(f"[green]Cleared {count} project cache(s)[/green]")
            else:
                console.print("[yellow]No caches to clear[/yellow]")
            for error in errors:
                console.print(f"[red]  {error}[/red]")
            return "cache_cleared_all"

        # Clear cache for current project
        cache_dir = get_project_cache_dir(root_path)
        if not cache_dir.exists():
            console.print(f"[yellow]No cache found for: {root_path}[/yellow]")
            return "cache_not_found"

        if CacheManager.clear_cache_by_source_dir(root_path):
            console.print(f"[green]Cache cleared for: {root_path}[/green]")
            return "cache_cleared"
        else:
            console.print("[red]Failed to clear cache[/red]")
            return "cache_clear_failed"

    elif sub_cmd == "status":
        cache = CacheManager(root_path)
        status = cache.get_cache_status()

        if not status.get("exists"):
            console.print("[yellow]No cache for current project[/yellow]")
            console.print(f"[dim]Project: {root_path}[/dim]")
            return "cache_not_found"

        console.print("[bold]Cache Status:[/bold]")
        console.print(f"  Project: [cyan]{root_path}[/cyan]")
        console.print(f"  Cache: [dim]{status.get('cache_dir', 'N/A')}[/dim]")
        console.print(f"  Entities: {status.get('entity_count', 0):,}")
        console.print(f"  Files: {status.get('file_count', 0)}")
        if status.get("updated_at"):
            console.print(f"  Updated: {status['updated_at']}")

        changes = status.get("changes", {})
        if changes.get("added") or changes.get("modified") or changes.get("deleted"):
            console.print()
            console.print("  [yellow]Changes detected:[/yellow]")
            console.print(f"    Added: {changes.get('added', 0)}, Modified: {changes.get('modified', 0)}, Deleted: {changes.get('deleted', 0)}")

        return "cache_status"

    else:
        console.print(f"[red]Unknown subcommand: {sub_cmd}[/red]")
        console.print("[dim]Usage: /cache [list|clear|status][/dim]")
        return "cache_unknown_subcommand"


@registry.register(
    "cache-clear",
    "Clear cache for current project",
    usage="/cache-clear",
    aliases=["cc"],
)
def cmd_cache_clear(args: list[str], context: "CommandContext") -> str:
    """Shortcut to clear cache for current project."""
    return cmd_cache(["clear"], context)


@registry.register(
    "cache-list",
    "List all cached projects",
    usage="/cache-list",
    aliases=["cl"],
)
def cmd_cache_list(args: list[str], context: "CommandContext") -> str:
    """Shortcut to list all cached projects."""
    return cmd_cache(["list"], context)


def execute_command(user_input: str, context: "CommandContext") -> tuple[bool, str]:
    """Execute a slash command if the input is a command.

    Args:
        user_input: The raw user input.
        context: CommandContext object with agent state (modified directly).

    Returns:
        Tuple of (was_command, result_string).
        was_command is True if input was a slash command.
        result_string is the command result or empty string.
    """
    command, args = registry.parse(user_input)

    if command is None:
        # Check if it looks like a command (starts with /)
        if user_input.startswith("/"):
            partial = user_input[1:].split()[0] if user_input[1:] else ""
            suggestions = registry.get_suggestions(partial)

            console.print()
            console.print(f"[red]Unknown command: {user_input}[/red]")
            if suggestions:
                console.print(f"[dim]Did you mean: /{suggestions[0]} ?[/dim]")
            else:
                console.print("[dim]Type /help for available commands[/dim]")
            console.print()
            return True, "unknown_command"

        return False, ""

    try:
        result = command.handler(args, context)
        return True, result
    except Exception as e:
        console.print(f"[red]Command error: {e}[/red]")
        return True, "command_error"
