"""Rich console utilities for professional CLI output.

Provides styled console components for the coding agent:
- Color palette for different message types
- Formatted panels for ReAct steps
- Streaming text display with Live updates
- User input prompts
"""

import io
import sys
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from ._constants import WALL_CLOCK_FORMAT
from .models import ReActStep

if TYPE_CHECKING:
    from .shell_exec import ShellResult

# Fix Windows console encoding for Unicode support
if sys.platform == "win32":
    try:
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Older Python or non-standard environment

AGENT_THEME = Theme({
    "thought": "italic dim cyan",
    "action": "bold yellow",
    "action.tool": "bold yellow",
    "action.args": "yellow",
    "observation.success": "green",
    "observation.error": "bold red",
    "observation.border": "dim green",
    "step.number": "bold magenta",
    "user.prompt": "bold blue",
    "agent.prefix": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "info": "dim",
    "tokens": "dim cyan",
})

# Create themed console with force_terminal for better Windows compatibility
console = Console(theme=AGENT_THEME, force_terminal=True)

# Global reference to active Live display to allow pausing it
# This prevents rendering conflicts between Rich Live and prompt_toolkit
_active_live_display: Live | None = None
_active_live_lock = threading.Lock()


def set_active_live(live: Live | None) -> None:
    """Set the currently active Live display.

    Thread-safe: Uses a lock to prevent race conditions when called
    from the async event loop and thread pool executors simultaneously.

    Args:
        live: The Live display to register, or None to unregister.
    """
    global _active_live_display
    with _active_live_lock:
        _active_live_display = live


def get_active_live() -> Live | None:
    """Get the currently active Live display.

    Thread-safe: Uses a lock to ensure consistent reads when the
    display is being registered/unregistered from another thread.

    Returns:
        The currently active Live display, or None if no display is active.
    """
    with _active_live_lock:
        return _active_live_display


# Maximum characters for truncated content (set high to show full tool calls)
MAX_ARGS_DISPLAY = 10000  # Show full tool call arguments
MAX_OBSERVATION_DISPLAY = 10000  # Show full tool output

# ASCII art logo - Golden Retriever with code bone tag
# Generated from images/logo1.jpg (smaller version from images/ascii_logo.txt)
DOG_LOGO = """[yellow]                                =%%%%%%%%%%%%+
                           :%%#-.....::::.....-#%%-
                      =%%%%-..::::::::::::::::::..:%%%%+
                    %#:...::::::::::::::::::::::::::...:#%
                  *%:::::::::::::::::::::::::::::::::::::.##
                 #%.:::-%-:::::::::::::::::::::::::::%-:::.##
                %#.:::-%+::::::::::::::::::::::::::::=%-::::*%
              =%-:::::+%::::::-::::::::::::::::--:::::#*::::::%+
             %*.::::::%+::::+%%%=::::::::::::-%%%*::::=%::::::.*%
           -%::::::::-%-:::::%%#::::::::::::::#%%-:::::%-::::::::%+
          -%.::::::::-%::::::::::::::::::::::::::::::::%=::::::::.%+
          **:::::::::-%-:::::::::::::::::::::::::::::::%-:::::::::*#
          +%::::::::::%=:::::::::::-#%%%%%-:::::::::::-%::::::::::#*
           %*:::::::::#*::::::::::#%%%%%%%%%::::::::::+%:::::::::+%
            %*::::::::##::::::::::+%%%%%%%%*::::::::::*#::::::::+%
             +%-::::::##:::::::::::=%%%%%%+:::::::::::+%::::::-%*
               %*:::::%=:::*#:::::::::*#:::::::::#*:::-%:::::+%
                ##:::*%:::::##::::::::*#::::::::*#:::::#*:::*#
                -%::=%-::::::#%+:::+%%#%#%+:::+%%::::::-%+::#+
                 %#%%%::::::::%%%%*+==##===*%%%%::::::::#%%#%                 [/yellow]
[red]                     #%::::::::%%%====##====%%%::::::::%%
                     %%%%-:::..:*#====##====##:...:::#%%%                     [/red]
[cyan]                     %%%%%%*....*#=====+====*#....*%%%%%%:
                    %-+%%%%%%%%%*%==========###%%%%%%%%+-%
                   *#:::+%%%%%%%%%#========#%%%%%%%%%*:::**
                   %=%-:...*%%%%%%%%#====#%%%%%%%%*...::%=%
                   #+%-:....:#%%%#%%%%##%%%%#%%%#:....:-%+#
                     #*:::.%+===+%...%%%%...#*===+%.:::+#
                      %=::=#======++++++*+++======*+::=%
                       %*::%*======..==.=..======+%::+%
                        *%-=%+===-..-=.:=-..-====%=-%#
                          #%========-=.==-========#%
                           %=====#%%%%%%%%%%#=====#:
                           :%#*#%-%%-....-%%-%#*#%-
                                     %%%%:                                    [/cyan]"""


def format_step_rich(step: ReActStep) -> Group:
    """Format a single ReAct step as Rich renderables.

    Args:
        step: The ReAct step to format.

    Returns:
        A Rich Group containing the formatted step elements.
    """
    elements: list[Any] = []

    # Step header with number
    step_header = Text()
    step_header.append(f"[Step {step.step_number}] ", style="step.number")

    # Thought (italicized, dim cyan)
    if step.thought:
        thought_text = Text()
        thought_text.append(step_header)
        thought_text.append("💭 Thought: ", style="thought")
        thought_text.append(escape(step.thought.reasoning), style="thought")
        elements.append(thought_text)

    # Action (bold yellow with tool name highlighted)
    if step.action:
        action_text = Text()
        action_text.append(f"[Step {step.step_number}] ", style="step.number")
        action_text.append("🔧 Action: ", style="action")
        action_text.append(escape(step.action.tool_name), style="action.tool")

        # Format arguments
        args_str = ", ".join(f"{k}={v!r}" for k, v in step.action.tool_input.items())
        if len(args_str) > MAX_ARGS_DISPLAY:
            args_str = args_str[:MAX_ARGS_DISPLAY - 3] + "..."
        action_text.append(f"({escape(args_str)})", style="action.args")
        elements.append(action_text)

    # Observation (in a bordered panel)
    if step.observation:
        if step.observation.success:
            status = "[ok]"
            content = step.observation.result or ""
            style = "observation.success"
            border_style = "dim green"
        else:
            status = "[x]"
            content = step.observation.error or ""
            style = "observation.error"
            border_style = "dim red"

        # Truncate long content
        content_str = str(content)
        if len(content_str) > MAX_OBSERVATION_DISPLAY:
            content_str = content_str[:MAX_OBSERVATION_DISPLAY - 3] + "..."

        obs_text = Text()
        obs_text.append(f"{status} ", style=style)
        obs_text.append(escape(content_str))

        panel = Panel(
            obs_text,
            title="Observation",
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
        )
        elements.append(panel)

    return Group(*elements)


def print_step_rich(step: ReActStep) -> None:
    """Print a single ReAct step with Rich formatting."""
    console.print(format_step_rich(step))


def print_steps_rich(steps: list[ReActStep]) -> None:
    """Print all ReAct steps with Rich formatting."""
    if not steps:
        return

    console.print()  # Add spacing before steps
    for step in steps:
        print_step_rich(step)
    console.print()  # Add spacing after steps


print_steps = print_steps_rich


def print_agent_response(response_text: str) -> None:
    """Print the agent's response with Markdown rendering.

    Args:
        response_text: The agent's response (may contain Markdown).
    """
    console.print()

    # Create prefix text
    prefix = Text()
    prefix.append("Agent: ", style="agent.prefix")
    console.print(prefix, end="")

    # Render response as Markdown (handles code blocks with syntax highlighting)
    md = Markdown(response_text)
    console.print(md)
    console.print()


def get_user_input() -> str:
    """Get styled user input with Rich prompt.

    Returns:
        The user's input string, stripped.
    """
    console.print()
    console.print(Rule(style="dim"))
    console.print("[dim]Type 'tools' to list available tools, 'run' to execute one manually, 'exit' to quit[/dim]")

    try:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        return user_input.strip()
    except EOFError:
        return "exit"


def print_welcome(
    root_directory: str,
    max_steps: int,
    tool_count: int | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> None:
    """Print the welcome dashboard when the agent starts.

    Args:
        root_directory: The working directory path.
        max_steps: Maximum tool calls per query.
        tool_count: Number of available MCP tools, or None if still loading.
        model_name: The model being used (e.g., "ollama:llama3").
        base_url: The base URL for the API (if applicable).
    """
    # Print ASCII art dog logo
    console.print(DOG_LOGO, markup=True)

    # Build welcome content
    welcome_lines = Text()
    welcome_lines.append("Woof! coden-retriever is now in agent mode.\n")

    # Tip line
    welcome_lines.append("TIP: ", style="bold cyan")
    welcome_lines.append("When using ollama, first do: ", style="dim")
    welcome_lines.append("ollama pull <model>\n", style="cyan")
    welcome_lines.append("     Then run local models in agent model: ", style="dim")
    welcome_lines.append("/model ollama:qwen2.5-coder:14b\n", style="cyan")
    welcome_lines.append("     Or cloud models (ollama signin): ", style="dim")
    welcome_lines.append("/model ollama:gemma4:31b-cloud", style="cyan")
    welcome_lines.append(" or ", style="dim")
    welcome_lines.append("/model ollama:gpt-oss:20b-cloud\n", style="cyan")
    welcome_lines.append("     For GGUF: ", style="dim")
    welcome_lines.append("llama-server -m model.gguf", style="cyan")
    welcome_lines.append(", then in agent mode:", style="dim")
    welcome_lines.append("/model llamacpp:model\n", style="cyan")
    welcome_lines.append("     For remote servers: ", style="dim")
    welcome_lines.append("--base-url <url>", style="cyan")
    welcome_lines.append(" (e.g. http://192.168.1.100:11434/v1)\n", style="dim italic")

    # Model info
    if model_name:
        welcome_lines.append("Model: ", style="dim")
        welcome_lines.append(f"{model_name}", style="bold cyan")
        if base_url:
            welcome_lines.append(" @ ", style="dim")
            welcome_lines.append(f"{base_url}", style="dim")
        welcome_lines.append("\n")

    # Working directory info
    welcome_lines.append("Working directory: ", style="dim")
    welcome_lines.append(f"{root_directory}\n", style="bold")
    welcome_lines.append("Tools: ", style="dim")
    if tool_count is None:
        welcome_lines.append("...", style="dim italic")
    else:
        welcome_lines.append(f"{tool_count}", style="bold")
    welcome_lines.append(" | ", style="dim")
    welcome_lines.append("Max steps: ", style="dim")
    welcome_lines.append(f"{max_steps}\n", style="bold")

    # Commands line
    welcome_lines.append("Commands: ", style="bold green")
    commands = ["/help", "/cd", "/config", "/model", "/tools", "/run", "/study", "/copy", "/clear", "/exit"]
    for i, cmd in enumerate(commands):
        welcome_lines.append(cmd, style="cyan")
        if i < len(commands) - 1:
            welcome_lines.append("  ", style="dim")
    welcome_lines.append("\n")

    # Starter questions hint
    welcome_lines.append("Press ", style="dim")
    welcome_lines.append("Tab", style="bold magenta")
    welcome_lines.append(" to try prepared questions about this codebase", style="dim")

    panel = Panel(
        welcome_lines,
        title="[bold green]CODEN RETRIEVER AGENT[/bold green]",
        border_style="green",
        box=box.DOUBLE,
        padding=(0, 1),
    )
    console.print(panel)


def print_goodbye() -> None:
    """Print the goodbye message."""
    console.print()
    console.print("[bold green]Goodbye![/bold green]")
    console.print()


def print_token_usage(
    context_tokens: int,
    estimated: bool = False,
    elapsed_seconds: float | None = None,
    wall_start: datetime | None = None,
    wall_end: datetime | None = None,
) -> None:
    """Print a compact context-usage line after the agent's response.

    Args:
        context_tokens: Retained tokens in the active context window after the turn.
        estimated: Whether the count is a local estimate (prefix with ~).
        elapsed_seconds: Total response duration in seconds (monotonic).
        wall_start: Wall-clock time when the query started.
        wall_end: Wall-clock time when the response finished.
    """
    if context_tokens == 0:
        return

    prefix = "~" if estimated else ""
    line = Text()
    line.append(f"  {prefix}{context_tokens:,}", style="cyan")
    line.append(" tokens in context", style="dim")

    if elapsed_seconds is not None and wall_start is not None and wall_end is not None:
        start_str = wall_start.strftime(WALL_CLOCK_FORMAT)
        end_str = wall_end.strftime(WALL_CLOCK_FORMAT)
        line.append("  \u2502  ", style="dim")
        line.append(f"{elapsed_seconds:.1f}s", style="bold green")
        line.append(f"  {start_str} \u2192 {end_str}", style="dim")

    console.print(line)


def print_session_baseline(baseline_tokens: int, tool_count: int) -> None:
    """Print the measured system+tools context floor after the first turn."""
    if baseline_tokens == 0:
        return

    line = Text()
    line.append("  base context: ", style="dim")
    line.append(f"{baseline_tokens:,}", style="cyan")
    line.append(" tokens (system + ", style="dim")
    line.append(f"{tool_count}", style="cyan")
    line.append(" tool defs, measured)", style="dim")
    console.print(line)


def print_shell_result(result: "ShellResult") -> None:
    """Render a ShellResult in a styled panel.

    Border colour encodes outcome at-a-glance: green=success, red=non-zero,
    yellow=timeout. ShellResult is imported under TYPE_CHECKING only so there
    is no runtime circular dependency between rich_console and shell_exec.
    """
    title = Text()
    title.append("$ ", style="dim")
    title.append(result.cmd, style="bold")
    title.append(f"   [{result.shell_kind.value}]", style="dim")

    body = Text()
    if result.stdout:
        body.append(result.stdout.rstrip("\n"))
    if result.stderr.strip():
        if body.plain:
            body.append("\n")
        body.append("[stderr] ", style="observation.error")
        body.append(result.stderr.rstrip("\n"), style="observation.error")
    if not body.plain:
        body.append("(no output)", style="dim")

    if result.timed_out:
        border = "yellow"
    elif result.returncode != 0:
        border = "red"
    else:
        border = "dim green"

    footer_bits = [f"exit {result.returncode}", f"{result.elapsed_s:.2f}s"]
    if result.timed_out:
        footer_bits.append("timed out")
    if result.truncated:
        footer_bits.append("truncated")
    subtitle = Text(" · ".join(footer_bits), style="dim")

    console.print()
    console.print(title)
    console.print(Panel(
        body, border_style=border, box=box.MINIMAL,
        padding=(0, 1), subtitle=subtitle, subtitle_align="right",
    ))


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: The warning message to display.
    """
    warning_text = Text()
    warning_text.append("[!] Warning: ", style="warning")
    warning_text.append(message, style="warning")
    console.print(warning_text)


def format_exception_message(e: BaseException, limit: int = 5) -> str:
    """Format exception showing the root cause(s) with full context.

    Handles ExceptionGroups and nested __cause__/__context__ chains automatically.

    Args:
        e: The exception to format.
        limit: Maximum number of errors to show (default 5).

    Returns:
        Human-readable error message showing the actual root cause(s).
    """
    def get_root_cause(exc: BaseException) -> BaseException:
        """Trace the exception chain to the absolute bottom (with cycle safety)."""
        visited = {id(exc)}
        while True:
            next_exc = exc.__cause__ or exc.__context__
            if next_exc is None or id(next_exc) in visited:
                return exc
            visited.add(id(next_exc))
            exc = next_exc

    def collect_errors(exc: BaseException, visited: set[int] | None = None) -> list[BaseException]:
        """Flatten ExceptionGroups and find root causes for regular exceptions."""
        if visited is None:
            visited = set()

        # Cycle detection for group traversal
        if id(exc) in visited:
            return []
        visited.add(id(exc))

        errors = []
        if hasattr(exc, 'exceptions') and exc.exceptions:
            for sub_exc in exc.exceptions:
                errors.extend(collect_errors(sub_exc, visited))
        else:
            root = get_root_cause(exc)
            if root is not exc and hasattr(root, 'exceptions') and root.exceptions:
                errors.extend(collect_errors(root, visited))
            else:
                errors.append(root)
        return errors

    all_roots = collect_errors(e)

    # Deduplicate while preserving order
    unique_roots = []
    seen = set()
    for error in all_roots:
        err_id = (type(error), str(error))
        if err_id not in seen:
            seen.add(err_id)
            unique_roots.append(error)

    if not unique_roots:
        return "Unknown Error"

    if len(unique_roots) == 1:
        root = unique_roots[0]
        msg = str(root) or "No details"
        return f"{type(root).__name__}: {msg}"

    parts = [f"Multiple errors ({len(unique_roots)}):"]
    for i, root in enumerate(unique_roots[:limit], 1):
        msg = str(root) or "No details"
        parts.append(f"  {i}. {type(root).__name__}: {msg}")

    if len(unique_roots) > limit:
        parts.append(f"  ... and {len(unique_roots) - limit} more")

    return "\n".join(parts)


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: The error message to display.
    """
    error_text = Text()
    error_text.append("❌ Error: ", style="error")
    error_text.append(str(message), style="error")
    console.print(error_text)
    console.print()


def print_fatal_error(e: BaseException, show_traceback: bool = True) -> None:
    """Print a fatal error with full traceback.

    Use this for unrecoverable errors where the user needs
    full debugging information.

    Args:
        e: The exception that occurred.
        show_traceback: Whether to show the full traceback.
    """
    import traceback

    # Print the formatted error message
    error_msg = format_exception_message(e)
    print_error(error_msg)

    if show_traceback:
        # Print full traceback for debugging
        console.print()
        console.print("[dim]--- Traceback ---[/dim]")

        # For ExceptionGroup, format all nested exceptions
        if hasattr(e, 'exceptions') and e.exceptions:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        else:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        # Print traceback in dim style so error message stands out
        console.print(Text(tb_str, style="dim"))
