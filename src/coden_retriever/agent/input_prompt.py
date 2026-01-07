"""Interactive input prompt with tab auto-completion for slash commands.

Uses prompt_toolkit to provide:
- Tab completion for /commands
- Directory path completion for /cd
- Starter question completion (Tab on empty prompt)
- Command suggestions while typing
- Styled input prompt matching Rich theme
"""

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.rule import Rule

from ..config_loader import SETTING_METADATA
from .commands import registry
from .rich_console import console
from .starter_questions import get_starter_questions


class SlashCommandCompleter(Completer):
    """Auto-completer for slash commands, directory paths, and starter questions.

    Provides completions when user types / followed by partial command name.
    Also provides directory path completion for /cd command.
    Shows command descriptions in the completion menu.

    When input is empty or starts with a regular query, shows starter questions
    to help new developers explore the codebase.
    """

    def __init__(self, get_current_dir=None):
        """Initialize the completer.

        Args:
            get_current_dir: Callable that returns current working directory.
        """
        self.registry = registry
        self._get_current_dir = get_current_dir or (lambda: str(Path.cwd()))
        self._starter_questions = get_starter_questions()

    def _complete_path(self, partial_path: str):
        """Yield path completions for directory navigation.

        Args:
            partial_path: Partial path typed by user.

        Yields:
            Completion objects for matching directories.
        """
        # Handle special paths
        if partial_path == "~" or partial_path.startswith("~/"):
            base = Path.home()
            remainder = partial_path[2:] if partial_path.startswith("~/") else ""
            prefix = "~/"
        elif partial_path.startswith("/") or (len(partial_path) > 1 and partial_path[1] == ":"):
            # Absolute path (Unix or Windows)
            base = Path(partial_path)
            if not base.exists():
                base = base.parent
                remainder = base.name if base.name else ""
            else:
                remainder = ""
            prefix = ""
        else:
            # Relative path
            base = Path(self._get_current_dir())
            remainder = partial_path
            prefix = ""

        # Determine what to complete
        if remainder:
            # Complete within a directory
            parts = remainder.replace("\\", "/").split("/")
            if len(parts) > 1:
                # Navigate to parent directories first
                for part in parts[:-1]:
                    if part == "..":
                        base = base.parent
                    elif part and part != ".":
                        base = base / part
                partial_name = parts[-1]
            else:
                partial_name = parts[0]
        else:
            partial_name = ""

        if not base.exists() or not base.is_dir():
            return

        try:
            for entry in base.iterdir():
                if entry.is_dir():
                    name = entry.name
                    if name.startswith('.'):
                        continue  # Skip hidden directories
                    if partial_name and not name.lower().startswith(partial_name.lower()):
                        continue

                    # Build the completion text
                    if prefix:
                        completion_text = f"{prefix}{name}/"
                    elif remainder and "/" in remainder.replace("\\", "/"):
                        # Include the path prefix
                        path_prefix = "/".join(remainder.replace("\\", "/").split("/")[:-1])
                        completion_text = f"{path_prefix}/{name}/"
                    else:
                        completion_text = f"{name}/"

                    yield Completion(
                        completion_text,
                        start_position=-len(partial_path) if partial_path else 0,
                        display=f"{name}/",
                        display_meta="directory",
                    )
        except PermissionError:
            pass

    def _complete_questions(self, partial: str):
        """Yield question completions for starter questions.

        Args:
            partial: Partial text to match against question labels.

        Yields:
            Completion objects for matching starter questions.
        """
        partial_lower = partial.lower().strip()

        for q in self._starter_questions:
            # Match against label
            if not partial_lower or partial_lower in q.label.lower():
                yield Completion(
                    q.question,
                    start_position=-len(partial),
                    display=q.label,
                    display_meta=q.category,
                )

    def get_completions(self, document, complete_event):
        """Yield completions for the current input.

        Args:
            document: The prompt_toolkit Document with current text.
            complete_event: The completion event.

        Yields:
            Completion objects for matching commands or questions.
        """
        text = document.text_before_cursor

        # If input doesn't start with /, show starter questions
        if not text.startswith("/"):
            yield from self._complete_questions(text)
            return

        # Get the partial command name (without the leading /)
        partial = text[1:].split()[0] if len(text) > 1 else ""

        # Handle /cd path completion
        parts = text[1:].split(maxsplit=1)
        if len(parts) >= 1 and parts[0].lower() in ("cd", "dir", "chdir"):
            if len(parts) == 1 and text.endswith(" "):
                # Just typed "/cd ", show directory completions
                yield from self._complete_path("")
                return
            elif len(parts) == 2:
                # Completing a path
                path_partial = parts[1]
                yield from self._complete_path(path_partial)
                return

        # Handle subcommands for /config
        if len(parts) >= 1 and parts[0].lower() == "config":
            if len(parts) == 1 and not text.endswith(" "):
                # Still typing "config", complete it
                pass
            elif len(parts) >= 1:
                config_parts = text[1:].split()
                # Completing subcommand or key
                if len(config_parts) == 1 or (len(config_parts) == 2 and not text.endswith(" ")):
                    # Complete subcommands: set, reset
                    subcommands = [
                        ("set", "Set a configuration value"),
                        ("reset", "Reset all settings to defaults"),
                    ]
                    sub_partial = config_parts[1] if len(config_parts) > 1 else ""
                    for sub, desc in subcommands:
                        if sub.startswith(sub_partial.lower()):
                            yield Completion(
                                sub,
                                start_position=-len(sub_partial),
                                display=sub,
                                display_meta=desc,
                            )
                    return
                elif (len(config_parts) == 2 and config_parts[1].lower() == "set" and text.endswith(" ")) or \
                     (len(config_parts) == 3 and config_parts[1].lower() == "set" and not text.endswith(" ")):
                    # Complete config keys from centralized metadata
                    key_partial = config_parts[2] if len(config_parts) > 2 else ""
                    for key, meta in SETTING_METADATA.items():
                        if key.startswith(key_partial.lower()):
                            yield Completion(
                                key,
                                start_position=-len(key_partial) if key_partial else 0,
                                display=key,
                                display_meta=meta.short_desc,
                            )
                    return

        # Get matching commands
        completions = self.registry.get_completions(partial)

        for cmd_name, description in completions:
            # Calculate start position (how many characters to replace)
            start_pos = -len(text)  # Replace from /

            yield Completion(
                f"/{cmd_name}",
                start_position=start_pos,
                display=f"/{cmd_name}",
                display_meta=description,
            )


# Style matching the Rich AGENT_THEME
PROMPT_STYLE = Style.from_dict({
    # Prompt text
    "prompt": "bold #5f87ff",  # Blue, matching user.prompt

    # Completion menu
    "completion-menu": "bg:#333333 #ffffff",
    "completion-menu.completion": "bg:#333333 #ffffff",
    "completion-menu.completion.current": "bg:#00aa00 #000000 bold",
    "completion-menu.meta": "bg:#333333 #888888 italic",
    "completion-menu.meta.current": "bg:#00aa00 #000000 italic",

    # Scrollbar
    "scrollbar.background": "bg:#333333",
    "scrollbar.button": "bg:#666666",
})


def _create_key_bindings() -> KeyBindings:
    """Create custom key bindings for the prompt.

    Customizes Ctrl+C behavior:
    - If there's text in the buffer, clear it
    - If the buffer is empty, raise KeyboardInterrupt to exit

    Returns:
        KeyBindings with custom Ctrl+C handling.
    """
    bindings = KeyBindings()

    @bindings.add("c-c")
    def handle_ctrl_c(event):
        """Handle Ctrl+C: clear buffer if text present, otherwise exit."""
        buffer = event.app.current_buffer
        if buffer.text:
            # Clear the text field
            buffer.reset()
        else:
            # Raise KeyboardInterrupt to exit
            raise KeyboardInterrupt

    return bindings


def create_prompt_session(get_current_dir=None) -> PromptSession:
    """Create a prompt session with slash command completion.

    Args:
        get_current_dir: Optional callable that returns current working directory.
                        Used for /cd path completion.

    Returns:
        Configured PromptSession with auto-completion.
    """
    return PromptSession(
        completer=SlashCommandCompleter(get_current_dir=get_current_dir),
        style=PROMPT_STYLE,
        complete_while_typing=True,
        complete_in_thread=True,  # Non-blocking completion
        key_bindings=_create_key_bindings(),
    )


async def get_user_input_async(session: PromptSession) -> str:
    """Get user input with auto-completion support.

    Args:
        session: The prompt session to use.

    Returns:
        The user's input string, stripped.
    """
    console.print(Rule(style="dim"))
    console.print("[dim]Tab for starter questions, /help for commands[/dim]")

    try:
        # Use prompt_toolkit for input with completion
        user_input = await session.prompt_async(
            HTML("<prompt>You: </prompt>"),
        )
        return user_input.strip()
    except EOFError:
        return "/exit"
    except KeyboardInterrupt:
        return "/exit"


def get_user_input_sync(session: PromptSession) -> str:
    """Synchronous version of get_user_input_async.

    Args:
        session: The prompt session to use.

    Returns:
        The user's input string, stripped.
    """
    console.print(Rule(style="dim"))
    console.print("[dim]Tab for starter questions, /help for commands[/dim]")

    try:
        user_input = session.prompt(
            HTML("<prompt>You: </prompt>"),
        )
        return user_input.strip()
    except EOFError:
        return "/exit"
    except KeyboardInterrupt:
        return "/exit"
