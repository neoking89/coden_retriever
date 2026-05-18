"""Interactive tool permission picker with arrow-key navigation.

Provides a UI for users to approve or deny tool execution before it happens.
Uses up/down arrow keys for selection rather than letter-based shortcuts
for a more intuitive user experience.
"""

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Optional

from .protocols import PermissionChoice

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from .picker_styles import (
    CLASS_HEADER,
    CLASS_SELECTED,
    CLASS_SEPARATOR,
    KEY_CTRL_C,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_QUIT,
    KEY_UP,
    KEY_VIM_DOWN,
    KEY_VIM_UP,
    PERMISSION_SEPARATOR_WIDTH,
    SELECTED_PREFIX,
    STYLE_SELECTED,
    STYLE_SEPARATOR,
    STYLE_TOOLBAR,
    UNSELECTED_PREFIX,
)
from .rich_console import console

logger = logging.getLogger(__name__)

@dataclass
class ToolPermissionRequest:
    """A request to execute a tool that needs user permission."""
    tool_name: str
    tool_args: dict
    description: str = ""


class ToolPermissionPicker:
    """Interactive permission picker with arrow-key navigation.

    Allows users to approve or deny tool execution using up/down arrows
    to select between Allow and Deny options.
    """

    def __init__(self, request: ToolPermissionRequest):
        """Initialize the permission picker.

        Args:
            request: The tool permission request to display.
        """
        self.request = request
        self.selected_index = 0  # 0 = Allow, 1 = Always Allow, 2 = Deny
        self.choices = [PermissionChoice.ALLOW, PermissionChoice.ALWAYS_ALLOW, PermissionChoice.DENY]
        self.result: Optional[PermissionChoice] = None
        self.cancelled = False

    def get_selected_choice(self) -> PermissionChoice:
        """Get the currently selected choice."""
        return self.choices[self.selected_index]


def format_tool_args(args: dict[str, Any], max_length: int = 60) -> str:
    """Format tool arguments for display.

    Args:
        args: Dictionary of tool arguments.
        max_length: Maximum length per argument value before truncation.

    Returns:
        Formatted string of arguments.
    """
    if not args:
        return "(no arguments)"

    lines: list[str] = []
    for key, value in args.items():
        value_str = str(value)
        if len(value_str) > max_length:
            value_str = value_str[:max_length - 3] + "..."
        lines.append(f"    {key}: {value_str}")
    return "\n".join(lines)


def run_tool_permission_picker(request: ToolPermissionRequest) -> Optional[PermissionChoice]:
    """Run the interactive permission picker.

    Displays tool information and allows user to select Allow or Deny
    using arrow keys.

    Args:
        request: The tool permission request to display.

    Returns:
        PermissionChoice.ALLOW or PermissionChoice.DENY, or None if cancelled.
    """
    picker = ToolPermissionPicker(request)

    # Create key bindings
    kb = KeyBindings()

    @kb.add(KEY_UP)
    @kb.add(KEY_VIM_UP)
    def move_up(event):
        if picker.selected_index > 0:
            picker.selected_index -= 1

    @kb.add(KEY_DOWN)
    @kb.add(KEY_VIM_DOWN)
    def move_down(event):
        if picker.selected_index < len(picker.choices) - 1:
            picker.selected_index += 1

    @kb.add(KEY_ENTER)
    @kb.add(' ')
    def select_choice(event):
        picker.result = picker.get_selected_choice()
        event.app.exit()

    @kb.add('a')
    def quick_allow(event):
        picker.result = PermissionChoice.ALLOW
        event.app.exit()

    @kb.add('d')
    def quick_deny(event):
        picker.result = PermissionChoice.DENY
        event.app.exit()

    @kb.add('A')  # Shift+A for "Always Allow"
    @kb.add('s')  # 's' for session allow
    def quick_always_allow(event):
        picker.result = PermissionChoice.ALWAYS_ALLOW
        event.app.exit()

    @kb.add(KEY_ESCAPE)
    @kb.add(KEY_QUIT)
    def cancel(event):
        picker.cancelled = True
        event.app.exit()

    @kb.add(KEY_CTRL_C)
    def ctrl_c(event):
        picker.cancelled = True
        event.app.exit()

    def get_content():
        """Generate the picker content."""
        lines = []

        # Header with tool information
        lines.append((CLASS_HEADER, ' Tool Execution Request\n'))
        lines.append((CLASS_SEPARATOR, ' ' + '-' * PERMISSION_SEPARATOR_WIDTH + '\n\n'))

        # Tool name
        lines.append(('class:label', ' Tool: '))
        lines.append(('class:tool-name', f'{request.tool_name}\n\n'))

        # Tool arguments
        lines.append(('class:label', ' Arguments:\n'))
        args_str = format_tool_args(request.tool_args)
        lines.append(('class:args', f'{args_str}\n\n'))

        # Separator before choices
        lines.append((CLASS_SEPARATOR, ' ' + '-' * PERMISSION_SEPARATOR_WIDTH + '\n\n'))

        # Choice options with arrow indicator
        lines.append(('class:label', ' Select action:\n\n'))

        for i, choice in enumerate(picker.choices):
            is_selected = i == picker.selected_index

            if is_selected:
                prefix = SELECTED_PREFIX
                style = CLASS_SELECTED
            else:
                prefix = UNSELECTED_PREFIX
                style = 'class:choice'

            if choice == PermissionChoice.ALLOW:
                icon = '[ALLOW]'
                desc = ' - Execute this tool'
                icon_style = 'class:allow' if not is_selected else style
            elif choice == PermissionChoice.ALWAYS_ALLOW:
                icon = '[ALWAYS ALLOW]'
                desc = ' - Allow all tools for this session'
                icon_style = 'class:always-allow' if not is_selected else style
            else:
                icon = '[DENY]'
                desc = ' - Skip this tool'
                icon_style = 'class:deny' if not is_selected else style

            lines.append((style, prefix))
            lines.append((icon_style if not is_selected else style, icon))
            lines.append((style if is_selected else 'class:dim', desc + '\n'))

        lines.append(('', '\n'))

        return FormattedText(lines)

    def get_toolbar():
        """Generate the bottom toolbar."""
        return HTML(
            '<b>[Up/Down]</b> Navigate  '
            '<b>[Enter/Space]</b> Select  '
            '<b>[a]</b> Allow  '
            '<b>[s/A]</b> Always  '
            '<b>[d]</b> Deny  '
            '<b>[q/Esc]</b> Cancel'
        )

    # Create layout
    body = Window(
        content=FormattedTextControl(get_content),
        wrap_lines=True,
    )

    toolbar = Window(
        content=FormattedTextControl(get_toolbar),
        height=1,
        style='class:toolbar',
    )

    root_container = HSplit([
        Frame(
            body,
            title='Permission Required',
        ),
        toolbar,
    ])

    # Styles
    style = Style.from_dict({
        'header': 'bold #ffaa00',
        'separator': STYLE_SEPARATOR,
        'label': 'bold #00aaff',
        'tool-name': 'bold #ff6600',
        'args': '#cccccc',
        'choice': '#ffffff',
        'selected': STYLE_SELECTED,
        'allow': 'bold #00ff00',
        'always-allow': 'bold #00aaff',
        'deny': 'bold #ff4444',
        'dim': '#888888',
        'toolbar': STYLE_TOOLBAR,
        'frame.border': '#ffaa00',
    })

    # Create and run application
    app: Application[None] = Application(
        layout=Layout(root_container),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=True,
    )

    # Use thread pool for nested event loop compatibility
    # This handles both standalone and nested async contexts
    try:
        # Check if we're in a running event loop
        asyncio.get_running_loop()
        # We're inside an async context - run in thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(app.run)
            future.result()
    except RuntimeError:
        # No running loop - safe to use app.run() directly
        app.run()
    except Exception as e:
        # Log any unexpected errors but don't crash the agent
        logger.warning(f"Permission picker encountered an error: {e}")
        return None

    if picker.cancelled:
        return None

    return picker.result


async def run_tool_permission_picker_async(
    request: ToolPermissionRequest,
) -> Optional[PermissionChoice]:
    """Async version of the permission picker.

    Args:
        request: The tool permission request to display.

    Returns:
        PermissionChoice or None if cancelled.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        run_tool_permission_picker,
        request,
    )


async def ask_tool_permission(
    tool_name: str,
    tool_args: dict[str, Any],
    description: str = "",
) -> bool:
    """Ask user for permission to execute a tool.

    This is the main interface for the permission system. It displays
    the tool information and waits for user to approve or deny.

    Args:
        tool_name: Name of the tool to execute.
        tool_args: Arguments for the tool call.
        description: Optional description of what the tool does.

    Returns:
        True if user allows execution, False otherwise.
    """
    request = ToolPermissionRequest(
        tool_name=tool_name,
        tool_args=tool_args,
        description=description,
    )

    result = await run_tool_permission_picker_async(request)

    if result is None:
        # User cancelled - treat as deny
        console.print("[yellow]Tool execution cancelled[/yellow]")
        return False

    if result == PermissionChoice.ALLOW:
        return True
    else:
        console.print(f"[yellow]Tool '{tool_name}' execution denied[/yellow]")
        return False
