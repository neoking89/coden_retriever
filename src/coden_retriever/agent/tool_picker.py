"""Interactive tool picker with checkboxes for enabling/disabling MCP tools.

Provides a keyboard-navigable checkbox list using prompt_toolkit.
Users can toggle tools on/off with Space and apply changes with Enter.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

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
    CLASS_DIM,
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
    PICKER_SEPARATOR_WIDTH,
    SELECTED_PREFIX,
    STYLE_SELECTED,
    STYLE_SEPARATOR,
    STYLE_TOOLBAR,
    UNSELECTED_PREFIX,
)
from .rich_console import console
from .ui_utils import calculate_viewport, format_scroll_indicator
from ..mcp.constants import get_tool_categories


@dataclass
class ToolItem:
    """A tool item for the picker."""
    name: str
    description: str
    enabled: bool
    category: str = ""  # Category name for grouping


class ToolPicker:
    """Interactive tool picker with checkbox selection."""

    def __init__(self, tools: list[ToolItem]):
        """Initialize the tool picker.

        Args:
            tools: List of ToolItem objects to display.
        """
        self.tools = tools
        self.selected_index = 0
        self.cancelled = False
        self.applied = False

    def toggle_current(self) -> None:
        """Toggle the currently selected tool."""
        if self.tools:
            self.tools[self.selected_index].enabled = not self.tools[self.selected_index].enabled

    def get_enabled_tools(self) -> set[str]:
        """Get the set of enabled tool names."""
        return {t.name for t in self.tools if t.enabled}

    def get_disabled_tools(self) -> set[str]:
        """Get the set of disabled tool names."""
        return {t.name for t in self.tools if not t.enabled}


def run_tool_picker(
    available_tools: list,
    disabled_tools: set[str],
) -> Optional[set[str]]:
    """Run the interactive tool picker.

    Args:
        available_tools: List of available MCP tools (with .name and .description).
        disabled_tools: Set of currently disabled tool names.

    Returns:
        New set of disabled tools, or None if cancelled.
    """
    # Build tool lookup from available tools (registered with MCP server)
    tool_map = {}
    for tool in available_tools:
        desc = tool.description or ""
        if len(desc) > 50:
            desc = desc[:47] + "..."
        tool_map[tool.name] = desc

    # Build tool items organized by category
    # Include ALL tools from TOOL_CATEGORIES, even if not registered
    # This allows disabled tools to be visible and re-enabled
    tool_items: list[ToolItem] = []
    seen_names: set[str] = set()

    for category_name, category_tools in get_tool_categories():
        for tool_name in category_tools:
            if tool_name not in seen_names:
                # Use description from registered tool, or default for disabled
                if tool_name in tool_map:
                    description = tool_map[tool_name]
                else:
                    description = "(disabled - restart to enable)"
                tool_items.append(ToolItem(
                    name=tool_name,
                    description=description,
                    enabled=tool_name not in disabled_tools,
                    category=category_name,
                ))
                seen_names.add(tool_name)

    # Add any uncategorized tools (registered but not in TOOL_CATEGORIES)
    for tool in available_tools:
        if tool.name not in seen_names:
            desc = tool.description or ""
            if len(desc) > 50:
                desc = desc[:47] + "..."
            tool_items.append(ToolItem(
                name=tool.name,
                description=desc,
                enabled=tool.name not in disabled_tools,
                category="Other",
            ))
            seen_names.add(tool.name)

    if not tool_items:
        console.print("[yellow]No tools available[/yellow]")
        return None

    picker = ToolPicker(tool_items)

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
        if picker.selected_index < len(picker.tools) - 1:
            picker.selected_index += 1

    @kb.add(' ')
    @kb.add('x')
    def toggle_tool(event):
        picker.toggle_current()

    @kb.add(KEY_ENTER)
    def apply_changes(event):
        picker.applied = True
        event.app.exit()

    @kb.add('a')
    def enable_all(event):
        for tool in picker.tools:
            tool.enabled = True

    @kb.add('n')
    def disable_all(event):
        for tool in picker.tools:
            tool.enabled = False

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
        """Generate the picker content with category grouping and scrolling."""
        lines = []

        # Header
        enabled_count = sum(1 for t in picker.tools if t.enabled)
        total_count = len(picker.tools)
        lines.append((CLASS_HEADER, f' Tools: {enabled_count}/{total_count} enabled\n'))
        lines.append((CLASS_SEPARATOR, ' ' + '-' * PICKER_SEPARATOR_WIDTH + '\n'))

        # Calculate visible range (scrolling)
        max_visible = 12
        start, end = calculate_viewport(picker.selected_index, len(picker.tools), max_visible)

        # Tool list with checkboxes, grouped by category
        current_category = None
        for i in range(start, end):
            tool = picker.tools[i]
            # Add category header when category changes
            if tool.category != current_category:
                current_category = tool.category
                lines.append(('class:category', f'\n   --- {current_category} ---\n'))

            is_selected = i == picker.selected_index
            checkbox = '[X]' if tool.enabled else '[ ]'
            checkbox_style = 'class:checkbox-on' if tool.enabled else 'class:checkbox-off'

            if is_selected:
                prefix = SELECTED_PREFIX
                name_style = CLASS_SELECTED
            else:
                prefix = UNSELECTED_PREFIX
                name_style = 'class:tool-name'

            # Format: " > [X] tool_name - description"
            lines.append((name_style if is_selected else '', prefix))
            lines.append((checkbox_style, checkbox))
            lines.append((name_style, f' {tool.name}'))
            lines.append(('class:description', f' - {tool.description}\n'))

        # Scroll indicator
        if len(picker.tools) > max_visible:
            scroll_info = format_scroll_indicator(picker.selected_index, len(picker.tools))
            lines.append((CLASS_DIM, scroll_info + '\n'))

        lines.append((CLASS_SEPARATOR, '\n ' + '-' * PICKER_SEPARATOR_WIDTH + '\n'))

        return FormattedText(lines)

    def get_toolbar():
        """Generate the bottom toolbar."""
        return HTML(
            '<b>[Space/x]</b> Toggle  '
            '<b>[a]</b> All on  '
            '<b>[n]</b> All off  '
            '<b>[Enter]</b> Apply  '
            '<b>[q/Esc]</b> Cancel'
        )

    # Create layout
    body = Window(
        content=FormattedTextControl(get_content),
        wrap_lines=False,
    )

    toolbar = Window(
        content=FormattedTextControl(get_toolbar),
        height=1,
        style='class:toolbar',
    )

    root_container = HSplit([
        Frame(
            body,
            title='Tool Settings',
        ),
        toolbar,
    ])

    # Styles
    style = Style.from_dict({
        'header': 'bold #00aa00',
        'separator': STYLE_SEPARATOR,
        'category': 'bold #ffaa00',
        'tool-name': '#00aaff',
        'selected': STYLE_SELECTED,
        'checkbox-on': 'bold #00ff00',
        'checkbox-off': STYLE_SEPARATOR,
        'description': '#888888',
        'dim': STYLE_SEPARATOR,
        'toolbar': STYLE_TOOLBAR,
        'frame.border': '#00aa00',
    })

    # Create and run application
    app: Application[None] = Application(
        layout=Layout(root_container),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=True,
    )

    # Show instructions
    console.print()
    console.print("[bold green]Tool Settings[/bold green]")
    console.print("[dim]Use arrow keys to navigate, Space to toggle, Enter to apply[/dim]")
    console.print()

    app.run()

    if picker.cancelled:
        console.print("[yellow]Cancelled - no changes made[/yellow]")
        return None

    if picker.applied:
        return picker.get_disabled_tools()

    return None


async def run_tool_picker_async(
    available_tools: list,
    disabled_tools: set[str],
) -> Optional[set[str]]:
    """Async version of the tool picker.

    Args:
        available_tools: List of available MCP tools.
        disabled_tools: Set of currently disabled tool names.

    Returns:
        New set of disabled tools, or None if cancelled.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        run_tool_picker,
        available_tools,
        disabled_tools,
    )
