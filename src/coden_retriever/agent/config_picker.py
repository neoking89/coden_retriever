"""Interactive config picker — navigate, step, toggle, and edit settings.

Boolean rows toggle with Space. Numeric rows step with <,> (or arrows).
Any row can be overwritten by typing, trimmed with Backspace, or edited
inline. Enter at the outer level applies all staged changes; Esc/q
discards them.

Follows the same patterns as tool_picker.py and directory_browser.py.
State, layout, and key bindings live in sibling modules to keep each
file within the project's per-file size limit.
"""

import asyncio
from typing import Any

from prompt_toolkit import Application

from ..config_loader import SETTING_LOCATIONS, SETTING_METADATA
from .config_picker_keys import build_picker_keybindings
from .config_picker_layout import build_picker_layout
from .config_picker_state import ConfigItem, ConfigPicker
from .rich_console import console

# Map internal config section names to picker-display labels.
_SECTION_DISPLAY_NAMES: dict[str, str] = {
    "model": "Model",
    "agent": "Agent",
    "daemon": "Daemon",
    "search": "Search",
}


def build_config_items(current_values: dict[str, str]) -> list[ConfigItem]:
    """Assemble ConfigItem list from SETTING_METADATA + current values."""
    items: list[ConfigItem] = []
    for key, meta in SETTING_METADATA.items():
        section_name, _, _ = SETTING_LOCATIONS.get(key, ("other", "", None))
        display_section = _SECTION_DISPLAY_NAMES.get(section_name, section_name.title())
        items.append(ConfigItem(
            key=key,
            original_value=current_values.get(key, "?"),
            value_type=meta.value_type,
            description=meta.short_desc,
            section=display_section,
        ))
    return items


def run_config_picker(current_values: dict[str, str]) -> dict[str, Any] | None:
    """Run the interactive config picker.

    Args:
        current_values: Dict of setting key -> current display value string.

    Returns:
        Dict of parsed staged changes (key -> typed value), or None if cancelled.
    """
    items = build_config_items(current_values)
    if not items:
        console.print("[yellow]No settings available[/yellow]")
        return None

    picker = ConfigPicker(items)
    kb = build_picker_keybindings(picker)
    layout, style = build_picker_layout(picker)

    app: Application[None] = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=True,
    )

    app.run()

    if picker.cancelled:
        console.print("[yellow]Cancelled - no changes made[/yellow]")
        return None

    if picker.applied:
        changes = picker.get_all_changes()
        if not changes:
            console.print("[dim]No changes staged - nothing to apply.[/dim]")
            return None
        return changes

    return None


async def run_config_picker_async(
    current_values: dict[str, str],
) -> dict[str, Any] | None:
    """Async wrapper around run_config_picker (runs in a thread executor)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_config_picker, current_values)
