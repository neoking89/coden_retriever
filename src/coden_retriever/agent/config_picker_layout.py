"""Rendering for the interactive config picker.

Pure presentation: turns ConfigPicker state into prompt_toolkit
FormattedText and the surrounding layout. No state mutation here.
"""

from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.layout import (
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from .config_picker_state import NUMERIC_TYPES, ConfigItem, ConfigPicker
from .picker_styles import (
    CLASS_DIM,
    CLASS_HEADER,
    CLASS_SELECTED,
    CLASS_SEPARATOR,
    PICKER_SEPARATOR_WIDTH,
    SELECTED_PREFIX,
    STYLE_SELECTED,
    STYLE_SEPARATOR,
    STYLE_TOOLBAR,
    UNSELECTED_PREFIX,
)
from .ui_utils import calculate_viewport, format_scroll_indicator

# WHY 14: shows most of the ~20 settings without scrolling on a typical terminal.
MAX_VISIBLE_ITEMS: int = 14

# Column widths for the picker's list view. WHY: keeps columns aligned across rows
# regardless of value length; values longer than the column wrap visually via padding.
KEY_COL_WIDTH: int = 25
VALUE_COL_WIDTH: int = 14


def _render_bool_value(item: ConfigItem) -> tuple[str, str]:
    """Return (display_text, style_class) for a bool row."""
    on = item.display_value == "true"
    base = "[ON]" if on else "[OFF]"
    text = f"{base} *" if item.changed else base
    style = "class:bool-on" if on else "class:bool-off"
    return text, style


def _render_value(item: ConfigItem, is_editing: bool, buffer: str) -> tuple[str, str]:
    """Return (display_text, style_class) for any value cell."""
    if is_editing:
        return f"[{buffer}|]", "class:edit"
    if item.value_type == "bool":
        return _render_bool_value(item)
    text = item.display_value
    if item.changed:
        text = f"{text} *"
    return text, "class:value"


def _append_section_header(lines: list[tuple[str, str]], section: str) -> None:
    lines.append(("class:section", f"\n   --- {section} ---\n"))


def _append_item_row(
    lines: list[tuple[str, str]],
    item: ConfigItem,
    is_selected: bool,
    is_editing: bool,
    buffer: str,
) -> None:
    prefix = SELECTED_PREFIX if is_selected else UNSELECTED_PREFIX
    name_style = CLASS_SELECTED if is_selected else "class:key"
    val_text, val_style = _render_value(item, is_editing, buffer)
    shown_val_style = name_style if is_selected and not is_editing else val_style

    lines.append((name_style if is_selected else "", prefix))
    lines.append((name_style, f"{item.key:<{KEY_COL_WIDTH}}"))
    lines.append((shown_val_style, f" {val_text:<{VALUE_COL_WIDTH}}"))
    lines.append((CLASS_DIM, f" {item.description}\n"))


def _build_content(picker: ConfigPicker) -> FormattedText:
    """Generate picker content with section grouping and edit visuals."""
    lines: list[tuple[str, str]] = []
    changed_count = sum(1 for i in picker.items if i.changed)
    header_text = f" Settings ({changed_count} changed)" if changed_count else " Settings"
    lines.append((CLASS_HEADER, header_text + "\n"))
    lines.append((CLASS_SEPARATOR, " " + "-" * PICKER_SEPARATOR_WIDTH + "\n"))

    start, end = calculate_viewport(
        picker.selected_index, len(picker.items), MAX_VISIBLE_ITEMS,
    )
    current_section: str | None = None
    for i in range(start, end):
        item = picker.items[i]
        if item.section != current_section:
            current_section = item.section
            _append_section_header(lines, current_section)
        is_selected = i == picker.selected_index
        is_editing = is_selected and picker.edit.active
        _append_item_row(lines, item, is_selected, is_editing, picker.edit.buffer)

    if len(picker.items) > MAX_VISIBLE_ITEMS:
        lines.append((
            CLASS_DIM,
            format_scroll_indicator(picker.selected_index, len(picker.items)) + "\n",
        ))
    lines.append((CLASS_SEPARATOR, "\n " + "-" * PICKER_SEPARATOR_WIDTH + "\n"))

    if picker.edit.error:
        lines.append(("class:error", f" ! {picker.edit.error}\n"))
    return FormattedText(lines)


def _escape_html(text: str) -> str:
    """Escape angle-brackets and ampersands for prompt_toolkit HTML rendering."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_toolbar(picker: ConfigPicker) -> HTML:
    """Dynamic toolbar that reflects the current mode and row type."""
    # Validation errors take priority — nothing else matters if commit failed.
    if picker.edit.error:
        safe = _escape_html(picker.edit.error)
        return HTML(f"<b>! Error:</b> {safe}  <b>[Esc]</b> Cancel edit")
    if picker.edit.active:
        return HTML(
            "<b>[Enter]</b> Commit  "
            "<b>[Esc]</b> Cancel edit  "
            "<b>[Ctrl+U]</b> Clear  "
            "<b>[⌫]</b> Delete char"
        )
    item = picker.selected
    if item is None:
        return HTML("<b>[q/Esc]</b> Close")
    if item.value_type == "bool":
        return HTML(
            "<b>[Space/&lt;/&gt;]</b> Toggle  "
            "<b>[Enter]</b> Apply  "
            "<b>[q/Esc]</b> Cancel"
        )
    if item.value_type in NUMERIC_TYPES:
        return HTML(
            "<b>[&lt;/&gt;]</b> Step  "
            "<b>[type]</b> Overwrite  "
            "<b>[⌫]</b> Trim  "
            "<b>[Enter]</b> Apply  "
            "<b>[q/Esc]</b> Cancel"
        )
    return HTML(
        "<b>[type]</b> Overwrite  "
        "<b>[⌫]</b> Trim  "
        "<b>[Enter]</b> Apply  "
        "<b>[q/Esc]</b> Cancel"
    )


def build_picker_layout(picker: ConfigPicker) -> tuple[Layout, Style]:
    """Build the prompt_toolkit layout + style for the config picker."""
    body = Window(
        content=FormattedTextControl(lambda: _build_content(picker)),
        wrap_lines=False,
    )
    toolbar = Window(
        content=FormattedTextControl(lambda: _build_toolbar(picker)),
        height=1,
        style=lambda: "class:toolbar-error" if picker.edit.error else "class:toolbar",
    )
    root = HSplit([Frame(body, title="Configuration"), toolbar])
    layout = Layout(root)

    style = Style.from_dict({
        "header": "bold #00aa00",
        "separator": STYLE_SEPARATOR,
        "section": "bold #ffaa00",
        "key": "#00aaff",
        "value": "#cccccc",
        "bool-on": "bold #00ff00",
        "bool-off": "bold #ff4444",
        "edit": "bold #000000 bg:#ffaa00",
        "error": "bold #ff4444",
        "toolbar-error": "bg:#aa0000 #ffffff bold",
        "selected": STYLE_SELECTED,
        "dim": STYLE_SEPARATOR,
        "toolbar": STYLE_TOOLBAR,
        "frame.border": "#00aa00",
    })
    return layout, style
