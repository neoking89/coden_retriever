"""Rendering for the /undo picker — prompt_toolkit layout + styles."""

from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from ..picker_styles import (
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
from ..ui_utils import calculate_viewport, format_scroll_indicator
from .constants import MAX_VISIBLE_ROWS
from .picker_state import UndoPicker
from .tool_call_index import BranchHeaderRow, ToolCallRow

# WHY these widths: keep branch/tool-call columns aligned across rows.
_BRANCH_LABEL_WIDTH: int = 50
_ORDINAL_TAG_WIDTH: int = 8   # fits "[bNN:MM]" for small branch/ordinal counts
_EMPTY_STATE_HINT: str = " (no tool calls yet — use /clear to start fresh)"


def _branch_header_line(row: BranchHeaderRow, is_selected: bool) -> list[tuple[str, str]]:
    prefix = SELECTED_PREFIX if is_selected else UNSELECTED_PREFIX
    marker = "▸" if row.is_current else "▾"
    suffix = "  (current)" if row.is_current else ""
    label = f"{marker} {row.branch_id} — {row.label}{suffix}"
    style = "class:branch-current" if row.is_current else "class:branch-other"
    right = f"{row.child_count} fork(s)" if row.child_count else ""

    segments: list[tuple[str, str]] = []
    segments.append((CLASS_SELECTED if is_selected else "", prefix))
    segments.append((style, f"{label:<{_BRANCH_LABEL_WIDTH}}"))
    segments.append((CLASS_DIM, f" {right}\n"))
    return segments


def _tool_call_line(row: ToolCallRow, is_selected: bool) -> list[tuple[str, str]]:
    prefix = SELECTED_PREFIX if is_selected else UNSELECTED_PREFIX
    tag = f"[{row.branch_id}:{row.ordinal}]"
    name_style = CLASS_SELECTED if is_selected else "class:tool-name"
    args_style = CLASS_SELECTED if is_selected else "class:tool-args"

    segments: list[tuple[str, str]] = []
    segments.append((CLASS_SELECTED if is_selected else "", prefix))
    segments.append((CLASS_DIM, f"{tag:<{_ORDINAL_TAG_WIDTH}} "))
    segments.append((name_style, row.tool_name))
    segments.append((args_style, f"({row.args_preview})\n"))
    return segments


def _render_row(row, is_selected: bool) -> list[tuple[str, str]]:
    if isinstance(row, BranchHeaderRow):
        return _branch_header_line(row, is_selected)
    if isinstance(row, ToolCallRow):
        return _tool_call_line(row, is_selected)
    return []


def _render_directive(picker: UndoPicker) -> list[tuple[str, str]]:
    """Append the directive input line when the picker is in directive mode."""
    if not picker.directive.active:
        return []
    action = picker.directive.pending_action or ""
    target = picker.directive.pending_branch_id or ""
    header = f"\n {action.capitalize()} → {target}    Directive (optional, Enter to resume):\n"
    cursor_line = f"   {picker.directive.buffer}▋\n"
    return [
        (CLASS_SEPARATOR, "\n " + "-" * PICKER_SEPARATOR_WIDTH + "\n"),
        ("class:directive-prompt", header),
        ("class:directive", cursor_line),
    ]


def _build_content(picker: UndoPicker) -> FormattedText:
    """Render the whole body: header, row list, directive input, status."""
    lines: list[tuple[str, str]] = []
    total_rows = len(picker.rows)
    header_text = f" Undo — {total_rows} row(s) across {sum(1 for r in picker.rows if isinstance(r, BranchHeaderRow))} branch(es)"
    lines.append((CLASS_HEADER, header_text + "\n"))
    lines.append((CLASS_SEPARATOR, " " + "-" * PICKER_SEPARATOR_WIDTH + "\n"))

    if total_rows == 0:
        lines.append((CLASS_DIM, _EMPTY_STATE_HINT + "\n"))
        return FormattedText(lines)

    start, end = calculate_viewport(picker.selected_index, total_rows, MAX_VISIBLE_ROWS)
    for i in range(start, end):
        is_selected = i == picker.selected_index
        lines.extend(_render_row(picker.rows[i], is_selected))

    if total_rows > MAX_VISIBLE_ROWS:
        lines.append((
            CLASS_DIM,
            format_scroll_indicator(picker.selected_index, total_rows) + "\n",
        ))

    if picker.message:
        lines.append(("class:info", f"\n  {picker.message}\n"))

    lines.extend(_render_directive(picker))
    return FormattedText(lines)


def _build_toolbar(picker: UndoPicker) -> HTML:
    """Mode-aware hint bar at the bottom of the picker."""
    if picker.directive.active:
        return HTML(
            "<b>[Enter]</b> Apply  "
            "<b>[Esc]</b> Cancel directive  "
            "<b>[Ctrl+U]</b> Clear  "
            "<b>[⌫]</b> Delete char",
        )
    row = picker.selected
    if row is None:
        return HTML("<b>[Esc/q]</b> Close")
    if isinstance(row, BranchHeaderRow):
        if row.is_current:
            return HTML(
                "<b>[↑/↓]</b> Navigate  "
                "<b>[Esc/q]</b> Cancel  "
                "<i>(current branch — pick a tool call or another branch)</i>",
            )
        return HTML(
            "<b>[Enter]</b> Switch to branch  "
            "<b>[↑/↓]</b> Navigate  "
            "<b>[Esc/q]</b> Cancel",
        )
    return HTML(
        "<b>[Enter]</b> Fork here  "
        "<b>[↑/↓]</b> Navigate  "
        "<b>[Esc/q]</b> Cancel",
    )


def build_picker_layout(picker: UndoPicker) -> tuple[Layout, Style]:
    """Build the prompt_toolkit layout + style for the undo picker."""
    body = Window(
        content=FormattedTextControl(lambda: _build_content(picker)),
        wrap_lines=False,
    )
    toolbar = Window(
        content=FormattedTextControl(lambda: _build_toolbar(picker)),
        height=1,
        style="class:toolbar",
    )
    root = HSplit([Frame(body, title="Undo / Resume"), toolbar])
    layout = Layout(root)

    style = Style.from_dict({
        "header": "bold #00aa00",
        "separator": STYLE_SEPARATOR,
        "branch-current": "bold #ffaa00",
        "branch-other": "#888888",
        "tool-name": "#00aaff",
        "tool-args": "#cccccc",
        "selected": STYLE_SELECTED,
        "dim": STYLE_SEPARATOR,
        "info": "italic #888888",
        "directive-prompt": "bold #ffaa00",
        "directive": "bold #000000 bg:#ffaa00",
        "toolbar": STYLE_TOOLBAR,
        "frame.border": "#00aa00",
    })
    return layout, style
