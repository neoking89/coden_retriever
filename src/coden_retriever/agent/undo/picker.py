"""Interactive /undo picker — runs the prompt_toolkit app."""

import asyncio
from typing import Optional

from prompt_toolkit import Application

from ..rich_console import console
from .branch import ConversationTree
from .picker_keys import build_picker_keybindings
from .picker_layout import build_picker_layout
from .picker_state import UndoPicker, UndoResult
from .tool_call_index import ToolCallRow, build_rows


def run_undo_picker(tree: ConversationTree) -> Optional[UndoResult]:
    """Run the interactive undo picker.

    Returns the UndoResult chosen by the user, or None when cancelled or when
    there are no tool calls in the tree to pick from.
    """
    rows = build_rows(tree)
    if not any(isinstance(r, ToolCallRow) for r in rows):
        console.print("[yellow]No tool calls in history — nothing to undo.[/yellow]")
        return None

    picker = UndoPicker(rows)
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
        console.print("[yellow]Cancelled — no branch change.[/yellow]")
        return None

    if picker.applied and picker.result is not None:
        return picker.result
    return None


async def run_undo_picker_async(tree: ConversationTree) -> Optional[UndoResult]:
    """Async wrapper — runs the sync picker in a thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_undo_picker, tree)
