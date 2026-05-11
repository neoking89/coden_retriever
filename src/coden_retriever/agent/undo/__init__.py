"""Conversation-history branching for `/undo` — rewind to any prior tool call.

Lets the user fork the interactive session at any past tool call, try a
different path, and later switch between branches without losing history.
"""

from .branch import ConversationBranch, ConversationTree
from .picker import run_undo_picker, run_undo_picker_async
from .picker_state import UndoPicker, UndoResult

__all__ = [
    "ConversationBranch",
    "ConversationTree",
    "UndoPicker",
    "UndoResult",
    "run_undo_picker",
    "run_undo_picker_async",
]
