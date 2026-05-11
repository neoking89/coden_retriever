"""Pure state logic for the /undo picker — no prompt_toolkit, unit-testable."""

from dataclasses import dataclass
from typing import Optional

from .tool_call_index import BranchHeaderRow, PickerRow, ToolCallRow

ACTION_FORK: str = "fork"
ACTION_SWITCH: str = "switch"


@dataclass
class UndoResult:
    """Outcome of a confirmed /undo picker session."""

    action: str
    branch_id: str
    fork_message_index: int
    directive: str


@dataclass
class DirectiveBuffer:
    """Sub-state for the post-selection directive input."""

    active: bool = False
    buffer: str = ""
    pending_action: Optional[str] = None
    pending_branch_id: Optional[str] = None
    pending_fork_index: int = 0

    def start_fork(self, branch_id: str, fork_index: int) -> None:
        self.active = True
        self.buffer = ""
        self.pending_action = ACTION_FORK
        self.pending_branch_id = branch_id
        self.pending_fork_index = fork_index

    def start_switch(self, branch_id: str) -> None:
        self.active = True
        self.buffer = ""
        self.pending_action = ACTION_SWITCH
        self.pending_branch_id = branch_id
        self.pending_fork_index = 0

    def reset(self) -> None:
        self.active = False
        self.buffer = ""
        self.pending_action = None
        self.pending_branch_id = None
        self.pending_fork_index = 0

    def append(self, char: str) -> None:
        if self.active:
            self.buffer += char

    def backspace(self) -> None:
        if self.active and self.buffer:
            self.buffer = self.buffer[:-1]

    def clear(self) -> None:
        if self.active:
            self.buffer = ""


def _first_selectable_index(rows: list[PickerRow]) -> int:
    """Land the cursor on the first tool call if present, else row 0."""
    for i, row in enumerate(rows):
        if isinstance(row, ToolCallRow):
            return i
    return 0


class UndoPicker:
    """Navigation + selection state for the /undo picker."""

    def __init__(self, rows: list[PickerRow]):
        self.rows: list[PickerRow] = rows
        self.selected_index: int = _first_selectable_index(rows) if rows else 0
        self.cancelled: bool = False
        self.applied: bool = False
        self.directive: DirectiveBuffer = DirectiveBuffer()
        self.result: Optional[UndoResult] = None
        self.message: Optional[str] = None

    @property
    def selected(self) -> Optional[PickerRow]:
        return self.rows[self.selected_index] if self.rows else None

    def navigate(self, delta: int) -> None:
        """Move selection by delta; ignored while the directive input is active."""
        if self.directive.active or not self.rows:
            return
        self.message = None
        self.selected_index = max(
            0, min(len(self.rows) - 1, self.selected_index + delta),
        )

    def confirm(self) -> bool:
        """Handle Enter. Returns True iff the picker should exit after this call."""
        if self.directive.active:
            self.result = UndoResult(
                action=self.directive.pending_action or "",
                branch_id=self.directive.pending_branch_id or "",
                fork_message_index=self.directive.pending_fork_index,
                directive=self.directive.buffer.strip(),
            )
            self.applied = True
            return True

        row = self.selected
        if row is None:
            return False
        if isinstance(row, BranchHeaderRow):
            if row.is_current:
                self.message = "Already on this branch"
                return False
            self.directive.start_switch(row.branch_id)
            return False
        if isinstance(row, ToolCallRow):
            self.directive.start_fork(row.branch_id, row.fork_message_index)
            return False
        return False

    def cancel(self) -> bool:
        """Handle Esc. Returns True iff the entire picker should exit."""
        if self.directive.active:
            self.directive.reset()
            self.message = None
            return False
        self.cancelled = True
        return True
