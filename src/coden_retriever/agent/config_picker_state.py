"""Pure state logic for the interactive config picker.

Contains ConfigItem, the ConfigPicker state machine, and the step_numeric
helper. No prompt_toolkit / Rich — unit-testable without a TTY.
"""

from dataclasses import dataclass
from typing import Any, Optional

from ..config_loader import (
    DEFAULT_FLOAT_STEP,
    DEFAULT_INT_STEP,
    SETTING_CONSTRAINTS,
    SETTING_STEPS,
    parse_config_value,
    validate_config_value,
)

# Max digits kept after rounding float steps, to avoid 0.1+0.2=0.3000000004 artifacts.
# WHY 3: step sizes go as small as 0.01 for some future settings; 3 decimals covers that.
FLOAT_STEP_ROUND_DIGITS: int = 3

NUMERIC_TYPES: frozenset[str] = frozenset({"int", "float"})


@dataclass
class ConfigItem:
    """A single config setting row in the picker."""

    key: str
    original_value: str
    value_type: str                       # "bool", "int", "float", "str"
    description: str
    section: str
    staged_value: Optional[str] = None    # None = unchanged; else the new string

    @property
    def display_value(self) -> str:
        """The value string to render (staged if set, otherwise original)."""
        return self.staged_value if self.staged_value is not None else self.original_value

    @property
    def changed(self) -> bool:
        """Whether this item has a staged change that differs from the original.

        Float values are compared numerically to avoid false positives from
        formatting differences like '2.0' vs '2' (both represent the same value).
        """
        if self.staged_value is None:
            return False
        if self.staged_value == self.original_value:
            return False
        if self.value_type == "float":
            try:
                return float(self.staged_value) != float(self.original_value)
            except ValueError:
                pass
        return True


def step_numeric(current_str: str, key: str, value_type: str, direction: int) -> str:
    """Return the next value as a string after stepping by one delta.

    Clamps to SETTING_CONSTRAINTS when defined. Rounds floats to avoid
    binary drift. Raises ValueError if the current string isn't numeric.
    """
    step = SETTING_STEPS.get(
        key,
        DEFAULT_INT_STEP if value_type == "int" else DEFAULT_FLOAT_STEP,
    )
    current: float
    if value_type == "int":
        current = int(current_str)
    else:
        current = float(current_str)
    new = current + direction * step

    if key in SETTING_CONSTRAINTS:
        lo, hi, _ = SETTING_CONSTRAINTS[key]
        new = max(lo, min(hi, new))

    if value_type == "int":
        return str(int(new))
    return f"{round(new, FLOAT_STEP_ROUND_DIGITS):g}"


@dataclass
class EditState:
    """Inline-edit sub-state for ConfigPicker: buffer, active flag, last error."""

    active: bool = False
    buffer: str = ""
    error: Optional[str] = None

    def start(self, initial: str) -> None:
        self.active = True
        self.buffer = initial
        self.error = None

    def append(self, char: str) -> None:
        if self.active:
            self.buffer += char
            self.error = None

    def backspace(self) -> None:
        if self.active and self.buffer:
            self.buffer = self.buffer[:-1]
            self.error = None

    def clear(self) -> None:
        if self.active:
            self.buffer = ""
            self.error = None

    def reset(self) -> None:
        self.active = False
        self.buffer = ""
        self.error = None


class ConfigPicker:
    """Interactive config picker state — navigation, stepping, edit dispatch."""

    def __init__(self, items: list[ConfigItem]):
        self.items = items
        self.selected_index = 0
        self.cancelled = False
        self.applied = False
        self.edit = EditState()

    @property
    def selected(self) -> Optional[ConfigItem]:
        return self.items[self.selected_index] if self.items else None

    def navigate(self, delta: int) -> None:
        """Move selection by delta; cancels any active edit first."""
        if self.edit.active:
            self.edit.reset()
        self.selected_index = max(
            0, min(len(self.items) - 1, self.selected_index + delta),
        )

    def toggle_bool(self) -> None:
        item = self.selected
        if item is None or item.value_type != "bool":
            return
        item.staged_value = "false" if item.display_value == "true" else "true"

    def step_current(self, direction: int) -> None:
        item = self.selected
        if item is None or item.value_type not in NUMERIC_TYPES:
            return
        try:
            item.staged_value = step_numeric(
                item.display_value, item.key, item.value_type, direction,
            )
        except ValueError as exc:
            # Cannot step a non-numeric display value (e.g. '?' placeholder).
            # Surface as an edit error so the toolbar informs the user.
            self.edit.error = str(exc)

    def enter_edit(self, initial_buffer: str) -> None:
        item = self.selected
        if item is None or item.value_type == "bool":
            return
        self.edit.start(initial_buffer)

    def commit_edit(self) -> bool:
        """Validate buffer and stage it; returns True on success."""
        if not self.edit.active:
            return False
        item = self.selected
        if item is None:
            self.edit.reset()
            return False
        ok, parsed, err = parse_config_value(item.key, self.edit.buffer)
        if not ok:
            self.edit.error = err
            return False
        valid, v_err = validate_config_value(item.key, parsed, check_paths=True)
        if not valid:
            self.edit.error = v_err
            return False
        item.staged_value = self.edit.buffer
        self.edit.reset()
        return True

    def get_all_changes(self) -> dict[str, Any]:
        """Return parsed staged changes that differ from the original value."""
        changes: dict[str, Any] = {}
        for item in self.items:
            if not item.changed:
                continue
            ok, parsed, _ = parse_config_value(item.key, item.staged_value or "")
            if ok:
                changes[item.key] = parsed
        return changes
