"""Key bindings for the interactive config picker.

Maps keyboard input to ConfigPicker state mutations. All real logic lives
on ConfigPicker; handlers here are thin glue that route keys to mutations.
"""

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from .config_picker_state import NUMERIC_TYPES, ConfigPicker
from .picker_styles import (
    KEY_CTRL_C,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_QUIT,
    KEY_UP,
    KEY_VIM_DOWN,
    KEY_VIM_UP,
)

# Explicit control/navigation keys captured first; Keys.Any handles the rest.
# WHY an allowlist: catching a bare `q` during edit mode must append "q" to the
# buffer, not exit the picker — Keys.Any routes printable chars correctly.
_PRINTABLE_RANGE_START: int = 32   # ASCII space
_PRINTABLE_RANGE_END: int = 126    # ASCII tilde


def _is_printable(char: str) -> bool:
    """True if char is a single printable ASCII character."""
    return len(char) == 1 and _PRINTABLE_RANGE_START <= ord(char) <= _PRINTABLE_RANGE_END


def _step_or_edit(event, picker: ConfigPicker, direction: int) -> None:  # type: ignore[misc]
    """Arrow / </> / h-l handler: step numeric, toggle bool, overwrite on str."""
    char = event.data
    if picker.edit.active:
        if char and _is_printable(char):
            picker.edit.append(char)
        return
    item = picker.selected
    if item is None:
        return
    if item.value_type == "bool":
        picker.toggle_bool()
    elif item.value_type in NUMERIC_TYPES:
        picker.step_current(direction)
    elif item.value_type == "str" and char and _is_printable(char):
        picker.enter_edit(char)


def _bind_navigation(kb: KeyBindings, picker: ConfigPicker) -> None:
    @kb.add(KEY_UP)
    @kb.add(KEY_VIM_UP)
    def _up(event) -> None:  # type: ignore[misc]
        picker.navigate(-1)

    @kb.add(KEY_DOWN)
    @kb.add(KEY_VIM_DOWN)
    def _down(event) -> None:  # type: ignore[misc]
        picker.navigate(+1)


def _bind_stepping(kb: KeyBindings, picker: ConfigPicker) -> None:
    @kb.add("<")
    @kb.add("left")
    @kb.add("h")
    def _dec(event) -> None:  # type: ignore[misc]
        _step_or_edit(event, picker, direction=-1)

    @kb.add(">")
    @kb.add("right")
    @kb.add("l")
    def _inc(event) -> None:  # type: ignore[misc]
        _step_or_edit(event, picker, direction=+1)


def _bind_space(kb: KeyBindings, picker: ConfigPicker) -> None:
    @kb.add(" ")
    def _space(event) -> None:  # type: ignore[misc]
        if picker.edit.active:
            picker.edit.append(" ")
            return
        item = picker.selected
        if item is None:
            return
        if item.value_type == "bool":
            picker.toggle_bool()
        else:
            picker.enter_edit(" ")


def _bind_backspace(kb: KeyBindings, picker: ConfigPicker) -> None:
    @kb.add("backspace")
    def _backspace(event) -> None:  # type: ignore[misc]
        if picker.edit.active:
            picker.edit.backspace()
            return
        item = picker.selected
        if item is None or item.value_type == "bool":
            return
        # Start edit with one char trimmed off the current value ("delete tail").
        current = item.display_value
        trimmed = current[:-1] if current else ""
        picker.enter_edit(trimmed)


def _bind_commit_and_exit(kb: KeyBindings, picker: ConfigPicker) -> None:
    @kb.add(KEY_ENTER)
    def _enter(event) -> None:  # type: ignore[misc]
        if picker.edit.active:
            picker.commit_edit()  # stays in edit on failure, error shown in UI
            return
        picker.applied = True
        event.app.exit()

    @kb.add(KEY_ESCAPE)
    def _escape(event) -> None:  # type: ignore[misc]
        if picker.edit.active:
            picker.edit.reset()
            return
        picker.cancelled = True
        event.app.exit()

    @kb.add(KEY_QUIT)
    def _q(event) -> None:  # type: ignore[misc]
        if picker.edit.active:
            picker.edit.append("q")
            return
        picker.cancelled = True
        event.app.exit()

    @kb.add(KEY_CTRL_C)
    def _ctrl_c(event) -> None:  # type: ignore[misc]
        picker.cancelled = True
        event.app.exit()

    @kb.add("c-u")
    def _ctrl_u(event) -> None:  # type: ignore[misc]
        picker.edit.clear()


def _bind_printable_fallback(kb: KeyBindings, picker: ConfigPicker) -> None:
    @kb.add(Keys.Any)
    def _any(event) -> None:  # type: ignore[misc]
        char = event.data
        if not char or not _is_printable(char):
            return
        if picker.edit.active:
            picker.edit.append(char)
            return
        item = picker.selected
        if item is None or item.value_type == "bool":
            return
        picker.enter_edit(char)


def build_picker_keybindings(picker: ConfigPicker) -> KeyBindings:
    """Wire every supported key to its ConfigPicker action."""
    kb = KeyBindings()
    _bind_navigation(kb, picker)
    _bind_stepping(kb, picker)
    _bind_space(kb, picker)
    _bind_backspace(kb, picker)
    _bind_commit_and_exit(kb, picker)
    _bind_printable_fallback(kb, picker)
    return kb
