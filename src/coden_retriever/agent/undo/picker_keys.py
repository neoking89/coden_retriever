"""Keybindings for the /undo picker. Thin glue over UndoPicker state."""

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from ..picker_styles import (
    KEY_CTRL_C,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_QUIT,
    KEY_UP,
    KEY_VIM_DOWN,
    KEY_VIM_UP,
)
from .picker_state import UndoPicker

_PRINTABLE_RANGE_START: int = 32   # ASCII space
_PRINTABLE_RANGE_END: int = 126    # ASCII tilde


def _is_printable(char: str) -> bool:
    return len(char) == 1 and _PRINTABLE_RANGE_START <= ord(char) <= _PRINTABLE_RANGE_END


def _bind_navigation(kb: KeyBindings, picker: UndoPicker) -> None:
    @kb.add(KEY_UP)
    @kb.add(KEY_VIM_UP)
    def _up(event) -> None:  # type: ignore[misc]
        picker.navigate(-1)

    @kb.add(KEY_DOWN)
    @kb.add(KEY_VIM_DOWN)
    def _down(event) -> None:  # type: ignore[misc]
        picker.navigate(+1)


def _bind_confirm_and_cancel(kb: KeyBindings, picker: UndoPicker) -> None:
    @kb.add(KEY_ENTER)
    def _enter(event) -> None:  # type: ignore[misc]
        if picker.confirm():
            event.app.exit()

    @kb.add(KEY_ESCAPE)
    def _escape(event) -> None:  # type: ignore[misc]
        if picker.cancel():
            event.app.exit()

    @kb.add(KEY_QUIT)
    def _q(event) -> None:  # type: ignore[misc]
        if picker.directive.active:
            picker.directive.append("q")
            return
        picker.cancelled = True
        event.app.exit()

    @kb.add(KEY_CTRL_C)
    def _ctrl_c(event) -> None:  # type: ignore[misc]
        picker.cancelled = True
        event.app.exit()

    @kb.add("c-u")
    def _ctrl_u(event) -> None:  # type: ignore[misc]
        picker.directive.clear()


def _bind_backspace(kb: KeyBindings, picker: UndoPicker) -> None:
    @kb.add("backspace")
    def _backspace(event) -> None:  # type: ignore[misc]
        picker.directive.backspace()


def _bind_printable_fallback(kb: KeyBindings, picker: UndoPicker) -> None:
    @kb.add(Keys.Any)
    def _any(event) -> None:  # type: ignore[misc]
        char = event.data
        if not char or not _is_printable(char):
            return
        if picker.directive.active:
            picker.directive.append(char)


def build_picker_keybindings(picker: UndoPicker) -> KeyBindings:
    """Wire every supported key to its UndoPicker action."""
    kb = KeyBindings()
    _bind_navigation(kb, picker)
    _bind_confirm_and_cancel(kb, picker)
    _bind_backspace(kb, picker)
    _bind_printable_fallback(kb, picker)
    return kb
