"""System-clipboard integration and last-response memory.

The ``/copy`` slash-command exposes the most-recently-rendered agent answer.
:class:`~coden_retriever.agent.response_renderer.answer.AnswerRenderer`
is the only writer; the ``commands`` module is the reader.
"""

from __future__ import annotations

import subprocess
import sys
from enum import Enum
from typing import Optional


class ClipboardStatus(Enum):
    """Outcome of a :func:`copy_last_response` attempt.

    The string values are kept stable so legacy callers that ``==`` against
    the raw strings continue to work; new callers should compare against
    the enum members directly.
    """

    COPIED = "copied"
    NO_RESPONSE = "no_response"
    ERROR = "clipboard_error"


_last_response: Optional[str] = None


def remember_response(text: str) -> None:
    """Cache the latest agent answer so ``/copy`` can retrieve it later."""
    global _last_response
    _last_response = text


def copy_last_response() -> ClipboardStatus:
    """Copy the last remembered agent response to the system clipboard."""
    if _last_response is None:
        return ClipboardStatus.NO_RESPONSE
    if _copy_to_clipboard(_last_response):
        return ClipboardStatus.COPIED
    return ClipboardStatus.ERROR


def _copy_to_clipboard(text: str) -> bool:
    """Copy ``text`` to the system clipboard via the platform-native CLI tool."""
    try:
        _run_platform_clipboard(text)
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _run_platform_clipboard(text: str) -> None:
    payload = text.encode()
    if sys.platform == "win32":
        subprocess.run(
            ["clip"],
            input=payload,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=payload, check=True)
        return
    # Linux — xclip preferred, xsel as fallback.
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=payload,
            check=True,
        )
    except FileNotFoundError:
        subprocess.run(
            ["xsel", "--clipboard", "--input"],
            input=payload,
            check=True,
        )
