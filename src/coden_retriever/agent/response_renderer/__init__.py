"""Rendering of the agent's streaming output and final answer.

The package is split by domain so each concern lives in its own module:

* :mod:`streaming` — live in-place Rich ``Live`` panel driven by typed
  stream events (text, thinking, tool calls, tool results).
* :mod:`answer`    — final answer panel printed after the stream completes.
* :mod:`markdown`  — Rich-friendly whitespace + table normalization.
* :mod:`latex`     — LaTeX math → Unicode substitution for terminal display.
* :mod:`clipboard` — system-clipboard integration backing the ``/copy`` command.
"""

from .answer import AnswerRenderer
from .clipboard import copy_last_response
from .latex import replace_latex_symbols
from .markdown import normalize_markdown
from .streaming import (
    DEFAULT_STREAM_STYLE,
    StderrToolReporter,
    StdoutStreamWriter,
    StreamRenderer,
    StreamRendererStyle,
)

__all__ = [
    "AnswerRenderer",
    "DEFAULT_STREAM_STYLE",
    "StderrToolReporter",
    "StdoutStreamWriter",
    "StreamRenderer",
    "StreamRendererStyle",
    "copy_last_response",
    "normalize_markdown",
    "replace_latex_symbols",
]
