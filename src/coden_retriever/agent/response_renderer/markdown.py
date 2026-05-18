"""Whitespace + table normalization for Rich's Markdown renderer.

Rich shows LLM output verbatim — including the blank lines LLMs sprinkle
between table rows. That produces awkward vertical gaps. This module
pre-processes the markdown so the rendered panel stays tight:

* Collapses runs of 3+ blank lines to a paragraph break.
* Squeezes blank lines that fall inside markdown tables.
* Inlines :func:`replace_latex_symbols` so math notation renders too.
* Leaves fenced code blocks untouched.
"""

from __future__ import annotations

import re

from .latex import replace_latex_symbols


class _MarkdownPatterns:
    """All compiled patterns + sentinels used by :func:`normalize_markdown`."""

    # Runs of three or more consecutive newlines.
    EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")

    # Blank lines directly after a table header separator (``|---|---|``).
    TABLE_HEADER_GAP = re.compile(r"(?m)^(\s*\|[-:| ]+\|\s*)\n{2,}")

    # Blank lines between two table-body rows. The lookahead keeps the
    # next row's leading pipe from being consumed.
    TABLE_ROW_GAP = re.compile(r"(?m)^(\s*\|.*?\|\s*)\n{2,}(?=\s*\|)")

    # Fenced code blocks (``` or ~~~). Stashed so their internal
    # whitespace survives every other regex below.
    FENCED_CODE_BLOCK = re.compile(
        r"(```[\s\S]*?```|~~~[\s\S]*?~~~)",
        re.MULTILINE,
    )

    # Sentinel marker for stashed code blocks. NUL bytes do not occur in
    # well-formed markdown, so this cannot collide with legitimate content.
    CODE_BLOCK_PLACEHOLDER: str = "\x00CODE_BLOCK_{index}\x00"


def normalize_markdown(content: str) -> str:
    """Return ``content`` with table-friendly whitespace and Unicode math."""
    if not content:
        return ""

    stashed: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        stashed.append(match.group(0))
        return _MarkdownPatterns.CODE_BLOCK_PLACEHOLDER.format(index=len(stashed) - 1)

    content = _MarkdownPatterns.FENCED_CODE_BLOCK.sub(_stash, content)

    content = replace_latex_symbols(content)
    content = _MarkdownPatterns.EXCESSIVE_NEWLINES.sub("\n\n", content)
    content = _MarkdownPatterns.TABLE_HEADER_GAP.sub(r"\1\n", content)
    content = _MarkdownPatterns.TABLE_ROW_GAP.sub(r"\1\n", content)

    for index, block in enumerate(stashed):
        placeholder = _MarkdownPatterns.CODE_BLOCK_PLACEHOLDER.format(index=index)
        content = content.replace(placeholder, block)

    return content.strip("\n")
