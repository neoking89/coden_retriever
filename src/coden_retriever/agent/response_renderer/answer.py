"""Final-answer panel rendered after the live stream completes.

Stores the rendered text via :func:`remember_response` so the ``/copy``
command can later push it to the system clipboard.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.panel import Panel

from ..rich_console import console
from .clipboard import remember_response
from .markdown import normalize_markdown


class _AnswerPanelStyle:
    """Defaults for the answer panel's visible chrome."""

    DEFAULT_TITLE: str = "Agent"
    DEFAULT_BORDER_STYLE: str = "green"
    CLIPBOARD_HINT: str = "[dim]📋 /copy[/dim]"
    PADDING: tuple[int, int] = (0, 1)


class AnswerRenderer:
    """Renders the agent's final answer inside a styled Rich panel."""

    def __init__(
        self,
        title: str = _AnswerPanelStyle.DEFAULT_TITLE,
        border_style: str = _AnswerPanelStyle.DEFAULT_BORDER_STYLE,
    ) -> None:
        self.title = title
        self.border_style = border_style

    def render(self, text: str) -> None:
        """Render ``text`` as a markdown panel and remember it for ``/copy``."""
        if not text:
            return
        remember_response(text)
        console.print()
        console.print(self._create_panel(text))
        console.print()

    def _create_panel(self, content: str) -> Panel:
        return Panel(
            Markdown(normalize_markdown(content)),
            title=f"[bold {self.border_style}]{self.title}[/bold {self.border_style}]",
            title_align="left",
            subtitle=_AnswerPanelStyle.CLIPBOARD_HINT,
            subtitle_align="right",
            border_style=self.border_style,
            padding=_AnswerPanelStyle.PADDING,
        )
