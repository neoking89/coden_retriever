"""Response rendering for agent output.

Handles display of:
- Streaming text responses
- Final answer panels
- ReAct reasoning steps

Follows Single Responsibility Principle - only handles display.
"""

import re
from types import TracebackType
from typing import Optional, Type

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .rich_console import console, set_active_live

# special regex patterns for markdown normalization:

# Matches 3+ consecutive newlines (excessive blank lines)
_RE_EXCESSIVE_NEWLINES = re.compile(r'\n{3,}')

# Matches table header separator followed by blank lines
# Example: |---|---|\n\n  -> should become |---|---|\n
_RE_TABLE_HEADER_GAP = re.compile(
    r'(?m)^(\s*\|[-:| ]+\|\s*)\n{2,}'
)

# Matches table rows with blank lines between them
# Uses lookahead to not consume the next row's pipe
_RE_TABLE_ROW_GAP = re.compile(
    r'(?m)^(\s*\|.*?\|\s*)\n{2,}(?=\s*\|)'
)

# Matches fenced code blocks (``` or ~~~) to protect them from modification
_RE_FENCED_CODE_BLOCK = re.compile(
    r'(```[\s\S]*?```|~~~[\s\S]*?~~~)',
    re.MULTILINE
)


def _normalize_markdown(content: str) -> str:
    """Normalize markdown content to reduce blank space issues.

    Rich's Markdown renderer can produce excessive vertical space when
    rendering tables if the source markdown has gaps between rows.
    This function normalizes the content to mitigate the issue.

    Features:
    - Collapses excessive blank lines (3+ -> 2)
    - Removes blank lines within markdown tables
    - Protects fenced code blocks from modification
    - Preserves intentional formatting outside tables

    Args:
        content: Raw markdown content.

    Returns:
        Normalized markdown with reduced blank lines.
    """
    if not content:
        return ""

    # Step 1: Extract and protect fenced code blocks
    # Replace them with placeholders to prevent regex from modifying code
    code_blocks: list[str] = []

    def _preserve_code_block(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return f"\x00CODE_BLOCK_{len(code_blocks) - 1}\x00"

    content = _RE_FENCED_CODE_BLOCK.sub(_preserve_code_block, content)

    # Step 2: Collapse 3+ consecutive blank lines into 2
    # (Preserves distinct paragraph breaks while removing massive gaps)
    content = _RE_EXCESSIVE_NEWLINES.sub('\n\n', content)

    # Step 3: Fix Table Headers
    # Removes blank lines between the header separator (|---|) and the first row.
    content = _RE_TABLE_HEADER_GAP.sub(r'\1\n', content)

    # Step 4: Fix Table Rows
    # Removes blank lines between normal table rows.
    content = _RE_TABLE_ROW_GAP.sub(r'\1\n', content)

    # Step 5: Restore protected code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f"\x00CODE_BLOCK_{i}\x00", block)

    # Step 6: Strip only leading/trailing blank lines, preserve internal structure
    content = content.strip('\n')

    return content


class StreamRenderer:
    """Renders streaming text with Rich Live display.

    Uses vertical_overflow="visible" to allow content to scroll naturally
    instead of showing "..." ellipsis when content exceeds terminal height.
    """

    def __init__(
        self,
        refresh_per_second: int = 4,
        max_lines: Optional[int] = None,
    ) -> None:
        """Initialize the stream renderer.

        Args:
            refresh_per_second: How often to refresh the display (default: 4).
                Lower values reduce flickering but may feel less responsive.
            max_lines: Optional limit on displayed lines. When set, only the
                most recent N lines are shown, simulating auto-scroll behavior.
                If None, all content is shown (may cause overflow).
        """
        self.refresh_per_second = refresh_per_second
        self.max_lines = max_lines
        self._live: Optional[Live] = None

    def __enter__(self) -> "StreamRenderer":
        """Start the live display."""
        self._live = Live(
            Text(""),
            console=console,
            refresh_per_second=self.refresh_per_second,
            transient=True,
            # Use "visible" to allow content to scroll naturally instead of showing "..."
            vertical_overflow="visible",
        )
        # Register the Live display globally so it can be paused by permission picker
        set_active_live(self._live)
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop the live display."""
        if self._live:
            # Unregister the Live display
            set_active_live(None)
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None

    def update(self, text: str) -> None:
        """Update the display with new text.

        If max_lines is set, only shows the most recent N lines.
        """
        if self._live:
            display_text = text
            if self.max_lines is not None:
                lines = text.split("\n")
                if len(lines) > self.max_lines:
                    display_text = "\n".join(lines[-self.max_lines:])
            self._live.update(Text.from_markup(display_text))


class AnswerRenderer:
    """Renders final answer in a styled panel."""

    def __init__(
        self,
        title: str = "Agent",
        border_style: str = "green",
    ) -> None:
        """Initialize the answer renderer.

        Args:
            title: Panel title text.
            border_style: Rich border style.
        """
        self.title = title
        self.border_style = border_style

    def _create_panel(self, content: str) -> Panel:
        """Create a styled panel with markdown content."""
        normalized = _normalize_markdown(content)
        return Panel(
            Markdown(normalized),
            title=f"[bold {self.border_style}]{self.title}[/bold {self.border_style}]",
            title_align="left",
            border_style=self.border_style,
            padding=(0, 1),
        )

    def render(self, text: str) -> None:
        """Render the answer instantly."""
        if not text:
            return
        console.print()
        console.print(self._create_panel(text))
        console.print()
