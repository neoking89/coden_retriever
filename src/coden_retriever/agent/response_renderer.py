"""Response rendering for agent output.

Handles display of:
- Streaming text responses
- Final answer panels
- ReAct reasoning steps
"""

import re
import subprocess
import sys
from types import TracebackType
from typing import Optional, Type

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .rich_console import console, set_active_live

# Stores the last rendered answer so /copy can access it on demand
_last_response: Optional[str] = None


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to the system clipboard using platform-native tools.

    Args:
        text: The text to copy.

    Returns:
        True if the copy succeeded, False otherwise.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["clip"],
                input=text.encode(),
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        else:
            # Linux — try xclip first, fall back to xsel
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode(),
                    check=True,
                )
            except FileNotFoundError:
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=text.encode(),
                    check=True,
                )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def copy_last_response() -> str:
    """Copy the last agent response to the system clipboard.

    Returns:
        Status message describing the result.
    """
    if _last_response is None:
        return "no_response"
    if _copy_to_clipboard(_last_response):
        return "copied"
    return "clipboard_error"


# LaTeX math-mode symbols that LLMs commonly emit instead of Unicode.
# Frequency tiers based on observed LLM output across GPT-4, Claude, Gemma, LLaMA.
# All replacements use cross-platform Unicode (Latin-1, Dingbats, Math Operators, etc.).
_LATEX_SYMBOL_MAP: dict[str, str] = {
    # --- VERY COMMON: seen regularly across multiple LLMs ---
    # Arrows — chevron/triangle style for a modern terminal look
    r"\rightarrow": "\u276F",       # ❯  (Dingbats — used by Starship, Oh-My-Zsh)
    r"\to": "\u276F",               # ❯  (shorthand for \rightarrow, equally common)
    r"\leftarrow": "\u276E",        # ❮  (Dingbats)
    r"\Rightarrow": "\u00BB",       # »  (Latin-1 Supplement — universal)
    r"\Leftarrow": "\u00AB",        # «  (Latin-1 Supplement — universal)
    # Arithmetic
    r"\times": "\u00D7",            # ×  (Latin-1 Supplement)
    r"\div": "\u00F7",              # ÷  (Latin-1 Supplement)
    # Comparisons
    r"\neq": "\u2260",              # ≠  (Math Operators)
    r"\leq": "\u2264",              # ≤  (Math Operators)
    r"\geq": "\u2265",              # ≥  (Math Operators)
    r"\approx": "\u2248",           # ≈  (Math Operators)
    # Constants / big operators
    r"\infty": "\u221E",            # ∞  (Math Operators)
    r"\sum": "\u2211",              # ∑  (Math Operators)
    r"\prod": "\u220F",             # ∏  (Math Operators)
    # Greek letters (ML/stats contexts)
    r"\alpha": "\u03B1",            # α
    r"\beta": "\u03B2",             # β
    r"\gamma": "\u03B3",            # γ
    r"\delta": "\u03B4",            # δ
    r"\epsilon": "\u03B5",          # ε
    r"\lambda": "\u03BB",           # λ
    r"\mu": "\u03BC",               # μ
    r"\pi": "\u03C0",               # π
    r"\sigma": "\u03C3",            # σ
    r"\theta": "\u03B8",            # θ
    r"\phi": "\u03C6",              # φ
    r"\omega": "\u03C9",            # ω
    # Set membership
    r"\in": "\u2208",               # ∈  (Math Operators)
    r"\notin": "\u2209",            # ∉  (Math Operators)
    r"\subset": "\u2282",           # ⊂  (Math Operators)
    r"\subseteq": "\u2286",         # ⊆  (Math Operators)
    # Punctuation
    r"\cdot": "\u00B7",             # ·  (Latin-1 Supplement)
    r"\ldots": "\u2026",            # …  (General Punctuation)
    r"\dots": "\u2026",             # …  (alias for \ldots)
    # Brackets (LLMs use these for inner products, tuples)
    r"\langle": "\u27E8",           # ⟨  (Misc Math Symbols-A)
    r"\rangle": "\u27E9",           # ⟩  (Misc Math Symbols-A)

    # --- OCCASIONAL: seen sometimes, worth handling ---
    r"\pm": "\u00B1",               # ±  (Latin-1 Supplement)
    r"\leftrightarrow": "\u276E\u276F",  # ❮❯
    r"\Leftrightarrow": "\u00AB\u00BB",  # «»
    r"\uparrow": "\u25B2",          # ▲  (Geometric Shapes)
    r"\downarrow": "\u25BC",        # ▼  (Geometric Shapes)
    r"\cup": "\u222A",              # ∪  (Math Operators)
    r"\cap": "\u2229",              # ∩  (Math Operators)
    r"\emptyset": "\u2205",         # ∅  (Math Operators)
    r"\supset": "\u2283",           # ⊃  (Math Operators)
    r"\forall": "\u2200",           # ∀  (Math Operators)
    r"\exists": "\u2203",           # ∃  (Math Operators)
    r"\nabla": "\u2207",            # ∇  (Math Operators)
    r"\partial": "\u2202",          # ∂  (Math Operators)
    r"\cdots": "\u22EF",            # ⋯  (Math Operators)
    r"\star": "\u2605",             # ★  (Dingbats)
    r"\bullet": "\u2022",           # •  (General Punctuation)
    r"\checkmark": "\u2714",        # ✔  (Dingbats)
    r"\dagger": "\u2020",           # †  (General Punctuation)
}

# Matches $..$ or $$...$$ containing a single LaTeX command (with optional whitespace)
_RE_LATEX_INLINE = re.compile(
    r'\$\$?\s*('
    + '|'.join(re.escape(cmd) for cmd in sorted(_LATEX_SYMBOL_MAP, key=len, reverse=True))
    + r')\s*\$\$?'
)

# Labeled arrows: $\xrightarrow{Label}$ — seen from Gemma and some GPT-4 outputs.
# Only the two actually observed variants; exotic ones like \xLeftarrow are not worth handling.
_LATEX_LABELED_ARROW_MAP: dict[str, tuple[str, str]] = {
    r"\xrightarrow": ("\u2500[", "]\u276F"),  # ─[Label]❯
    r"\xleftarrow":  ("\u276E[", "]\u2500"),   # ❮[Label]─
}

_RE_LATEX_LABELED = re.compile(
    r'\$\$?\s*('
    + '|'.join(re.escape(cmd) for cmd in sorted(_LATEX_LABELED_ARROW_MAP, key=len, reverse=True))
    + r')\s*\{([^}]*)\}\s*\$\$?'
)

# $\frac{a}{b}$ — very common in math/stats explanations
_RE_LATEX_FRAC = re.compile(
    r'\$\$?\s*\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}\s*\$\$?'
)

# $\sqrt{x}$ — common in complexity/distance discussions
_RE_LATEX_SQRT = re.compile(
    r'\$\$?\s*\\sqrt\s*\{([^}]*)\}\s*\$\$?'
)

# $\mathbb{R}$, $\mathbb{N}$ etc. — number set notation
_RE_LATEX_MATHBB = re.compile(
    r'\$\$?\s*\\mathbb\s*\{([^}]*)\}\s*\$\$?'
)

# Unicode double-struck letters for common number sets
_MATHBB_MAP: dict[str, str] = {
    "R": "\u211D",  # ℝ
    "N": "\u2115",  # ℕ
    "Z": "\u2124",  # ℤ
    "Q": "\u211A",  # ℚ
    "C": "\u2102",  # ℂ
}


def _labeled_to_unicode(match: re.Match[str]) -> str:
    """Convert labeled arrow like $\\xrightarrow{Parser}$ to ─[Parser]❯."""
    cmd = match.group(1)
    label = match.group(2).strip()
    prefix, suffix = _LATEX_LABELED_ARROW_MAP[cmd]
    return f"{prefix}{label}{suffix}"


def _frac_to_unicode(match: re.Match[str]) -> str:
    """Convert $\\frac{a}{b}$ to a/b."""
    return f"{match.group(1).strip()}/{match.group(2).strip()}"


def _sqrt_to_unicode(match: re.Match[str]) -> str:
    """Convert $\\sqrt{x}$ to √(x)."""
    return f"\u221A({match.group(1).strip()})"


def _mathbb_to_unicode(match: re.Match[str]) -> str:
    """Convert $\\mathbb{R}$ to ℝ."""
    letter = match.group(1).strip()
    return _MATHBB_MAP.get(letter, letter)


def _simple_to_unicode(match: re.Match[str]) -> str:
    """Convert simple $\\command$ to its Unicode symbol."""
    return _LATEX_SYMBOL_MAP[match.group(1)]


def _replace_latex_symbols(text: str) -> str:
    """Replace common LaTeX math-mode symbols with Unicode equivalents.

    Handles:
    - Simple symbols: $\\rightarrow$ -> ❯
    - Labeled arrows: $\\xrightarrow{Parser}$ -> ─[Parser]❯
    - Fractions: $\\frac{a}{b}$ -> a/b
    - Square roots: $\\sqrt{x}$ -> √(x)
    - Number sets: $\\mathbb{R}$ -> ℝ
    """
    # Fast path: all LaTeX patterns require at least one '$' delimiter
    if "$" not in text:
        return text

    # Parameterised commands first (more specific patterns)
    text = _RE_LATEX_LABELED.sub(_labeled_to_unicode, text)
    text = _RE_LATEX_FRAC.sub(_frac_to_unicode, text)
    text = _RE_LATEX_SQRT.sub(_sqrt_to_unicode, text)
    text = _RE_LATEX_MATHBB.sub(_mathbb_to_unicode, text)
    # Then simple single-symbol replacements
    return _RE_LATEX_INLINE.sub(_simple_to_unicode, text)


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
    - Replaces LaTeX math symbols with Unicode equivalents
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

    # Protect code blocks from regex modification
    code_blocks: list[str] = []

    def _preserve_code_block(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return f"\x00CODE_BLOCK_{len(code_blocks) - 1}\x00"

    content = _RE_FENCED_CODE_BLOCK.sub(_preserve_code_block, content)

    content = _replace_latex_symbols(content)
    # Collapse 3+ blank lines to preserve paragraph breaks without massive gaps
    content = _RE_EXCESSIVE_NEWLINES.sub('\n\n', content)
    # Rich renders blank lines in tables as extra vertical space
    content = _RE_TABLE_HEADER_GAP.sub(r'\1\n', content)
    content = _RE_TABLE_ROW_GAP.sub(r'\1\n', content)

    for i, block in enumerate(code_blocks):
        content = content.replace(f"\x00CODE_BLOCK_{i}\x00", block)

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
            subtitle="[dim]\U0001f4cb /copy[/dim]",
            subtitle_align="right",
            border_style=self.border_style,
            padding=(0, 1),
        )

    def render(self, text: str) -> None:
        """Render the answer and store it for /copy."""
        global _last_response
        if not text:
            return
        _last_response = text
        console.print()
        console.print(self._create_panel(text))
        console.print()
