"""
Terminal styling utilities using Rich library for cross-platform support.

Provides:
- Clickable hyperlinks (file:// URLs)
- Score-based color mapping with Rich markup
- Works in all terminals that support Rich

Uses lazy imports to avoid loading Rich until actually needed.
Thread-safe initialization via double-checked locking.
"""
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console
    from rich.style import Style
    from rich.text import Text

# Lazy-loaded Rich module references
_rich_console: Any = None
_rich_style: Any = None
_rich_text: Any = None
_rich_load_lock = threading.Lock()


def _ensure_rich_loaded() -> None:
    """Lazily load Rich modules on first use. Thread-safe."""
    global _rich_console, _rich_style, _rich_text
    if _rich_console is None:
        with _rich_load_lock:
            # Double-checked locking pattern
            if _rich_console is None:
                from rich.console import Console as RichConsole
                from rich.style import Style as RichStyle
                from rich.text import Text as RichText
                _rich_console = RichConsole
                _rich_style = RichStyle
                _rich_text = RichText


# 10-tier color gradient for pronounced score visualization
# Uses a heat-map style gradient: green (hot/high) -> yellow -> orange -> red -> gray (cold/low)
# Each tier represents 10% of the score range
# Stored as tuples (color, bold, dim) to avoid loading Rich at import time
_SCORE_STYLE_DEFS = {
    "tier_10": ("#00ff00", True, False),   # 90-100%: Bright green, bold
    "tier_9": ("#40ff00", True, False),    # 80-90%:  Lime green, bold
    "tier_8": ("#80ff00", False, False),   # 70-80%:  Yellow-green
    "tier_7": ("#bfff00", False, False),   # 60-70%:  Chartreuse
    "tier_6": ("#ffff00", False, False),   # 50-60%:  Yellow
    "tier_5": ("#ffbf00", False, False),   # 40-50%:  Gold/amber
    "tier_4": ("#ff8000", False, False),   # 30-40%:  Orange
    "tier_3": ("#ff4000", False, False),   # 20-30%:  Red-orange
    "tier_2": ("#ff0040", False, True),    # 10-20%:  Red, dimmed
    "tier_1": ("#808080", False, True),    # 0-10%:   Gray, dimmed
}

# Cached Style objects (created lazily)
_SCORE_STYLES: dict[str, "Style"] | None = None
_score_styles_lock = threading.Lock()


def _get_score_styles() -> dict[str, "Style"]:
    """Get cached Style objects, creating them on first use. Thread-safe."""
    global _SCORE_STYLES
    if _SCORE_STYLES is None:
        with _score_styles_lock:
            # Double-checked locking pattern
            if _SCORE_STYLES is None:
                _ensure_rich_loaded()
                styles = {}
                for tier, (color, bold, dim) in _SCORE_STYLE_DEFS.items():
                    styles[tier] = _rich_style(color=color, bold=bold, dim=dim)
                _SCORE_STYLES = styles
    return _SCORE_STYLES


class TerminalStyle:
    """
    Terminal styling using Rich for cross-platform color and hyperlink support.
    """

    def __init__(self):
        """Initialize with a Rich console."""
        _ensure_rich_loaded()
        # Use force_terminal=True and color_system="truecolor" for full 24-bit color gradient
        self._console: "Console" = _rich_console(force_terminal=True, highlight=False, color_system="truecolor")

    def get_score_tier(self, score: float, max_score: float) -> str:
        """Determine the tier (1-10) for a given score relative to max."""
        if max_score <= 0:
            return "tier_1"

        ratio = score / max_score
        # Map ratio to tier 1-10 (each tier = 10% range)
        tier_num = min(10, max(1, int(ratio * 10) + 1))
        return f"tier_{tier_num}"

    def get_style(self, tier: str) -> "Style":
        """Get Rich style for a tier."""
        styles = _get_score_styles()
        return styles.get(tier, styles["tier_1"])

    def make_file_url(self, file_path: str | Path, line: int | None = None) -> str:
        """Create a file:// URL for the given path."""
        path = Path(file_path).resolve()
        uri = path.as_uri()
        if line is not None:
            uri = f"{uri}:{line}"
        return uri

    def colorize(self, text: str, tier: str) -> "Text":
        """Create colored Rich Text."""
        style = self.get_style(tier)
        return _rich_text(text, style=style)

    def make_link(
        self,
        text: str,
        file_path: str | Path,
        line: int | None = None,
        tier: str | None = None,
    ) -> "Text":
        """Create a clickable, optionally colored link."""
        url = self.make_file_url(file_path, line)
        style = self.get_style(tier) if tier else _rich_style()
        # Add link to the style
        style = style + _rich_style(link=url)
        return _rich_text(text, style=style)

    def render_to_string(self, text: "Text") -> str:
        """Render Rich Text to string with ANSI codes."""
        with self._console.capture() as capture:
            self._console.print(text, end="")
        return capture.get()

    def format_file_header(
        self,
        rel_path: str,
        abs_path: str | Path,
        score: float,
        max_score: float,
        icon: str = "📄",
    ) -> str:
        """Format a file header with clickable link and color."""
        tier = self.get_score_tier(score, max_score)
        display_text = f"{icon} {rel_path}"
        text = self.make_link(display_text, abs_path, tier=tier)
        return self.render_to_string(text)

    def format_entity(
        self,
        name: str,
        entity_type: str,
        file_path: str | Path,
        line: int,
        score: float,
        max_score: float,
    ) -> str:
        """Format an entity with clickable link and color."""
        tier = self.get_score_tier(score, max_score)
        icon = "🔶" if entity_type in ("class", "struct", "interface") else "🔹"
        display_text = f"{icon} {name} ({entity_type})"
        text = self.make_link(display_text, file_path, line, tier=tier)
        return self.render_to_string(text)

    def format_line_number(
        self,
        line_num: int,
        file_path: str | Path,
    ) -> str:
        """Format a clickable line number."""
        line_str = f"{line_num:4d}"
        text = self.make_link(line_str, file_path, line_num)
        return self.render_to_string(text)

    def format_rank(self, score: float, max_score: float) -> str:
        """Format a rank/score value with color."""
        tier = self.get_score_tier(score, max_score)
        text = self.colorize(f"{score:.4f}", tier)
        return self.render_to_string(text)

    def format_tree_entity(
        self,
        name: str,
        entity_type: str,
        file_path: str | Path,
        line: int,
        score: float | None = None,
        max_score: float | None = None,
    ) -> str:
        """Format an entity for tree view."""
        icon = "🔶" if entity_type in ("class", "struct", "interface") else "🔹"
        display_text = f"{icon} {name} (L{line})"

        tier = None
        if score is not None and max_score is not None and max_score > 0:
            tier = self.get_score_tier(score, max_score)

        text = self.make_link(display_text, file_path, line, tier=tier)
        return self.render_to_string(text)

    def format_tree_file(
        self,
        name: str,
        file_path: str | Path,
        score: float | None = None,
        max_score: float | None = None,
    ) -> str:
        """Format a file for tree view."""
        display_text = f"📄 {name}"

        tier = None
        if score is not None and max_score is not None and max_score > 0:
            tier = self.get_score_tier(score, max_score)

        text = self.make_link(display_text, file_path, tier=tier)
        return self.render_to_string(text)

    def format_stats_entity(
        self,
        name: str,
        file_path: str | Path,
        line: int,
        score: float,
        max_score: float,
        flags: str = "",
    ) -> str:
        """Format an entity name for stats table."""
        tier = self.get_score_tier(score, max_score)
        display_text = name + flags
        text = self.make_link(display_text, file_path, line, tier=tier)
        return self.render_to_string(text)


# Global instance
_default_style: TerminalStyle | None = None
_default_style_lock = threading.Lock()


def get_terminal_style() -> TerminalStyle:
    """Get the default terminal style instance. Thread-safe."""
    global _default_style
    if _default_style is None:
        with _default_style_lock:
            # Double-checked locking pattern
            if _default_style is None:
                _default_style = TerminalStyle()
    return _default_style


def reset_terminal_style() -> None:
    """Reset the default terminal style instance. Thread-safe."""
    global _default_style, _SCORE_STYLES
    with _default_style_lock:
        with _score_styles_lock:
            _default_style = None
            _SCORE_STYLES = None
