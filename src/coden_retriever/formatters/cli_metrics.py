"""
CLI metrics formatter protocol.

Defines the interface contract for CLI metric formatters (hotspots, clones, etc.).
All new CLI metrics MUST implement this protocol to ensure consistent:
- Colored output based on severity/importance
- Clickable VS Code hyperlinks for entities
- JSON output support
- Statistics formatting

IMPORTANT CONTRACT RULES:
1. CLI mode MUST pass token_limit=None (no limit) - users control results via -n/--limit
2. MCP mode should pass token_limit=4000 (or similar) for LLM context windows
3. Token budget should NEVER bottleneck CLI output - only -n/--limit controls result count
4. Colors use SeverityTier enum - higher severity = red, lower = green
5. All entity names MUST be clickable hyperlinks (vscode://file/...)
"""
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..constants import FORMATTER_WIDTH
from .terminal_style import TerminalStyle, get_terminal_style

FALSE_POSITIVE_WARNING = "Note: Results may contain false positives. Review before acting."

# Shared result limit warning strings used by all parameter headers
RESULT_LIMIT_ALL = "[!] Result Limit: ALL (may be slow for large repos)"
RESULT_LIMIT_TOP_FMT = "[!] Result Limit: TOP {} -- more results may exist (use -n -1 for all)"


def format_result_limit_line(limit: int | None) -> str:
    """Format the result limit warning line for parameter headers."""
    if limit is None:
        return RESULT_LIMIT_ALL
    return RESULT_LIMIT_TOP_FMT.format(limit)


def format_parameter_header(title: str, param_lines: list[str], limit: int | None) -> str:
    """Build a standard parameter header block used by all analysis formatters.

    Args:
        title: Header title (e.g. "DEAD CODE DETECTION PARAMETERS")
        param_lines: Analysis-specific parameter lines (e.g. "Confidence Threshold: >= 80%")
        limit: Result limit (None = unlimited)
    """
    sep = "=" * FORMATTER_WIDTH
    lines = [sep, title, sep]
    lines.extend(param_lines)
    lines.append(format_result_limit_line(limit))
    lines.append(FALSE_POSITIVE_WARNING)
    lines.append(sep)
    return "\n".join(lines)


def extract_filename(file_path: str) -> str:
    """Extract filename from a path, handling both Unix and Windows separators."""
    return file_path.split("/")[-1].split("\\")[-1]


class SeverityTier(Enum):
    """Color severity tiers for CLI metric output.

    Higher severity (CRITICAL, HIGH) = red tones (urgent action needed)
    Lower severity (SAFE, OPTIMAL) = green tones (healthy state)

    Usage:
        tier = SeverityTier.HIGH
        style.colorize(text, tier.value)
    """

    CRITICAL = "tier_1"   # Most severe - bright red
    HIGH = "tier_2"       # High severity - red
    ELEVATED = "tier_3"   # Elevated - dark orange
    MODERATE = "tier_4"   # Moderate - orange
    MEDIUM = "tier_5"     # Medium - yellow-orange
    NOTABLE = "tier_6"    # Notable - yellow
    LOW = "tier_7"        # Low - yellow-green
    MINIMAL = "tier_8"    # Minimal - light green
    SAFE = "tier_9"       # Safe - green
    OPTIMAL = "tier_10"   # Optimal - bright green

    @classmethod
    def from_score(cls, score: float, max_score: float) -> "SeverityTier":
        """Get tier based on normalized score (0.0 to 1.0 ratio).

        Args:
            score: Current score value
            max_score: Maximum possible score

        Returns:
            Appropriate SeverityTier based on score ratio
        """
        if max_score <= 0:
            return cls.OPTIMAL

        ratio = score / max_score
        if ratio >= 0.9:
            return cls.CRITICAL
        elif ratio >= 0.8:
            return cls.HIGH
        elif ratio >= 0.7:
            return cls.ELEVATED
        elif ratio >= 0.6:
            return cls.MODERATE
        elif ratio >= 0.5:
            return cls.MEDIUM
        elif ratio >= 0.4:
            return cls.NOTABLE
        elif ratio >= 0.3:
            return cls.LOW
        elif ratio >= 0.2:
            return cls.MINIMAL
        elif ratio >= 0.1:
            return cls.SAFE
        else:
            return cls.OPTIMAL


@runtime_checkable
class CLIMetricFormatter(Protocol):
    """Protocol for CLI metric formatters.

    All CLI metrics (hotspots, clones, etc.) must implement this interface
    to ensure consistent user experience with colors and hyperlinks.

    Example implementation:
        class CloneFormatter:
            def format_items(self, items, output_format, reverse) -> str:
                if output_format == "json":
                    return json.dumps(items, indent=2)
                style = get_terminal_style()
                # ... use style.colorize(), style.make_link() etc.

            def format_stats(self, summary) -> str:
                # ... format summary statistics

            def get_tier(self, item) -> SeverityTier:
                # ... return SeverityTier based on item severity
    """

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format metric items for CLI output.

        MUST use TerminalStyle for:
        - style.colorize(text, tier.value) for colored text
        - style.make_link(text, file_path, line, tier=tier.value) for hyperlinks
        - style.render_to_string(text) to convert to ANSI string

        Args:
            items: List of metric item dicts
            output_format: Output format ("tree", "json")
            reverse: If True, reverse display order

        Returns:
            Formatted string with ANSI colors and OSC 8 hyperlinks
        """
        ...

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format summary statistics for stderr output.

        Args:
            summary: Summary statistics dict

        Returns:
            Formatted statistics string
        """
        ...

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier for an item based on its severity/importance.

        Args:
            item: Single metric item dict

        Returns:
            SeverityTier enum value
        """
        ...


class BaseCLIMetricFormatter(ABC):
    """Abstract base class for CLI metric formatters.

    Provides common functionality and enforces the protocol contract.
    Extend this class for new CLI metrics to ensure consistency.

    Example:
        class CloneFormatter(BaseCLIMetricFormatter):
            def get_tier(self, item):
                sim = item.get("similarity", 0)
                if sim >= 0.9999:
                    return SeverityTier.HIGH
                if sim >= 0.98:
                    return SeverityTier.MODERATE
                return SeverityTier.LOW

            def _format_table_row(self, item, rank):
                tier = self.get_tier(item)
                # ... format single row with colors and links
    """

    def __init__(self) -> None:
        self._style: TerminalStyle | None = None

    @property
    def style(self) -> TerminalStyle:
        """Lazy-load terminal style."""
        if self._style is None:
            self._style = get_terminal_style()
        return self._style

    @abstractmethod
    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier for item. Must be implemented by subclasses."""
        pass

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Render items as a pipe-separated table (the default CLI format).

        This default works for any subclass that implements the table-builder
        hooks (`_build_header`, `_format_row`, `_empty_message`,
        `_table_width`, `_total_label`). Subclasses with non-table layouts
        (e.g. tree views) should override this method directly.
        """
        if output_format == "json":
            return json.dumps(items, indent=2)
        if not items:
            return self._empty_message()
        display_items = self._order_for_display(items, reverse)
        total = len(display_items)
        lines = [self._build_header(), "-" * self._table_width()]
        for i, item in enumerate(display_items, 1):
            rank = self._rank_for_position(i, total, reverse)
            lines.append(self._format_row(item, rank))
        lines.append("-" * self._table_width())
        lines.append(self._total_label(len(items)))
        return "\n".join(lines)

    @abstractmethod
    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format statistics. Must be implemented by subclasses."""
        pass

    @staticmethod
    def truncate_value(value: str, max_len: int) -> str:
        """Truncate a string for table display, appending '...' when cut."""
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."

    def colorize(self, text: str, tier: SeverityTier) -> str:
        """Colorize text using the tier color."""
        return self.style.render_to_string(self.style.colorize(text, tier.value))

    def make_hyperlink(
        self,
        text: str,
        file_path: str,
        line: int,
        tier: SeverityTier | None = None,
    ) -> str:
        """Create a clickable hyperlink with optional coloring."""
        tier_value = tier.value if tier else None
        link = self.style.make_link(text, file_path, line, tier=tier_value)
        return self.style.render_to_string(link)

    def _order_for_display(
        self,
        items: list[dict[str, Any]],
        reverse: bool,
    ) -> list[dict[str, Any]]:
        """Order items for display.

        Default assumes items arrive sorted ASCENDING by severity, so the
        default view (`reverse=False`) flips them to show the most severe
        first. Subclasses whose inputs arrive sorted descending should
        override.
        """
        return list(items) if reverse else list(reversed(items))

    def _rank_for_position(self, position: int, total: int, reverse: bool) -> int:
        """Compute the rank label for an item at `position` (1-based).

        Default convention (used by sensitive_value, tramp_data, dead_code):
        rank just numbers the displayed rows. Subclasses that want rank 1 to
        always mean "most significant" (e.g. magic_constant) override.
        """
        return position if reverse else total - position + 1

    # The following hooks have no sensible default; subclasses that use the
    # default `format_items` MUST implement them. Subclasses that override
    # `format_items` directly are free to ignore these.
    def _build_header(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_header() to use "
            "the default format_items template"
        )

    def _format_row(self, item: dict[str, Any], rank: int) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _format_row() to use "
            "the default format_items template"
        )

    def _empty_message(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _empty_message() to use "
            "the default format_items template"
        )

    def _table_width(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _table_width() to use "
            "the default format_items template"
        )

    def _total_label(self, count: int) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _total_label() to use "
            "the default format_items template"
        )
