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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .terminal_style import TerminalStyle, get_terminal_style

FALSE_POSITIVE_WARNING = "Note: Results may contain false positives. Review before acting."


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

    @abstractmethod
    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format items for output. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format statistics. Must be implemented by subclasses."""
        pass

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
