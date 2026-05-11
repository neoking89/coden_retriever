"""Magic constant detection CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for magic constant detection results with:
- Pipe-separated table matching hotspots/tramp-data style
- Colored output based on occurrence frequency
- JSON output support
- Summary statistics
"""
from typing import Any

from ..constants import (
    FORMATTER_WIDTH,
    MAGIC_CONSTANT_TIER_HIGH,
    MAGIC_CONSTANT_TIER_MODERATE,
)
from .cli_metrics import (
    BaseCLIMetricFormatter,
    SeverityTier,
    extract_filename,
    format_parameter_header,
)

# 110 chars matches the hotspots table width for visual consistency
_TABLE_WIDTH = 110

# 30 chars for value column (quoted strings can be long)
_VALUE_COL_WIDTH = 30

# 3 example locations shown per row
_MAX_DISPLAY_LOCATIONS = 3


def format_magic_constant_parameters_header(
    min_occurrences: int,
    min_files: int,
    exclude_tests: bool,
    limit: int | None,
    constant_type: str = "all",
) -> str:
    """Format parameter summary header for magic constant detection."""
    return format_parameter_header(
        "MAGIC CONSTANT DETECTION PARAMETERS",
        [
            f"Min Occurrences: >= {min_occurrences} times",
            f"Min Files: >= {min_files} distinct files",
            f"Exclude Tests: {exclude_tests}",
            f"Constant Type: {constant_type}",
        ],
        limit,
    )


class MagicConstantFormatter(BaseCLIMetricFormatter):
    """Formatter for magic constant detection CLI output.

    Color tiers by occurrence frequency:
    - HIGH (10+): red - scattered everywhere, strong naming signal
    - MODERATE (5-9): orange - worth investigating
    - LOW (3-4): yellow-green - may be intentional
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on occurrence count."""
        count = item.get("count", 0)
        if count >= MAGIC_CONSTANT_TIER_HIGH:
            return SeverityTier.HIGH
        if count >= MAGIC_CONSTANT_TIER_MODERATE:
            return SeverityTier.MODERATE
        return SeverityTier.LOW

    def _empty_message(self) -> str:
        return "No magic constants detected at the specified threshold."

    def _table_width(self) -> int:
        return _TABLE_WIDTH

    def _total_label(self, count: int) -> str:
        return f"Total: {count} magic constants"

    def _order_for_display(
        self,
        items: list[dict[str, Any]],
        reverse: bool,
    ) -> list[dict[str, Any]]:
        # Items arrive sorted DESC by count. Default view shows highest first;
        # --reverse flips to ascending (lowest first).
        return list(reversed(items)) if reverse else list(items)

    def _rank_for_position(self, position: int, total: int, reverse: bool) -> int:
        # Rank 1 stays = most significant (highest count), independent of display order.
        return total - position + 1 if reverse else position

    def _build_header(self) -> str:
        """Build table header row."""
        return (
            f"{'Rank':<4} | {'Count':<5} | {'Files':<5} | "
            f"{'Value':<{_VALUE_COL_WIDTH}} | {'Example Locations'}"
        )

    def _format_row(self, item: dict[str, Any], rank: int) -> str:
        """Format a single table row."""
        count = item.get("count", 0)
        files = item.get("files", 0)
        value = self.truncate_value(item.get("value", "?"), _VALUE_COL_WIDTH)
        tier = self.get_tier(item)

        colored_count = self.colorize(f"{count:<5}", tier)
        locations = self._format_locations(item.get("occurrences", []), tier)

        return (
            f"{rank:<4} | {colored_count} | {files:<5} | "
            f"{value:<{_VALUE_COL_WIDTH}} | {locations}"
        )

    def _format_locations(
        self,
        occurrences: list[dict[str, Any]],
        tier: SeverityTier,
    ) -> str:
        """Format example file locations as clickable hyperlinks."""
        if not occurrences:
            return ""

        seen_keys: list[str] = []
        links: list[str] = []
        for occ in occurrences:
            file_path = occ.get("file", "")
            fname = extract_filename(file_path)
            line = occ.get("line", 0)
            key = f"{fname}:{line}"
            if key not in seen_keys:
                seen_keys.append(key)
                links.append(self.make_hyperlink(key, file_path, line, tier))
            if len(seen_keys) >= _MAX_DISPLAY_LOCATIONS:
                break

        result = ", ".join(links)
        remaining = len(occurrences) - len(seen_keys)
        if remaining > 0:
            result += f" +{remaining} more"
        return result

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format magic constant detection summary statistics."""
        total_entities = summary.get("total_entities_analyzed", 0)
        constants_found = summary.get("magic_constants_found", 0)

        distribution = summary.get("distribution", {})
        high_count = distribution.get("high", 0)
        moderate_count = distribution.get("moderate", 0)
        low_count = distribution.get("low", 0)

        lines = [
            "",
            "=" * FORMATTER_WIDTH,
            f"Magic Constant Analysis | {total_entities:,} entities"
            f" | {constants_found:,} magic constants found",
            "-" * FORMATTER_WIDTH,
            "Frequency Distribution:",
            f"  HIGH (10+):     {high_count:>4}  (scattered everywhere -- name it)",
            f"  MODERATE (5+):  {moderate_count:>4}  (worth investigating)",
            f"  LOW (3+):       {low_count:>4}  (may be intentional)",
            "-" * FORMATTER_WIDTH,
            "Tip: Extract repeated literals into named constants"
            " (e.g., DEFAULT_PORT = 8080).",
            "=" * FORMATTER_WIDTH,
        ]
        return "\n".join(lines)
