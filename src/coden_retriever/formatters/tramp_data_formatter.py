"""Tramp data detection CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for tramp data detection results with:
- Pipe-separated table matching hotspots/dead-code style
- Colored output based on occurrence frequency
- Clickable VS Code hyperlinks for function navigation
- JSON output support
- Summary statistics
"""

from typing import Any

from ..constants import (
    FORMATTER_WIDTH,
    TRAMP_DATA_MAX_FUNCTIONS_DISPLAY,
    TRAMP_DATA_TIER_HIGH,
    TRAMP_DATA_TIER_MODERATE,
)
from .cli_metrics import (
    BaseCLIMetricFormatter,
    SeverityTier,
    extract_filename,
    format_parameter_header,
)


# 110 chars matches the hotspots table width for visual consistency across commands
_TABLE_WIDTH = 110

# 35 chars allows most param groups to display without truncation
_GROUP_COL_WIDTH = 35

# 4 params fit within _GROUP_COL_WIDTH (35 chars) for typical short names
_MAX_DISPLAY_PARAMS = 4


def _format_group_label(group: list[str]) -> str:
    """Format a parameter group as a comma-separated display string.

    Truncates long groups with '...+N more' suffix.
    """
    if len(group) <= _MAX_DISPLAY_PARAMS:
        return ", ".join(group)

    shown = ", ".join(group[:_MAX_DISPLAY_PARAMS])
    remaining = len(group) - _MAX_DISPLAY_PARAMS
    return f"{shown} ...+{remaining} more"


def format_tramp_data_parameters_header(
    min_occurrences: int,
    exclude_tests: bool,
    limit: int | None,
    min_group_size: int = 2,
) -> str:
    """Format parameter summary header for tramp data detection."""
    return format_parameter_header(
        "TRAMP DATA DETECTION PARAMETERS",
        [
            f"Min Occurrences: >= {min_occurrences} functions",
            f"Min Group Size: >= {min_group_size} params",
            f"Exclude Tests: {exclude_tests}",
        ],
        limit,
    )


class TrampDataFormatter(BaseCLIMetricFormatter):
    """Formatter for tramp data detection CLI output.

    Uses pipe-separated table matching the hotspots/dead-code table style.
    Function names are shown as clickable hyperlinks on a single row.

    Color tiers by occurrence frequency:
    - HIGH (20+): red - passed everywhere, strong refactoring signal
    - MODERATE (10-19): orange - crosses many boundaries
    - LOW (<10): yellow-green - mild pattern
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on occurrence count."""
        count = item.get("count", 0)

        if count >= TRAMP_DATA_TIER_HIGH:
            return SeverityTier.HIGH
        if count >= TRAMP_DATA_TIER_MODERATE:
            return SeverityTier.MODERATE
        return SeverityTier.LOW

    def _empty_message(self) -> str:
        return "No tramp data groups detected at the specified threshold."

    def _table_width(self) -> int:
        return _TABLE_WIDTH

    def _total_label(self, count: int) -> str:
        return f"Total: {count} tramp data groups"

    def _build_header(self) -> str:
        """Build table header row matching hotspots style."""
        return (
            f"{'Rank':<4} | {'Count':<5} | "
            f"{'Parameter Group':<{_GROUP_COL_WIDTH}} | {'Top Functions'}"
        )

    def _format_row(self, item: dict[str, Any], rank: int) -> str:
        """Format a single table row with inline function links."""
        count = item.get("count", 0)
        group = item.get("group", [])
        functions = item.get("functions", [])
        tier = self.get_tier(item)

        colored_count = self.colorize(f"{count:<5}", tier)
        group_label = _format_group_label(group)
        funcs_str = self._format_functions_inline(functions, tier)

        return (
            f"{rank:<4} | {colored_count} | "
            f"{group_label:<{_GROUP_COL_WIDTH}} | {funcs_str}"
        )

    def _format_functions_inline(
        self,
        functions: list[dict[str, Any]],
        tier: SeverityTier,
    ) -> str:
        """Format function list as compact inline hyperlinks (names only)."""
        if not functions:
            return ""

        display_funcs = functions[:TRAMP_DATA_MAX_FUNCTIONS_DISPLAY]
        parts = []

        for func in display_funcs:
            name = func.get("name", "?")
            file_path = func.get("file", "")
            line = func.get("line", 0)
            file_short = extract_filename(file_path)
            link = self.make_hyperlink(name, file_path, line, tier)
            parts.append(f"{link}({file_short}:{line})")

        result = ", ".join(parts)

        remaining = len(functions) - TRAMP_DATA_MAX_FUNCTIONS_DISPLAY
        if remaining > 0:
            result += f" +{remaining} more"

        return result

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format tramp data detection summary statistics."""
        total_analyzed = summary.get("total_functions_analyzed", 0)
        total_params = summary.get("total_unique_params", 0)
        groups_found = summary.get("tramp_groups_found", 0)

        distribution = summary.get("distribution", {})
        high_count = distribution.get("high", 0)
        moderate_count = distribution.get("moderate", 0)
        low_count = distribution.get("low", 0)

        lines = [
            "",
            "=" * FORMATTER_WIDTH,
            f"Tramp Data Analysis | {total_analyzed:,} functions"
            f" | {total_params:,} unique params"
            f" | {groups_found:,} tramp groups",
            "-" * FORMATTER_WIDTH,
            "Frequency Distribution:",
            f"  HIGH (20+):     {high_count:>4}  (passed everywhere -- refactoring signal)",
            f"  MODERATE (10+): {moderate_count:>4}  (crosses many boundaries)",
            f"  LOW (5+):       {low_count:>4}  (mild tramp data pattern)",
            "-" * FORMATTER_WIDTH,
            "Tip: Consider encapsulating high-frequency param groups"
            " into config/context objects.",
            "=" * FORMATTER_WIDTH,
        ]

        return "\n".join(lines)
