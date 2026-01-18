"""Dead code detection CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for dead code detection results with:
- Colored output based on confidence levels
- Clickable VS Code hyperlinks for function navigation
- JSON output support
- Summary statistics
"""

import json
from typing import Any

from ..constants import DEAD_CODE_CONFIDENCE_HIGH, DEAD_CODE_CONFIDENCE_MEDIUM
from .cli_metrics import BaseCLIMetricFormatter, FALSE_POSITIVE_WARNING, SeverityTier


# Table width set to 110 chars to fit standard terminal width with margins
_TABLE_WIDTH = 110
# Function name column fits most identifiers while leaving room for location
_NAME_COL_WIDTH = 35


def _extract_filename(file_path: str) -> str:
    """Extract filename from a path, handling both Unix and Windows separators."""
    return file_path.split("/")[-1].split("\\")[-1]


def format_dead_code_parameters_header(
    confidence_threshold: float,
    exclude_tests: bool,
    limit: int | None,
) -> str:
    """Format parameter summary header for dead code detection."""
    lines = []
    lines.append("=" * 80)
    lines.append("DEAD CODE DETECTION PARAMETERS")
    lines.append("=" * 80)
    lines.append(f"Confidence Threshold: >= {confidence_threshold * 100:.0f}%")
    lines.append(f"Exclude Tests: {exclude_tests}")

    if limit is None:
        lines.append("[!] Result Limit: ALL (may be slow for large repos)")
    else:
        lines.append(f"[!] Result Limit: TOP {limit} -- more results may exist (use -n -1 for all)")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * 80)
    return "\n".join(lines)


class DeadCodeFormatter(BaseCLIMetricFormatter):
    """Formatter for dead code detection CLI output.

    Uses SeverityTier to color dead code by confidence:
    - HIGH (>80%): red - definitely dead
    - MODERATE (50-80%): orange - probably dead
    - LOW (<50%): yellow-green - uncertain
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on confidence score."""
        confidence = item.get("confidence", 0)

        if confidence >= DEAD_CODE_CONFIDENCE_HIGH:
            return SeverityTier.HIGH
        if confidence >= DEAD_CODE_CONFIDENCE_MEDIUM:
            return SeverityTier.MODERATE
        return SeverityTier.LOW

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format dead code results for CLI output with clickable hyperlinks."""
        if output_format == "json":
            return json.dumps(items, indent=2)

        if not items:
            return "No dead code detected at the specified confidence threshold."

        display_items = items if reverse else list(reversed(items))

        lines = []
        header = self._build_header()
        lines.append(header)
        lines.append("-" * _TABLE_WIDTH)

        for i, item in enumerate(display_items, 1):
            rank = i if reverse else len(display_items) - i + 1
            row = self._format_row(item, rank)
            lines.append(row)

        lines.append("-" * _TABLE_WIDTH)
        lines.append(f"Total: {len(items)} potentially dead functions")

        return "\n".join(lines)

    def _build_header(self) -> str:
        """Build table header row."""
        return (
            f"{'Rank':<4} | {'Conf':<6} | {'Type':<8} | "
            f"{'Lines':<5} | {'Function':<{_NAME_COL_WIDTH}} | {'Location'}"
        )

    def _format_row(self, item: dict[str, Any], rank: int) -> str:
        """Format a single result row."""
        confidence = item.get("confidence", 0)
        tier = self.get_tier(item)

        conf_str = f"{confidence * 100:.0f}%"
        colored_conf = self.colorize(f"{conf_str:<6}", tier)

        name = item.get("name", "?")
        file_path = item.get("file", "")
        line = item.get("line", 0)
        entity_type = item.get("type", "function")[:8]
        line_count = item.get("lines", 0)

        # Truncate long names
        display_name = self._truncate_name(name)

        # Create clickable hyperlink for function name
        func_link = self.make_hyperlink(display_name, file_path, line, tier)

        # Extract filename for location column
        file_short = _extract_filename(file_path)
        location = f"{file_short}:{line}"

        return (
            f"{rank:<4} | {colored_conf} | {entity_type:<8} | "
            f"{line_count:<5} | {func_link:<{_NAME_COL_WIDTH}} | {location}"
        )

    def _truncate_name(self, name: str) -> str:
        """Truncate name to fit column width."""
        max_len = _NAME_COL_WIDTH - 3
        if len(name) > max_len:
            return "..." + name[-(max_len - 3):]
        return name

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format dead code detection summary statistics."""
        total_analyzed = summary.get("total_functions_analyzed", 0)
        dead_found = summary.get("dead_code_found", 0)
        dead_pct = (dead_found / total_analyzed * 100) if total_analyzed > 0 else 0

        distribution = summary.get("distribution", {})
        high_conf = distribution.get("high", 0)
        medium_conf = distribution.get("medium", 0)
        low_conf = distribution.get("low", 0)

        lines = [
            "",
            "=" * 80,
            f"Dead Code Analysis | {total_analyzed:,} functions | {dead_found:,} potentially dead ({dead_pct:.1f}%)",
            "-" * 80,
            "Confidence Distribution:",
            f"  HIGH (>80%):     {high_conf:>4}  (likely truly dead)",
            f"  MEDIUM (50-80%): {medium_conf:>4}  (investigate further)",
            f"  LOW (<50%):      {low_conf:>4}  (may be entry points)",
            "-" * 80,
            "Note: Entry points (main, handlers) have lower confidence by design.",
            "      Use --include-private to include private functions.",
            "=" * 80,
        ]

        return "\n".join(lines)
