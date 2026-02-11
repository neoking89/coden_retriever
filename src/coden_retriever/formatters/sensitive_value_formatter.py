"""Sensitive value detection CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for sensitive value detection results with:
- Pipe-separated table matching hotspots/dead-code style
- Colored output based on confidence level
- Clickable VS Code hyperlinks for file navigation
- JSON output support
- Summary statistics
"""

import json
from typing import Any

from ..constants import (
    SENSITIVE_VALUE_TABLE_DISPLAY_LENGTH,
    SENSITIVE_VALUE_TABLE_WIDTH,
    SENSITIVE_VALUE_TIER_HIGH,
    SENSITIVE_VALUE_TIER_MODERATE,
    SENSITIVE_VALUE_VALUE_COLUMN_WIDTH,
)
from .cli_metrics import BaseCLIMetricFormatter, FALSE_POSITIVE_WARNING, SeverityTier


def _extract_filename(file_path: str) -> str:
    """Extract filename from a path, handling both Unix and Windows separators."""
    return file_path.split("/")[-1].split("\\")[-1]


def _truncate_value(value: str, max_len: int = SENSITIVE_VALUE_TABLE_DISPLAY_LENGTH) -> str:
    """Truncate a value for preview, masking the middle portion."""
    if len(value) <= max_len:
        return value
    return value[:max_len - 3] + "..."


def format_sensitive_value_parameters_header(
    confidence_threshold: float,
    exclude_tests: bool,
    limit: int | None,
    replace_value: str | None = None,
) -> str:
    """Format parameter summary header for sensitive value detection."""
    lines = []
    lines.append("=" * 80)
    lines.append("SENSITIVE VALUE DETECTION PARAMETERS")
    lines.append("=" * 80)
    lines.append(f"Confidence Threshold: >= {confidence_threshold * 100:.0f}%")
    lines.append(f"Exclude Tests: {exclude_tests}")

    if replace_value is not None:
        lines.append(f"Replace Mode: ON (replacement: \"{replace_value}\")")
    else:
        lines.append("Replace Mode: OFF (detection only)")

    if limit is None:
        lines.append("[!] Result Limit: ALL (may be slow for large repos)")
    else:
        lines.append(f"[!] Result Limit: TOP {limit} -- more results may exist (use -n -1 for all)")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * 80)
    return "\n".join(lines)


class SensitiveValueFormatter(BaseCLIMetricFormatter):
    """Formatter for sensitive value detection CLI output.

    Color tiers by confidence:
    - HIGH (>=80%): red - very likely a real secret
    - MODERATE (>=50%): orange - worth investigating
    - LOW (<50%): yellow-green - possible false positive
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on confidence score."""
        confidence = item.get("confidence", 0)

        if confidence >= SENSITIVE_VALUE_TIER_HIGH:
            return SeverityTier.HIGH
        if confidence >= SENSITIVE_VALUE_TIER_MODERATE:
            return SeverityTier.MODERATE
        return SeverityTier.LOW

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format sensitive value results as pipe-separated table."""
        if output_format == "json":
            return json.dumps(items, indent=2)

        if not items:
            return "No sensitive values detected at the specified threshold."

        display_items = items if reverse else list(reversed(items))

        lines = [self._build_header(), "-" * SENSITIVE_VALUE_TABLE_WIDTH]

        for i, item in enumerate(display_items, 1):
            rank = i if reverse else len(display_items) - i + 1
            lines.append(self._format_row(item, rank))

        lines.append("-" * SENSITIVE_VALUE_TABLE_WIDTH)
        lines.append(f"Total: {len(items)} sensitive values detected")

        return "\n".join(lines)

    def _build_header(self) -> str:
        """Build table header row."""
        return (
            f"{'Rank':<4} | {'Conf':<5} | "
            f"{'Variable':<15} | {'Value':<{SENSITIVE_VALUE_VALUE_COLUMN_WIDTH}} | {'Location'}"
        )

    def _format_row(self, item: dict[str, Any], rank: int) -> str:
        """Format a single table row."""
        confidence = item.get("confidence", 0)
        var_name = item.get("variable_name") or "?"
        value_preview = item.get("value_preview", "?")
        file_path = item.get("file", "")
        line = item.get("line", 0)
        name = item.get("name", "?")

        tier = self.get_tier(item)

        colored_conf = self.colorize(f"{confidence * 100:.0f}%  ", tier)
        truncated_value = _truncate_value(value_preview, SENSITIVE_VALUE_TABLE_DISPLAY_LENGTH)

        file_short = _extract_filename(file_path)
        name_link = self.make_hyperlink(name, file_path, line, tier)
        location = f"{name_link} ({file_short}:{line})"

        return (
            f"{rank:<4} | {colored_conf} | "
            f"{var_name:<15} | {truncated_value:<{SENSITIVE_VALUE_VALUE_COLUMN_WIDTH}} | {location}"
        )

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format sensitive value detection summary statistics."""
        total_entities = summary.get("total_entities_analyzed", 0)
        total_strings = summary.get("total_strings_scanned", 0)
        values_found = summary.get("sensitive_values_found", 0)

        distribution = summary.get("distribution", {})
        high_count = distribution.get("high", 0)
        moderate_count = distribution.get("moderate", 0)
        low_count = distribution.get("low", 0)

        lines = [
            "",
            "=" * 80,
            f"Sensitive Value Analysis | {total_entities:,} entities"
            f" | {total_strings:,} strings scanned"
            f" | {values_found:,} sensitive values",
            "-" * 80,
            "Confidence Distribution:",
            f"  HIGH (80%+):     {high_count:>4}  (very likely a real secret)",
            f"  MODERATE (50%+): {moderate_count:>4}  (worth investigating)",
            f"  LOW (<50%):      {low_count:>4}  (possible false positive)",
            "-" * 80,
            "Tip: Move secrets to environment variables or a secrets manager.",
            "=" * 80,
        ]

        return "\n".join(lines)
