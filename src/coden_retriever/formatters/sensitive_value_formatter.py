"""Sensitive value detection CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for sensitive value detection results with:
- Pipe-separated table matching hotspots/dead-code style
- Colored output based on confidence level
- Clickable VS Code hyperlinks for file navigation
- JSON output support
- Summary statistics
"""

from typing import Any

from ..constants import (
    FORMATTER_WIDTH,
    SENSITIVE_VALUE_TABLE_DISPLAY_LENGTH,
    SENSITIVE_VALUE_TABLE_WIDTH,
    SENSITIVE_VALUE_TIER_HIGH,
    SENSITIVE_VALUE_TIER_MODERATE,
    SENSITIVE_VALUE_VALUE_COLUMN_WIDTH,
)
from .cli_metrics import (
    BaseCLIMetricFormatter,
    SeverityTier,
    extract_filename,
    format_parameter_header,
)


def format_sensitive_value_parameters_header(
    confidence_threshold: float,
    exclude_tests: bool,
    limit: int | None,
    replace_value: str | None = None,
    whitelist: list[str] | None = None,
) -> str:
    """Format parameter summary header for sensitive value detection."""
    param_lines = [
        f"Confidence Threshold: >= {confidence_threshold * 100:.0f}%",
        f"Exclude Tests: {exclude_tests}",
    ]
    if replace_value is not None:
        param_lines.append(f'Replace Mode: ON (replacement: "{replace_value}")')
    else:
        param_lines.append("Replace Mode: OFF (detection only)")
    if whitelist:
        param_lines.append(f"Whitelist Patterns: {', '.join(whitelist)}")
    return format_parameter_header(
        "SENSITIVE VALUE DETECTION PARAMETERS", param_lines, limit
    )


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

    def _empty_message(self) -> str:
        return "No sensitive values detected at the specified threshold."

    def _table_width(self) -> int:
        return SENSITIVE_VALUE_TABLE_WIDTH

    def _total_label(self, count: int) -> str:
        return f"Total: {count} sensitive values detected"

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
        truncated_value = self.truncate_value(value_preview, SENSITIVE_VALUE_TABLE_DISPLAY_LENGTH)

        file_short = extract_filename(file_path)
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
            "=" * FORMATTER_WIDTH,
            f"Sensitive Value Analysis | {total_entities:,} entities"
            f" | {total_strings:,} strings scanned"
            f" | {values_found:,} sensitive values",
            "-" * FORMATTER_WIDTH,
            "Confidence Distribution:",
            f"  HIGH (80%+):     {high_count:>4}  (very likely a real secret)",
            f"  MODERATE (50%+): {moderate_count:>4}  (worth investigating)",
            f"  LOW (<50%):      {low_count:>4}  (possible false positive)",
            "-" * FORMATTER_WIDTH,
            "Tip: Move secrets to environment variables or a secrets manager.",
            "=" * FORMATTER_WIDTH,
        ]

        return "\n".join(lines)
