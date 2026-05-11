"""Flag command CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for flag command results with:
- Colored output based on flag type (HOTSPOT > PROPAGATION > CLONE)
- Clickable VS Code hyperlinks for file navigation
- JSON output support
- Summary statistics
"""

import json
from typing import Any

from ..constants import (
    DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD,
    FORMATTER_WIDTH,
    MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    MAGIC_CONSTANT_TIER_HIGH,
    MAGIC_CONSTANT_TIER_MODERATE,
    SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    SENSITIVE_VALUE_TIER_HIGH,
    SENSITIVE_VALUE_TIER_MODERATE,
    TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    TRAMP_DATA_TIER_HIGH,
    TRAMP_DATA_TIER_LOW,
    TRAMP_DATA_TIER_MODERATE,
)
from .cli_metrics import BaseCLIMetricFormatter, FALSE_POSITIVE_WARNING, SeverityTier, extract_filename, format_parameter_header


_FLAG_TABLE_WIDTH = 130
_FLAG_FILE_COL_WIDTH = 40
_FLAG_NAME_COL_WIDTH = 25
_FLAG_METRIC_COL_WIDTH = 28


def _truncate(text: str | None, max_len: int) -> str:
    """Truncate text to max length with ellipsis."""
    if text is None:
        text = "?"
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def format_parameters_header(
    active_flags: list[str],
    risk_threshold: float,
    propagation_threshold: float,
    clone_threshold: float,
    echo_threshold: float,
    limit: int | None,
    dry_run: bool,
    dead_code_threshold: float = DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD,
    min_occurrences: int = TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    sensitive_threshold: float = SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    min_constant_occurrences: int = MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    min_constant_files: int = MAGIC_CONSTANT_DEFAULT_MIN_FILES,
) -> str:
    """Format parameter summary header showing active analysis and thresholds.

    Args:
        active_flags: List of active flags (e.g., ["-H", "-P", "-C", "-E", "-D", "-T", "-S"])
        risk_threshold: Risk threshold for hotspots
        propagation_threshold: Propagation cost threshold (0-1 scale)
        clone_threshold: Clone similarity threshold (0-1 scale)
        echo_threshold: Echo comment similarity threshold (0-1 scale)
        limit: Result limit (None for no limit)
        dry_run: Whether this is a dry-run
        dead_code_threshold: Dead code confidence threshold (0-1 scale)
        min_occurrences: Minimum function occurrences for tramp data
        sensitive_threshold: Sensitive value confidence threshold (0-1 scale)

    Returns:
        Formatted parameter header string
    """
    lines = []
    lines.append("=" * FORMATTER_WIDTH)
    lines.append("ANALYSIS PARAMETERS")
    lines.append("=" * FORMATTER_WIDTH)

    # Active analysis types with their thresholds
    analysis_names = {
        "-H": f"Hotspots (risk >= {risk_threshold})",
        "-P": f"Propagation (cost >= {propagation_threshold * 100:.0f}%)",
        "-C": f"Clones (similarity >= {clone_threshold * 100:.0f}%)",
        "-E": f"Echo Comments (similarity >= {echo_threshold * 100:.0f}%)",
        "-D": f"Dead Code (confidence >= {dead_code_threshold * 100:.0f}%)",
        "-T": f"Tramp Data (occurrences >= {min_occurrences})",
        "-S": f"Sensitive Values (confidence >= {sensitive_threshold * 100:.0f}%)",
        "-K": f"Magic Constants (occurrences >= {min_constant_occurrences}, files >= {min_constant_files})",
    }
    active = [analysis_names[flag] for flag in active_flags if flag in analysis_names]

    if active:
        lines.append(f"Active Flags: {', '.join(active)}")
    else:
        lines.append("Active Flags: None")

    # Result limit behavior
    if limit is not None:
        if dry_run:
            lines.append(f"[!] Preview Limit: TOP {limit} ITEMS ONLY (dry-run mode)")
        else:
            lines.append(f"[!] WARNING: -n {limit} specified but ignored (not in dry-run mode)")
            lines.append("           ALL matching items will be flagged")
    else:
        if dry_run:
            lines.append("Preview Limit: ALL matching items")
        else:
            lines.append("Result Limit: ALL matching items will be flagged")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * FORMATTER_WIDTH)
    return "\n".join(lines)


def format_echo_parameters_header(
    echo_threshold: float,
    exclude_tests: bool,
    limit: int | None,
) -> str:
    """Format parameter summary header for echo comment detection.

    Args:
        echo_threshold: Echo comment similarity threshold (0-1 scale)
        exclude_tests: Whether tests are excluded
        limit: Result limit (None = show all)

    Returns:
        Formatted parameter header string
    """
    return format_parameter_header(
        "ECHO COMMENT DETECTION PARAMETERS",
        [
            f"Similarity Threshold: >= {echo_threshold * 100:.0f}%",
            f"Exclude Tests: {exclude_tests}",
        ],
        limit,
    )


class FlagFormatter(BaseCLIMetricFormatter):
    """Formatter for flag command CLI output.

    Uses SeverityTier to color flags by type:
    - HOTSPOT: HIGH severity - red (high coupling risk)
    - PROPAGATION: ELEVATED severity - orange (architectural concern)
    - CLONE: MODERATE severity - yellow (code duplication)
    - ECHO: MODERATE-ELEVATED severity - orange/yellow (redundant comments)
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on flag type.

        Hotspots are most critical (high coupling risk),
        propagation issues are architectural concerns,
        clones are less urgent but still important.
        """
        flag_type = item.get("type", "unknown")
        if flag_type == "hotspot":
            risk = item.get("risk_score", 0)
            if risk >= 0.8:
                return SeverityTier.CRITICAL
            elif risk >= 0.6:
                return SeverityTier.HIGH
            else:
                return SeverityTier.ELEVATED
        elif flag_type == "propagation":
            cost = item.get("propagation_cost", 0)
            if cost >= 40:
                return SeverityTier.HIGH
            elif cost >= 25:
                return SeverityTier.ELEVATED
            else:
                return SeverityTier.MODERATE
        elif flag_type == "clone":
            similarity = item.get("similarity", 0)
            if similarity >= 0.99:
                return SeverityTier.ELEVATED
            else:
                return SeverityTier.MODERATE
        elif flag_type in ("echo", "echo_remove"):
            similarity = item.get("similarity_score", 0)
            if similarity >= 0.95:
                return SeverityTier.CRITICAL
            elif similarity >= 0.90:
                return SeverityTier.HIGH
            elif similarity >= 0.85:
                return SeverityTier.ELEVATED
            else:
                return SeverityTier.MODERATE
        elif flag_type in ("dead_code", "dead_code_remove"):
            confidence = item.get("confidence", 0)
            if confidence >= 0.80:
                return SeverityTier.HIGH
            elif confidence >= 0.50:
                return SeverityTier.MODERATE
            else:
                return SeverityTier.LOW
        elif flag_type == "tramp_data":
            occurrences = item.get("occurrences", 0)
            if occurrences >= TRAMP_DATA_TIER_HIGH:
                return SeverityTier.HIGH
            elif occurrences >= TRAMP_DATA_TIER_MODERATE:
                return SeverityTier.ELEVATED
            elif occurrences >= TRAMP_DATA_TIER_LOW:
                return SeverityTier.MODERATE
            else:
                return SeverityTier.LOW
        elif flag_type in ("sensitive_value", "sensitive_value_replace"):
            confidence = item.get("confidence", 0)
            if confidence >= SENSITIVE_VALUE_TIER_HIGH:
                return SeverityTier.HIGH
            elif confidence >= SENSITIVE_VALUE_TIER_MODERATE:
                return SeverityTier.MODERATE
            else:
                return SeverityTier.LOW
        elif flag_type == "magic_constant":
            occurrences = item.get("occurrences", 0)
            if occurrences >= MAGIC_CONSTANT_TIER_HIGH:
                return SeverityTier.HIGH
            elif occurrences >= MAGIC_CONSTANT_TIER_MODERATE:
                return SeverityTier.MODERATE
            else:
                return SeverityTier.LOW
        return SeverityTier.LOW

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format flagged items for CLI output.

        Args:
            items: List of flagged item dicts
            output_format: Output format ("tree", "json")
            reverse: If True, show highest severity first

        Returns:
            Formatted string with ANSI colors and OSC 8 hyperlinks
        """
        if output_format == "json":
            return json.dumps(items, indent=2)

        if not items:
            return "No items to flag at the specified thresholds."

        # Sort by severity (type priority: hotspot > propagation > clone > dead_code > echo)
        type_priority = {
            "hotspot": 0, "propagation": 1, "clone": 2,
            "sensitive_value": 3, "sensitive_value_replace": 3,
            "dead_code": 4, "dead_code_remove": 4,
            "tramp_data": 5,
            "magic_constant": 6,
            "echo": 7, "echo_remove": 7,
        }
        sorted_items = sorted(
            items,
            key=lambda x: (
                type_priority.get(x.get("type", "unknown"), 99),
                -x.get("risk_score", 0),
                -x.get("propagation_cost", 0),
                -x.get("similarity", 0),
                -x.get("confidence", 0),
                -x.get("occurrences", 0),
                -x.get("similarity_score", 0),
            ),
        )

        if reverse:
            sorted_items = list(reversed(sorted_items))

        lines = []
        header = f"{'#':<3} | {'Type':<12} | {'Metric':<{_FLAG_METRIC_COL_WIDTH}} | {'Name':<{_FLAG_NAME_COL_WIDTH}} | {'File':<{_FLAG_FILE_COL_WIDTH}}"
        lines.append(header)
        lines.append("-" * _FLAG_TABLE_WIDTH)

        for i, item in enumerate(sorted_items, 1):
            flag_type = item.get("type", "unknown").upper()
            name = item.get("name") or "?"
            file_path = item.get("file") or ""
            line = item.get("line", 0)

            tier = self.get_tier(item)

            # Format metric based on type
            if item.get("type") == "hotspot":
                metric = f"Risk: {item.get('risk_score', 0):.2f}"
            elif item.get("type") == "propagation":
                metric = f"Cost: {item.get('propagation_cost', 0):.1f}%"
            elif item.get("type") == "clone":
                similarity = item.get("similarity", 0)
                semantic_sim = item.get("semantic_sim")
                syntactic_pct = item.get("syntactic_pct")
                # Format: "Score (Sem|Syn)" e.g. "100.0% (100|100)"
                if semantic_sim is not None and syntactic_pct is not None:
                    metric = f"{similarity * 100:.1f}% ({semantic_sim * 100:.0f}|{syntactic_pct * 100:.0f})"
                elif semantic_sim is not None:
                    metric = f"{similarity * 100:.1f}% (S:{semantic_sim * 100:.0f})"
                elif syntactic_pct is not None:
                    metric = f"{similarity * 100:.1f}% (Y:{syntactic_pct * 100:.0f})"
                else:
                    metric = f"Sim: {similarity * 100:.1f}%"
            elif item.get("type") in ("echo", "echo_remove"):
                metric = f"Echo: {item.get('similarity_score', 0) * 100:.1f}%"
            elif item.get("type") in ("dead_code", "dead_code_remove"):
                metric = f"Conf: {item.get('confidence', 0) * 100:.0f}%"
            elif item.get("type") == "tramp_data":
                occurrences = item.get("occurrences", 0)
                group_size = len(item.get("group", []))
                metric = f"Tramp: {occurrences} funcs, {group_size} params"
            elif item.get("type") in ("sensitive_value", "sensitive_value_replace"):
                metric = f"Conf: {item.get('confidence', 0) * 100:.0f}%"
            elif item.get("type") == "magic_constant":
                occ = item.get("occurrences", 0)
                files = item.get("files", 0)
                metric = f"Magic: {occ}x in {files} files"
            else:
                metric = "N/A"

            # Color type and metric
            colored_type = self.colorize(f"{flag_type:<12}", tier)
            colored_metric = self.colorize(f"{metric:<{_FLAG_METRIC_COL_WIDTH}}", tier)

            # Create clickable hyperlink for name
            name_link = self.make_hyperlink(
                _truncate(name, _FLAG_NAME_COL_WIDTH),
                file_path,
                line,
                tier,
            )

            # Short file path
            file_short = extract_filename(file_path)
            file_display = f"{file_short}:{line}"

            lines.append(
                f"{i:<3} | {colored_type} | {colored_metric} | {name_link:<{_FLAG_NAME_COL_WIDTH}} | {file_display:<{_FLAG_FILE_COL_WIDTH}}"
            )

        lines.append("-" * _FLAG_TABLE_WIDTH)
        return "\n".join(lines)

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format flag command summary statistics.

        Args:
            summary: Summary statistics dict from flag_code

        Returns:
            Formatted statistics string for stderr
        """
        flagged_count = summary.get("flagged_count", 0)
        files_modified = summary.get("files_modified", 0)
        dry_run = summary.get("dry_run", False)

        type_summary = summary.get("summary", {})
        hotspots = type_summary.get("hotspots", 0)
        propagation = type_summary.get("propagation", 0)
        clones = type_summary.get("clones", 0)
        dead_code = type_summary.get("dead_code", 0)
        echo_comments = type_summary.get("echo_comments", 0)
        tramp_data = type_summary.get("tramp_data", 0)
        sensitive_values = type_summary.get("sensitive_values", 0)
        magic_constants = type_summary.get("magic_constants", 0)

        mode = "[DRY-RUN] Would flag" if dry_run else "Flagged"

        lines = [
            "",
            "=" * FORMATTER_WIDTH,
            f"Flag Command | {mode} {flagged_count} objects across {files_modified} files",
            "-" * FORMATTER_WIDTH,
            f"  HOTSPOT flags: {hotspots}",
            f"  PROPAGATION flags: {propagation}",
            f"  CLONE flags: {clones}",
            f"  SENSITIVE_VALUE flags: {sensitive_values}",
            f"  DEAD_CODE flags: {dead_code}",
            f"  TRAMP_DATA flags: {tramp_data}",
            f"  MAGIC_CONSTANT flags: {magic_constants}",
            f"  ECHO flags: {echo_comments}",
        ]

        if dry_run:
            lines.append("-" * FORMATTER_WIDTH)
            lines.append("Run without --dry-run to apply changes.")

        lines.append("=" * FORMATTER_WIDTH)
        return "\n".join(lines)

    def format_clear_stats(self, result: dict[str, Any]) -> str:
        """Format flag clear command summary.

        Args:
            result: Result dict from flag_clear

        Returns:
            Formatted statistics string
        """
        files_cleaned = result.get("files_cleaned", 0)
        comments_removed = result.get("comments_removed", 0)
        dry_run = result.get("dry_run", False)

        mode = "[DRY-RUN] Would remove" if dry_run else "Removed"

        lines = [
            "",
            "=" * FORMATTER_WIDTH,
            f"Flag Clear | {mode} {comments_removed} [CODEN] comments from {files_cleaned} files",
        ]

        if dry_run:
            lines.append("-" * FORMATTER_WIDTH)
            lines.append("Run without --dry-run to apply changes.")

        lines.append("=" * FORMATTER_WIDTH)
        return "\n".join(lines)
