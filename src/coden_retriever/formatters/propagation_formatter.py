"""Propagation cost CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for propagation cost results with:
- Colored output based on status severity (CRITICAL > WARNING > PASS)
- Clickable VS Code hyperlinks for critical path functions
- JSON output support
- Summary statistics
"""

import json
from typing import Any

from .cli_metrics import BaseCLIMetricFormatter, FALSE_POSITIVE_WARNING, SeverityTier


_PROP_TABLE_WIDTH = 80


def format_propagation_parameters_header(
    propagation_threshold: float,
    exclude_tests: bool,
    limit: int | None,
) -> str:
    """Format parameter summary header for propagation cost analysis.

    Args:
        propagation_threshold: Propagation cost threshold (0-1 scale)
        exclude_tests: Whether tests are excluded
        limit: Result limit for module breakdown (None = show all)

    Returns:
        Formatted parameter header string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PROPAGATION COST ANALYSIS PARAMETERS")
    lines.append("=" * 80)
    lines.append(f"Cost Threshold (Module Breakdown): >= {propagation_threshold * 100:.0f}%")
    lines.append(f"Exclude Tests: {exclude_tests}")

    if limit is None:
        lines.append("[!] Module Breakdown Limit: ALL (may be slow for large repos)")
    else:
        lines.append(f"[!] Module Breakdown Limit: TOP {limit} -- more results may exist (use -n -1 for all)")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * 80)
    return "\n".join(lines)


class PropagationFormatter(BaseCLIMetricFormatter):
    """Formatter for propagation cost CLI output.

    Uses SeverityTier to color output by status:
    - CRITICAL (>43%): CRITICAL severity - bright red
    - WARNING (25-43%): MODERATE severity - orange
    - PASS (<25%): OPTIMAL severity - green
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on propagation cost status."""
        status = item.get("status", "N/A")
        if status == "CRITICAL":
            return SeverityTier.CRITICAL
        elif status == "WARNING":
            return SeverityTier.MODERATE
        return SeverityTier.OPTIMAL

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format propagation cost result for CLI output.

        Args:
            items: List containing a single result dict
            output_format: Output format ("tree", "json")
            reverse: Not used for propagation cost (single result)

        Returns:
            Formatted string with ANSI colors and OSC 8 hyperlinks
        """
        if output_format == "json":
            return json.dumps(items, indent=2)

        if not items:
            return "No propagation cost data available."

        result = items[0]  # Single result dict
        tier = self.get_tier(result)

        lines = []
        lines.append("Propagation Cost Analysis")
        lines.append("=" * _PROP_TABLE_WIDTH)
        lines.append("")

        # Overall metric with color
        pc = result.get("propagation_cost", 0)
        status = result.get("status", "N/A")
        interp = result.get("interpretation", "")

        colored_status = self.colorize(status, tier)
        colored_pc = self.colorize(f"{pc*100:.2f}%", tier)

        lines.append("Overall Metric:")
        lines.append(f"  Propagation Cost: {colored_pc}")
        lines.append(f"  Status: {colored_status}")
        lines.append(f"  {interp}")
        lines.append("")

        # Graph stats
        stats = result.get("graph_stats", {})
        nodes = stats.get("nodes", "N/A")
        edges = stats.get("edges", "N/A")
        reachable = stats.get("reachable_pairs", "N/A")
        possible = stats.get("possible_pairs", "N/A")

        lines.append("Graph Statistics:")
        lines.append(f"  Total functions: {nodes:,}" if isinstance(nodes, int) else f"  Total functions: {nodes}")
        lines.append(f"  Total call edges: {edges:,}" if isinstance(edges, int) else f"  Total call edges: {edges}")
        reachable_str = f"{reachable:,}" if isinstance(reachable, int) else str(reachable)
        possible_str = f"{possible:,}" if isinstance(possible, int) else str(possible)
        lines.append(f"  Reachable pairs: {reachable_str} / {possible_str}")
        lines.append("")

        # Module breakdown
        breakdown = result.get("module_breakdown", [])
        if breakdown:
            lines.append("Module Breakdown (Top Contributors):")
            lines.append("-" * 60)
            lines.append(f"  {'Module':<20} | {'Functions':>10} | {'Coupling':>10}")
            lines.append("-" * 60)
            for mod in breakdown[:5]:
                module_name = mod.get("module", "?")
                functions = mod.get("functions", 0)
                coupling = mod.get("internal_coupling", 0)
                lines.append(f"  {module_name:<20} | {functions:>10} | {coupling*100:>9.1f}%")
            lines.append("")

        # Critical paths with hyperlinks
        paths = result.get("critical_paths", [])
        if paths:
            lines.append("Critical Paths (Highest Impact):")
            for i, path in enumerate(paths, 1):
                name = path.get("start", "?")
                file_path = path.get("file", "")
                line = path.get("line", 1)
                downstream = path.get("downstream_count", 0)
                link = self.make_hyperlink(name, file_path, line, tier)
                lines.append(f"  {i}. {link} -> {downstream} downstream functions")
            lines.append("")

        # Recommendations
        recs = result.get("recommendations", [])
        if recs:
            lines.append("Recommendations:")
            for rec in recs:
                lines.append(f"  {rec}")

        lines.append("=" * _PROP_TABLE_WIDTH)
        return "\n".join(lines)

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format propagation cost summary statistics.

        Args:
            summary: Result dict from propagation_cost

        Returns:
            Formatted statistics string for stderr
        """
        pc = summary.get("propagation_cost", 0)
        status = summary.get("status", "N/A")
        nodes = summary.get("graph_stats", {}).get("nodes", 0)

        lines = [
            "",
            "=" * 80,
            f"Propagation Cost | {pc*100:.2f}% | Status: {status} | {nodes:,} functions",
            "=" * 80,
        ]
        return "\n".join(lines)
