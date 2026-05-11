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
from ..constants import FORMATTER_WIDTH, PC_APPROXIMATE_STATUS, PC_THRESHOLD_WARNING

_MODULE_TABLE_WIDTH = 70


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
    lines.append("=" * FORMATTER_WIDTH)
    lines.append("PROPAGATION COST ANALYSIS")
    lines.append("=" * FORMATTER_WIDTH)
    lines.append("")
    lines.append("What this measures:")
    lines.append("  Propagation Cost (PC) = % of codebase reachable from any function")
    lines.append("  High PC means changes ripple widely - harder to modify safely")
    lines.append("")
    lines.append("How it's calculated:")
    lines.append("  1. Build call graph from all function calls")
    lines.append("  2. Count reachable pairs (A can reach B through calls)")
    lines.append("  3. PC = reachable pairs / total possible pairs")
    lines.append("")
    lines.append("Research thresholds (MacCormack et al. 2006):")
    lines.append("  < 10%  PASS     Low coupling")
    lines.append("  10-25% WARNING  Moderate coupling, monitor trends")
    lines.append("  25-43% WARNING  High coupling, consider refactoring")
    lines.append("  > 43%  CRITICAL Architectural decay, action needed")
    lines.append("")
    lines.append("-" * FORMATTER_WIDTH)
    lines.append(f"High-Coupling Threshold: >= {propagation_threshold * 100:.0f}% (modules above are flagged)")
    lines.append(f"Exclude Tests: {exclude_tests}")

    if limit is None:
        lines.append("[!] Module Breakdown Limit: ALL")
    else:
        lines.append(f"[!] Module Breakdown Limit: TOP {limit} (use -n -1 for all)")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * FORMATTER_WIDTH)
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
        elif status == PC_APPROXIMATE_STATUS:
            return SeverityTier.NOTABLE
        return SeverityTier.OPTIMAL

    def _format_approximation_banner(self, result: dict[str, Any]) -> list[str]:
        """Loud banner shown when reachable_pairs was approximated by direct edges."""
        if not result.get("approximation_used"):
            return []
        stats = result.get("graph_stats", {})
        nodes = stats.get("nodes", "?")
        limit = stats.get("closure_node_limit", "?")
        nodes_str = f"{nodes:,}" if isinstance(nodes, int) else str(nodes)
        limit_str = f"{limit:,}" if isinstance(limit, int) else str(limit)
        bar = "=" * FORMATTER_WIDTH
        return [
            bar,
            self.colorize("[APPROXIMATION] DIRECT-EDGES LOWER BOUND", SeverityTier.NOTABLE),
            bar,
            f"Graph too large for transitive closure ({nodes_str} nodes > {limit_str}).",
            "Reachable-pairs count is approximated by direct edges only.",
            "PASS / WARNING / CRITICAL verdict is suppressed - the approximation",
            "systematically under-counts. Use the per-module breakdown below for",
            "actionable signal that does not depend on total graph size.",
            bar,
            "",
        ]

    def _format_max_module_coupling(self, result: dict[str, Any]) -> list[str]:
        """Headline figure that doesn't shrink with total ``n``."""
        summary = result.get("max_module_coupling")
        if not summary:
            return []
        value = summary.get("value", 0)
        module = summary.get("module", "?")
        funcs = summary.get("functions", 0)
        return [
            f"  Max module coupling: {value*100:.2f}% (in `{module}`, {funcs} functions)",
            "",
        ]

    def _format_overall_metric(
        self, result: dict[str, Any], tier: SeverityTier
    ) -> list[str]:
        """Format the overall propagation cost metric section."""
        pc = result.get("propagation_cost", 0)
        status = result.get("status", "N/A")
        interp = result.get("interpretation", "")

        if status == PC_APPROXIMATE_STATUS:
            status_line = (
                f"  Status: {self.colorize('APPROXIMATE (verdict suppressed)', tier)}"
            )
            pc_label = "Propagation Cost (lower bound)"
        else:
            status_line = f"  Status: {self.colorize(status, tier)}"
            pc_label = "Propagation Cost"

        return [
            "Overall Metric:",
            f"  {pc_label}: {self.colorize(f'{pc*100:.2f}%', tier)}",
            status_line,
            f"  {interp}",
            "",
        ]

    def _format_graph_stats(self, result: dict[str, Any]) -> list[str]:
        """Format the graph statistics section."""
        stats = result.get("graph_stats", {})
        nodes = stats.get("nodes", "N/A")
        edges = stats.get("edges", "N/A")
        reachable = stats.get("reachable_pairs", "N/A")
        possible = stats.get("possible_pairs", "N/A")

        nodes_str = f"{nodes:,}" if isinstance(nodes, int) else str(nodes)
        edges_str = f"{edges:,}" if isinstance(edges, int) else str(edges)
        reachable_str = f"{reachable:,}" if isinstance(reachable, int) else str(reachable)
        possible_str = f"{possible:,}" if isinstance(possible, int) else str(possible)

        return [
            "Graph Statistics:",
            f"  Total functions: {nodes_str}",
            f"  Total call edges: {edges_str}",
            f"  Reachable pairs: {reachable_str} / {possible_str}",
            "",
        ]

    def _format_module_breakdown(self, result: dict[str, Any]) -> list[str]:
        """Format the module breakdown section."""
        breakdown = result.get("module_breakdown", [])
        if not breakdown:
            return []

        threshold = result.get("coupling_threshold", PC_THRESHOLD_WARNING)
        above_count = sum(1 for m in breakdown if m.get("above_threshold", False))
        header_suffix = f" ({above_count} above {threshold*100:.0f}% threshold)" if above_count else ""

        lines = [
            f"Module Breakdown{header_suffix}:",
            "-" * _MODULE_TABLE_WIDTH,
            f"  {'Module':<25} | {'Functions':>10} | {'Coupling':>12} | Flag",
            "-" * _MODULE_TABLE_WIDTH,
        ]

        for mod in breakdown:
            module_name = mod.get("module", "?")
            functions = mod.get("functions", 0)
            coupling = mod.get("internal_coupling", 0)
            above = mod.get("above_threshold", False)
            flag = "[!]" if above else ""
            coupling_str = (
                self.colorize(f"{coupling*100:>11.1f}%", SeverityTier.MODERATE)
                if above else f"{coupling*100:>11.1f}%"
            )
            lines.append(f"  {module_name:<25} | {functions:>10} | {coupling_str} | {flag}")

        lines.append("")
        return lines

    def _format_critical_paths(
        self, result: dict[str, Any], tier: SeverityTier
    ) -> list[str]:
        """Format the critical paths section with hyperlinks."""
        paths = result.get("critical_paths", [])
        if not paths:
            return []

        lines = ["Critical Paths (Highest Impact):"]
        for i, path in enumerate(paths, 1):
            name = path.get("start", "?")
            file_path = path.get("file", "")
            line = path.get("line", 1)
            downstream = path.get("downstream_count", 0)
            link = self.make_hyperlink(name, file_path, line, tier)
            lines.append(f"  {i}. {link} -> {downstream} downstream functions")

        lines.append("")
        return lines

    def _format_recommendations(self, result: dict[str, Any]) -> list[str]:
        """Format the recommendations section."""
        recs = result.get("recommendations", [])
        if not recs:
            return []
        return ["Recommendations:"] + [f"  {rec}" for rec in recs]

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format propagation cost result for CLI output."""
        if output_format == "json":
            return json.dumps(items, indent=2)

        if not items:
            return "No propagation cost data available."

        result = items[0]
        tier = self.get_tier(result)

        lines: list[str] = []
        lines.extend(self._format_approximation_banner(result))
        lines.extend(["Propagation Cost Analysis", "=" * FORMATTER_WIDTH, ""])
        lines.extend(self._format_overall_metric(result, tier))
        lines.extend(self._format_max_module_coupling(result))
        lines.extend(self._format_graph_stats(result))
        lines.extend(self._format_module_breakdown(result))
        lines.extend(self._format_critical_paths(result, tier))
        lines.extend(self._format_recommendations(result))
        lines.append("=" * FORMATTER_WIDTH)

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
            "=" * FORMATTER_WIDTH,
            f"Propagation Cost | {pc*100:.2f}% | Status: {status} | {nodes:,} functions",
            "=" * FORMATTER_WIDTH,
        ]
        return "\n".join(lines)
