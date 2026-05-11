"""Hotspot output formatting for CLI display."""
import json

from ..constants import FORMATTER_WIDTH
from .cli_metrics import FALSE_POSITIVE_WARNING, format_result_limit_line
from .terminal_style import TerminalStyle, get_terminal_style

# Table separator width matching the column layout
_TABLE_SEPARATOR_WIDTH = 110
# Truncation limit for entity names in table display
_ENTITY_NAME_MAX_LEN = 35
# Characters to keep from end of truncated name (after "...")
_ENTITY_NAME_TAIL_LEN = 32
# Tier inversion base for risk score coloring (11 - tier_num)
_TIER_INVERSION_BASE = 11


def format_hotspots_parameters_header(
    risk_threshold: float,
    exclude_tests: bool,
    limit: int | None,
) -> str:
    """Format parameter summary header for hotspots analysis."""
    lines = []
    lines.append("=" * FORMATTER_WIDTH)
    lines.append("HOTSPOTS ANALYSIS PARAMETERS")
    lines.append("-" * FORMATTER_WIDTH)
    lines.append("Analysis: Coupling Hotspots (Fan-in/Fan-out + Cyclomatic Complexity)")
    lines.append("=" * FORMATTER_WIDTH)
    lines.append(f"Risk Threshold: >= {risk_threshold}")
    lines.append(f"Exclude Tests: {exclude_tests}")
    lines.append(format_result_limit_line(limit))
    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * FORMATTER_WIDTH)
    return "\n".join(lines)


def format_hotspots_output(
    hotspots: list[dict],
    output_format: str = "tree",
    reverse: bool = False,
) -> str:
    """Format hotspots result for CLI output."""
    if output_format == "json":
        return json.dumps(hotspots, indent=2)

    if not hotspots:
        return "No refactoring hotspots found."

    style = get_terminal_style()

    max_risk = max(h.get("risk_score", 0) for h in hotspots) if hotspots else 1.0
    display_hotspots = hotspots if reverse else list(reversed(hotspots))

    lines = []

    header = f"{'Rank':<4} | {'Risk':<7} | {'Coupling':<13} | {'CC':<4} | {'Category':<12} | {'Lines':<5} | {'Entity'}"
    lines.append(header)
    lines.append("-" * _TABLE_SEPARATOR_WIDTH)

    for i, h in enumerate(display_hotspots, 1):
        rank = i if reverse else len(display_hotspots) - i + 1
        lines.append(_format_hotspot_row(h, rank, max_risk, style))

    lines.append("-" * _TABLE_SEPARATOR_WIDTH)
    return "\n".join(lines)


def _format_hotspot_row(
    h: dict,
    rank: int,
    max_risk: float,
    style: TerminalStyle,
) -> str:
    """Format a single hotspot row for the table."""
    category = h.get("category", "Unknown")
    risk_score = h.get("risk_score", 0)
    name = h.get("name", "unknown")
    file_path = h.get("file", "")
    line = h.get("line", 0)
    fan_in = h.get("fan_in", 0)
    fan_out = h.get("fan_out", 0)
    complexity = h.get("complexity", 1)
    line_count = h.get("lines", 0)

    if len(name) > _ENTITY_NAME_MAX_LEN:
        name = "..." + name[-_ENTITY_NAME_TAIL_LEN:]

    tier = style.get_score_tier(risk_score, max_risk)
    tier_num = int(tier.split('_')[1])
    inverted_tier = f"tier_{_TIER_INVERSION_BASE - tier_num}"
    risk_str = f"{risk_score:>6.1f}"
    colored_risk = style.render_to_string(style.colorize(risk_str, inverted_tier))

    colored_entity = style.format_stats_entity(
        name, file_path, line, max_risk - risk_score + 1, max_risk
    )

    coupling_str = f"{fan_in}in/{fan_out}out"

    return (
        f"{rank:<4} | {colored_risk} | {coupling_str:<13} | {complexity:<4} | "
        f"{category:<12} | {line_count:<5} | {colored_entity}"
    )


def format_hotspots_stats(summary: dict) -> str:
    """Format hotspots summary statistics."""
    total = summary.get('total_functions_analyzed', 0)
    above_threshold = summary.get('functions_above_threshold', 0)

    category_dist = summary.get("category_distribution", {})
    danger = category_dist.get("Danger Zone", 0)
    traffic = category_dist.get("Traffic Jam", 0)
    local = category_dist.get("Local Mess", 0)
    low = category_dist.get("Low Risk", 0)

    lines = [
        "",
        "=" * FORMATTER_WIDTH,
        f"Hotspots Analysis | {total:,} functions analyzed | {above_threshold:,} above threshold",
        "-" * FORMATTER_WIDTH,
        f"Risk: avg {summary.get('average_risk_score', 0):.1f} / max {summary.get('max_risk_score', 0):.1f}",
        f"Coupling: avg {summary.get('average_coupling_score', 0):.1f} / max {summary.get('highest_coupling_score', 0)}",
        f"Complexity: avg {summary.get('average_complexity', 1):.1f} / max {summary.get('max_complexity', 1)}",
        "-" * FORMATTER_WIDTH,
        f"Categories: Danger Zone: {danger} | Traffic Jam: {traffic} | Local Mess: {local} | Low Risk: {low}",
        "-" * FORMATTER_WIDTH,
        "Legend: Danger Zone = high coupling + high complexity (hardest to maintain)",
        "        Traffic Jam = high coupling, low complexity (architectural bottleneck)",
        "        Local Mess = low coupling, high complexity (hard to test/understand)",
    ]

    if summary.get("token_budget_exceeded"):
        lines.append("-" * FORMATTER_WIDTH)
        lines.append("Note: Results truncated due to token budget")

    lines.append("=" * FORMATTER_WIDTH)
    return "\n".join(lines)
