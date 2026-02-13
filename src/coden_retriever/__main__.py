"""
Main entry point for CodenRetriever.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
import io
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from .utils.optional_deps import MissingDependencyError, require_feature

if TYPE_CHECKING:
    from .formatters.flag_formatter import FlagFormatter


def _get_asyncio():
    """Lazy import asyncio to avoid 18ms startup cost."""
    import asyncio
    return asyncio


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for a CLI threshold argument."""
    name: str  # CLI argument name without -- (e.g., "clone-threshold")
    default: float
    analysis_flag: str  # Short flag (e.g., "-C")
    analysis_name: str  # Human-readable name (e.g., "Code Clones")
    short_help: str  # Brief help for search parser
    detailed_help: str  # Full help for flag parser (includes range, examples)
    example_value: float  # Value to use in examples
    validate_0_1: bool = True  # If True, validate range 0.0-1.0


# Threshold definitions - update these to change help text everywhere
THRESHOLD_CONFIGS = {
    "risk": ThresholdConfig(
        name="risk-threshold",
        default=50.0,
        analysis_flag="-H",
        analysis_name="Hotspots",
        short_help="Hotspot min risk score (raw score, typically 50-200+)",
        detailed_help="Hotspots (-H): min risk score for flagging. Raw score = coupling * log(complexity). Default: 50",
        example_value=50.0,
        validate_0_1=False,
    ),
    "propagation": ThresholdConfig(
        name="propagation-threshold",
        default=0.25,
        analysis_flag="-P",
        analysis_name="Propagation Cost",
        short_help="Propagation cost threshold (0.0-1.0)",
        detailed_help="Propagation (-P): min internal coupling %% for flagging modules. Range: 0-1. Default: 0.25 (25%%)",
        example_value=0.25,
    ),
    "clone": ThresholdConfig(
        name="clone-threshold",
        default=0.95,
        analysis_flag="-C",
        analysis_name="Code Clones",
        short_help="Clone similarity threshold (0.0-1.0)",
        detailed_help="Clones (-C): min semantic similarity for flagging. Range: 0-1. Default: 0.95 (very similar)",
        example_value=0.90,
    ),
    "echo": ThresholdConfig(
        name="echo-threshold",
        default=0.85,
        analysis_flag="-E",
        analysis_name="Echo Comments",
        short_help="Echo comment similarity threshold (0.0-1.0)",
        detailed_help="Echo Comments (-E): semantic similarity threshold. Range: 0-1. Default: 0.85. Stricter (0.95) = near-identical only, Looser (0.75) = more detections",
        example_value=0.85,
    ),
    "dead_code": ThresholdConfig(
        name="dead-code-threshold",
        default=0.5,
        analysis_flag="-D",
        analysis_name="Dead Code",
        short_help="Dead code confidence threshold (0.0-1.0)",
        detailed_help="Dead Code (-D): min confidence score for flagging. Range: 0-1. Default: 0.5 (medium confidence)",
        example_value=0.7,
    ),
    "sensitive_value": ThresholdConfig(
        name="sensitive-threshold",
        default=0.35,
        analysis_flag="-S",
        analysis_name="Sensitive Values",
        short_help="Sensitive value confidence threshold (0.0-1.0)",
        detailed_help="Sensitive Values (-S): min confidence for flagging. Range: 0-1. Default: 0.35. Lower = more recall, higher = more precision",
        example_value=0.50,
    ),
}


def _validate_threshold(value: str) -> float:
    """Validate threshold is between 0.0 and 1.0."""
    fval = float(value)
    if not (0.0 <= fval <= 1.0):
        raise argparse.ArgumentTypeError(f"threshold must be between 0.0 and 1.0, got {fval}")
    return fval


def _validate_positive_float(value: str) -> float:
    """Validate threshold is a positive number (no upper bound)."""
    fval = float(value)
    if fval < 0:
        raise argparse.ArgumentTypeError(f"threshold must be non-negative, got {fval}")
    return fval


def normalize_limit(limit: int | None) -> int | None:
    """Convert negative values to None (unlimited) for limit arguments.

    This allows users to explicitly request all results via -n -1.
    Any negative value is treated as unlimited for robustness.
    """
    if limit is not None and limit < 0:
        return None
    return limit


def add_threshold_argument(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    config: ThresholdConfig,
    use_detailed_help: bool = False,
) -> None:
    """Add a threshold argument to a parser using centralized config.

    Args:
        parser: The argparse parser or argument group to add to
        config: ThresholdConfig with argument settings
        use_detailed_help: If True, use detailed_help; otherwise use short_help
    """
    help_text = config.detailed_help if use_detailed_help else config.short_help
    validator = _validate_threshold if config.validate_0_1 else _validate_positive_float
    parser.add_argument(
        f"--{config.name}",
        type=validator,
        default=config.default,
        metavar="FLOAT",
        help=help_text,
    )


# =============================================================================

from .cache import CacheManager
from .cli_metrics_contract import apply_defensive_limit, print_metric_output
from .config import get_central_cache_root, get_project_cache_dir
from .config import OutputFormat
from .config_loader import (
    AppConfig,
    get_config,
    load_config,
    save_config,
    get_config_file,
    reset_config,
    set_config_value,
    SETTING_LOCATIONS,
    _config_to_dict,
)
from .daemon.client import (
    DaemonClient,
    get_daemon_status,
    stop_daemon,
    try_daemon_search,
    try_daemon_hotspots,
    try_daemon_clones,
    try_daemon_dead_code,
    try_daemon_propagation_cost,
    try_daemon_sensitive_values,
    try_daemon_tramp_data,
    try_daemon_flag,
    try_daemon_flag_clear,
)
from .daemon.protocol import (
    WINDOWS_CREATE_NEW_PROCESS_GROUP,
    WINDOWS_DETACHED_PROCESS,
    CloneDetectionParams,
    DeadCodeParams,
    FlagClearParams,
    FlagParams,
    GraphAnalysisParams,
    PropagationCostParams,
    SearchParams,
    SensitiveValueParams,
    TrampDataParams,
)
from .daemon.server import get_log_file, is_daemon_running, run_daemon
from .formatters import CloneFormatter, DeadCodeFormatter, PropagationFormatter, SensitiveValueFormatter, TrampDataFormatter
from .formatters.cli_metrics import FALSE_POSITIVE_WARNING
from .formatters.dead_code_formatter import format_dead_code_parameters_header
from .formatters.sensitive_value_formatter import format_sensitive_value_parameters_header
from .formatters.tramp_data_formatter import format_tramp_data_parameters_header
from .formatters.flag_formatter import FlagFormatter
from .formatters.terminal_style import get_terminal_style
from .pipeline import SearchConfig, SearchPipeline

logger = logging.getLogger(__name__)


def parse_duration(duration_str: str) -> int:
    """Parse a duration string (e.g., '30m', '1h', '90s') to seconds."""
    if not duration_str:
        return 0

    duration_str = duration_str.strip().lower()

    if duration_str.endswith('s'):
        return int(duration_str[:-1])
    elif duration_str.endswith('m'):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 3600
    else:
        return int(duration_str)


def print_search_output(
    formatted_output: str,
    tree_output: str | None,
    stats_output: str | None,
    reverse: bool,
) -> None:
    """Print search output in correct order based on reverse flag.

    Args:
        formatted_output: The main formatted search results
        tree_output: Optional directory tree output
        stats_output: Optional ranking statistics (always printed to stderr)
        reverse: If True, show results first then tree then stats.
                 If False, show stats first then tree then results.
    """
    if reverse:
        # Reversed mode: results first, then tree, then stats (highest score last)
        print(formatted_output)

        if tree_output:
            print("\n" + "=" * 60 + "\n")
            print(tree_output)

        if stats_output:
            print(stats_output, file=sys.stderr)
    else:
        # Normal mode: stats first, then tree, then results
        if stats_output:
            print(stats_output, file=sys.stderr)

        if tree_output:
            print(tree_output)
            print("\n" + "=" * 60 + "\n")

        print(formatted_output)


def format_semantic_search_header(query: str) -> str:
    """Format a header indicating semantic search mode is active.

    Args:
        query: The search query being used

    Returns:
        Formatted semantic search header string
    """
    lines = [
        "=" * 60,
        "SEMANTIC SEARCH MODE",
        "-" * 60,
        "Using Model2Vec embeddings for semantic similarity matching.",
        f'Query: "{query}"',
        "=" * 60,
    ]
    return "\n".join(lines)


def format_hotspots_parameters_header(
    risk_threshold: float,
    exclude_tests: bool,
    limit: int | None,
) -> str:
    """Format parameter summary header for hotspots analysis.

    Args:
        risk_threshold: Risk score threshold
        exclude_tests: Whether tests are excluded
        limit: Result limit (None = show all)

    Returns:
        Formatted parameter header string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("HOTSPOTS ANALYSIS PARAMETERS")
    lines.append("=" * 80)
    lines.append(f"Risk Threshold: >= {risk_threshold}")
    lines.append(f"Exclude Tests: {exclude_tests}")

    if limit is None:
        lines.append("[!] Result Limit: ALL (may be slow for large repos)")
    else:
        lines.append(f"[!] Result Limit: TOP {limit} -- more results may exist (use -n -1 for all)")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * 80)
    return "\n".join(lines)


def format_hotspots_output(
    hotspots: list[dict],
    output_format: str = "tree",
    reverse: bool = False,
) -> str:
    """Format hotspots result for CLI output.

    Args:
        hotspots: List of hotspot dicts from daemon
        output_format: Output format (tree, json, etc.)
        reverse: If True, show highest risk first (opposite of other modes)

    Returns:
        Formatted string for display
    """
    if output_format == "json":
        return json.dumps(hotspots, indent=2)

    if not hotspots:
        return "No refactoring hotspots found."

    # Get terminal style for coloring
    style = get_terminal_style()

    # Find max risk for color scaling
    max_risk = max(h.get("risk_score", 0) for h in hotspots) if hotspots else 1.0
    display_hotspots = hotspots if reverse else list(reversed(hotspots))

    lines = []

    # Table header
    header = f"{'Rank':<4} | {'Risk':<7} | {'Coupling':<13} | {'CC':<4} | {'Category':<12} | {'Lines':<5} | {'Entity'}"
    lines.append(header)
    lines.append("-" * 110)

    for i, h in enumerate(display_hotspots, 1):
        # Calculate display rank (accounts for reverse)
        rank = i if reverse else len(display_hotspots) - i + 1

        category = h.get("category", "Unknown")
        risk_score = h.get("risk_score", 0)
        name = h.get("name", "unknown")
        file_path = h.get("file", "")
        line = h.get("line", 0)
        fan_in = h.get("fan_in", 0)
        fan_out = h.get("fan_out", 0)
        complexity = h.get("complexity", 1)
        line_count = h.get("lines", 0)

        # Truncate long entity names
        if len(name) > 35:
            name = "..." + name[-32:]

        # Color the risk score based on its value relative to max
        tier = style.get_score_tier(risk_score, max_risk)
        tier_num = int(tier.split('_')[1])
        inverted_tier = f"tier_{11 - tier_num}"
        risk_str = f"{risk_score:>6.1f}"
        colored_risk = style.render_to_string(style.colorize(risk_str, inverted_tier))

        # Color the entity name
        colored_entity = style.format_stats_entity(
            name, file_path, line, max_risk - risk_score + 1, max_risk
        )

        # Coupling display
        coupling_str = f"{fan_in}in/{fan_out}out"

        lines.append(
            f"{rank:<4} | {colored_risk} | {coupling_str:<13} | {complexity:<4} | {category:<12} | {line_count:<5} | {colored_entity}"
        )

    lines.append("-" * 110)
    return "\n".join(lines)


def format_hotspots_stats(summary: dict) -> str:
    """Format hotspots summary statistics.

    Args:
        summary: Summary dict from daemon

    Returns:
        Formatted stats string for stderr
    """
    total = summary.get('total_functions_analyzed', 0)
    above_threshold = summary.get('functions_above_threshold', 0)

    # Category distribution
    category_dist = summary.get("category_distribution", {})
    danger = category_dist.get("Danger Zone", 0)
    traffic = category_dist.get("Traffic Jam", 0)
    local = category_dist.get("Local Mess", 0)
    low = category_dist.get("Low Risk", 0)

    lines = [
        "",
        "=" * 80,
        f"Hotspots Analysis | {total:,} functions analyzed | {above_threshold:,} above threshold",
        "-" * 80,
        f"Risk: avg {summary.get('average_risk_score', 0):.1f} / max {summary.get('max_risk_score', 0):.1f}",
        f"Coupling: avg {summary.get('average_coupling_score', 0):.1f} / max {summary.get('highest_coupling_score', 0)}",
        f"Complexity: avg {summary.get('average_complexity', 1):.1f} / max {summary.get('max_complexity', 1)}",
        "-" * 80,
        f"Categories: Danger Zone: {danger} | Traffic Jam: {traffic} | Local Mess: {local} | Low Risk: {low}",
        "-" * 80,
        "Legend: Danger Zone = high coupling + high complexity (hardest to maintain)",
        "        Traffic Jam = high coupling, low complexity (architectural bottleneck)",
        "        Local Mess = low coupling, high complexity (hard to test/understand)",
    ]

    if summary.get("token_budget_exceeded"):
        lines.append("-" * 80)
        lines.append("Note: Results truncated due to token budget")

    lines.append("=" * 80)
    return "\n".join(lines)


def print_hotspots_output(
    formatted_output: str,
    stats_output: str | None,
    reverse: bool,
) -> None:
    """Print hotspots output in correct order based on reverse flag.

    Args:
        formatted_output: The main formatted hotspots results
        stats_output: Optional ranking statistics (always printed to stderr)
        reverse: If True, show results first then stats.
                 If False, show stats first then results.
    """
    if reverse:
        print(formatted_output)
        if stats_output:
            print(stats_output, file=sys.stderr)
    else:
        if stats_output:
            print(stats_output, file=sys.stderr)
        print(formatted_output)


def _get_clone_mode(args: argparse.Namespace) -> Literal["combined", "semantic", "syntactic"]:
    """Determine clone detection mode from CLI flags."""
    clone_semantic = getattr(args, "clone_semantic", False)
    clone_syntactic = getattr(args, "clone_syntactic", False)

    if clone_semantic and clone_syntactic:
        return "combined"  # Both flags = combined (explicit)
    if clone_semantic:
        return "semantic"
    if clone_syntactic:
        return "syntactic"
    return "combined"  # Default


def handle_clones_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle clone detection command using CloneFormatter for output."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = CloneFormatter()

    # Determine clone mode
    mode = _get_clone_mode(args)

    # Clone detection requires semantic feature unless syntactic-only mode
    if mode in ("semantic", "combined"):
        try:
            require_feature("semantic")
        except MissingDependencyError as e:
            print(str(e), file=sys.stderr)
            return 1
    line_threshold = getattr(args, "line_threshold", 0.70)
    func_threshold = getattr(args, "func_threshold", 0.50)
    semantic_weight = getattr(args, "semantic_weight", 0.65)
    syntactic_weight = getattr(args, "syntactic_weight", 0.35)

    # Print parameter header before results
    from .formatters.clone_formatter import format_clone_parameters_header
    header = format_clone_parameters_header(
        mode=mode,
        similarity_threshold=args.clone_threshold,
        line_threshold=line_threshold,
        func_threshold=func_threshold,
        min_lines=args.min_lines,
        limit=args.limit,
        exclude_tests=True,
    )
    print(header)
    print()  # Blank line

    params = CloneDetectionParams(
        source_dir=str(root_path),
        mode=mode,
        similarity_threshold=args.clone_threshold,
        line_threshold=line_threshold,
        func_threshold=func_threshold,
        limit=args.limit,
        exclude_tests=True,
        min_lines=args.min_lines,
        token_limit=args.tokens,  # None = no limit for CLI
        semantic_weight=semantic_weight,
        syntactic_weight=syntactic_weight,
    )

    # Use daemon_timeout from config (adjustable via: coden config set daemon_timeout <seconds>)
    # Default 60s for heavy analysis; increase if clone detection times out on large repos
    daemon_result = try_daemon_clones(
        params, host=config.daemon.host, port=config.daemon.port,
        timeout=max(config.daemon.daemon_timeout, 60.0)
    )

    if daemon_result is not None:
        if "error" in daemon_result:
            logger.error(f"Clone detection error: {daemon_result['error']}")
            return 1

        all_clones = daemon_result.get("clones", [])
        summary = daemon_result.get("summary", {})

        # Apply defensive limit (ensures contract compliance)
        clones = apply_defensive_limit(all_clones, args.limit)

        formatted_output = formatter.format_items(clones, args.format, args.reverse)
        stats_output = formatter.format_stats(summary) if args.stats else None
        print_metric_output(formatted_output, stats_output, args.reverse)
        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\n[Daemon mode] Clone detection time: {elapsed_ms:.1f}ms, Pairs: {len(clones)}", file=sys.stderr)
        return 0

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from .mcp.clone_detection import detect_clones as mcp_detect_clones

        result = _get_asyncio().run(mcp_detect_clones(
            root_directory=str(root_path),
            mode=mode,
            similarity_threshold=args.clone_threshold,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=args.limit,
            exclude_tests=True,
            min_lines=args.min_lines,
            token_limit=args.tokens,  # None = no limit for CLI
            semantic_weight=semantic_weight,
            syntactic_weight=syntactic_weight,
        ))

        if "error" in result:
            logger.error(f"Clone detection error: {result['error']}")
            return 1

        all_clones = result.get("clones", [])
        summary = result.get("summary", {})

        # Apply defensive limit (ensures contract compliance)
        clones = apply_defensive_limit(all_clones, args.limit)

        formatted_output = formatter.format_items(clones, args.format, args.reverse)
        stats_output = formatter.format_stats(summary) if args.stats else None
        print_metric_output(formatted_output, stats_output, args.reverse)
        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\n[Direct mode] Clone detection time: {elapsed_ms:.1f}ms, Pairs: {len(clones)}", file=sys.stderr)
        return 0

    except Exception as e:
        logger.error(f"Clone detection failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_echo_comments_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle echo comment detection command (read-only analysis or file modification with --remove-comments)."""
    # Echo comment detection requires semantic feature (Model2Vec)
    try:
        require_feature("semantic")
    except MissingDependencyError as e:
        print(str(e), file=sys.stderr)
        return 1

    args.limit = normalize_limit(args.limit)
    start_time = time.time()

    try:
        from .cache import CacheManager

        cache = CacheManager(root_path)
        indices = cache.load_or_rebuild()

        # Print parameter header before results (only for read-only analysis, not for --remove-comments)
        if not args.remove_comments:
            from .formatters.flag_formatter import format_echo_parameters_header
            header = format_echo_parameters_header(
                echo_threshold=args.echo_threshold,
                exclude_tests=not args.include_tests,
                limit=args.limit,
            )
            print(header)
            print()  # Blank line

        # If --remove-comments is specified, use flag_code for file modification
        if args.remove_comments:
            from .mcp.flag_insertion import flag_code

            result = flag_code(
                entities=indices.entities,
                graph=indices.graph,
                pagerank=indices.pagerank,
                source_dir=str(root_path),
                echo_comments=True,
                echo_threshold=args.echo_threshold,
                dry_run=args.dry_run,
                backup=args.backup,
                verbose=args.verbose,
                exclude_tests=not args.include_tests,
                remove_comments=True,
            )

            if "error" in result:
                logger.error(f"Echo comment removal failed: {result['error']}")
                return 1

            # Print the output (formatted by flag_code)
            from .formatters.flag_formatter import FlagFormatter
            formatter = FlagFormatter()
            formatted_output = formatter.format_items(result.get("items", []), args.format, args.reverse)

            stats_output = None
            if args.stats:
                stats_lines = [
                    "",
                    "=" * 80,
                    f"Echo Comment Removal | {result.get('flagged_count', 0)} items",
                    "-" * 80,
                    f"Files modified: {result.get('files_modified', 0)}",
                    f"Comments removed: {result.get('flagged_count', 0)}",
                    "=" * 80,
                ]
                stats_output = "\n".join(stats_lines)

            print_metric_output(formatted_output, stats_output, args.reverse)

            elapsed_ms = (time.time() - start_time) * 1000
            if args.verbose:
                print(f"\nEcho comment removal time: {elapsed_ms:.1f}ms", file=sys.stderr)
            return 0

        # Read-only analysis (original behavior)
        from .mcp.echo_comments import compute_echo_comments
        from .formatters.flag_formatter import FlagFormatter
        formatter = FlagFormatter()

        result = compute_echo_comments(
            entities=indices.entities,
            echo_threshold=args.echo_threshold,
            token_limit=args.tokens,  # None = no limit for CLI
            include_tests=args.include_tests,
            include_private=False,
        )

        if "error" in result:
            logger.error(f"Echo comment detection error: {result['error']}")
            return 1

        all_echo_comments = result.get("echo_comments", [])
        summary = result.get("summary", {})

        # Apply defensive limit (ensures contract compliance)
        echo_comments = apply_defensive_limit(all_echo_comments, args.limit)

        # Convert echo_comments to flag format for display
        items = []
        for echo in echo_comments:
            items.append({
                "type": "echo",
                "file": echo.get("file_path"),  # Note: formatter expects "file" not "file_path"
                "line": echo.get("line"),
                "name": echo.get("context_identifier"),
                "similarity_score": echo.get("similarity_score"),
                "comment_text": echo.get("comment_text"),
                "severity": echo.get("severity"),
            })

        formatted_output = formatter.format_items(items, args.format, args.reverse)

        # Create custom stats output for echo analysis
        if args.stats:
            total_comments = summary.get("total_comments_found", 0)
            # Use original count before limiting for ratio calculation
            total_echo_count = len(all_echo_comments)
            echo_ratio = total_echo_count / total_comments if total_comments > 0 else 0
            distribution = summary.get("distribution", {})

            stats_lines = [
                "",
                "=" * 80,
                f"Echo Comment Analysis | {len(echo_comments):,} shown ({total_echo_count:,} total)",
                "-" * 80,
                f"Total comments analyzed: {total_comments:,}",
                f"Echo ratio: {echo_ratio * 100:.1f}%",
                f"Files affected: {summary.get('files_affected', 0)}",
                f"Avg similarity: {summary.get('avg_similarity', 0) * 100:.1f}%",
                "-" * 80,
                "Distribution:",
                f"  CRITICAL (>95%): {distribution.get('critical', 0)}",
                f"  HIGH (90-95%): {distribution.get('high', 0)}",
                f"  ELEVATED (85-90%): {distribution.get('elevated', 0)}",
                f"  MODERATE (<85%): {distribution.get('moderate', 0)}",
                "=" * 80,
            ]
            stats_output = "\n".join(stats_lines)
        else:
            stats_output = None

        # Print with contract-compliant ordering
        print_metric_output(formatted_output, stats_output, args.reverse)

        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\nEcho comment detection time: {elapsed_ms:.1f}ms, Found: {len(echo_comments)}", file=sys.stderr)
        return 0

    except Exception as e:
        logger.error(f"Echo comment detection failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _filter_propagation_by_threshold(result: dict, threshold: float) -> dict:
    """Mark modules above threshold in propagation result.

    Args:
        result: Propagation cost result dict
        threshold: Minimum internal_coupling (0-1) to flag as high-coupling

    Returns:
        Result with module_breakdown annotated (not filtered)
    """
    if "module_breakdown" not in result:
        return result

    # Don't filter - annotate modules above threshold instead
    filtered = result.copy()
    filtered["module_breakdown"] = [
        {**m, "above_threshold": m.get("internal_coupling", 0) >= threshold}
        for m in result["module_breakdown"]
    ]
    filtered["coupling_threshold"] = threshold
    return filtered


def handle_propagation_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle propagation cost command using PropagationFormatter for output."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = PropagationFormatter()

    # Print parameter header before results
    from .formatters.propagation_formatter import format_propagation_parameters_header
    exclude_tests = not getattr(args, 'include_tests', False)
    header = format_propagation_parameters_header(
        propagation_threshold=args.propagation_threshold,
        exclude_tests=exclude_tests,
        limit=args.limit,
    )
    print(header)
    print()  # Blank line

    params = PropagationCostParams(
        source_dir=str(root_path),
        include_breakdown=args.breakdown,
        show_critical_paths=args.critical_paths,
        exclude_tests=exclude_tests,
        token_limit=args.tokens,  # None = no limit for CLI
    )

    daemon_result = try_daemon_propagation_cost(params, host=config.daemon.host, port=config.daemon.port)

    if daemon_result is not None:
        if "error" in daemon_result:
            logger.error(f"Propagation cost error: {daemon_result['error']}")
            return 1

        # Filter module_breakdown by threshold
        filtered_result = _filter_propagation_by_threshold(daemon_result, args.propagation_threshold)

        formatted_output = formatter.format_items([filtered_result], args.format, args.reverse)
        stats_output = formatter.format_stats(filtered_result) if args.stats else None
        print_metric_output(formatted_output, stats_output, args.reverse)
        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            pc = daemon_result.get('propagation_cost', 0)
            print(f"[Daemon mode] Propagation cost: {pc*100:.2f}% ({elapsed_ms:.1f}ms)", file=sys.stderr)
        return 0

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from .mcp.propagation_cost import propagation_cost as mcp_propagation_cost

        result = _get_asyncio().run(mcp_propagation_cost(
            root_directory=str(root_path),
            include_breakdown=args.breakdown,
            show_critical_paths=args.critical_paths,
            exclude_tests=exclude_tests,
            token_limit=args.tokens,  # None = no limit for CLI
        ))

        if "error" in result:
            logger.error(f"Propagation cost error: {result['error']}")
            return 1

        filtered_result = _filter_propagation_by_threshold(result, args.propagation_threshold)

        formatted_output = formatter.format_items([filtered_result], args.format, args.reverse)
        stats_output = formatter.format_stats(filtered_result) if args.stats else None
        print_metric_output(formatted_output, stats_output, args.reverse)
        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            pc = result.get('propagation_cost', 0)
            print(f"[Direct mode] Propagation cost: {pc*100:.2f}% ({elapsed_ms:.1f}ms)", file=sys.stderr)
        return 0

    except Exception as e:
        logger.error(f"Propagation cost analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _print_propagation_output(formatted_output: str, stats_output: str | None, reverse: bool) -> None:
    """Print propagation output with proper ordering based on reverse flag."""
    if reverse:
        print(formatted_output)
        if stats_output:
            print(stats_output, file=sys.stderr)
    else:
        if stats_output:
            print(stats_output, file=sys.stderr)
        print(formatted_output)


def _print_flag_output(formatted_output: str, stats_output: str | None, reverse: bool) -> None:
    """Print flag output with proper ordering based on reverse flag."""
    if reverse:
        print(formatted_output)
        if stats_output:
            print(stats_output, file=sys.stderr)
    else:
        if stats_output:
            print(stats_output, file=sys.stderr)
        print(formatted_output)


def _validate_flag_args(args: argparse.Namespace, root_path: Path) -> int:
    """Validate flag command arguments. Returns 0 on success, 1 on error."""
    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}", file=sys.stderr)
        return 1
    if not root_path.is_dir():
        print(f"Error: Path is not a directory: {root_path}", file=sys.stderr)
        return 1

    # Create temporary params to validate flags
    from .daemon.protocol import FlagParams
    temp_params = FlagParams(
        source_dir=str(root_path),
        hotspots=args.hotspots,
        propagation=args.propagation,
        clones=args.clones,
        echo_comments=args.echo_comments,
        dead_code=args.dead_code,
        tramp_data=args.tramp_data,
        sensitive_values=args.sensitive_values,
    )
    if error := temp_params.validate():
        print(f"Error: {error}", file=sys.stderr)
        return 1

    clone_mode = _get_clone_mode(args)
    needs_semantic = args.echo_comments or (args.clones and clone_mode in ("semantic", "combined"))
    if needs_semantic:
        try:
            require_feature("semantic")
        except MissingDependencyError as e:
            print(str(e), file=sys.stderr)
            return 1

    return 0


# Centralized flag mapping enables consistent validation and formatting
# Prevents string literal duplication across codebase
_FLAG_ATTR_TO_SHORT: list[tuple[str, str]] = [
    ("hotspots", "-H"),
    ("propagation", "-P"),
    ("clones", "-C"),
    ("echo_comments", "-E"),
    ("dead_code", "-D"),
    ("tramp_data", "-T"),
    ("sensitive_values", "-S"),
]

# Syntactic clone detection thresholds
# Line threshold: Jaccard similarity for line-level matching (higher = stricter)
# Based on empirical testing to balance precision (avoid false positives) vs recall (catch real clones)
DEFAULT_SYNTACTIC_LINE_THRESHOLD = 0.70

# Function threshold: Percentage of lines that must match (lower = more lenient)
# Allows for some structural variation while catching meaningful clones
DEFAULT_SYNTACTIC_FUNC_THRESHOLD = 0.50


def _build_flag_active_flags(args: argparse.Namespace) -> list[str]:
    """Build list of active short-flag strings from args."""
    return [short for attr, short in _FLAG_ATTR_TO_SHORT if getattr(args, attr, False)]


def _process_flag_result(
    result: dict,
    args: argparse.Namespace,
    formatter: "FlagFormatter",
    start_time: float,
    mode_label: str,
) -> int:
    """Process flag_code result: format, print, and return exit code."""
    if "error" in result:
        logger.error(f"Flag command error: {result['error']}")
        return 1

    items = result.get("items", [])

    if args.dry_run and args.limit is not None:
        items = items[: args.limit]

    formatted_output = formatter.format_items(items, args.format, args.reverse)
    stats_output = formatter.format_stats(result) if args.stats else None
    _print_flag_output(formatted_output, stats_output, args.reverse)

    if args.verbose:
        elapsed_ms = (time.time() - start_time) * 1000
        count = result.get("flagged_count", 0)
        files = result.get("files_modified", 0)
        run_mode = "preview" if args.dry_run else "applied"
        print(
            f"\n[{mode_label}] Flagged {count} objects in {files} files ({run_mode}) in {elapsed_ms:.1f}ms",
            file=sys.stderr,
        )
    return 0


def _build_flag_params(args: argparse.Namespace, root_path: Path) -> FlagParams:
    """Build FlagParams from CLI args."""
    return FlagParams(
        source_dir=str(root_path),
        hotspots=args.hotspots,
        propagation=args.propagation,
        clones=args.clones,
        echo_comments=args.echo_comments,
        dead_code=args.dead_code,
        tramp_data=args.tramp_data,
        risk_threshold=args.risk_threshold,
        propagation_threshold=args.propagation_threshold,
        clone_threshold=args.clone_threshold,
        echo_threshold=args.echo_threshold,
        dead_code_threshold=args.dead_code_threshold,
        clone_mode=_get_clone_mode(args),
        line_threshold=getattr(args, "line_threshold", DEFAULT_SYNTACTIC_LINE_THRESHOLD),
        func_threshold=getattr(args, "func_threshold", DEFAULT_SYNTACTIC_FUNC_THRESHOLD),
        dry_run=args.dry_run,
        backup=args.backup,
        verbose=args.verbose,
        exclude_tests=not args.include_tests,
        remove_comments=args.remove_comments,
        remove_dead_code=args.remove_dead_code,
        output_format=args.format,
        limit=args.limit,
        min_occurrences=args.min_occurrences,
        min_group_size=args.min_group_size,
        sensitive_values=args.sensitive_values,
        sensitive_threshold=args.sensitive_threshold,
        replace_value=getattr(args, "replace", None),
    )


def handle_flag_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle flag command to insert [CODEN] comments."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = FlagFormatter()

    validation_code = _validate_flag_args(args, root_path)
    if validation_code != 0:
        return validation_code

    from .formatters.flag_formatter import format_parameters_header

    active_flags = _build_flag_active_flags(args)
    header = format_parameters_header(
        active_flags=active_flags,
        risk_threshold=args.risk_threshold,
        propagation_threshold=args.propagation_threshold,
        clone_threshold=args.clone_threshold,
        echo_threshold=args.echo_threshold,
        limit=args.limit,
        dry_run=args.dry_run,
        dead_code_threshold=args.dead_code_threshold,
        min_occurrences=args.min_occurrences,
        sensitive_threshold=args.sensitive_threshold,
    )
    print(header)
    print()

    params = _build_flag_params(args, root_path)

    daemon_result = try_daemon_flag(params, host=config.daemon.host, port=config.daemon.port)
    if daemon_result is not None:
        return _process_flag_result(daemon_result, args, formatter, start_time, "Daemon mode")

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from .mcp.flag_insertion import flag_code
        from .cache import CacheManager

        cache = CacheManager(root_path)
        indices = cache.load_or_rebuild()

        result = flag_code(
            entities=indices.entities,
            graph=indices.graph,
            pagerank=indices.pagerank,
            params=params,
        )
        return _process_flag_result(result, args, formatter, start_time, "Direct mode")

    except Exception as e:
        logger.error(f"Flag command failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_flag_clear_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle flag clear command to remove [CODEN] comments."""
    start_time = time.time()
    formatter = FlagFormatter()

    # Validate root path
    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}", file=sys.stderr)
        return 1
    if not root_path.is_dir():
        print(f"Error: Path is not a directory: {root_path}", file=sys.stderr)
        return 1

    params = FlagClearParams(
        source_dir=str(root_path),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    daemon_result = try_daemon_flag_clear(params, host=config.daemon.host, port=config.daemon.port)

    if daemon_result is not None:
        if "error" in daemon_result:
            logger.error(f"Flag clear error: {daemon_result['error']}")
            return 1

        stats_output = formatter.format_clear_stats(daemon_result)
        print(stats_output)
        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\n[Daemon mode] Clear completed in {elapsed_ms:.1f}ms", file=sys.stderr)
        return 0

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from .mcp.flag_insertion import flag_clear

        result = flag_clear(
            source_dir=str(root_path),
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if "error" in result:
            logger.error(f"Flag clear error: {result['error']}")
            return 1

        stats_output = formatter.format_clear_stats(result)
        print(stats_output)
        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\n[Direct mode] Clear completed in {elapsed_ms:.1f}ms", file=sys.stderr)
        return 0

    except Exception as e:
        logger.error(f"Flag clear failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _daemon_start(host: str, port: int, max_projects: int, idle_timeout: str | None, verbose: bool, no_watch: bool = False) -> int:
    """Start daemon in background."""
    running, pid = is_daemon_running()
    if running:
        print(f"Daemon is already running (PID: {pid})")
        return 0

    # On Windows, use pythonw.exe to avoid console window
    if sys.platform == "win32":
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")
    else:
        python_exe = sys.executable

    cmd = [
        python_exe, "-m", "coden_retriever",
        "daemon", "run",
        "--daemon-host", host,
        "--daemon-port", str(port),
    ]

    if max_projects:
        cmd.extend(["--max-projects", str(max_projects)])
    if idle_timeout:
        cmd.extend(["--idle-timeout", str(idle_timeout)])
    if verbose:
        cmd.append("--verbose")
    if no_watch:
        cmd.append("--no-watch")

    # Start daemon process (platform-specific)
    if sys.platform == "win32":
        subprocess.Popen(
            cmd,
            creationflags=WINDOWS_DETACHED_PROCESS | WINDOWS_CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Wait and verify it started
    for _ in range(20):  # Wait up to 2 seconds
        time.sleep(0.1)
        status = get_daemon_status(host, port)
        if status:
            _, pid = is_daemon_running()
            print(f"Daemon started (PID: {pid})")
            print(f"  Address: {host}:{port}")
            print(f"  Log: {get_log_file()}")
            return 0

    print("Daemon failed to start. Check log:", file=sys.stderr)
    print(f"  {get_log_file()}", file=sys.stderr)
    return 1


def _daemon_stop(host: str, port: int) -> int:
    """Stop the daemon."""
    running, pid = is_daemon_running()
    if not running:
        print("Daemon is not running")
        return 0

    if stop_daemon(host, port):
        print(f"Daemon stopped (was PID: {pid})")
        print(f"  Address: {host}:{port}")
        return 0
    else:
        print(f"Failed to stop daemon (PID: {pid})", file=sys.stderr)
        return 1


def _daemon_status(host: str, port: int) -> int:
    """Show daemon status."""
    status = get_daemon_status(host, port)
    if status:
        print("Daemon is running")
        print(json.dumps(status, indent=2))
        return 0

    running, pid = is_daemon_running()
    if running:
        print(f"Daemon process exists (PID: {pid}) but not responding")
        return 1
    else:
        print("Daemon is not running")
        return 1


def _daemon_restart(host: str, port: int, max_projects: int, idle_timeout: str | None, verbose: bool, no_watch: bool = False) -> int:
    """Restart the daemon."""
    stop_daemon(host, port)
    time.sleep(0.5)
    return _daemon_start(host, port, max_projects, idle_timeout, verbose, no_watch)


def _daemon_run(host: str, port: int, max_projects: int, idle_timeout: str | None, verbose: bool, no_watch: bool = False) -> int:
    """Run daemon in foreground."""
    config = get_config()
    max_projects = max_projects or config.daemon.max_projects
    timeout_seconds = parse_duration(idle_timeout) if idle_timeout else None

    return run_daemon(
        host=host,
        port=port,
        max_projects=max_projects,
        idle_timeout=timeout_seconds,
        verbose=verbose,
        foreground=True,
        enable_watch=not no_watch,
    )


def _daemon_clear_cache(host: str, port: int, clear_path: str | None, clear_all: bool) -> int:
    """Clear daemon cache."""
    client = DaemonClient(host=host, port=port, timeout=5.0)
    try:
        result = client.invalidate(source_dir=clear_path, all=clear_all)
        print(f"Cache cleared: {result.get('invalidated', 'none')}")
        return 0
    except Exception as e:
        print(f"Failed to clear cache: {e}", file=sys.stderr)
        return 1


def handle_daemon_command(args: argparse.Namespace) -> int:
    """Handle daemon subcommands by dispatching to specific handlers."""
    config = get_config()
    host = getattr(args, 'daemon_host', config.daemon.host)
    port = getattr(args, 'daemon_port', config.daemon.port)
    verbose = getattr(args, 'verbose', False)
    no_watch = getattr(args, 'no_watch', False)

    action = args.daemon_action

    if action == "start":
        return _daemon_start(
            host, port,
            getattr(args, 'max_projects', config.daemon.max_projects),
            getattr(args, 'idle_timeout', None),
            verbose,
            no_watch
        )
    elif action == "stop":
        return _daemon_stop(host, port)
    elif action == "status":
        return _daemon_status(host, port)
    elif action == "restart":
        return _daemon_restart(
            host, port,
            getattr(args, 'max_projects', config.daemon.max_projects),
            getattr(args, 'idle_timeout', None),
            verbose,
            no_watch
        )
    elif action == "run":
        return _daemon_run(
            host, port,
            getattr(args, 'max_projects', config.daemon.max_projects),
            getattr(args, 'idle_timeout', None),
            verbose,
            no_watch
        )
    elif action == "clear-cache":
        return _daemon_clear_cache(
            host, port,
            getattr(args, 'clear_path', None),
            getattr(args, 'clear_all', False)
        )

    return 0


def handle_config_command(args: list[str]) -> int:
    """Handle config subcommands: show, path, reset, set."""
    if not args or args[0] == "show":
        config = load_config()
        print(json.dumps(_config_to_dict(config), indent=2))
        return 0

    elif args[0] == "path":
        print(get_config_file())
        return 0

    elif args[0] == "reset":
        if reset_config():
            print("Configuration reset to defaults")
            return 0
        else:
            print("Failed to reset configuration", file=sys.stderr)
            return 1

    elif args[0] == "set" and len(args) >= 3:
        # config set <key> <value>
        # Supports both "section.key" format (e.g., agent.debug) and flat keys (e.g., debug)
        key_path = args[1]
        value = args[2]

        config = load_config()
        parts = key_path.split(".")

        # Determine the actual key to look up
        if len(parts) == 2:
            section, key = parts
            # Validate section matches expected location
            if key in SETTING_LOCATIONS:
                expected_section = SETTING_LOCATIONS[key][0]
                if section != expected_section:
                    print(f"Key '{key}' belongs to section '{expected_section}', not '{section}'", file=sys.stderr)
                    return 1
        elif len(parts) == 1:
            key = parts[0]
        else:
            print(f"Invalid key format: {key_path}. Use key or section.key (e.g., debug or agent.debug)", file=sys.stderr)
            return 1

        success, error = set_config_value(config, key, value)
        if not success:
            print(error, file=sys.stderr)
            return 1

        save_config(config)
        print(f"Set {key} = {value}")
        return 0

    else:
        print("Usage: coden config [show|path|reset|set <key> <value>]")
        print("\nCommands:")
        print("  show             Show current configuration")
        print("  path             Show config file path")
        print("  reset            Reset configuration to defaults")
        print("  set <key> <val>  Set a configuration value")
        print("\nKeys:")
        print("  model.default, model.base_url")
        print("  agent.max_steps, agent.max_retries, agent.debug")
        print("  daemon.host, daemon.port, daemon.daemon_timeout, daemon.max_projects")
        print("  search.default_tokens, search.default_limit, search.semantic_model_path")
        return 1


def handle_cache_command(args: list[str]) -> int:
    """Handle cache subcommands: list, clear, status, path."""
    if not args or args[0] == "list":
        # List all cached projects
        caches = CacheManager.list_all_caches()
        if not caches:
            print("No cached projects found.")
            print(f"Cache directory: {get_central_cache_root()}")
            return 0

        print(f"Cached projects ({len(caches)}):")
        print(f"Cache directory: {get_central_cache_root()}\n")

        total_size = 0
        for cache_info in caches:
            total_size += cache_info["size_mb"]
            source = cache_info["source_dir"]
            # Truncate long paths
            if len(source) > 60:
                source = "..." + source[-57:]
            print(f"  {source}")
            print(f"    Entities: {cache_info['entity_count']:,} | Files: {cache_info['file_count']:,} | Size: {cache_info['size_mb']:.1f} MB")
            if cache_info.get("updated_at"):
                print(f"    Updated: {cache_info['updated_at']}")
            print()

        print(f"Total cache size: {total_size:.1f} MB")
        return 0

    elif args[0] == "clear":
        # Check for --all flag
        clear_all = "--all" in args or "-a" in args

        if clear_all:
            # Clear all caches
            count, errors = CacheManager.clear_all_caches()
            if count > 0:
                print(f"Cleared {count} project cache(s)")
            else:
                print("No caches to clear")
            for error in errors:
                print(f"  Warning: {error}", file=sys.stderr)
            return 0 if not errors else 1

        # Clear cache for specific path or current directory
        # Check if a path was provided (argument that's not a flag)
        path_arg = None
        for arg in args[1:]:
            if not arg.startswith("-"):
                path_arg = arg
                break

        target_path = Path(path_arg).resolve() if path_arg else Path.cwd()

        if not target_path.exists():
            print(f"Path does not exist: {target_path}", file=sys.stderr)
            return 1

        if not target_path.is_dir():
            print(f"Path is not a directory: {target_path}", file=sys.stderr)
            return 1

        cache_dir = get_project_cache_dir(target_path)
        if not cache_dir.exists():
            print(f"No cache found for: {target_path}")
            return 0

        if CacheManager.clear_cache_by_source_dir(target_path):
            print(f"Cache cleared for: {target_path}")
            return 0
        else:
            print(f"Failed to clear cache for: {target_path}", file=sys.stderr)
            return 1

    elif args[0] == "status":
        # Show cache status for specific path or current directory
        path_arg = args[1] if len(args) > 1 else None
        target_path = Path(path_arg).resolve() if path_arg else Path.cwd()

        if not target_path.exists():
            print(f"Path does not exist: {target_path}", file=sys.stderr)
            return 1

        if not target_path.is_dir():
            print(f"Path is not a directory: {target_path}", file=sys.stderr)
            return 1

        cache = CacheManager(target_path)
        status = cache.get_cache_status()
        print(json.dumps(status, indent=2))
        return 0

    elif args[0] == "path":
        # Show cache path for specific path or current directory
        path_arg = args[1] if len(args) > 1 else None
        target_path = Path(path_arg).resolve() if path_arg else Path.cwd()

        cache_dir = get_project_cache_dir(target_path)
        print(f"Project: {target_path}")
        print(f"Cache:   {cache_dir}")
        print(f"Exists:  {cache_dir.exists()}")
        return 0

    else:
        print("Usage: coden cache [list|clear|status|path]")
        print("\nCommands:")
        print("  list              List all cached projects")
        print("  clear             Clear cache for current directory")
        print("  clear <path>      Clear cache for specific project")
        print("  clear --all       Clear ALL cached projects")
        print("  status [path]     Show cache status for project")
        print("  path [path]       Show cache directory path for project")
        print(f"\nCache location: {get_central_cache_root()}")
        return 1


def handle_reset_command() -> int:
    """Handle reset command: clear all caches, stop daemon, reset config."""
    exit_code = 0
    config = get_config()

    # 1. Clear all caches
    print("Clearing all caches...")
    count, errors = CacheManager.clear_all_caches()
    if count > 0:
        print(f"  Cleared {count} project cache(s)")
    else:
        print("  No caches to clear")
    for error in errors:
        print(f"  Warning: {error}", file=sys.stderr)
        exit_code = 1

    # 2. Stop daemon
    print("Stopping daemon...")
    running, pid = is_daemon_running()
    if not running:
        print("  Daemon is not running")
    else:
        if stop_daemon(config.daemon.host, config.daemon.port):
            print(f"  Daemon stopped (was PID: {pid})")
        else:
            print(f"  Failed to stop daemon (PID: {pid})", file=sys.stderr)
            exit_code = 1

    # 3. Reset configuration
    print("Resetting configuration...")
    if reset_config():
        print("  Configuration reset to defaults")
    else:
        print("  Failed to reset configuration", file=sys.stderr)
        exit_code = 1

    if exit_code == 0:
        print("\nReset complete.")
    else:
        print("\nReset completed with warnings.", file=sys.stderr)

    return exit_code


class DefaultValueHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Argparse help formatter that appends default values to each argument help."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help
        if not help_text:
            base_help = super()._get_help_string(action)
            return base_help if base_help is not None else ""

        if (
            "%(default)" not in help_text
            and action.default is not argparse.SUPPRESS
        ):
            default_value = action.default
            default_str = '""' if default_value == "" else str(default_value)
            help_text = f"{help_text} (default: {default_str})"

        return help_text


def _create_common_daemon_parser() -> argparse.ArgumentParser:
    """Create parent parser with common daemon arguments."""
    config = get_config()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--daemon-host", default=config.daemon.host, help="Daemon host address")
    parser.add_argument("--daemon-port", type=int, default=config.daemon.port, help="Daemon port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def _create_daemon_settings_parser() -> argparse.ArgumentParser:
    """Create parent parser with daemon settings arguments."""
    config = get_config()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-projects", type=int, default=config.daemon.max_projects,
                        help="Max projects to cache")
    parser.add_argument("--idle-timeout", type=str,
                        help="Auto-shutdown after idle (e.g., 30m, 1h)")
    parser.add_argument("--no-watch", action="store_true",
                        help="Disable automatic file watching for index updates")
    return parser


def create_daemon_parser() -> argparse.ArgumentParser:
    """Create parser for daemon commands."""
    common_parser = _create_common_daemon_parser()
    settings_parser = _create_daemon_settings_parser()

    parser = argparse.ArgumentParser(
        prog="coden-retriever daemon",
        description="Manage the daemon for fast responses",
        formatter_class=DefaultValueHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="daemon_action", help="Daemon action")

    # daemon start (inherits common + settings)
    subparsers.add_parser(
        "start",
        parents=[common_parser, settings_parser],
        help="Start daemon in background"
    )

    # daemon stop (inherits common only)
    subparsers.add_parser(
        "stop",
        parents=[common_parser],
        help="Stop the daemon"
    )

    # daemon status (inherits common only)
    subparsers.add_parser(
        "status",
        parents=[common_parser],
        help="Show daemon status"
    )

    # daemon restart (inherits common + settings)
    subparsers.add_parser(
        "restart",
        parents=[common_parser, settings_parser],
        help="Restart the daemon"
    )

    # daemon run (inherits common + settings)
    subparsers.add_parser(
        "run",
        parents=[common_parser, settings_parser],
        help="Run daemon in foreground (for debugging)"
    )

    # daemon clear-cache (inherits common + custom args)
    clear_cache_parser = subparsers.add_parser(
        "clear-cache",
        parents=[common_parser],
        help="Clear daemon cache"
    )
    clear_cache_parser.add_argument("clear_path", nargs="?", help="Path to clear from cache")
    clear_cache_parser.add_argument("--all", dest="clear_all", action="store_true",
                                    help="Clear all cached projects")

    return parser


def create_flag_parser(config) -> argparse.ArgumentParser:
    """Create parser for flag command with clear subcommand."""
    parser = argparse.ArgumentParser(
        prog="coden flag",
        description="Insert [CODEN] comments in source code based on analysis results",
        formatter_class=DefaultValueHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="flag_action", help="Flag action")

    # Main flag command (default)
    flag_parser = subparsers.add_parser(
        "add",
        help="Add [CODEN] flags to code (default if path given)"
    )
    _add_flag_arguments(flag_parser, config)

    # Clear subcommand
    clear_parser = subparsers.add_parser(
        "clear",
        help="Remove all [CODEN] flags from code"
    )
    clear_parser.add_argument("root", nargs="?", default=".",
                              help="Repository root directory")
    clear_parser.add_argument("--dry-run", action="store_true",
                              help="Preview changes without modifying files")
    clear_parser.add_argument("-v", "--verbose", action="store_true",
                              help="Verbose output")
    clear_parser.add_argument("-f", "--format", default="tree",
                              choices=["tree", "json"],
                              help="Output format")
    clear_parser.add_argument("-r", "--reverse", action="store_true",
                              help="Reverse output order")
    clear_parser.add_argument("--stats", action="store_true",
                              help="Show summary statistics")

    return parser


def _add_flag_arguments(parser: argparse.ArgumentParser, config) -> None:
    """Add common flag arguments to a parser."""
    parser.add_argument("root", nargs="?", default=".",
                        help="Repository root directory")

    # Analysis type flags
    analysis_group = parser.add_argument_group("Analysis Types (at least one required)")
    analysis_group.add_argument("-H", "--hotspots", action="store_true",
                                help="Flag coupling hotspots")
    analysis_group.add_argument("-P", "--propagation", action="store_true",
                                help="Flag high propagation cost functions")
    analysis_group.add_argument("-C", "--clones", action="store_true",
                                help="Flag code clones. Requires [semantic] extra for semantic/combined modes")
    analysis_group.add_argument("--clone-semantic", action="store_true",
                                help="Clone detection: semantic only (Model2Vec embeddings)")
    analysis_group.add_argument("--clone-syntactic", action="store_true",
                                help="Clone detection: syntactic only (line-by-line Jaccard)")
    analysis_group.add_argument("--line-threshold", type=float, default=0.70,
                                help="Line similarity threshold for syntactic clones")
    analysis_group.add_argument("--func-threshold", type=float, default=0.50,
                                help="Function match threshold for syntactic clones")
    analysis_group.add_argument("--semantic-weight", type=float, default=0.65,
                                help="Weight for semantic similarity in combined score")
    analysis_group.add_argument("--syntactic-weight", type=float, default=0.35,
                                help="Weight for syntactic similarity in combined score")
    analysis_group.add_argument("-E", "--echo-comments", action="store_true",
                                help="Detect and flag echo comments. Requires [semantic] extra")
    analysis_group.add_argument("-D", "--dead-code", action="store_true",
                                help="Flag dead code - functions/methods with no callers in the codebase")
    analysis_group.add_argument("-T", "--tramp-data", action="store_true",
                                help="Flag tramp data - parameters appearing across many functions")
    analysis_group.add_argument("-S", "--sensitive-values", action="store_true",
                                help="Flag sensitive values - hardcoded secrets, API keys, credentials")

    # Threshold options (uses centralized THRESHOLD_CONFIGS for maintainability)
    threshold_group = parser.add_argument_group("Threshold Options")
    for threshold_config in THRESHOLD_CONFIGS.values():
        add_threshold_argument(threshold_group, threshold_config, use_detailed_help=True)
    threshold_group.add_argument("--min-occurrences", type=int, default=3,
                                 help="Minimum function count for tramp data detection (default: 3)")
    threshold_group.add_argument("--min-group-size", type=int, default=2,
                                 help="Minimum parameters in a tramp data group (default: 2)")

    # Behavior options
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying files")
    parser.add_argument("--backup", action="store_true",
                        help="Create .coden-backup files before modifying")
    parser.add_argument("--remove-comments", action="store_true",
                        help="Delete detected echo comments entirely instead of flagging with [CODEN] markers (use with -E)")
    parser.add_argument("--remove-dead-code", action="store_true",
                        help="Delete dead code functions entirely instead of flagging with [CODEN] markers (DESTRUCTIVE - use with --backup)")
    parser.add_argument("--replace", nargs="?", const="***REDACTED***", default=None,
                        help="Replace detected sensitive values with placeholder (default: ***REDACTED***, or specify custom value)")
    parser.add_argument("--include-tests", action="store_true",
                        help="Include test files in analysis. By default, test files are excluded to focus on production code")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("-f", "--format", default="tree",
                        choices=["tree", "json"],
                        help="Output format")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse output order (highest severity last)")
    parser.add_argument("--stats", action="store_true",
                        help="Show summary statistics")
    parser.add_argument("-n", "--limit", type=int, default=config.search.default_limit,
                        help="Limit results (default: 20, use -n -1 for all; dry-run preview only)")


def run_direct_search(args: argparse.Namespace, root_path: Path, cache: CacheManager, app_config) -> int:
    """Run search directly (fallback when daemon not available)."""
    args.limit = normalize_limit(args.limit)
    try:
        # Create search config from CLI args
        config = SearchConfig(
            root_path=root_path,
            query=args.query or "",
            token_limit=args.tokens,
            output_format=OutputFormat(args.format),
            enable_semantic=args.enable_semantic,
            model_path=app_config.search.semantic_model_path,
            show_deps=args.show_deps,
            dir_tree=args.dir_tree,
            map_mode=args.map,
            find_mode=args.find,
            limit=args.limit,
            verbose=args.verbose,
            show_stats=args.stats,
            reverse=args.reverse,
        )

        # Create and execute pipeline (reuse provided cache)
        pipeline = SearchPipeline(config, cache=cache)
        engine = pipeline.create_engine()
        stats = engine.get_stats()

        if args.verbose:
            print(f"\n{stats}\n", file=sys.stderr)

        if stats.total_entities == 0:
            logger.warning("No code entities found")
            return 0

        # Execute pipeline
        result = pipeline.execute()

        if args.enable_semantic and args.query:
            print(format_semantic_search_header(args.query))
            print()

        # Print output in correct order based on reverse flag
        print_search_output(
            formatted_output=result.formatted_output,
            tree_output=result.tree_output,
            stats_output=result.stats,
            reverse=args.reverse,
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def create_serve_parser(config) -> argparse.ArgumentParser:
    """Create parser for 'serve' subcommand."""
    parser = argparse.ArgumentParser(
        prog="coden serve",
        description="Run as MCP server. Requires [mcp] extra: pip install 'coden-retriever[mcp]'",
        formatter_class=DefaultValueHelpFormatter,
    )
    parser.add_argument("--transport", choices=["stdio", "http", "sse", "streamable-http"],
                        default="stdio", help="Transport protocol")
    parser.add_argument("--host", type=str, default=config.daemon.host,
                        help="Host address (for http/sse transport)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="Port (for http/sse transport)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def create_agent_parser(config) -> argparse.ArgumentParser:
    """Create parser for 'agent' subcommand."""
    parser = argparse.ArgumentParser(
        prog="coden agent",
        description="Interactive coding agent with ReAct reasoning. Requires [agent] extra: pip install 'coden-retriever[agent]'",
        formatter_class=DefaultValueHelpFormatter,
    )
    parser.add_argument("root", nargs="?", default=".",
                        help="Repository root directory")
    parser.add_argument("--model", "-m", type=str, default=config.model.default,
                        help="LLM model (ollama:model, openai:model, or model with --base-url)")
    parser.add_argument("--base-url", type=str, default=config.model.base_url,
                        help="Base URL for OpenAI-compatible endpoints")
    parser.add_argument("--max-steps", type=int, default=config.agent.max_steps,
                        help="Max tool calls per query")
    parser.add_argument("--mcp-timeout", type=float, default=config.agent.mcp_server_timeout,
                        help="MCP server startup timeout (seconds)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def create_search_parser(config) -> argparse.ArgumentParser:
    """Create parser for search (default) mode."""
    parser = argparse.ArgumentParser(
        prog="coden",
        description="Coden - code search and context generation",
        formatter_class=DefaultValueHelpFormatter,
        epilog="""
Subcommands:
  serve                 Run as MCP server (requires [mcp] extra)
  agent (-a)            Interactive coding agent (requires [agent] extra)
  daemon                Manage daemon (start, stop, status)
  cache                 Manage caches (list, clear, status)
  config                Manage configuration (show, set, reset)
  flag                  Add/remove [CODEN] comments based on analysis
  reset                 Reset everything (clear caches, stop daemon, reset config)

Examples:
  coden                              # Context map of current directory
  coden /path/to/repo -q "auth"      # Search for "auth"
  coden -q "database" -sr --stats    # Semantic search, reversed, with stats
  coden --find UserAuth --show-deps  # Find identifier with dependencies
  coden -H -r --stats -n 20          # Top 20 refactoring hotspots
  coden -C --clone-threshold 0.90          # Find code clones (90% similarity)
  coden -C --clone-semantic               # Semantic-only clone detection (Model2Vec)
  coden -C --clone-syntactic              # Syntactic-only clone detection (Jaccard)
  coden -C --semantic-weight 0.5          # Adjust combined mode weights
  coden -P --breakdown               # Architecture health with module breakdown
  coden -P --critical-paths --stats  # Propagation cost with critical paths
  coden -E --echo-threshold 0.85     # Detect echo comments (redundant comments)
  coden -E --remove-comments         # Remove echo comments directly (no preview)
  coden -E --remove-comments --dry-run  # Preview echo comment removal
  coden flag -E --remove-comments    # Alternative: use flag subcommand
  coden flag -C --dry-run            # Preview clone flags without modifying
  coden flag -HPCE --backup          # Flag all issues with backup files
  coden flag clear                   # Remove all [CODEN] comments
  coden serve                        # MCP server (stdio)
  coden -a                           # Interactive agent
        """
    )

    parser.add_argument("root", nargs="?", default=".",
                        help="Repository root directory")
    parser.add_argument("-q", "--query", default="",
                        help="Search query")
    parser.add_argument("--map", action="store_true",
                        help="Generate context map (default when no query)")
    parser.add_argument("--find", metavar="IDENT",
                        help="Find specific identifier")
    parser.add_argument("-H", "--hotspots", action="store_true",
                        help="Find refactoring hotspots (high coupling + complexity)")
    parser.add_argument("-C", "--clones", action="store_true",
                        help="Detect code clones - find semantically similar functions for refactoring. Requires [semantic] extra for semantic/combined modes")
    parser.add_argument("--clone-semantic", action="store_true",
                        help="Clone detection: semantic only (Model2Vec embeddings)")
    parser.add_argument("--clone-syntactic", action="store_true",
                        help="Clone detection: syntactic only (line-by-line Jaccard)")
    parser.add_argument("--line-threshold", type=float, default=0.70,
                        help="Line similarity threshold for syntactic clone detection (0.0-1.0, default: 0.70)")
    parser.add_argument("--func-threshold", type=float, default=0.50,
                        help="Function match threshold for syntactic clone detection (0.0-1.0, default: 0.50)")
    parser.add_argument("--semantic-weight", type=float, default=0.65,
                        help="Weight for semantic similarity in combined score (0.0-1.0, default: 0.65)")
    parser.add_argument("--syntactic-weight", type=float, default=0.35,
                        help="Weight for syntactic similarity in combined score (0.0-1.0, default: 0.35)")
    parser.add_argument("-P", "--propagation", action="store_true",
                        help="Analyze propagation cost (architecture coupling)")
    parser.add_argument("-E", "--echo-comments", action="store_true",
                        help="Detect echo comments - comments that merely restate what the code already says. Requires [semantic] extra")
    parser.add_argument("-D", "--dead-code", action="store_true",
                        help="Detect dead code - functions/methods with no callers")
    parser.add_argument("-T", "--tramp-data", action="store_true",
                        help="Detect tramp data - parameters appearing across many functions")
    parser.add_argument("--min-occurrences", type=int, default=3,
                        help="Minimum function count for tramp data detection (default: 3)")
    parser.add_argument("--min-group-size", type=int, default=2,
                        help="Minimum parameters in a tramp data group (default: 2)")
    parser.add_argument("-S", "--sensitive-values", action="store_true",
                        help="Detect sensitive values - hardcoded secrets, API keys, credentials")
    parser.add_argument("--replace", nargs="?", const="***REDACTED***", default=None,
                        help="Replace detected sensitive values (default replacement: ***REDACTED***, or specify custom)")
    parser.add_argument("--breakdown", action="store_true",
                        help="Include per-module breakdown (with -P)")
    parser.add_argument("--critical-paths", action="store_true",
                        help="Show most connected paths (with -P)")

    # Threshold options for all analysis modes (uses centralized THRESHOLD_CONFIGS)
    add_threshold_argument(parser, THRESHOLD_CONFIGS["risk"], use_detailed_help=False)
    add_threshold_argument(parser, THRESHOLD_CONFIGS["propagation"], use_detailed_help=False)
    add_threshold_argument(parser, THRESHOLD_CONFIGS["clone"], use_detailed_help=False)
    add_threshold_argument(parser, THRESHOLD_CONFIGS["echo"], use_detailed_help=False)
    add_threshold_argument(parser, THRESHOLD_CONFIGS["dead_code"], use_detailed_help=False)
    add_threshold_argument(parser, THRESHOLD_CONFIGS["sensitive_value"], use_detailed_help=False)
    def _validate_min_lines(value):
        ival = int(value)
        if ival < 1:
            raise argparse.ArgumentTypeError(f"min-lines must be at least 1, got {ival}")
        return ival
    parser.add_argument("--min-lines", type=_validate_min_lines, default=3,
                        metavar="INT",
                        help="Minimum function lines to consider (default: 3, minimum: 1)")
    parser.add_argument("--tokens", type=int, default=None,
                        help="Token budget (default: unlimited, only -n/--limit controls result count)")
    parser.add_argument("-n", "--limit", type=int, default=config.search.default_limit,
                        help="Max results (default: 20, use -n -1 for all)")
    parser.add_argument("-f", "--format", choices=["xml", "markdown", "tree", "json"],
                        default="tree", help="Output format")
    parser.add_argument("--show-deps", action="store_true",
                        help="Include dependency context")
    parser.add_argument("--dir-tree", action=argparse.BooleanOptionalAction, default=True,
                        help="Show directory tree")
    parser.add_argument("--stats", action="store_true",
                        help="Print ranking statistics")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse result order (highest score last)")
    parser.add_argument("-s", "--semantic", dest="enable_semantic", action="store_true",
                        help="Enable semantic search (Model2Vec). Requires [semantic] extra")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    # File modification options (for -E with --remove-comments)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying files (use with -E --remove-comments)")
    parser.add_argument("--backup", action="store_true",
                        help="Create .coden-backup files before modifying (use with -E --remove-comments)")
    parser.add_argument("--remove-comments", action="store_true",
                        help="Delete detected echo comments entirely instead of just displaying them (use with -E)")
    parser.add_argument("--include-tests", action="store_true",
                        help="Include test files in analysis (use with -E). By default, test files are excluded")
    return parser


def handle_serve_command(args: argparse.Namespace) -> int:
    """Handle 'serve' subcommand."""
    try:
        require_feature("mcp")
    except MissingDependencyError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .mcp.server import create_mcp_server
    mcp = create_mcp_server()
    if mcp:
        logger.info(f"Starting MCP server with {args.transport} transport...")
        # Disable banner for stdio transport to avoid corrupting the MCP protocol
        # (fastmcp 2.14.2+ prints banner to stdout which breaks stdio JSON-RPC)
        show_banner = args.transport != "stdio"
        if args.transport in ["http", "sse", "streamable-http"]:
            logger.info(f"Server will be available at: http://{args.host}:{args.port}")
            mcp.run(transport=args.transport, host=args.host, port=args.port, show_banner=show_banner)
        else:
            mcp.run(transport=args.transport, show_banner=show_banner)
        return 0
    return 1


def handle_agent_command(args: argparse.Namespace, config) -> int:
    """Handle 'agent' subcommand."""
    try:
        require_feature("agent")
    except MissingDependencyError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .agent import run_interactive

    root_path = Path(args.root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Invalid root path: {root_path}")
        return 1

    # Save settings if user explicitly provided them
    user_provided_model = args.model != config.model.default
    user_provided_base_url = args.base_url != config.model.base_url
    user_provided_mcp_timeout = args.mcp_timeout != config.agent.mcp_server_timeout

    if user_provided_model or user_provided_base_url or user_provided_mcp_timeout:
        if user_provided_model:
            config.model.default = args.model
        if user_provided_base_url:
            config.model.base_url = args.base_url
        if user_provided_mcp_timeout:
            config.agent.mcp_server_timeout = args.mcp_timeout
        save_config(config)

    try:
        _get_asyncio().run(run_interactive(
            str(root_path),
            args.model,
            args.base_url,
            args.max_steps,
            disabled_tools=config.agent.disabled_tools,
        ))
    except KeyboardInterrupt:
        pass
    return 0


def handle_hotspots_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle hotspots mode (-H/--hotspots flag).

    Args:
        args: Parsed CLI arguments
        root_path: Resolved root path
        config: Application configuration

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args.limit = normalize_limit(args.limit)
    start_time = time.time()

    # Print parameter header before results
    header = format_hotspots_parameters_header(
        risk_threshold=args.risk_threshold,
        exclude_tests=True,
        limit=args.limit,
    )
    print(header)
    print()  # Blank line

    # Create GraphAnalysisParams for hotspots
    params = GraphAnalysisParams(
        source_dir=str(root_path),
        limit=args.limit,
        exclude_tests=True,
        token_limit=args.tokens,  # None = no limit for CLI
        min_coupling_score=10,
        exclude_private=False,
    )

    # Try daemon mode first
    daemon_result = try_daemon_hotspots(params, host=config.daemon.host, port=config.daemon.port)

    if daemon_result is not None:
        all_hotspots = daemon_result.get("hotspots", [])
        summary = daemon_result.get("summary", {})

        # Filter by threshold (only include hotspots above risk threshold)
        threshold_filtered = [
            h for h in all_hotspots
            if h.get("risk_score", 0) >= args.risk_threshold
        ]

        # Apply defensive limit (ensures contract compliance)
        hotspots = apply_defensive_limit(threshold_filtered, args.limit)

        # Format output
        formatted_output = format_hotspots_output(
            hotspots,
            output_format=args.format,
            reverse=args.reverse,
        )

        # Format stats if requested
        stats_output = format_hotspots_stats(summary) if args.stats else None

        # Print output with contract-compliant ordering
        print_metric_output(formatted_output, stats_output, args.reverse)

        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\n[Daemon mode] Hotspots time: {elapsed_ms:.1f}ms, "
                  f"Results: {len(hotspots)}", file=sys.stderr)
        return 0

    # If daemon not available, try direct mode via MCP tool
    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from .mcp.graph_analysis import coupling_hotspots

        result = _get_asyncio().run(coupling_hotspots(
            root_directory=str(root_path),
            limit=args.limit,
            min_coupling_score=10,
            exclude_tests=True,
            exclude_private=False,
            token_limit=args.tokens,  # None = no limit for CLI
        ))

        all_hotspots = result.get("hotspots", [])
        summary = result.get("summary", {})

        # Filter by threshold (only include hotspots above risk threshold)
        threshold_filtered = [
            h for h in all_hotspots
            if h.get("risk_score", 0) >= args.risk_threshold
        ]

        # Apply defensive limit (ensures contract compliance)
        hotspots = apply_defensive_limit(threshold_filtered, args.limit)

        formatted_output = format_hotspots_output(
            hotspots,
            output_format=args.format,
            reverse=args.reverse,
        )
        stats_output = format_hotspots_stats(summary) if args.stats else None
        print_metric_output(formatted_output, stats_output, args.reverse)

        elapsed_ms = (time.time() - start_time) * 1000
        if args.verbose:
            print(f"\n[Direct mode] Hotspots time: {elapsed_ms:.1f}ms, "
                  f"Results: {len(hotspots)}", file=sys.stderr)
        return 0

    except Exception as e:
        logger.error(f"Hotspots analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _filter_dead_code_results(
    results: list[dict],
    threshold: float,
    limit: int | None,
) -> list[dict]:
    """Filter dead code results by confidence threshold and apply limit."""
    threshold_filtered = [
        d for d in results
        if d.get("confidence", 0) >= threshold
    ]
    return apply_defensive_limit(threshold_filtered, limit)


def _process_dead_code_result(
    result: dict,
    formatter: DeadCodeFormatter,
    args: argparse.Namespace,
    start_time: float,
    mode: str,
) -> int:
    """Process dead code detection result and output."""
    if "error" in result:
        logger.error(f"Dead code detection error: {result['error']}")
        return 1

    all_dead_code = result.get("dead_code", [])
    summary = result.get("summary", {})

    dead_code = _filter_dead_code_results(all_dead_code, args.dead_code_threshold, args.limit)

    formatted_output = formatter.format_items(dead_code, args.format, args.reverse)
    stats_output = formatter.format_stats(summary) if args.stats else None
    print_metric_output(formatted_output, stats_output, args.reverse)

    elapsed_ms = (time.time() - start_time) * 1000
    if args.verbose:
        print(f"\n[{mode} mode] Dead code detection: {elapsed_ms:.1f}ms, Found: {len(dead_code)}", file=sys.stderr)
    return 0


def _run_direct_dead_code(
    root_path: Path,
    formatter: DeadCodeFormatter,
    args: argparse.Namespace,
    start_time: float,
) -> int:
    """Run dead code detection directly (non-daemon)."""
    from .mcp.dead_code import detect_dead_code as mcp_detect_dead_code
    try:
        result = _get_asyncio().run(mcp_detect_dead_code(
            root_directory=str(root_path),
            confidence_threshold=args.dead_code_threshold,
            limit=args.limit,
            exclude_tests=True,
            include_private=False,
            min_lines=args.min_lines,
        ))
        return _process_dead_code_result(result, formatter, args, start_time, "Direct")
    except Exception as e:
        logger.error(f"Dead code detection failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_dead_code_command(
    args: argparse.Namespace,
    root_path: Path,
    config: AppConfig,
) -> int:
    """Handle dead code detection (-D/--dead-code flag)."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = DeadCodeFormatter()

    header = format_dead_code_parameters_header(
        confidence_threshold=args.dead_code_threshold,
        exclude_tests=True,
        limit=args.limit,
    )
    print(header)
    print()

    params = DeadCodeParams(
        source_dir=str(root_path),
        confidence_threshold=args.dead_code_threshold,
        limit=args.limit,
        exclude_tests=True,
        include_private=False,
        min_lines=args.min_lines,
        token_limit=args.tokens,
    )

    daemon_result = try_daemon_dead_code(params, host=config.daemon.host, port=config.daemon.port)
    if daemon_result is not None:
        return _process_dead_code_result(daemon_result, formatter, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    return _run_direct_dead_code(root_path, formatter, args, start_time)


def _filter_tramp_data_results(
    results: list[dict],
    min_occurrences: int,
    limit: int | None,
) -> list[dict]:
    """Filter tramp data results by min occurrences and apply limit."""
    threshold_filtered = [
        d for d in results
        if d.get("count", 0) >= min_occurrences
    ]
    return apply_defensive_limit(threshold_filtered, limit)


def _process_tramp_data_result(
    result: dict,
    formatter: TrampDataFormatter,
    args: argparse.Namespace,
    start_time: float,
    mode: str,
) -> int:
    """Process tramp data detection result and output."""
    if "error" in result:
        logger.error(f"Tramp data detection error: {result['error']}")
        return 1

    all_tramp_data = result.get("tramp_data", [])
    summary = result.get("summary", {})

    tramp_data = _filter_tramp_data_results(all_tramp_data, args.min_occurrences, args.limit)

    formatted_output = formatter.format_items(tramp_data, args.format, args.reverse)
    stats_output = formatter.format_stats(summary) if args.stats else None
    print_metric_output(formatted_output, stats_output, args.reverse)

    elapsed_ms = (time.time() - start_time) * 1000
    if args.verbose:
        print(f"\n[{mode} mode] Tramp data detection: {elapsed_ms:.1f}ms, Groups: {len(tramp_data)}", file=sys.stderr)
    return 0


def _run_direct_tramp_data(
    root_path: Path,
    formatter: TrampDataFormatter,
    args: argparse.Namespace,
    start_time: float,
) -> int:
    """Run tramp data detection directly (non-daemon)."""
    from .mcp.tramp_data import detect_tramp_data_tool
    try:
        result = _get_asyncio().run(detect_tramp_data_tool(
            root_directory=str(root_path),
            min_occurrences=args.min_occurrences,
            limit=args.limit,
            exclude_tests=True,
            min_group_size=args.min_group_size,
        ))
        return _process_tramp_data_result(result, formatter, args, start_time, "Direct")
    except Exception as e:
        logger.error(f"Tramp data detection failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_tramp_data_command(
    args: argparse.Namespace,
    root_path: Path,
    config: AppConfig,
) -> int:
    """Handle tramp data detection (-T/--tramp-data flag)."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = TrampDataFormatter()

    header = format_tramp_data_parameters_header(
        min_occurrences=args.min_occurrences,
        exclude_tests=True,
        limit=args.limit,
        min_group_size=args.min_group_size,
    )
    print(header)
    print()

    params = TrampDataParams(
        source_dir=str(root_path),
        min_occurrences=args.min_occurrences,
        limit=args.limit,
        exclude_tests=True,
        token_limit=args.tokens,
        min_group_size=args.min_group_size,
    )

    daemon_result = try_daemon_tramp_data(params, host=config.daemon.host, port=config.daemon.port)
    if daemon_result is not None:
        return _process_tramp_data_result(daemon_result, formatter, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    return _run_direct_tramp_data(root_path, formatter, args, start_time)


def _process_sensitive_value_result(
    result: dict,
    formatter: SensitiveValueFormatter,
    args: argparse.Namespace,
    start_time: float,
    mode: str,
) -> int:
    """Process sensitive value detection result and output."""
    if "error" in result:
        logger.error(f"Sensitive value detection error: {result['error']}")
        return 1

    all_values = result.get("sensitive_values", [])
    summary = result.get("summary", {})

    values = apply_defensive_limit(all_values, args.limit)

    formatted_output = formatter.format_items(values, args.format, args.reverse)
    stats_output = formatter.format_stats(summary) if args.stats else None
    print_metric_output(formatted_output, stats_output, args.reverse)

    elapsed_ms = (time.time() - start_time) * 1000
    if args.verbose:
        print(f"\n[{mode} mode] Sensitive value detection: {elapsed_ms:.1f}ms, Found: {len(values)}", file=sys.stderr)
    return 0


def _run_direct_sensitive_values(
    root_path: Path,
    formatter: SensitiveValueFormatter,
    args: argparse.Namespace,
    start_time: float,
) -> int:
    """Run sensitive value detection directly (non-daemon)."""
    from .mcp.sensitive_values import detect_sensitive_values_tool
    try:
        result = _get_asyncio().run(detect_sensitive_values_tool(
            root_directory=str(root_path),
            confidence_threshold=args.sensitive_threshold,
            limit=args.limit,
            exclude_tests=True,
            replace_value=getattr(args, "replace", None),
        ))
        return _process_sensitive_value_result(result, formatter, args, start_time, "Direct")
    except Exception as e:
        logger.error(f"Sensitive value detection failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_sensitive_values_command(
    args: argparse.Namespace,
    root_path: Path,
    config: AppConfig,
) -> int:
    """Handle sensitive value detection (-S/--sensitive-values flag)."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = SensitiveValueFormatter()

    header = format_sensitive_value_parameters_header(
        confidence_threshold=args.sensitive_threshold,
        exclude_tests=True,
        limit=args.limit,
        replace_value=getattr(args, "replace", None),
    )
    print(header)
    print()

    params = SensitiveValueParams(
        source_dir=str(root_path),
        confidence_threshold=args.sensitive_threshold,
        limit=args.limit,
        exclude_tests=True,
        token_limit=args.tokens,
        replace_value=getattr(args, "replace", None),
    )

    daemon_result = try_daemon_sensitive_values(params, host=config.daemon.host, port=config.daemon.port)
    if daemon_result is not None:
        return _process_sensitive_value_result(daemon_result, formatter, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    return _run_direct_sensitive_values(root_path, formatter, args, start_time)


def handle_search_command(args: argparse.Namespace, config) -> int:
    """Handle search (default) mode."""
    args.limit = normalize_limit(args.limit)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root_path = Path(args.root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Invalid root path: {args.root}")
        return 1

    # Check semantic feature for --semantic flag
    if args.enable_semantic:
        try:
            require_feature("semantic")
        except MissingDependencyError as e:
            print(str(e), file=sys.stderr)
            return 1

    # Handle propagation mode separately
    if args.propagation:
        return handle_propagation_command(args, root_path, config)

    # Handle clones mode separately
    if args.clones:
        return handle_clones_command(args, root_path, config)

    # Handle hotspots mode separately
    if args.hotspots:
        return handle_hotspots_command(args, root_path, config)

    # Handle echo comments mode separately
    if args.echo_comments:
        return handle_echo_comments_command(args, root_path, config)

    # Handle dead code mode separately
    if args.dead_code:
        return handle_dead_code_command(args, root_path, config)

    # Handle tramp data mode separately
    if args.tramp_data:
        return handle_tramp_data_command(args, root_path, config)

    # Handle sensitive values mode separately
    if args.sensitive_values:
        return handle_sensitive_values_command(args, root_path, config)

    # Create cache manager (use config for model_path)
    cache = CacheManager(
        root_path,
        enable_semantic=args.enable_semantic,
        model_path=config.search.semantic_model_path,
        verbose=args.verbose
    )

    # Try daemon mode first (use config for host/port)
    params = SearchParams(
        source_dir=str(root_path),
        query=args.query,
        enable_semantic=args.enable_semantic,
        model_path=config.search.semantic_model_path,
        limit=args.limit,
        tokens=args.tokens,
        show_deps=args.show_deps,
        output_format=args.format,
        find_identifier=args.find,
        map_mode=args.map or not args.query,
        dir_tree=args.dir_tree,
        stats=args.stats,
        reverse=args.reverse,
    )
    daemon_result = try_daemon_search(params, host=config.daemon.host, port=config.daemon.port)

    if daemon_result is not None:
        if args.enable_semantic and args.query:
            print(format_semantic_search_header(args.query))
            print()

        print_search_output(
            formatted_output=daemon_result.get("output", ""),
            tree_output=None,
            stats_output=daemon_result.get("stats_output") if args.stats else None,
            reverse=args.reverse,
        )

        if args.verbose:
            print(f"\n[Daemon mode] Search time: {daemon_result.get('search_time_ms', 0):.1f}ms, "
                  f"Results: {daemon_result.get('result_count', 0)}/{daemon_result.get('total_matched', 0)}, "
                  f"Tokens: {daemon_result.get('tokens_used', 0)}", file=sys.stderr)
        return 0

    # Direct mode (fallback)
    return run_direct_search(args, root_path, cache, config)


def main() -> int:
    """Main CLI entry point."""
    # Fix Windows console encoding for Unicode output
    if sys.platform == "win32":
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Load configuration
    config = get_config()

    # Ensure config file exists
    config_file = get_config_file()
    if not config_file.exists():
        print(f"Warning: Config file was missing, recreating at {config_file}", file=sys.stderr)
        save_config(config)

    # Route to subcommands
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        # Shortcuts: -a/--agent -> agent subcommand
        if cmd in ("-a", "--agent"):
            sys.argv[1] = "agent"
            cmd = "agent"

        if cmd == "serve":
            parser = create_serve_parser(config)
            args = parser.parse_args(sys.argv[2:])
            return handle_serve_command(args)

        if cmd == "agent":
            parser = create_agent_parser(config)
            args = parser.parse_args(sys.argv[2:])
            return handle_agent_command(args, config)

        if cmd == "daemon":
            daemon_parser = create_daemon_parser()
            args = daemon_parser.parse_args(sys.argv[2:])
            if not args.daemon_action:
                daemon_parser.print_help()
                return 1
            return handle_daemon_command(args)

        if cmd == "flag":
            flag_parser = create_flag_parser(config)
            # Check if there's a subcommand or just a path
            remaining_args = sys.argv[2:]
            if remaining_args and remaining_args[0] == "clear":
                args = flag_parser.parse_args(remaining_args)
                root_path = Path(args.root).resolve()
                return handle_flag_clear_command(args, root_path, config)
            else:
                # Default to "add" action - parse as if "add" was specified
                # Create a simple parser for the add command
                add_parser = argparse.ArgumentParser(
                    prog="coden flag",
                    description="Insert [CODEN] comments in source code based on analysis results",
                    formatter_class=DefaultValueHelpFormatter,
                )
                _add_flag_arguments(add_parser, config)
                args = add_parser.parse_args(remaining_args)
                root_path = Path(args.root).resolve()
                return handle_flag_command(args, root_path, config)

        if cmd == "config":
            return handle_config_command(sys.argv[2:])

        if cmd == "cache":
            return handle_cache_command(sys.argv[2:])

        if cmd == "reset":
            return handle_reset_command()

    # Default: search mode
    parser = create_search_parser(config)
    args = parser.parse_args()
    return handle_search_command(args, config)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    sys.exit(main())
