"""Handler for echo comment detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...constants import MILLISECONDS_PER_SECOND, PERCENT, STATS_SEPARATOR_WIDTH
from ...utils.optional_deps import MissingDependencyError, require_feature
from ..utils import normalize_limit

logger = logging.getLogger(__name__)


def handle_echo_comments_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle echo comment detection command (read-only analysis or file modification with --remove-comments)."""
    try:
        require_feature("semantic")
    except MissingDependencyError as e:
        print(str(e), file=sys.stderr)
        return 1

    args.limit = normalize_limit(args.limit)
    start_time = time.time()

    try:
        from ...cache import CacheManager

        cache = CacheManager(root_path)
        indices = cache.load_or_rebuild()

        if not args.remove_comments:
            from ...formatters.flag_formatter import format_echo_parameters_header
            header = format_echo_parameters_header(
                echo_threshold=args.echo_threshold,
                exclude_tests=not args.include_tests,
                limit=args.limit,
            )
            print(header)
            print()

        if args.remove_comments:
            return _handle_remove_comments(args, indices, root_path, start_time)

        return _handle_read_only_analysis(args, indices, start_time)

    except Exception as e:
        logger.error(f"Echo comment detection failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _handle_remove_comments(args: argparse.Namespace, indices, root_path: Path, start_time: float) -> int:
    """Handle echo comment removal mode (--remove-comments)."""
    from ...mcp.flag_insertion import flag_code
    from ...daemon.protocol import FlagParams
    from ...formatters.flag_formatter import FlagFormatter

    params = FlagParams(
        source_dir=str(root_path),
        echo_comments=True,
        echo_threshold=args.echo_threshold,
        dry_run=args.dry_run,
        backup=args.backup,
        verbose=args.verbose,
        exclude_tests=not args.include_tests,
        remove_comments=True,
    )

    result = flag_code(
        entities=indices.entities,
        graph=indices.graph,
        pagerank=indices.pagerank,
        params=params,
    )

    if "error" in result:
        logger.error(f"Echo comment removal failed: {result['error']}")
        return 1

    formatter = FlagFormatter()
    formatted_output = formatter.format_items(result.get("items", []), args.format, args.reverse)

    stats_output = None
    if args.stats:
        stats_lines = [
            "",
            "=" * STATS_SEPARATOR_WIDTH,
            f"Echo Comment Removal | {result.get('flagged_count', 0)} items",
            "-" * STATS_SEPARATOR_WIDTH,
            f"Files modified: {result.get('files_modified', 0)}",
            f"Comments removed: {result.get('flagged_count', 0)}",
            "=" * STATS_SEPARATOR_WIDTH,
        ]
        stats_output = "\n".join(stats_lines)

    print_metric_output(formatted_output, stats_output, args.reverse)

    elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
    if args.verbose:
        print(f"\nEcho comment removal time: {elapsed_ms:.1f}ms", file=sys.stderr)
    return 0


def _handle_read_only_analysis(args: argparse.Namespace, indices, start_time: float) -> int:
    """Handle read-only echo comment analysis."""
    from ...mcp.echo_comments import compute_echo_comments
    from ...formatters.flag_formatter import FlagFormatter

    formatter = FlagFormatter()

    result = compute_echo_comments(
        entities=indices.entities,
        echo_threshold=args.echo_threshold,
        token_limit=args.tokens,
        include_tests=args.include_tests,
        include_private=False,
    )

    if "error" in result:
        logger.error(f"Echo comment detection error: {result['error']}")
        return 1

    all_echo_comments = result.get("echo_comments", [])
    summary = result.get("summary", {})

    echo_comments = apply_defensive_limit(all_echo_comments, args.limit)

    items = []
    for echo in echo_comments:
        items.append({
            "type": "echo",
            "file": echo.get("file_path"),
            "line": echo.get("line"),
            "name": echo.get("context_identifier"),
            "similarity_score": echo.get("similarity_score"),
            "comment_text": echo.get("comment_text"),
            "severity": echo.get("severity"),
        })

    formatted_output = formatter.format_items(items, args.format, args.reverse)

    if args.stats:
        stats_output = _format_echo_stats(echo_comments, all_echo_comments, summary)
    else:
        stats_output = None

    print_metric_output(formatted_output, stats_output, args.reverse)

    elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
    if args.verbose:
        print(f"\nEcho comment detection time: {elapsed_ms:.1f}ms, Found: {len(echo_comments)}", file=sys.stderr)
    return 0


def _format_echo_stats(
    echo_comments: list[dict],
    all_echo_comments: list[dict],
    summary: dict,
) -> str:
    """Format echo comment analysis statistics."""
    total_comments = summary.get("total_comments_found", 0)
    total_echo_count = len(all_echo_comments)
    echo_ratio = total_echo_count / total_comments if total_comments > 0 else 0
    distribution = summary.get("distribution", {})

    stats_lines = [
        "",
        "=" * STATS_SEPARATOR_WIDTH,
        f"Echo Comment Analysis | {len(echo_comments):,} shown ({total_echo_count:,} total)",
        "-" * STATS_SEPARATOR_WIDTH,
        f"Total comments analyzed: {total_comments:,}",
        f"Echo ratio: {echo_ratio * PERCENT:.1f}%",
        f"Files affected: {summary.get('files_affected', 0)}",
        f"Avg similarity: {summary.get('avg_similarity', 0) * PERCENT:.1f}%",
        "-" * STATS_SEPARATOR_WIDTH,
        "Distribution:",
        f"  CRITICAL (>95%): {distribution.get('critical', 0)}",
        f"  HIGH (90-95%): {distribution.get('high', 0)}",
        f"  ELEVATED (85-90%): {distribution.get('elevated', 0)}",
        f"  MODERATE (<85%): {distribution.get('moderate', 0)}",
        "=" * STATS_SEPARATOR_WIDTH,
    ]
    return "\n".join(stats_lines)
