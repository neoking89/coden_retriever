"""Handler for dead code detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...config_loader import AppConfig, daemon_enabled
from ...daemon.client import try_daemon_dead_code
from ...daemon.protocol import DeadCodeParams
from ...formatters import DeadCodeFormatter
from ...formatters.dead_code_formatter import format_dead_code_parameters_header
from ..utils import get_asyncio, normalize_limit

logger = logging.getLogger(__name__)


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
    from ...mcp.dead_code import detect_dead_code as mcp_detect_dead_code
    try:
        result = get_asyncio().run(mcp_detect_dead_code(
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

    daemon_result = None
    if daemon_enabled(args):
        daemon_result = try_daemon_dead_code(params, address=config.daemon.address)
    if daemon_result is not None:
        return _process_dead_code_result(daemon_result, formatter, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    return _run_direct_dead_code(root_path, formatter, args, start_time)
