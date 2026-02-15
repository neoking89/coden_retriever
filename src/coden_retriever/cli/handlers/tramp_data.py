"""Handler for tramp data detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...config_loader import AppConfig
from ...daemon.client import try_daemon_tramp_data
from ...daemon.protocol import TrampDataParams
from ...formatters import TrampDataFormatter
from ...formatters.tramp_data_formatter import format_tramp_data_parameters_header
from ..utils import get_asyncio, normalize_limit

logger = logging.getLogger(__name__)


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
    from ...mcp.tramp_data import detect_tramp_data_tool
    try:
        result = get_asyncio().run(detect_tramp_data_tool(
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
