"""Handler for hotspots analysis command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...config_loader import daemon_enabled
from ...daemon.client import try_daemon_hotspots
from ...daemon.protocol import GraphAnalysisParams
from ...formatters.hotspots_formatter import (
    format_hotspots_output,
    format_hotspots_parameters_header,
    format_hotspots_stats,
)
from ..utils import get_asyncio, normalize_limit

logger = logging.getLogger(__name__)

# Minimum coupling score for hotspot detection
_MIN_COUPLING_SCORE = 10


def handle_hotspots_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle hotspots mode (-H/--hotspots flag)."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()

    header = format_hotspots_parameters_header(
        risk_threshold=args.risk_threshold,
        exclude_tests=True,
        limit=args.limit,
    )
    print(header)
    print()

    params = GraphAnalysisParams(
        source_dir=str(root_path),
        limit=args.limit,
        exclude_tests=True,
        token_limit=args.tokens,
        min_coupling_score=_MIN_COUPLING_SCORE,
        exclude_private=False,
    )

    daemon_result = None
    if daemon_enabled(args):
        daemon_result = try_daemon_hotspots(params, address=config.daemon.address)

    if daemon_result is not None:
        return _process_hotspots_result(daemon_result, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from ...mcp.graph_analysis import coupling_hotspots

        result = get_asyncio().run(coupling_hotspots(
            root_directory=str(root_path),
            limit=args.limit,
            min_coupling_score=_MIN_COUPLING_SCORE,
            exclude_tests=True,
            exclude_private=False,
            token_limit=args.tokens,
        ))
        return _process_hotspots_result(result, args, start_time, "Direct")

    except Exception as e:
        logger.error(f"Hotspots analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _process_hotspots_result(
    result: dict,
    args: argparse.Namespace,
    start_time: float,
    mode: str,
) -> int:
    """Process hotspots result: filter, format, print, return exit code."""
    all_hotspots = result.get("hotspots", [])
    summary = result.get("summary", {})

    threshold_filtered = [
        h for h in all_hotspots
        if h.get("risk_score", 0) >= args.risk_threshold
    ]

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
        print(f"\n[{mode} mode] Hotspots time: {elapsed_ms:.1f}ms, "
              f"Results: {len(hotspots)}", file=sys.stderr)
    return 0
