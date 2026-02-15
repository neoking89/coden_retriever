"""Handler for clone detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...daemon.client import try_daemon_clones
from ...daemon.protocol import CloneDetectionParams
from ...formatters import CloneFormatter
from ...utils.optional_deps import MissingDependencyError, require_feature
from ..utils import get_asyncio, get_clone_mode, normalize_limit

logger = logging.getLogger(__name__)

# Minimum timeout for clone detection (seconds); overrides config if lower
_MIN_CLONE_TIMEOUT = 60.0


def handle_clones_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle clone detection command using CloneFormatter for output."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = CloneFormatter()

    mode = get_clone_mode(args)

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

    from ...formatters.clone_formatter import format_clone_parameters_header
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
    print()

    params = CloneDetectionParams(
        source_dir=str(root_path),
        mode=mode,
        similarity_threshold=args.clone_threshold,
        line_threshold=line_threshold,
        func_threshold=func_threshold,
        limit=args.limit,
        exclude_tests=True,
        min_lines=args.min_lines,
        token_limit=args.tokens,
        semantic_weight=semantic_weight,
        syntactic_weight=syntactic_weight,
    )

    daemon_result = try_daemon_clones(
        params, host=config.daemon.host, port=config.daemon.port,
        timeout=max(config.daemon.daemon_timeout, _MIN_CLONE_TIMEOUT),
    )

    if daemon_result is not None:
        if "error" in daemon_result:
            logger.error(f"Clone detection error: {daemon_result['error']}")
            return 1

        all_clones = daemon_result.get("clones", [])
        summary = daemon_result.get("summary", {})

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
        from ...mcp.clone_detection import detect_clones as mcp_detect_clones

        result = get_asyncio().run(mcp_detect_clones(
            root_directory=str(root_path),
            mode=mode,
            similarity_threshold=args.clone_threshold,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=args.limit,
            exclude_tests=True,
            min_lines=args.min_lines,
            token_limit=args.tokens,
            semantic_weight=semantic_weight,
            syntactic_weight=syntactic_weight,
        ))

        if "error" in result:
            logger.error(f"Clone detection error: {result['error']}")
            return 1

        all_clones = result.get("clones", [])
        summary = result.get("summary", {})

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
