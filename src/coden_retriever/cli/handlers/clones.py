"""Handler for clone detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...constants import (
    DEFAULT_CLONE_SEMANTIC_WEIGHT,
    DEFAULT_CLONE_SYNTACTIC_WEIGHT,
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
)
from ...config_loader import daemon_enabled
from ...daemon.client import try_daemon_clones
from ...daemon.protocol import CloneDetectionParams
from ...formatters import CloneFormatter
from ...utils.progress import encoding_progress
from ..utils import get_clone_mode, normalize_limit

logger = logging.getLogger(__name__)

# Minimum timeout for clone detection (seconds); overrides config if lower
_MIN_CLONE_TIMEOUT = 60.0


def handle_clones_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle clone detection command using CloneFormatter for output."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = CloneFormatter()

    mode = get_clone_mode(args)

    line_threshold = getattr(args, "line_threshold", DEFAULT_SYNTACTIC_LINE_THRESHOLD)
    func_threshold = getattr(args, "func_threshold", DEFAULT_SYNTACTIC_FUNC_THRESHOLD)
    semantic_weight = getattr(args, "semantic_weight", DEFAULT_CLONE_SEMANTIC_WEIGHT)
    syntactic_weight = getattr(args, "syntactic_weight", DEFAULT_CLONE_SYNTACTIC_WEIGHT)

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

    daemon_result = None
    if daemon_enabled(args):
        daemon_result = try_daemon_clones(
            params, address=config.daemon.address,
            timeout=max(config.daemon.daemon_timeout, _MIN_CLONE_TIMEOUT),
            auto_start=False,
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
        result = _run_direct_clone_detection(
            root_path, mode, args, line_threshold, func_threshold,
            semantic_weight, syntactic_weight,
        )

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


def _run_direct_clone_detection(
    root_path: Path,
    mode: str,
    args: argparse.Namespace,
    line_threshold: float,
    func_threshold: float,
    semantic_weight: float,
    syntactic_weight: float,
) -> dict:
    """Run clone detection directly with progress bar for encoding."""
    from ...cache import CacheManager
    from ...clone import detect_clones_combined, detect_clones_semantic, detect_clones_syntactic

    cache = CacheManager(root_path)
    indices = cache.load_or_rebuild()

    # Count functions for progress bar (texts + names = 2x)
    func_count = sum(
        1 for e in indices.entities.values()
        if e.entity_type in ("function", "method")
        and e.source_code
        and (e.line_end - e.line_start + 1) >= args.min_lines
    )
    # Each function encodes source_code + name = 2 encoding batches
    _ENCODE_MULTIPLIER = 2
    total_encode = func_count * _ENCODE_MULTIPLIER

    if mode == "syntactic":
        result = detect_clones_syntactic(
            entities=indices.entities,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=args.limit,
            exclude_tests=True,
            min_lines=args.min_lines,
            token_limit=args.tokens,
        )
    elif mode == "semantic":
        with encoding_progress("Encoding functions", total_encode) as advance:
            result = detect_clones_semantic(
                entities=indices.entities,
                model_path="",
                threshold=args.clone_threshold,
                limit=args.limit,
                exclude_tests=True,
                min_lines=args.min_lines,
                token_limit=args.tokens,
                embedding_cache=indices.embedding_cache,
                on_encode_progress=advance,
            )
    else:
        with encoding_progress("Encoding functions", total_encode) as advance:
            result = detect_clones_combined(
                entities=indices.entities,
                model_path="",
                semantic_threshold=args.clone_threshold,
                line_threshold=line_threshold,
                func_threshold=func_threshold,
                limit=args.limit,
                exclude_tests=True,
                min_lines=args.min_lines,
                token_limit=args.tokens,
                semantic_weight=semantic_weight,
                syntactic_weight=syntactic_weight,
                embedding_cache=indices.embedding_cache,
                on_encode_progress=advance,
            )

    if indices.embedding_cache is not None:
        indices.embedding_cache.save(cache.cache_dir)

    return result
