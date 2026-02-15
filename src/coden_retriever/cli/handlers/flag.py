"""Handler for flag insert and flag clear commands."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...constants import MILLISECONDS_PER_SECOND
from ...daemon.client import try_daemon_flag, try_daemon_flag_clear
from ...daemon.protocol import FlagClearParams
from ...formatters.flag_formatter import FlagFormatter, format_parameters_header
from ..utils import normalize_limit
from .flag_utils import (
    build_flag_active_flags,
    build_flag_params,
    process_flag_result,
    validate_flag_args,
)

logger = logging.getLogger(__name__)


def handle_flag_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle flag command to insert [CODEN] comments."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = FlagFormatter()

    validation_code = validate_flag_args(args, root_path)
    if validation_code != 0:
        return validation_code

    active_flags = build_flag_active_flags(args)
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

    params = build_flag_params(args, root_path)

    daemon_result = try_daemon_flag(params, host=config.daemon.host, port=config.daemon.port)
    if daemon_result is not None:
        return process_flag_result(daemon_result, args, formatter, start_time, "Daemon mode")

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from ...cache import CacheManager
        from ...mcp.flag_insertion import flag_code

        cache = CacheManager(root_path)
        indices = cache.load_or_rebuild()

        result = flag_code(
            entities=indices.entities,
            graph=indices.graph,
            pagerank=indices.pagerank,
            params=params,
        )
        return process_flag_result(result, args, formatter, start_time, "Direct mode")

    except Exception as e:
        logger.error(f"Flag command failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_flag_clear_command(args: argparse.Namespace, root_path: Path, config) -> int:
    """Handle flag clear command to remove [CODEN] comments."""
    start_time = time.time()
    formatter = FlagFormatter()

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
        elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
        if args.verbose:
            print(f"\n[Daemon mode] Clear completed in {elapsed_ms:.1f}ms", file=sys.stderr)
        return 0

    logger.warning("Daemon not available, falling back to direct analysis...")
    try:
        from ...mcp.flag_insertion import flag_clear

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
        elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
        if args.verbose:
            print(f"\n[Direct mode] Clear completed in {elapsed_ms:.1f}ms", file=sys.stderr)
        return 0

    except Exception as e:
        logger.error(f"Flag clear failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
