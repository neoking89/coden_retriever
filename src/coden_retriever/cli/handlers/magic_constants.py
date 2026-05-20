"""Handler for magic constant detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...config_loader import AppConfig, daemon_enabled
from ...constants import MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT, MAGIC_CONSTANT_MAX_RESULTS
from ...daemon.client import try_daemon_magic_constants
from ...daemon.protocol import MagicConstantParams
from ...formatters.magic_constant_formatter import (
    MagicConstantFormatter,
    format_magic_constant_parameters_header,
)
from ..utils import get_asyncio, normalize_limit

logger = logging.getLogger(__name__)


# Internal limit used when the user's --constant-type filter needs the full pool.
# Without this, the detector returns only DEFAULT limit items, which may contain
# zero matches after type filtering — yielding fewer results than -n requested.
def _compute_internal_limit(user_limit: int | None, constant_type: str = "all") -> int:
    """Compute the internal detector limit based on user options."""
    needs_full_pool = user_limit is None or constant_type != "all"
    return MAGIC_CONSTANT_MAX_RESULTS if needs_full_pool else MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT


def _filter_results(
    results: list[dict],
    min_occurrences: int,
    limit: int | None,
    constant_type: str = "all",
) -> list[dict]:
    """Filter magic constant results by min occurrences, type, and apply limit."""
    filtered = [r for r in results if r.get("count", 0) >= min_occurrences]
    if constant_type != "all":
        filtered = [r for r in filtered if r.get("node_type_category") == constant_type]
    return apply_defensive_limit(filtered, limit)


def _process_result(
    result: dict,
    formatter: MagicConstantFormatter,
    args: argparse.Namespace,
    start_time: float,
    mode: str,
) -> int:
    """Process magic constant detection result and output."""
    if "error" in result:
        logger.error("Magic constant detection error: %s", result["error"])
        return 1

    all_constants = result.get("magic_constants", [])
    summary = result.get("summary", {})

    constant_type = getattr(args, "constant_type", "all")
    constants = _filter_results(
        all_constants, args.min_constant_occurrences, args.limit, constant_type,
    )

    formatted_output = formatter.format_items(constants, args.format, args.reverse)
    stats_output = formatter.format_stats(summary) if args.stats else None
    print_metric_output(formatted_output, stats_output, args.reverse)

    elapsed_ms = (time.time() - start_time) * 1000
    if args.verbose:
        print(
            f"\n[{mode} mode] Magic constant detection: {elapsed_ms:.1f}ms, "
            f"Constants: {len(constants)}",
            file=sys.stderr,
        )
    return 0


def _run_direct(
    root_path: Path,
    formatter: MagicConstantFormatter,
    args: argparse.Namespace,
    start_time: float,
) -> int:
    """Run magic constant detection directly (non-daemon)."""
    from ...mcp.magic_constants import detect_magic_constants_tool
    try:
        constant_type = getattr(args, "constant_type", "all")
        internal_limit = _compute_internal_limit(args.limit, constant_type)
        result = get_asyncio().run(detect_magic_constants_tool(
            root_directory=str(root_path),
            min_occurrences=args.min_constant_occurrences,
            min_files=args.min_constant_files,
            limit=internal_limit,
            exclude_tests=True,
        ))
        return _process_result(result, formatter, args, start_time, "Direct")
    except Exception as e:
        logger.error("Magic constant detection failed: %s", e)
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_magic_constants_command(
    args: argparse.Namespace,
    root_path: Path,
    config: AppConfig,
) -> int:
    """Handle magic constant detection (-K/--magic-constants flag)."""
    args.limit = normalize_limit(args.limit)
    start_time = time.time()
    formatter = MagicConstantFormatter()

    header = format_magic_constant_parameters_header(
        min_occurrences=args.min_constant_occurrences,
        min_files=args.min_constant_files,
        exclude_tests=True,
        limit=args.limit,
        constant_type=getattr(args, "constant_type", "all"),
    )
    print(header)
    print()

    constant_type = getattr(args, "constant_type", "all")
    daemon_limit = _compute_internal_limit(args.limit, constant_type)
    params = MagicConstantParams(
        source_dir=str(root_path),
        min_occurrences=args.min_constant_occurrences,
        min_files=args.min_constant_files,
        limit=daemon_limit,
        exclude_tests=True,
    )

    daemon_result = None
    if daemon_enabled(args):
        daemon_result = try_daemon_magic_constants(
            params, address=config.daemon.address,
        )
    if daemon_result is not None:
        return _process_result(daemon_result, formatter, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    return _run_direct(root_path, formatter, args, start_time)
