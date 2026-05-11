"""Handler for sensitive value detection command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...config_loader import AppConfig
from ...daemon.client import try_daemon_sensitive_values
from ...daemon.protocol import SensitiveValueParams
from ...formatters import SensitiveValueFormatter
from ...formatters.sensitive_value_formatter import format_sensitive_value_parameters_header
from ..utils import get_asyncio, normalize_limit

logger = logging.getLogger(__name__)


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
    from ...mcp.sensitive_values import detect_sensitive_values_tool
    try:
        result = get_asyncio().run(detect_sensitive_values_tool(
            root_directory=str(root_path),
            confidence_threshold=args.sensitive_threshold,
            limit=args.limit,
            exclude_tests=True,
            replace_value=getattr(args, "replace", None),
            whitelist=getattr(args, "whitelist", None),
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

    whitelist = getattr(args, "whitelist", None)

    header = format_sensitive_value_parameters_header(
        confidence_threshold=args.sensitive_threshold,
        exclude_tests=True,
        limit=args.limit,
        replace_value=getattr(args, "replace", None),
        whitelist=whitelist,
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
        whitelist=whitelist,
    )

    daemon_result = try_daemon_sensitive_values(params, address=config.daemon.address)
    if daemon_result is not None:
        return _process_sensitive_value_result(daemon_result, formatter, args, start_time, "Daemon")

    logger.warning("Daemon not available, falling back to direct analysis...")
    return _run_direct_sensitive_values(root_path, formatter, args, start_time)
