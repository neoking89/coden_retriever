"""Handler for propagation cost analysis command."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from ...cli_metrics_contract import apply_defensive_limit, print_metric_output
from ...daemon.client import try_daemon_propagation_cost
from ...daemon.protocol import PropagationCostParams
from ...formatters import PropagationFormatter
from ..utils import get_asyncio, normalize_limit

logger = logging.getLogger(__name__)


def _filter_propagation_by_threshold(result: dict, threshold: float) -> dict:
    """Mark modules above threshold in propagation result.

    Annotates (not filters) modules above threshold for display purposes.
    """
    if "module_breakdown" not in result:
        return result

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

    from ...formatters.propagation_formatter import format_propagation_parameters_header
    exclude_tests = not getattr(args, 'include_tests', False)
    header = format_propagation_parameters_header(
        propagation_threshold=args.propagation_threshold,
        exclude_tests=exclude_tests,
        limit=args.limit,
    )
    print(header)
    print()

    params = PropagationCostParams(
        source_dir=str(root_path),
        include_breakdown=args.breakdown,
        show_critical_paths=args.critical_paths,
        exclude_tests=exclude_tests,
        token_limit=args.tokens,
    )

    daemon_result = try_daemon_propagation_cost(params, host=config.daemon.host, port=config.daemon.port)

    if daemon_result is not None:
        if "error" in daemon_result:
            logger.error(f"Propagation cost error: {daemon_result['error']}")
            return 1

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
        from ...mcp.propagation_cost import propagation_cost as mcp_propagation_cost

        result = get_asyncio().run(mcp_propagation_cost(
            root_directory=str(root_path),
            include_breakdown=args.breakdown,
            show_critical_paths=args.critical_paths,
            exclude_tests=exclude_tests,
            token_limit=args.tokens,
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
