"""Helper functions for flag command processing."""
import argparse
import logging
import sys
import time
from pathlib import Path

from ...cli_metrics_contract import print_metric_output
from ...constants import (
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    MILLISECONDS_PER_SECOND,
)
from ...daemon.protocol import FlagParams
from ...formatters.flag_formatter import FlagFormatter
from ...utils.optional_deps import MissingDependencyError, require_feature
from ..utils import get_clone_mode

# Centralized flag mapping enables consistent validation and formatting
FLAG_ATTR_TO_SHORT: list[tuple[str, str]] = [
    ("hotspots", "-H"),
    ("propagation", "-P"),
    ("clones", "-C"),
    ("echo_comments", "-E"),
    ("dead_code", "-D"),
    ("tramp_data", "-T"),
    ("sensitive_values", "-S"),
]


def build_flag_active_flags(args: argparse.Namespace) -> list[str]:
    """Build list of active short-flag strings from args."""
    return [short for attr, short in FLAG_ATTR_TO_SHORT if getattr(args, attr, False)]


def validate_flag_args(args: argparse.Namespace, root_path: Path) -> int:
    """Validate flag command arguments. Returns 0 on success, 1 on error."""
    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}", file=sys.stderr)
        return 1
    if not root_path.is_dir():
        print(f"Error: Path is not a directory: {root_path}", file=sys.stderr)
        return 1

    temp_params = FlagParams(
        source_dir=str(root_path),
        hotspots=args.hotspots,
        propagation=args.propagation,
        clones=args.clones,
        echo_comments=args.echo_comments,
        dead_code=args.dead_code,
        tramp_data=args.tramp_data,
        sensitive_values=args.sensitive_values,
    )
    if error := temp_params.validate():
        print(f"Error: {error}", file=sys.stderr)
        return 1

    clone_mode = get_clone_mode(args)
    needs_semantic = args.echo_comments or (args.clones and clone_mode in ("semantic", "combined"))
    if needs_semantic:
        try:
            require_feature("semantic")
        except MissingDependencyError as e:
            print(str(e), file=sys.stderr)
            return 1

    return 0


def build_flag_params(args: argparse.Namespace, root_path: Path) -> FlagParams:
    """Build FlagParams from CLI args."""
    return FlagParams(
        source_dir=str(root_path),
        hotspots=args.hotspots,
        propagation=args.propagation,
        clones=args.clones,
        echo_comments=args.echo_comments,
        dead_code=args.dead_code,
        tramp_data=args.tramp_data,
        risk_threshold=args.risk_threshold,
        propagation_threshold=args.propagation_threshold,
        clone_threshold=args.clone_threshold,
        echo_threshold=args.echo_threshold,
        dead_code_threshold=args.dead_code_threshold,
        clone_mode=get_clone_mode(args),
        line_threshold=getattr(args, "line_threshold", DEFAULT_SYNTACTIC_LINE_THRESHOLD),
        func_threshold=getattr(args, "func_threshold", DEFAULT_SYNTACTIC_FUNC_THRESHOLD),
        dry_run=args.dry_run,
        backup=args.backup,
        verbose=args.verbose,
        exclude_tests=not args.include_tests,
        remove_comments=args.remove_comments,
        remove_dead_code=args.remove_dead_code,
        output_format=args.format,
        limit=args.limit,
        min_occurrences=args.min_occurrences,
        min_group_size=args.min_group_size,
        sensitive_values=args.sensitive_values,
        sensitive_threshold=args.sensitive_threshold,
        replace_value=getattr(args, "replace", None),
        sensitive_whitelist=getattr(args, "whitelist", None),
    )


def process_flag_result(
    result: dict,
    args: argparse.Namespace,
    formatter: FlagFormatter,
    start_time: float,
    mode_label: str,
) -> int:
    """Process flag_code result: format, print, and return exit code."""
    if "error" in result:
        logger = logging.getLogger(__name__)
        logger.error(f"Flag command error: {result['error']}")
        return 1

    items = result.get("items", [])

    if args.dry_run and args.limit is not None:
        items = items[: args.limit]

    formatted_output = formatter.format_items(items, args.format, args.reverse)
    stats_output = formatter.format_stats(result) if args.stats else None
    print_metric_output(formatted_output, stats_output, args.reverse)

    if args.verbose:
        elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
        count = result.get("flagged_count", 0)
        files = result.get("files_modified", 0)
        run_mode = "preview" if args.dry_run else "applied"
        print(
            f"\n[{mode_label}] Flagged {count} objects in {files} files ({run_mode}) in {elapsed_ms:.1f}ms",
            file=sys.stderr,
        )
    return 0
