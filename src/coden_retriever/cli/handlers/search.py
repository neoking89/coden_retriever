"""Handler for search command and output formatting."""
import argparse
import logging
import sys
import traceback
from pathlib import Path

from ...cache import CacheManager
from ...config import OutputFormat
from ...constants import OUTPUT_SEPARATOR_WIDTH
from ...daemon.client import try_daemon_search
from ...daemon.protocol import SearchParams
from ...pipeline import SearchConfig, SearchPipeline
from ...utils.optional_deps import MissingDependencyError, require_feature
from ..utils import normalize_limit

logger = logging.getLogger(__name__)


def print_search_output(
    formatted_output: str,
    tree_output: str | None,
    stats_output: str | None,
    reverse: bool,
) -> None:
    """Print search output in correct order based on reverse flag."""
    if reverse:
        print(formatted_output)
        if tree_output:
            print("\n" + "=" * OUTPUT_SEPARATOR_WIDTH + "\n")
            print(tree_output)
        if stats_output:
            print(stats_output, file=sys.stderr)
    else:
        if stats_output:
            print(stats_output, file=sys.stderr)
        if tree_output:
            print(tree_output)
            print("\n" + "=" * OUTPUT_SEPARATOR_WIDTH + "\n")
        print(formatted_output)


def format_semantic_search_header(query: str) -> str:
    """Format a header indicating semantic search mode is active."""
    lines = [
        "=" * OUTPUT_SEPARATOR_WIDTH,
        "SEMANTIC SEARCH MODE",
        "-" * OUTPUT_SEPARATOR_WIDTH,
        "Using Model2Vec embeddings for semantic similarity matching.",
        f'Query: "{query}"',
        "=" * OUTPUT_SEPARATOR_WIDTH,
    ]
    return "\n".join(lines)


def run_direct_search(args: argparse.Namespace, root_path: Path, cache: CacheManager, app_config) -> int:
    """Run search directly (fallback when daemon not available)."""
    args.limit = normalize_limit(args.limit)
    try:
        config = SearchConfig(
            root_path=root_path,
            query=args.query or "",
            token_limit=args.tokens,
            output_format=OutputFormat(args.format),
            enable_semantic=args.enable_semantic,
            model_path=app_config.search.semantic_model_path,
            show_deps=args.show_deps,
            dir_tree=args.dir_tree,
            map_mode=args.map,
            find_mode=args.find,
            limit=args.limit,
            verbose=args.verbose,
            show_stats=args.stats,
            reverse=args.reverse,
        )

        pipeline = SearchPipeline(config, cache=cache)
        engine = pipeline.create_engine()
        stats = engine.get_stats()

        if args.verbose:
            print(f"\n{stats}\n", file=sys.stderr)

        if stats.total_entities == 0:
            logger.warning("No code entities found")
            return 0

        result = pipeline.execute()

        if args.enable_semantic and args.query:
            print(format_semantic_search_header(args.query))
            print()

        print_search_output(
            formatted_output=result.formatted_output,
            tree_output=result.tree_output,
            stats_output=result.stats,
            reverse=args.reverse,
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_search_command(args: argparse.Namespace, config) -> int:
    """Handle search (default) mode."""
    from .clones import handle_clones_command
    from .dead_code import handle_dead_code_command
    from .echo_comments import handle_echo_comments_command
    from .hotspots import handle_hotspots_command
    from .propagation import handle_propagation_command
    from .sensitive_values import handle_sensitive_values_command
    from .tramp_data import handle_tramp_data_command

    args.limit = normalize_limit(args.limit)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root_path = Path(args.root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Invalid root path: {args.root}")
        return 1

    if args.enable_semantic:
        try:
            require_feature("semantic")
        except MissingDependencyError as e:
            print(str(e), file=sys.stderr)
            return 1

    if args.propagation:
        return handle_propagation_command(args, root_path, config)
    if args.clones:
        return handle_clones_command(args, root_path, config)
    if args.hotspots:
        return handle_hotspots_command(args, root_path, config)
    if args.echo_comments:
        return handle_echo_comments_command(args, root_path, config)
    if args.dead_code:
        return handle_dead_code_command(args, root_path, config)
    if args.tramp_data:
        return handle_tramp_data_command(args, root_path, config)
    if args.sensitive_values:
        return handle_sensitive_values_command(args, root_path, config)

    cache = CacheManager(
        root_path,
        enable_semantic=args.enable_semantic,
        model_path=config.search.semantic_model_path,
        verbose=args.verbose,
    )

    params = SearchParams(
        source_dir=str(root_path),
        query=args.query,
        enable_semantic=args.enable_semantic,
        model_path=config.search.semantic_model_path,
        limit=args.limit,
        tokens=args.tokens,
        show_deps=args.show_deps,
        output_format=args.format,
        find_identifier=args.find,
        map_mode=args.map or not args.query,
        dir_tree=args.dir_tree,
        stats=args.stats,
        reverse=args.reverse,
    )
    daemon_result = try_daemon_search(params, host=config.daemon.host, port=config.daemon.port)

    if daemon_result is not None:
        if args.enable_semantic and args.query:
            print(format_semantic_search_header(args.query))
            print()

        print_search_output(
            formatted_output=daemon_result.get("output", ""),
            tree_output=None,
            stats_output=daemon_result.get("stats_output") if args.stats else None,
            reverse=args.reverse,
        )

        if args.verbose:
            print(f"\n[Daemon mode] Search time: {daemon_result.get('search_time_ms', 0):.1f}ms, "
                  f"Results: {daemon_result.get('result_count', 0)}/{daemon_result.get('total_matched', 0)}, "
                  f"Tokens: {daemon_result.get('tokens_used', 0)}", file=sys.stderr)
        return 0

    return run_direct_search(args, root_path, cache, config)
