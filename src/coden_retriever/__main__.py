"""Main entry point for CodenRetriever."""

import argparse
import io
import logging
import sys
from pathlib import Path

from .cli.arguments import (
    create_agent_parser,
    create_daemon_parser,
    create_flag_parser,
    create_serve_parser,
)
from .cli.arguments_search import add_flag_arguments, create_search_parser
from .cli.handlers.cache import handle_cache_command
from .cli.handlers.daemon import handle_daemon_command
from .cli.handlers.debug_availability import handle_debug_availability_command
from .cli.handlers.flag import handle_flag_clear_command, handle_flag_command
from .cli.handlers.search import handle_search_command
from .cli.handlers.serve import handle_agent_command, handle_serve_command
from .cli.handlers.system import handle_config_command, handle_reset_command
from .cli.utils import DefaultValueHelpFormatter
from .config_loader import get_config, get_config_file, save_config

logger = logging.getLogger(__name__)


def main() -> int:
    """Main CLI entry point."""
    if sys.platform == "win32":
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    config = get_config()

    config_file = get_config_file()
    if not config_file.exists():
        print(
            f"Warning: Config file was missing, recreating at {config_file}",
            file=sys.stderr,
        )
        save_config(config)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd in ("-a", "--agent"):
            sys.argv[1] = "agent"
            cmd = "agent"

        if cmd == "serve":
            parser = create_serve_parser(config)
            args = parser.parse_args(sys.argv[2:])
            return handle_serve_command(args)

        if cmd == "agent":
            parser = create_agent_parser(config)
            args = parser.parse_args(sys.argv[2:])
            return handle_agent_command(args, config)

        if cmd == "daemon":
            daemon_parser = create_daemon_parser()
            args = daemon_parser.parse_args(sys.argv[2:])
            if not args.daemon_action:
                daemon_parser.print_help()
                return 1
            return handle_daemon_command(args)

        if cmd == "flag":
            flag_parser = create_flag_parser(config)
            remaining_args = sys.argv[2:]
            if remaining_args and remaining_args[0] == "clear":
                args = flag_parser.parse_args(remaining_args)
                root_path = Path(args.root).resolve()
                return handle_flag_clear_command(args, root_path, config)
            else:
                add_parser = argparse.ArgumentParser(
                    prog="coden flag",
                    description="Insert [CODEN] comments in source code based on analysis results",
                    formatter_class=DefaultValueHelpFormatter,
                )
                add_flag_arguments(add_parser, config)
                args = add_parser.parse_args(remaining_args)
                root_path = Path(args.root).resolve()
                return handle_flag_command(args, root_path, config)

        if cmd == "config":
            return handle_config_command(sys.argv[2:])

        if cmd == "cache":
            return handle_cache_command(sys.argv[2:])

        if cmd == "debug-availability":
            return handle_debug_availability_command(sys.argv[2:])

        if cmd == "reset":
            return handle_reset_command()

    parser = create_search_parser(config)
    args = parser.parse_args()
    return handle_search_command(args, config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    sys.exit(main())
