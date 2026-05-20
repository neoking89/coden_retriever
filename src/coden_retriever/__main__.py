"""Main entry point for CodenRetriever."""

import argparse
import importlib
import io
import logging
import sys
from pathlib import Path
from typing import Any

from .cli.utils import DefaultValueHelpFormatter
from .config_loader import (
    get_config,
    get_config_file,
    has_config_override,
    save_config,
    set_config_file,
)

logger = logging.getLogger(__name__)

KNOWN_SUBCOMMANDS = frozenset({
    "serve", "agent", "daemon", "flag", "config", "cache",
    "debug-availability", "reset", "architecture",
})

# Handler functions and parser factories — resolved on first access via
# __getattr__ below. Keeping them off module top is the entire point of T01:
# `coden reset` should not pay tree-sitter / pydantic-ai / SearchEngine import
# costs just to clear caches.
_LAZY_BINDINGS: dict[str, str] = {
    "handle_agent_command": ".cli.handlers.serve",
    "handle_architecture_command": ".cli.handlers.architecture",
    "handle_cache_command": ".cli.handlers.cache",
    "handle_config_command": ".cli.handlers.system",
    "handle_daemon_command": ".cli.handlers.daemon",
    "handle_debug_availability_command": ".cli.handlers.debug_availability",
    "handle_flag_clear_command": ".cli.handlers.flag",
    "handle_flag_command": ".cli.handlers.flag",
    "handle_reset_command": ".cli.handlers.system",
    "handle_search_command": ".cli.handlers.search",
    "handle_serve_command": ".cli.handlers.serve",
    "create_agent_parser": ".cli.arguments",
    "create_daemon_parser": ".cli.arguments",
    "create_flag_parser": ".cli.arguments",
    "create_search_parser": ".cli.arguments_search",
    "create_serve_parser": ".cli.arguments",
    "add_flag_arguments": ".cli.arguments_search",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy resolver. Mirrors agent/__init__.py's pattern.

    Critical: bare-name references inside main() use LOAD_GLOBAL, which is a
    __dict__ lookup that does NOT fire __getattr__. Every lazy binding must
    therefore be dispatched as `_self.<name>(...)` via sys.modules[__name__]
    so the attribute-access path hits this function (or the cached value).
    """
    module_path = _LAZY_BINDINGS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, package=__package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def _consume_agent_config_override(argv: list[str]) -> Path | None:
    """Strip `--config <path>` from argv when present in agent mode.

    Only fires for `coden -a/--agent ...` invocations — `--config` is an
    agent-mode convenience flag, not a global one. Returns the expanded
    path or None. Mutates argv in place so the downstream agent parser
    does not see (and re-parse) the consumed flag.

    Exits non-zero on `--config` with no value (matches argparse behavior).
    """
    if len(argv) < 2 or argv[1] not in ("-a", "--agent", "agent"):
        return None

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    parsed, remaining = pre_parser.parse_known_args(argv[2:])

    if parsed.config is None:
        return None

    argv[2:] = remaining
    return Path(parsed.config).expanduser().resolve()


def main() -> int:
    """Main CLI entry point."""
    _self = sys.modules[__name__]

    if sys.platform == "win32":
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    override_path = _consume_agent_config_override(sys.argv)
    if override_path is not None:
        set_config_file(override_path)

    try:
        config = get_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Error reading config file: {exc}", file=sys.stderr)
        return 2

    if not has_config_override():
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
            parser = _self.create_serve_parser(config)
            args = parser.parse_args(sys.argv[2:])
            return _self.handle_serve_command(args)

        if cmd == "agent":
            parser = _self.create_agent_parser(config)
            args = parser.parse_args(sys.argv[2:])
            return _self.handle_agent_command(args, config)

        if cmd == "daemon":
            daemon_parser = _self.create_daemon_parser()
            args = daemon_parser.parse_args(sys.argv[2:])
            if not args.daemon_action:
                daemon_parser.print_help()
                return 1
            return _self.handle_daemon_command(args)

        if cmd == "flag":
            flag_parser = _self.create_flag_parser(config)
            remaining_args = sys.argv[2:]
            if remaining_args and remaining_args[0] == "clear":
                args = flag_parser.parse_args(remaining_args)
                root_path = Path(args.root).resolve()
                return _self.handle_flag_clear_command(args, root_path, config)
            else:
                add_parser = argparse.ArgumentParser(
                    prog="coden flag",
                    description="Insert [CODEN] comments in source code based on analysis results",
                    formatter_class=DefaultValueHelpFormatter,
                )
                _self.add_flag_arguments(add_parser, config)
                args = add_parser.parse_args(remaining_args)
                root_path = Path(args.root).resolve()
                return _self.handle_flag_command(args, root_path, config)

        if cmd == "config":
            return _self.handle_config_command(sys.argv[2:])

        if cmd == "cache":
            return _self.handle_cache_command(sys.argv[2:])

        if cmd == "debug-availability":
            return _self.handle_debug_availability_command(sys.argv[2:])

        if cmd == "reset":
            reset_parser = argparse.ArgumentParser(
                prog="coden reset",
                description="Clear all caches, stop daemon, and reset config to defaults.",
                formatter_class=DefaultValueHelpFormatter,
            )
            reset_parser.add_argument(
                "--keep-config",
                action="store_true",
                help="Clear caches and stop daemon, but leave the config file untouched",
            )
            reset_args = reset_parser.parse_args(sys.argv[2:])
            return _self.handle_reset_command(keep_config=reset_args.keep_config)

        if cmd == "architecture":
            return _self.handle_architecture_command(sys.argv[2:])

        if (
            cmd not in KNOWN_SUBCOMMANDS
            and not cmd.startswith("-")
            and not any(sep in cmd for sep in ("/", "\\", "."))
            and not Path(cmd).exists()
        ):
            print(f"Error: unknown subcommand '{cmd}'", file=sys.stderr)
            print(f"Available subcommands: {', '.join(sorted(KNOWN_SUBCOMMANDS))}", file=sys.stderr)
            print("Run 'coden --help' for usage.", file=sys.stderr)
            return 2

    parser = _self.create_search_parser(config)
    args = parser.parse_args()
    return _self.handle_search_command(args, config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    sys.exit(main())
