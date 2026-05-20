"""Argument parser creation for serve, agent, daemon, and flag commands."""
import argparse

from ..config_loader import get_config
from ..constants import MCP_DEFAULT_HTTP_PORT
from .utils import DefaultValueHelpFormatter


def create_serve_parser(config) -> argparse.ArgumentParser:
    """Create parser for 'serve' subcommand."""
    parser = argparse.ArgumentParser(
        prog="coden serve",
        description="Run as MCP server.",
        formatter_class=DefaultValueHelpFormatter,
    )
    parser.add_argument("--transport", choices=["stdio", "http", "sse", "streamable-http"],
                        default="stdio", help="Transport protocol")
    parser.add_argument("--host", type=str, default=config.daemon.host,
                        help="Host address (for http/sse transport)")
    parser.add_argument("--port", "-p", type=int, default=MCP_DEFAULT_HTTP_PORT,
                        help="Port (for http/sse transport)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def create_agent_parser(config) -> argparse.ArgumentParser:
    """Create parser for 'agent' subcommand."""
    parser = argparse.ArgumentParser(
        prog="coden agent",
        description="Interactive coding agent with ReAct reasoning.",
        formatter_class=DefaultValueHelpFormatter,
    )
    parser.add_argument("root", nargs="?", default=".",
                        help="Repository root directory")
    parser.add_argument("--model", "-m", type=str, default=config.model.default,
                        help="LLM model (ollama:model, openai:model, or model with --base-url)")
    parser.add_argument("--base-url", type=str, default=config.model.base_url,
                        help="Base URL for OpenAI-compatible endpoints")
    parser.add_argument("--max-steps", type=int, default=config.agent.max_steps,
                        help="Max tool calls per query")
    parser.add_argument("--mcp-timeout", type=float, default=config.agent.mcp_server_timeout,
                        help="MCP server startup timeout (seconds)")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="Path to custom config JSON. Create via `coden config new <path>`.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def _create_common_daemon_parser() -> argparse.ArgumentParser:
    """Create parent parser with common daemon arguments."""
    config = get_config()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--daemon-host", default=config.daemon.host, help="Daemon host address")
    parser.add_argument("--daemon-port", type=int, default=config.daemon.port, help="Daemon port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def _create_daemon_settings_parser() -> argparse.ArgumentParser:
    """Create parent parser with daemon settings arguments."""
    config = get_config()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-projects", type=int, default=config.daemon.max_projects,
                        help="Max projects to cache")
    parser.add_argument("--idle-timeout", type=str,
                        help="Auto-shutdown after idle (e.g., 30m, 1h)")
    parser.add_argument("--no-watch", action="store_true",
                        help="Disable automatic file watching for index updates")
    parser.add_argument("--daemon-timeout", type=float, default=config.daemon.daemon_timeout,
                        help="Socket timeout for client connections (seconds)")
    return parser


def create_daemon_parser() -> argparse.ArgumentParser:
    """Create parser for daemon commands."""
    common_parser = _create_common_daemon_parser()
    settings_parser = _create_daemon_settings_parser()

    parser = argparse.ArgumentParser(
        prog="coden-retriever daemon",
        description="Manage the daemon for fast responses",
        formatter_class=DefaultValueHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="daemon_action", help="Daemon action")

    subparsers.add_parser(
        "start",
        parents=[common_parser, settings_parser],
        help="Start daemon in background"
    )
    subparsers.add_parser(
        "stop",
        parents=[common_parser],
        help="Stop the daemon"
    )
    subparsers.add_parser(
        "status",
        parents=[common_parser],
        help="Show daemon status"
    )
    subparsers.add_parser(
        "restart",
        parents=[common_parser, settings_parser],
        help="Restart the daemon"
    )
    subparsers.add_parser(
        "run",
        parents=[common_parser, settings_parser],
        help="Run daemon in foreground (for debugging)"
    )

    clear_cache_parser = subparsers.add_parser(
        "clear-cache",
        parents=[common_parser],
        help="Clear daemon cache"
    )
    clear_cache_parser.add_argument("clear_path", nargs="?", help="Path to clear from cache")
    clear_cache_parser.add_argument("--all", dest="clear_all", action="store_true",
                                    help="Clear all cached projects")

    return parser


def create_flag_parser(config) -> argparse.ArgumentParser:
    """Create parser for flag command with clear subcommand."""
    from .arguments_search import add_flag_arguments

    parser = argparse.ArgumentParser(
        prog="coden flag",
        description="Insert [CODEN] comments in source code based on analysis results",
        formatter_class=DefaultValueHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="flag_action", help="Flag action")

    flag_parser = subparsers.add_parser(
        "add",
        help="Add [CODEN] flags to code (default if path given)"
    )
    add_flag_arguments(flag_parser, config)

    clear_parser = subparsers.add_parser(
        "clear",
        help="Remove all [CODEN] flags from code"
    )
    clear_parser.add_argument("root", nargs="?", default=".",
                              help="Repository root directory")
    clear_parser.add_argument("--dry-run", action="store_true",
                              help="Preview changes without modifying files")
    clear_parser.add_argument("-v", "--verbose", action="store_true",
                              help="Verbose output")
    clear_parser.add_argument("-f", "--format", default="tree",
                              choices=["tree", "json"],
                              help="Output format")
    clear_parser.add_argument("-r", "--reverse", action="store_true",
                              help="Reverse output order")
    clear_parser.add_argument("--stats", action="store_true",
                              help="Show summary statistics")
    clear_parser.add_argument("--no-daemon", dest="no_daemon", action="store_true",
                              help="Skip the daemon for this invocation; use the in-process direct path.")

    return parser
