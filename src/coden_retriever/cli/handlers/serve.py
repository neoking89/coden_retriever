"""Handlers for serve and agent commands."""
import logging
import sys
from pathlib import Path

from ...config_loader import daemon_enabled, save_config
from ..utils import get_asyncio

logger = logging.getLogger(__name__)


def _resolve_print_prompt(value: str) -> str:
    """Resolve the prompt for `coden -a -p`.

    A non-empty flag value is used as-is. A bare `-p` (value `""`) reads
    stdin when it is piped (`echo q | coden -a -p`), else returns `""` so
    the caller can error out cleanly.
    """
    if value:
        return value
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def handle_serve_command(args) -> int:
    """Handle 'serve' subcommand."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from ...mcp.server import create_mcp_server
    mcp = create_mcp_server()
    logger.info(f"Starting MCP server with {args.transport} transport...")
    show_banner = args.transport != "stdio"
    if args.transport in ["http", "sse", "streamable-http"]:
        logger.info(f"Server will be available at: http://{args.host}:{args.port}")
        mcp.run(transport=args.transport, host=args.host, port=args.port, show_banner=show_banner)
    else:
        mcp.run(transport=args.transport, show_banner=show_banner)
    return 0


def handle_agent_command(args, config) -> int:
    """Handle 'agent' subcommand."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from ...agent import run_interactive, run_once

    root_path = Path(args.root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Invalid root path: {root_path}")
        return 1

    user_provided_model = args.model != config.model.default
    user_provided_base_url = args.base_url != config.model.base_url
    user_provided_mcp_timeout = args.mcp_timeout != config.agent.mcp_server_timeout

    def _apply_model_overrides() -> None:
        if user_provided_model:
            config.model.default = args.model
        if user_provided_base_url:
            config.model.base_url = args.base_url
        if user_provided_mcp_timeout:
            config.agent.mcp_server_timeout = args.mcp_timeout

    start_daemon = daemon_enabled(args)  # --no-daemon > env > config > True

    if args.prompt is not None:  # print mode
        prompt_text = _resolve_print_prompt(args.prompt)
        if not prompt_text.strip():
            logger.error("No prompt provided for --print")
            return 1
        # Apply overrides for this run, but do NOT persist them: a scripted
        # one-shot should not rewrite the user's config on every call.
        _apply_model_overrides()
        return get_asyncio().run(run_once(
            str(root_path),
            args.model,
            args.base_url,
            args.max_steps,
            prompt_text,
            disabled_tools=config.agent.disabled_tools,
            start_daemon=start_daemon,
        ))

    if user_provided_model or user_provided_base_url or user_provided_mcp_timeout:
        _apply_model_overrides()
        save_config(config)

    try:
        get_asyncio().run(run_interactive(
            str(root_path),
            args.model,
            args.base_url,
            args.max_steps,
            disabled_tools=config.agent.disabled_tools,
            start_daemon=start_daemon,
        ))
    except KeyboardInterrupt:
        pass
    return 0
