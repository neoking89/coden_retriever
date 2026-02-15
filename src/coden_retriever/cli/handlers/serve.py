"""Handlers for serve and agent commands."""
import logging
from pathlib import Path

from ...config_loader import save_config
from ...utils.optional_deps import MissingDependencyError, require_feature
from ..utils import get_asyncio

logger = logging.getLogger(__name__)


def handle_serve_command(args) -> int:
    """Handle 'serve' subcommand."""
    try:
        require_feature("mcp")
    except MissingDependencyError as e:
        print(str(e), file=__import__('sys').stderr)
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from ...mcp.server import create_mcp_server
    mcp = create_mcp_server()
    if mcp:
        logger.info(f"Starting MCP server with {args.transport} transport...")
        show_banner = args.transport != "stdio"
        if args.transport in ["http", "sse", "streamable-http"]:
            logger.info(f"Server will be available at: http://{args.host}:{args.port}")
            mcp.run(transport=args.transport, host=args.host, port=args.port, show_banner=show_banner)
        else:
            mcp.run(transport=args.transport, show_banner=show_banner)
        return 0
    return 1


def handle_agent_command(args, config) -> int:
    """Handle 'agent' subcommand."""
    try:
        require_feature("agent")
    except MissingDependencyError as e:
        print(str(e), file=__import__('sys').stderr)
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from ...agent import run_interactive

    root_path = Path(args.root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Invalid root path: {root_path}")
        return 1

    user_provided_model = args.model != config.model.default
    user_provided_base_url = args.base_url != config.model.base_url
    user_provided_mcp_timeout = args.mcp_timeout != config.agent.mcp_server_timeout

    if user_provided_model or user_provided_base_url or user_provided_mcp_timeout:
        if user_provided_model:
            config.model.default = args.model
        if user_provided_base_url:
            config.model.base_url = args.base_url
        if user_provided_mcp_timeout:
            config.agent.mcp_server_timeout = args.mcp_timeout
        save_config(config)

    try:
        get_asyncio().run(run_interactive(
            str(root_path),
            args.model,
            args.base_url,
            args.max_steps,
            disabled_tools=config.agent.disabled_tools,
        ))
    except KeyboardInterrupt:
        pass
    return 0
