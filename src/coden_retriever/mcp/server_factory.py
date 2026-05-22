"""
MCP Server Factory.

Provides a generic factory function for creating MCP servers with different configurations.
Eliminates code duplication across server creation modules.
"""
import logging
from typing import Callable

from fastmcp import FastMCP

from .tool_timeout import WORKER_SAFE_ATTR, wrap_with_timeout

logger = logging.getLogger(__name__)


def _install_tool_timeout(mcp: FastMCP, timeout_s: float) -> None:
    """Patch ``mcp.tool`` so worker-safe tools get the kill-on-timeout wrapper.

    Single chokepoint: every built-in register fn and the runtime
    ``create_dynamic_tool`` path register via ``mcp.tool(...)(func)``. Patching
    the bound method before the register loop routes each registration based on the
    declarative :data:`WORKER_SAFE_ATTR` marker: marked stateless tools are wrapped to
    run in the worker subprocess (``wrap_with_timeout`` is idempotent), while unmarked
    and stateful tools register unchanged and run in-process. ``decorator_mode``
    defaults to ``"function"``, so ``original_tool(...)(fn)`` returns ``fn`` — the
    ``decorator(wrap(func))`` composition below is the correct order.
    """
    original_tool = mcp.tool

    def tool(*args, **kwargs):  # no functools.wraps: shim is never introspected as mcp.tool
        decorator = original_tool(*args, **kwargs)

        def register(func):
            if getattr(func, WORKER_SAFE_ATTR, False):
                return decorator(wrap_with_timeout(func, timeout_s))
            return decorator(func)

        return register

    mcp.tool = tool  # type: ignore[method-assign]


def register_health_endpoint(mcp: FastMCP) -> None:
    """Register a health check endpoint for HTTP transport.

    Note: Starlette is a transitive dependency of MCP/FastMCP, so it's always available.

    Args:
        mcp: FastMCP server instance
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        """Health check endpoint for container orchestration and load balancers."""
        return JSONResponse({"status": "healthy", "service": mcp.name})


def create_mcp_server_with_config(
    server_name: str,
    instructions: str,
    register_functions: list[Callable],
    tool_timeout: float,
) -> FastMCP:
    """Generic factory function for creating MCP servers.

    Args:
        server_name: Name of the MCP server
        instructions: Instructions text for the server
        register_functions: List of registration functions to call with the mcp instance
        tool_timeout: Per-call timeout (seconds) applied globally to every tool.

    Returns:
        FastMCP instance.
    """
    mcp = FastMCP(
        name=server_name,
        instructions=instructions
    )

    # Wrap mcp.tool BEFORE any registration so every tool inherits the timeout.
    _install_tool_timeout(mcp, tool_timeout)

    # Register health endpoint for HTTP transport
    register_health_endpoint(mcp)

    # Register all tools
    for register_func in register_functions:
        register_func(mcp)

    return mcp
