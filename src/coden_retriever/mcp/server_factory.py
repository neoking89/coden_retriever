"""
MCP Server Factory.

Provides a generic factory function for creating MCP servers with different configurations.
Eliminates code duplication across server creation modules.

Requires the 'mcp' extra:
    pip install 'coden-retriever[mcp]'
"""
import logging
from typing import TYPE_CHECKING, Callable, Optional

from ..utils.optional_deps import MissingDependencyError

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_health_endpoint(mcp: "FastMCP") -> None:
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
    install_dependency: str,
    register_functions: list[Callable],
) -> Optional["FastMCP"]:
    """Generic factory function for creating MCP servers.

    Args:
        server_name: Name of the MCP server
        instructions: Instructions text for the server
        install_dependency: Dependency identifier for error message (e.g., 'code-search', 'all')
        register_functions: List of registration functions to call with the mcp instance

    Returns:
        FastMCP instance or None if FastMCP is not installed

    Raises:
        MissingDependencyError: If FastMCP is not installed
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        raise MissingDependencyError("mcp")

    mcp = FastMCP(
        name=server_name,
        instructions=instructions
    )

    # Register health endpoint for HTTP transport
    register_health_endpoint(mcp)

    # Register all tools
    for register_func in register_functions:
        register_func(mcp)

    return mcp
