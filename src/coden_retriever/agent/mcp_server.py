"""MCP server setup utilities.

Provides helpers for creating and configuring MCP server connections.
Centralizes the MCP server creation pattern used across the agent.

Public API:
    create_mcp_server: Factory function for creating MCP server instances.
    mcp_server_context: Async context manager for managed server lifecycle.
"""

__all__ = ["create_mcp_server", "mcp_server_context"]

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from pydantic_ai.mcp import MCPServerStdio

from ..constants import DEFAULT_MAX_RETRIES


def create_mcp_server(
    disabled_tools: Optional[list[str]] = None,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> MCPServerStdio:
    """Create an MCP server instance for coden-retriever.

    Uses -I (isolated mode) to prevent site-packages from printing
    debug output (e.g. [Monitor]) that corrupts MCP stdio protocol.

    Args:
        disabled_tools: Optional list of tool names to disable.
        timeout: Optional timeout in seconds.
        max_retries: Maximum retry attempts for tool calls (default: DEFAULT_MAX_RETRIES).

    Returns:
        Configured MCPServerStdio instance.
    """
    env = os.environ.copy()
    if disabled_tools:
        env["CODEN_RETRIEVER_DISABLED_TOOLS"] = ",".join(disabled_tools)

    # Use -OO for optimized mode (faster startup, no docstrings/asserts)
    # Use -I for isolated mode (prevents debug output from site-packages)
    kwargs = {
        "args": ["-I", "-OO", "-m", "coden_retriever", "serve"],
        "env": env,
        "max_retries": max_retries,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    return MCPServerStdio(sys.executable, **kwargs)


@asynccontextmanager
async def mcp_server_context(
    disabled_tools: Optional[list[str]] = None,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AsyncIterator[MCPServerStdio]:
    """Context manager for MCP server with automatic cleanup.

    Args:
        disabled_tools: Optional list of tool names to disable.
        timeout: Optional timeout in seconds.
        max_retries: Maximum retry attempts for tool calls (default: DEFAULT_MAX_RETRIES).

    Yields:
        Connected MCPServerStdio instance.
    """
    server = create_mcp_server(disabled_tools, timeout, max_retries)
    async with server:
        yield server
