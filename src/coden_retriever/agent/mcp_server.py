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

from ._constants import DEFAULT_MAX_RETRIES


def create_mcp_server(
    disabled_tools: Optional[list[str]] = None,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    tool_timeout: Optional[float] = None,
) -> MCPServerStdio:
    """Create an MCP server instance for coden-retriever.

    Uses -I (isolated mode) to prevent site-packages from printing
    debug output (e.g. [Monitor]) that corrupts MCP stdio protocol.

    Args:
        disabled_tools: Optional list of tool names to disable.
        timeout: Optional connection timeout in seconds (MCPServerStdio init).
        max_retries: Maximum retry attempts for tool calls (default: DEFAULT_MAX_RETRIES).
        tool_timeout: Optional global per-call tool timeout (seconds), forwarded to
            the server subprocess via CODEN_RETRIEVER_TOOL_TIMEOUT.

    Returns:
        Configured MCPServerStdio instance.
    """
    env = os.environ.copy()
    if disabled_tools:
        env["CODEN_RETRIEVER_DISABLED_TOOLS"] = ",".join(disabled_tools)
    if tool_timeout is not None:
        env["CODEN_RETRIEVER_TOOL_TIMEOUT"] = str(tool_timeout)

    # Use -OO for optimized mode (faster startup, no docstrings/asserts)
    # Use -I for isolated mode (prevents debug output from site-packages)
    args = ["-I", "-OO", "-m", "coden_retriever", "serve"]

    if timeout is not None:
        return MCPServerStdio(
            sys.executable, args=args, env=env, max_retries=max_retries, timeout=timeout
        )
    return MCPServerStdio(sys.executable, args=args, env=env, max_retries=max_retries)


@asynccontextmanager
async def mcp_server_context(
    disabled_tools: Optional[list[str]] = None,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    tool_timeout: Optional[float] = None,
) -> AsyncIterator[MCPServerStdio]:
    """Context manager for MCP server with automatic cleanup.

    Args:
        disabled_tools: Optional list of tool names to disable.
        timeout: Optional connection timeout in seconds (MCPServerStdio init).
        max_retries: Maximum retry attempts for tool calls (default: DEFAULT_MAX_RETRIES).
        tool_timeout: Optional global per-call tool timeout (seconds).

    Yields:
        Connected MCPServerStdio instance.
    """
    server = create_mcp_server(disabled_tools, timeout, max_retries, tool_timeout)
    async with server:
        yield server
