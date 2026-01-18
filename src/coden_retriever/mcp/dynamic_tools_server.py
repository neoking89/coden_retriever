"""
Dynamic Tools MCP Server module.

Provides the Model Context Protocol server for CodenRetriever dynamic tools functionality only.
"""
from typing import TYPE_CHECKING

from .constants import DYNAMIC_TOOLS_INSTRUCTIONS, SERVER_NAME_DYNAMIC_TOOLS

if TYPE_CHECKING:
    from fastmcp import FastMCP
from .dynamic_tools import register_dynamic_tools
from .server_factory import create_mcp_server_with_config


def create_dynamic_tools_server() -> "FastMCP | None":
    """Create an MCP server with only dynamic tools."""
    return create_mcp_server_with_config(
        server_name=SERVER_NAME_DYNAMIC_TOOLS,
        instructions=DYNAMIC_TOOLS_INSTRUCTIONS,
        install_dependency="dynamic-tools",
        register_functions=[register_dynamic_tools],
    )
