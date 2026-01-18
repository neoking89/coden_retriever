"""
Code Search MCP Server module.

Provides the Model Context Protocol server for CodenRetriever code search functionality only.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

from .code_search import register_code_search_tools
from .constants import CODE_SEARCH_INSTRUCTIONS, SERVER_NAME_CODE_SEARCH
from .server_factory import create_mcp_server_with_config


def create_code_search_server() -> "FastMCP | None":
    """Create an MCP server with only code search tools."""
    return create_mcp_server_with_config(
        server_name=SERVER_NAME_CODE_SEARCH,
        instructions=CODE_SEARCH_INSTRUCTIONS,
        install_dependency="code-search",
        register_functions=[register_code_search_tools],
    )
