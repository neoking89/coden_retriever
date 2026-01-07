"""
MCP (Model Context Protocol) module for CodenRetriever.

This module provides MCP servers with different tool configurations:
- create_mcp_server(): Full server with all tools (default)
- create_code_search_server(): Server with only code search tools
- create_dynamic_tools_server(): Server with only dynamic tools

Public API:
    Servers:
        - create_mcp_server
        - create_code_search_server
        - create_dynamic_tools_server

    Tool Registration:
        - register_code_search_tools
        - register_dynamic_tools
        - register_file_edit_tools
        - register_all_tools (registers all tools)

    Tool Filtering:
        - ToolFilter
        - ToolMetadata
        - FilteredTool
        - FilterResult
        - CORE_TOOLS
        - is_tool_filter_enabled
        - create_tool_filter_from_functions
        - display_filtered_tools
"""

from .code_search import register_code_search_tools
from .code_search_server import create_code_search_server
from .dynamic_tools import register_dynamic_tools
from .dynamic_tools_server import create_dynamic_tools_server
from .file_edit import register_file_edit_tools
from .server import create_mcp_server
from .tool_filter import (
    CORE_TOOLS,
    FilteredTool,
    FilterResult,
    ToolFilter,
    ToolMetadata,
    create_tool_filter_from_functions,
    display_filtered_tools,
    is_tool_filter_enabled,
)


def register_all_tools(mcp) -> None:
    """Register all MCP tools on the given FastMCP instance.

    This is a convenience function that registers all available tools:
    code search, dynamic tools, and file editing tools.

    Args:
        mcp: FastMCP instance to register tools on.
    """
    register_code_search_tools(mcp)
    register_dynamic_tools(mcp)
    register_file_edit_tools(mcp)


__all__ = [
    # Server creation functions
    "create_mcp_server",
    "create_code_search_server",
    "create_dynamic_tools_server",
    # Tool registration functions
    "register_code_search_tools",
    "register_dynamic_tools",
    "register_file_edit_tools",
    "register_all_tools",
    # Tool filtering
    "CORE_TOOLS",
    "ToolFilter",
    "ToolMetadata",
    "FilteredTool",
    "FilterResult",
    "is_tool_filter_enabled",
    "create_tool_filter_from_functions",
    "display_filtered_tools",
]
