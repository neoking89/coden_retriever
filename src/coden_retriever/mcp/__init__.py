"""
MCP (Model Context Protocol) module for CodenRetriever.

Provides the full MCP server with all tools (code search, debugging,
file editing, dead code, clones, etc.).

Public API:
    Server:
        - create_mcp_server (full server with all tools)

    Tool Registration:
        - register_code_search_tools
        - register_dynamic_tools
        - register_file_edit_tools
        - register_all_tools (registers all tools)

    Tool Filtering:
        - LLMToolRouter
        - ToolMetadata
        - FilteredTool
        - FilterResult
        - CORE_TOOLS
        - display_filtered_tools
"""

def __getattr__(name: str):
    """Lazy load heavy modules only when accessed."""
    if name == "register_code_search_tools":
        from .code_search import register_code_search_tools
        return register_code_search_tools
    if name == "register_dynamic_tools":
        from .dynamic_tools import register_dynamic_tools
        return register_dynamic_tools
    if name == "register_file_edit_tools":
        from .file_edit import register_file_edit_tools
        return register_file_edit_tools
    if name == "create_mcp_server":
        from .server import create_mcp_server
        return create_mcp_server
    if name in ("CORE_TOOLS", "FilteredTool", "FilterResult", "TOOL_QUERY_DESCRIPTIONS",
                "ToolMetadata", "display_filtered_tools"):
        from . import tool_filter
        return getattr(tool_filter, name)
    if name == "LLMToolRouter":
        from .llm_tool_router import LLMToolRouter
        return LLMToolRouter
    if name == "register_all_tools":
        return register_all_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_all_tools(mcp) -> None:
    """Register all MCP tools on the given FastMCP instance.

    This is a convenience function that registers all available tools:
    code search, dynamic tools, and file editing tools.

    Args:
        mcp: FastMCP instance to register tools on.
    """
    from .code_search import register_code_search_tools
    from .dynamic_tools import register_dynamic_tools
    from .file_edit import register_file_edit_tools
    register_code_search_tools(mcp)
    register_dynamic_tools(mcp)
    register_file_edit_tools(mcp)


__all__ = [
    # Server creation functions
    "create_mcp_server",
    # Tool registration functions
    "register_code_search_tools",
    "register_dynamic_tools",
    "register_file_edit_tools",
    "register_all_tools",
    # Tool filtering
    "CORE_TOOLS",
    "TOOL_QUERY_DESCRIPTIONS",
    "LLMToolRouter",
    "ToolMetadata",
    "FilteredTool",
    "FilterResult",
    "display_filtered_tools",
]
