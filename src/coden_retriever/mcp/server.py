"""
MCP Server module.

Provides the Model Context Protocol server for CodenRetriever with all tools.
This is the default server that includes both code search and dynamic tools.
"""
import os

from .clone_detection import register_clone_detection_tools
from .code_search import register_code_search_tools
from .constants import FULL_SERVER_INSTRUCTIONS, SERVER_NAME_FULL, _is_dynamic_tools_enabled
from .debug_trace import register_debug_tools
from .dynamic_tools import register_dynamic_tools
from .echo_comments import register_echo_comment_tools
from .file_edit import register_file_edit_tools
from .flag_insertion import register_flag_tools
from .graph_analysis import register_graph_analysis_tools
from .propagation_cost import register_propagation_cost_tools
from .server_factory import create_mcp_server_with_config


def get_disabled_tools() -> set[str]:
    """Get the set of disabled tools from environment variable.

    Reads CODEN_RETRIEVER_DISABLED_TOOLS env var (comma-separated list).
    """
    disabled_str = os.environ.get("CODEN_RETRIEVER_DISABLED_TOOLS", "")
    if not disabled_str:
        return set()
    return set(name.strip() for name in disabled_str.split(",") if name.strip())


def create_mcp_server() -> "FastMCP":
    """Create an MCP server with all tools (code search + dynamic tools).

    This is the default server that includes both code search and dynamic tools functionality.
    For specialized servers, use create_code_search_server() or create_dynamic_tools_server().

    Dynamic tools require enable_dynamic_tools = true in pyproject.toml [tool.coden-retriever].
    Respects CODEN_RETRIEVER_DISABLED_TOOLS env var to filter out specific tools.
    """
    disabled_tools = get_disabled_tools()

    register_functions = [
        lambda mcp: register_clone_detection_tools(mcp, disabled_tools),
        lambda mcp: register_code_search_tools(mcp, disabled_tools),
        lambda mcp: register_debug_tools(mcp, disabled_tools),
        lambda mcp: register_echo_comment_tools(mcp, disabled_tools),
        lambda mcp: register_file_edit_tools(mcp, disabled_tools),
        lambda mcp: register_flag_tools(mcp, disabled_tools),
        lambda mcp: register_graph_analysis_tools(mcp, disabled_tools),
        lambda mcp: register_propagation_cost_tools(mcp, disabled_tools),
    ]

    # Only register dynamic tools if explicitly enabled in pyproject.toml
    if _is_dynamic_tools_enabled():
        register_functions.append(lambda mcp: register_dynamic_tools(mcp, disabled_tools))

    return create_mcp_server_with_config(
        server_name=SERVER_NAME_FULL,
        instructions=FULL_SERVER_INSTRUCTIONS,
        install_dependency="all",
        register_functions=register_functions,
    )
