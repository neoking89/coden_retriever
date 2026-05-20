"""
MCP Server module.

Provides the Model Context Protocol server for CodenRetriever with all tools.
This is the default server that includes both code search and dynamic tools.
"""
import os

from fastmcp import FastMCP

from ..config_loader import get_config, resolve_or_default
from ..constants import DEFAULT_FULL_SERVER_INSTRUCTIONS_TEMPLATE
from .constants import SERVER_NAME_FULL, _is_dynamic_tools_enabled, get_tool_timeout
from .server_factory import create_mcp_server_with_config


def get_disabled_tools() -> set[str]:
    """Get the set of disabled tools from environment variable.

    Reads CODEN_RETRIEVER_DISABLED_TOOLS env var (comma-separated list).
    """
    disabled_str = os.environ.get("CODEN_RETRIEVER_DISABLED_TOOLS", "")
    if not disabled_str:
        return set()
    return set(name.strip() for name in disabled_str.split(",") if name.strip())


def create_mcp_server() -> FastMCP:
    """Create an MCP server with all tools (code search + dynamic tools).

    Dynamic tools require enable_dynamic_tools = true in pyproject.toml [tool.coden-retriever].
    Respects CODEN_RETRIEVER_DISABLED_TOOLS env var to filter out specific tools.
    """
    # Lazy imports to avoid heavy dependencies at module load time
    from .architecture import register_architecture_tools
    from .clone_detection import register_clone_detection_tools
    from .code_search import register_code_search_tools
    from .dead_code import register_dead_code_tools
    from .debug_registration import register_debug_tools
    from .dynamic_tools import register_dynamic_tools
    from .echo_comments import register_echo_comment_tools
    from .file_edit import register_file_edit_tools
    from .flag_insertion import register_flag_tools
    from .graph_analysis import register_graph_analysis_tools
    from .propagation_cost import register_propagation_cost_tools
    from .sensitive_values import register_sensitive_value_tools
    from .magic_constants import register_magic_constant_tools
    from .tramp_data import register_tramp_data_tools

    disabled_tools = get_disabled_tools()

    register_functions = [
        lambda mcp: register_architecture_tools(mcp, disabled_tools),
        lambda mcp: register_clone_detection_tools(mcp, disabled_tools),
        lambda mcp: register_code_search_tools(mcp, disabled_tools),
        lambda mcp: register_dead_code_tools(mcp, disabled_tools),
        lambda mcp: register_debug_tools(mcp, disabled_tools),
        lambda mcp: register_echo_comment_tools(mcp, disabled_tools),
        lambda mcp: register_file_edit_tools(mcp, disabled_tools),
        lambda mcp: register_flag_tools(mcp, disabled_tools),
        lambda mcp: register_graph_analysis_tools(mcp, disabled_tools),
        lambda mcp: register_propagation_cost_tools(mcp, disabled_tools),
        lambda mcp: register_sensitive_value_tools(mcp, disabled_tools),
        lambda mcp: register_tramp_data_tools(mcp, disabled_tools),
        lambda mcp: register_magic_constant_tools(mcp, disabled_tools),
    ]

    # Only register dynamic tools if explicitly enabled in pyproject.toml
    if _is_dynamic_tools_enabled():
        register_functions.append(lambda mcp: register_dynamic_tools(mcp, disabled_tools))

    return create_mcp_server_with_config(
        server_name=SERVER_NAME_FULL,
        instructions=resolve_or_default(
            get_config().mcp.full_server_instructions_template,
            DEFAULT_FULL_SERVER_INSTRUCTIONS_TEMPLATE,
            "full_server_instructions_template",
        ),
        register_functions=register_functions,
        tool_timeout=get_tool_timeout(),
    )
