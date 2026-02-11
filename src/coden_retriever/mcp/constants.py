"""
MCP Server Constants.

Centralized configuration for server names, instructions, and error messages.
"""

# Server names
SERVER_NAME_FULL = "CodenRetriever"
SERVER_NAME_CODE_SEARCH = "CodenRetriever-CodeSearch"
SERVER_NAME_DYNAMIC_TOOLS = "CodenRetriever-DynamicTools"

# Instruction templates
# NOTE: Tool descriptions are dynamically provided via MCP protocol.
# Each tool contains its own "WHEN TO USE" guidance in its description.

CODE_SEARCH_INSTRUCTIONS = """Code intelligence server with tools for working with codebases.

Use tools to get real information - never make up code or file contents.
Each tool's description explains when to use it.
All paths must be absolute."""

DYNAMIC_TOOLS_INSTRUCTIONS = """Dynamic tools management server for creating custom MCP tools at runtime.

Creates tools that persist across server restarts in ~/.coden-retriever/dynamic_tools.py.
All tools must include type hints and docstrings."""

FULL_SERVER_INSTRUCTIONS = """Code intelligence server with tools for working with codebases.

Use tools to get real information - never make up code or file contents.
Each tool's description explains when to use it.
All paths must be absolute."""

# Error messages
ERROR_FASTMCP_NOT_INSTALLED = "FastMCP not installed. Install with: pip install 'coden-retriever[{}]'"

# Tool groupings for organized display in CLI
# Format: (category_name, [tool_names])
_BASE_TOOL_CATEGORIES = [
    ("Code Discovery", ["code_map", "code_search", "find_hotspots"]),
    ("Graph Analysis", ["change_impact_radius", "coupling_hotspots", "architectural_bottlenecks"]),
    ("Symbol Lookup", ["find_identifier", "trace_dependency_path"]),
    ("Code Inspection", ["read_source_range", "read_source_ranges", "git_history_context", "code_evolution"]),
    ("Security", ["detect_sensitive_values"]),
    ("File Editing", ["write_file", "edit_file", "delete_file", "undo_file_change"]),
    ("Debugging", [
        "debug_stacktrace",
        # DAP debugging tools (3 consolidated, auto-return rich context)
        "debug_session",   # Lifecycle: launch, stop, status
        "debug_action",    # Execution: step, continue (returns stack+vars+code automatically)
        "debug_state",     # Inspection: eval, variables, breakpoints
        # Source injection tools (simple, no DAP session needed)
        "add_breakpoint",
        "remove_injections",
        "list_injections",
        "inject_trace",
        "debug_server",
    ]),
    ("Python Environment", ["check_python_virtual_env", "get_python_package_path"]),
]

_DYNAMIC_TOOLS_CATEGORY = ("Dynamic Tools", ["create_dynamic_tool", "remove_dynamic_tool"])


def _is_dynamic_tools_enabled() -> bool:
    """Check if dynamic tools are enabled via environment variable."""
    import os
    return os.environ.get("CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS", "").lower() in ("1", "true", "yes")


def get_tool_categories() -> list[tuple[str, list[str]]]:
    """Get tool categories, excluding Dynamic Tools if not enabled."""
    if _is_dynamic_tools_enabled():
        return _BASE_TOOL_CATEGORIES + [_DYNAMIC_TOOLS_CATEGORY]
    return _BASE_TOOL_CATEGORIES
