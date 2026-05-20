"""
MCP Server Constants.

Centralized configuration for server names, instructions, and error messages.
"""
import logging
import os

logger = logging.getLogger(__name__)

# Server name
SERVER_NAME_FULL = "CodenRetriever"

# Global per-call timeout applied to EVERY MCP tool (built-in + dynamic).
# 120 s — generous enough that a legitimately slow whole-repo built-in
# (dead-code / graph / propagation analysis on a large repo) completes, while
# still bounding an obviously-wedged tool within a normal conversation turn.
DEFAULT_TOOL_TIMEOUT_S: float = 120.0

# Env var name follows the existing CODEN_RETRIEVER_DISABLED_TOOLS convention.
TOOL_TIMEOUT_ENV_VAR: str = "CODEN_RETRIEVER_TOOL_TIMEOUT"

# pydantic-ai MCPServerStdio per-call ceiling (`read_timeout` default, mcp.py:314).
# Our structured timeout payload only reaches the agent if the tool timeout
# fires BEFORE this client-side transport timeout — i.e. the load-bearing
# invariant is `get_tool_timeout() < MCP_CLIENT_READ_TIMEOUT_S`.
MCP_CLIENT_READ_TIMEOUT_S: float = 300.0


def get_tool_timeout() -> float:
    """Read the global tool timeout from env, falling back to the default.

    Invalid or non-positive values fall back to the default (with a warning)
    so a typo in the environment can never silently disable the timeout layer.
    A value at/above the client read-timeout ceiling is honored but warned
    about, since the transport would time out first and the structured error
    would never surface.
    """
    raw = os.environ.get(TOOL_TIMEOUT_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_TOOL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %.1fs.",
            TOOL_TIMEOUT_ENV_VAR, raw, DEFAULT_TOOL_TIMEOUT_S,
        )
        return DEFAULT_TOOL_TIMEOUT_S
    if value <= 0:
        logger.warning(
            "Non-positive %s=%r; falling back to %.1fs.",
            TOOL_TIMEOUT_ENV_VAR, raw, DEFAULT_TOOL_TIMEOUT_S,
        )
        return DEFAULT_TOOL_TIMEOUT_S
    if value >= MCP_CLIENT_READ_TIMEOUT_S:
        logger.warning(
            "%s=%r is at/above the MCP client read-timeout ceiling (%.1fs); "
            "the transport will time out first and the structured tool-timeout "
            "error will not surface.",
            TOOL_TIMEOUT_ENV_VAR, raw, MCP_CLIENT_READ_TIMEOUT_S,
        )
    return value

# Note: the server "instructions" string is user-configurable. The default lives
# in coden_retriever.constants (DEFAULT_FULL_SERVER_INSTRUCTIONS_TEMPLATE), and
# the runtime value comes from MCPConfig in config_loader.

# Error messages
ERROR_FASTMCP_NOT_INSTALLED = "FastMCP not installed. Install with: pip install 'coden-retriever[{}]'"

# Tool groupings for organized display in CLI
# Format: (category_name, [tool_names])
_BASE_TOOL_CATEGORIES = [
    ("Code Discovery", ["code_map", "code_search", "find_hotspots"]),
    ("Graph Analysis", ["change_impact_radius", "coupling_hotspots", "architectural_bottlenecks"]),
    ("Symbol Lookup", ["find_identifier", "trace_dependency_path"]),
    ("Code Inspection", ["read_source_range", "read_source_ranges", "git_history_context", "code_evolution"]),
    ("Code Quality", [
        "detect_clones",
        "detect_dead_code",
        "detect_echo_comments",
        "propagation_cost",
        "detect_tramp_data",
        "detect_magic_constants",
        "flag_code",
        "flag_clear",
    ]),
    ("Security", ["detect_sensitive_values"]),
    ("File Editing", ["write_file", "edit_file", "delete_file", "undo_file_change"]),
    ("Debugging - DAP Session", [
        "debug_stacktrace",
        "debug_guide",        # Workflow guidance — call first when unsure
        "debug_session",      # Lifecycle: launch, attach, stop, status
        "debug_action",       # Execution: step, continue, pause (auto-returns context)
        "debug_eval",         # Evaluate expressions
        "debug_variables",    # Inspect frame variables (+ expand nested objects)
        "debug_stack",        # Full call stack (thread-aware)
        "debug_threads",      # List all threads in debug session
        "debug_breakpoint",   # Manage DAP breakpoints (set/list/clear/save/load)
    ]),
    ("Debugging - Source Injection", [
        "source_add_breakpoint",      # Inject breakpoint()/debugger; into source
        "source_remove_injections",   # Clean up injected code
        "source_list_injections",     # List active injections
        "source_inject_trace",        # Inject print/console.log
        "debug_server",               # Start debugpy for IDE attachment
    ]),
    ("Python Environment", ["check_python_virtual_env", "get_python_package_path"]),
]

_DYNAMIC_TOOLS_CATEGORY = ("Dynamic Tools", ["create_dynamic_tool", "remove_dynamic_tool"])


def _is_dynamic_tools_enabled() -> bool:
    """Check if dynamic tools are enabled via environment variable."""
    return os.environ.get("CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS", "").lower() in ("1", "true", "yes")


def get_tool_categories() -> list[tuple[str, list[str]]]:
    """Get tool categories, excluding Dynamic Tools if not enabled."""
    if _is_dynamic_tools_enabled():
        return _BASE_TOOL_CATEGORIES + [_DYNAMIC_TOOLS_CATEGORY]
    return _BASE_TOOL_CATEGORIES
