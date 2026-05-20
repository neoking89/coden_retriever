"""
MCP Server Constants.

Centralized configuration for server names, instructions, and error messages.
"""
import logging
import os

logger = logging.getLogger(__name__)

# Server name
SERVER_NAME_FULL = "CodenRetriever"

# Default per-tool timeout for dynamic tools.
# 30 s — long enough for legitimate shell calls and HTTP fetches, short enough
# that an obviously-wedged tool surfaces within a normal conversation turn
# instead of hanging the agent indefinitely.
DEFAULT_DYNAMIC_TOOL_TIMEOUT_S: float = 30.0

# Env var name follows the existing CODEN_RETRIEVER_DISABLED_TOOLS convention.
DYNAMIC_TOOL_TIMEOUT_ENV_VAR: str = "CODEN_DYNAMIC_TOOL_TIMEOUT_S"


def get_dynamic_tool_timeout() -> float:
    """Read the dynamic-tool timeout from env, falling back to the default.

    Invalid or non-positive values fall back to the default (with a warning)
    so a typo in the environment can never silently disable the timeout layer.
    """
    raw = os.environ.get(DYNAMIC_TOOL_TIMEOUT_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_DYNAMIC_TOOL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %.1fs.",
            DYNAMIC_TOOL_TIMEOUT_ENV_VAR, raw, DEFAULT_DYNAMIC_TOOL_TIMEOUT_S,
        )
        return DEFAULT_DYNAMIC_TOOL_TIMEOUT_S
    if value <= 0:
        logger.warning(
            "Non-positive %s=%r; falling back to %.1fs.",
            DYNAMIC_TOOL_TIMEOUT_ENV_VAR, raw, DEFAULT_DYNAMIC_TOOL_TIMEOUT_S,
        )
        return DEFAULT_DYNAMIC_TOOL_TIMEOUT_S
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
