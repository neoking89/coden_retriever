"""MCP tool filtering data structures and display utilities.

Contains shared constants (CORE_TOOLS, TOOL_GROUPS, TOOL_QUERY_DESCRIPTIONS),
data classes (ToolMetadata, FilteredTool, FilterResult), and console display
logic used by the LLM-based tool router.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Any

from rich.console import Console

logger = logging.getLogger(__name__)

# Semantic tool groups: when ANY tool in a group passes the filter,
# ALL tools in that group are included. This ensures logically related tools
# (e.g. all debug tools) appear together — missing one from a group is worse
# than showing an extra tool.
TOOL_GROUPS: dict[str, frozenset[str]] = {
    "debugging": frozenset({
        "debug_stacktrace", "debug_guide", "debug_session", "debug_action",
        "debug_eval", "debug_variables", "debug_stack", "debug_breakpoint",
    }),
    "injection": frozenset({
        "source_add_breakpoint", "source_inject_trace",
        "source_list_injections", "source_remove_injections",
    }),
    "flagging": frozenset({"flag_code", "flag_clear"}),
    "security": frozenset({"detect_sensitive_values", "source_list_injections"}),
    "architecture": frozenset({"coupling_hotspots", "find_hotspots", "architectural_bottlenecks"}),
    "impact": frozenset({"change_impact_radius", "propagation_cost"}),
    "history": frozenset({"code_evolution", "git_history_context"}),
}

# CORE tools - always shown, never filtered.
# These are essential tools that the LLM needs for basic code operations.
CORE_TOOLS = frozenset({
    # Code Discovery - essential for exploring codebases
    "code_search",
    "code_map",
    # Symbol Lookup - essential for finding definitions
    "find_identifier",
    # Code Inspection - essential for reading code
    "read_source_range",
    "read_source_ranges",
    # File Editing - essential for modifying code
    "write_file",
    "edit_file",
    "delete_file",
    "undo_file_change",
})

# Short, user-facing descriptions optimized for LLM tool routing.
# Written in the language users actually type (query-style) rather than verbose
# docstring prose, giving the router clearer signal about tool relevance.
TOOL_QUERY_DESCRIPTIONS: dict[str, str] = {
    "detect_clones": "Find duplicate or similar code, copy-paste detection, code clones",
    "trace_dependency_path": "Trace call paths between functions, who calls what, dependency chain",
    "check_python_virtual_env": "Check Python virtual environment, venv, virtualenv",
    "get_python_package_path": "Find where a Python package is installed, library source path",
    "debug_stacktrace": "Parse error stacktrace, map traceback frames to code",
    "git_history_context": "Git blame, who changed this line, commit history for a file",
    "find_hotspots": "Find frequently changed files, git churn analysis, code hotspots",
    "code_evolution": "How a function changed over time, function history across commits",
    "detect_dead_code": "Find unused functions, dead code, unreferenced code",
    "debug_guide": "How to debug, debugging workflow, which debug tool to use",
    "debug_session": "Start interactive debugger, launch debug session, step through code",
    "debug_action": "Step over, step into, continue execution in debugger",
    "debug_eval": "Evaluate expression in debugger, check variable value at runtime",
    "debug_variables": "Get all variables in current debug frame, inspect locals",
    "debug_stack": "Get full call stack, who called this function, call chain",
    "debug_breakpoint": "Set DAP breakpoint, list breakpoints, clear breakpoints in debug session",
    "source_add_breakpoint": "Inject breakpoint() or debugger; into source code file",
    "source_remove_injections": "Remove injected breakpoints and traces from source files",
    "source_list_injections": "List injected breakpoints and traces in source files",
    "source_inject_trace": "Inject print/console.log to trace variable values at runtime",
    "debug_server": "Start debug server for IDE attachment, VS Code debugger",
    "detect_echo_comments": "Find useless comments that just repeat the code, redundant comments",
    "flag_code": "Insert code quality markers, flag problematic functions with comments",
    "flag_clear": "Remove CODEN flag markers from source files",
    "change_impact_radius": "What breaks if I change this function, blast radius analysis",
    "coupling_hotspots": "Find highly coupled functions, high fan-in fan-out, refactoring targets",
    "architectural_bottlenecks": "Find architectural bottlenecks, bridge functions, critical paths",
    "propagation_cost": "Measure architecture health, coupling metrics, propagation cost",
    "detect_sensitive_values": "Find hardcoded secrets, passwords, API keys, credentials in code",
    "detect_tramp_data": "Find tramp data, parameters passed through many functions unnecessarily",
    "detect_magic_constants": "Find magic constants, repeated literal values across files, unnamed numbers/strings",
}


@dataclass
class ToolMetadata:
    """Metadata about a tool for filtering."""

    name: str
    description: str
    category: str = ""
    parameters_schema: str = ""
    example_code: str = ""
    query_description: str = ""


@dataclass
class FilteredTool:
    """Result of tool filtering with selection information."""

    metadata: ToolMetadata
    score: float
    is_core: bool = False


@dataclass
class FilterResult:
    """Result containing both core and filtered domain tools."""

    core_tools: list[FilteredTool]
    domain_tools: list[FilteredTool]

    @property
    def all_tools(self) -> list[FilteredTool]:
        """Get all tools (core + domain) for backwards compatibility."""
        return self.core_tools + self.domain_tools

    def __len__(self) -> int:
        return len(self.core_tools) + len(self.domain_tools)


def extract_tool_metadata(
    func: Callable[..., Any],
    query_descriptions: dict[str, str] | None = None,
) -> ToolMetadata:
    """Extract ToolMetadata from a tool function.

    Args:
        func: The tool function (typically decorated with @mcp.tool()).
        query_descriptions: Optional mapping of tool name to short description.

    Returns:
        ToolMetadata with name, description, and query_description populated.
    """
    name = func.__name__
    description = (func.__doc__ or "").strip()
    query_desc = (query_descriptions or {}).get(name, "")

    return ToolMetadata(name=name, description=description, query_description=query_desc)


def display_filtered_tools(
    result: FilterResult,
    console: Console | None = None,
) -> None:
    """Display the filtered tools to the console in agent mode.

    Args:
        result: FilterResult from LLMToolRouter.filter().
        console: Rich Console instance. If None, creates a default one.
    """
    if console is None:
        console = Console()

    core_names = [f"[cyan]{t.metadata.name}[/cyan]" for t in result.core_tools]
    domain_names = [
        f"[green]{t.metadata.name}[/green]" for t in result.domain_tools
    ]

    console.print(f"[dim]>> Core tools ({len(result.core_tools)}):[/dim] " + ", ".join(core_names))
    if domain_names:
        console.print(f"[dim]>> Routed tools ({len(result.domain_tools)}):[/dim] " + ", ".join(domain_names))
    else:
        console.print("[dim]>> Routed tools: (none matched)[/dim]")
