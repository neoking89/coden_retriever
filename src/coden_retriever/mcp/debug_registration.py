"""Registration aggregator for all MCP debug tools.

Lives in its own module so `debug_trace.py` does not need to import from
`debug_simplified.py`/`debug_inspect.py`/`debug_guide.py`. That import direction
would create a cycle now that `debug_simplified.py` calls into `debug_trace.py`
for source-injection cleanup on session stop.
"""
from typing import TYPE_CHECKING, Any, Callable

from .debug_guide import debug_guide
from .debug_inspect import debug_breakpoint, debug_eval, debug_stack, debug_threads, debug_variables
from .debug_simplified import debug_action, debug_session
from .debug_trace import (
    debug_server,
    source_add_breakpoint,
    source_inject_region_trace,
    source_inject_trace,
    source_list_injections,
    source_remove_injections,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register_debug_tools(mcp: "FastMCP", disabled_tools: set[str] | None = None) -> None:
    """Register all debug MCP tools.

    1. DAP SESSION TOOLS — lifecycle, stepping, and focused inspection tools:
       - debug_guide: situation-aware workflow guidance
       - debug_session: lifecycle management (launch, stop, status)
       - debug_action: execution flow (step, continue) with auto-context
       - debug_eval, debug_variables, debug_stack, debug_breakpoint

    2. SOURCE INJECTION TOOLS — modify source code directly, no DAP needed:
       - source_add_breakpoint, source_inject_trace,
       - source_inject_region_trace (bash/powershell flow trace),
       - source_list_injections, source_remove_injections, debug_server
    """
    disabled = disabled_tools or set()

    dap_tools: list[Callable[..., Any]] = [
        debug_guide,
        debug_session,
        debug_action,
        debug_eval,
        debug_variables,
        debug_stack,
        debug_threads,
        debug_breakpoint,
    ]

    source_tools: list[Callable[..., Any]] = [
        source_add_breakpoint,
        source_remove_injections,
        source_list_injections,
        source_inject_trace,
        source_inject_region_trace,
        debug_server,
    ]

    for func in dap_tools + source_tools:
        if func.__name__ not in disabled:
            mcp.tool()(func)
