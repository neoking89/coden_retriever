"""Debug guidance meta-tool for MCP clients.

Returns situation-specific debugging workflows so LLMs know which tools
to call and in what order. Especially useful for external MCP clients
that don't receive the system prompt tool_instructions.
"""
import logging
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field

from .dap_client import get_dap_client
from .debug_errors import ErrorCategory

logger = logging.getLogger(__name__)


# Map each ErrorCategory to its one-line recovery action — LLMs pattern-match
# on `category` first (Phase 7 Step 6 injection) and only read `message` for
# parameters. A single source of truth here keeps the recovery text aligned
# with the enum; adding a category to debug_errors.py without extending this
# dict fails the _CATEGORY_RECOVERY test below.
_CATEGORY_RECOVERY: dict[ErrorCategory, str] = {
    ErrorCategory.INSTALL_MISSING: (
        "Run the command in `suggested_action` to install the adapter "
        "toolchain, then retry."
    ),
    ErrorCategory.CONFIG_ERROR: (
        "Fix the session shape — start a session first "
        "(debug_session action='launch'), pass the right language=, or "
        "correct the host/port for attach."
    ),
    ErrorCategory.PROGRAM_ERROR: (
        "The user's program broke — surface the `message` to the caller "
        "and fix the source (syntax, launch command, breakpoint expression)."
    ),
    ErrorCategory.ADAPTER_INTERNAL: (
        "Adapter-side issue the caller cannot fix directly. Try "
        "debug_session(action='stop') then relaunch; if it persists, "
        "the adapter version / toolchain is the root cause."
    ),
    ErrorCategory.TOOL_MISUSE: (
        "Your tool call was wrong — re-read `message` and re-issue with "
        "the right parameter (missing arg, bad path, conflicting flags)."
    ),
}
# The category↔recovery coverage invariant is enforced at test time
# (tests/mcp/test_debug_guide.py::test_category_recovery_covers_every_enum_value)
# rather than via a module-level `assert`: python -O strips asserts, and we'd
# rather the test suite flag drift than crash prod at import time if a future
# ErrorCategory addition slips through without a recovery entry.

# Reusable tool lists referenced in multiple situations
_DAP_TOOLS = [
    "debug_session", "debug_action", "debug_eval",
    "debug_variables", "debug_stack", "debug_breakpoint",
]
_SOURCE_TOOLS = [
    "source_add_breakpoint", "source_inject_trace",
    "source_list_injections", "source_remove_injections",
]

_RUNTIME_ERROR_WORKFLOW = [
    "1. find_identifier(root_directory='...', identifier='suspect_function') — get the line number",
    "2. debug_session(action='launch', program='script.py', stop_on_entry=True)",
    "3. debug_breakpoint(action='set', file_path='...', lines=[<line from step 1>])",
    "4. debug_action(action='continue') — run to the breakpoint",
    "5. Inspect: debug_eval(expression='suspect_variable') or debug_variables()",
    "6. debug_action(action='step_over') — step through to find the bug",
    "7. debug_session(action='stop') — end session",
    "NOTE: On termination, breakpoint_summary shows which breakpoints were hit/missed.",
    "NOTE: stop_on_entry=True keeps the program paused so breakpoints land before it finishes.",
    "NOTE: the SUBMIT target is the line where the error manifests in "
    "execution — the top frame of the stack, the crash/dereference/exception "
    "line. Walk UP to callers (debug_variables(frame_index=1,2,...)) to "
    "UNDERSTAND how the bad state got there, but submit the crash-site line.",
]

# Counters a common LLM confusion on runtime errors: submitting the
# setup/data-insertion line (where the bad value was born) instead of the
# crash-site line (where execution actually broke). Both matter — they
# belong in different fields of the submission.
_SUBMIT_CRASH_SITE_TIPS = [
    "IMPORTANT: submit the line where the error MANIFESTS during execution "
    "(the top frame of the stack: the crash line, the dereference line, the "
    "exception line).",
    "The setup/data line that introduced the bad value is the ROOT CAUSE — "
    "explain it in `root_cause`, but do NOT submit it as `line`. Submit the "
    "site where the program actually broke.",
    "Use debug_stack() and debug_variables(frame_index=1,2,...) to walk UP "
    "through callers and understand HOW the bad state was seeded. That "
    "context belongs in `root_cause`; the top-of-stack line is your submit "
    "target.",
]

_WRONG_OUTPUT_WORKFLOW = [
    "1. find_identifier(root_directory='...', identifier='function_name') — get the line number",
    "2. debug_session(action='launch', program='script.py', stop_on_entry=True)",
    "3. debug_breakpoint(action='set', file_path='...', lines=[<line from step 1>])",
    "4. debug_action(action='continue') — jump to breakpoint",
    "5. debug_action(action='step_over') — step line by line, watch variables change",
    "6. Compare actual vs expected values with debug_eval()",
    "7. debug_session(action='stop') — end session",
]

_VARIABLE_MYSTERY_WORKFLOW = [
    "1. find_identifier(root_directory='...', identifier='function_name') — get the line number",
    "2. debug_session(action='launch', program='script.py', stop_on_entry=True)",
    "3. debug_breakpoint(action='set', file_path='...', lines=[<line from step 1>])",
    "4. debug_action(action='continue') — run to breakpoint",
    "5. debug_eval(expression='variable_name') — check actual value",
    "6. debug_eval(expression='type(variable_name).__name__') — check type",
    "7. debug_stack() — see who called this function and trace the value upstream",
    "8. debug_variables(frame_index=1) — check caller's variables",
]

_TROUBLESHOOTING_WORKFLOW = [
    "1. Check debugpy installed: pip list | grep debugpy (or pip install debugpy)",
    "2. Check file path: must be absolute or relative to working directory",
    "3. If syntax error: fix the error before launching (pre-launch validation catches this)",
    "4. If port conflict: debug_session uses auto-allocated ports for launch, specify port for attach",
    "5. If breakpoint never hit: check breakpoint_summary after termination for suggestions",
    "6. If conditional breakpoint never fires: try without condition first to verify line is reached",
    "7. For multithreaded issues: use debug_threads() to list threads, debug_stack(thread_id=N)",
]


def _session_status_tip() -> str:
    """Check if a DAP session is active and return a tip."""
    try:
        client = get_dap_client()
        if client.is_connected:
            return "A debug session is already active. You can use debug_action and debug_eval directly."
        return "No debug session active. Start one with debug_session(action='launch', program='...')."
    except Exception:
        logger.warning("Failed to check DAP session status", exc_info=True)
        return "Start a debug session with debug_session(action='launch', program='...')."


async def debug_guide(
    situation: Annotated[
        Literal["runtime_error", "wrong_output", "variable_mystery", "choose_approach", "tool_help", "troubleshooting"],
        Field(
            validation_alias=AliasChoices("situation", "task"),
            description=(
                "'runtime_error': TypeError/ValueError/KeyError — need to see actual values; "
                "'wrong_output': Code runs but produces incorrect result; "
                "'variable_mystery': Variable has unexpected value (None, wrong type); "
                "'choose_approach': Not sure whether to use debugger or trace injection; "
                "'tool_help': List all debug tools with when to use each; "
                "'troubleshooting': Debug session won't start, breakpoints don't work, or other issues"
            ),
        ),
    ],
    error_message: Annotated[
        str | None,
        Field(description="The error message or traceback (for 'runtime_error')"),
    ] = None,
) -> dict[str, Any]:
    """Get debugging guidance — which tools to use and in what order.

    Call this FIRST when you need to debug something.
    Returns a step-by-step workflow tailored to your situation.
    """
    session_tip = _session_status_tip()

    if situation == "runtime_error":
        tips = [
            *_SUBMIT_CRASH_SITE_TIPS,
            "Set breakpoint 1-2 lines BEFORE the error line",
            "Check variable types with debug_eval(expression='type(x).__name__')",
            "Check dict keys with debug_eval(expression='list(d.keys())')",
        ]
        if error_message and "NoneType" in error_message:
            tips.insert(0, "A variable is None — trace back to where it was assigned")
        if error_message and "KeyError" in error_message:
            tips.insert(0, "Dict missing a key — check contents with debug_eval")

        return {
            "situation": "runtime_error",
            "session_status": session_tip,
            "workflow": _RUNTIME_ERROR_WORKFLOW,
            "tips": tips,
        }

    if situation == "wrong_output":
        return {
            "situation": "wrong_output",
            "session_status": session_tip,
            "workflow": _WRONG_OUTPUT_WORKFLOW,
            "tips": [
                "Step through loops with debug_action(action='step_over')",
                "Check accumulator variables after each iteration",
                "Use debug_eval to compare actual vs expected values",
            ],
        }

    if situation == "variable_mystery":
        return {
            "situation": "variable_mystery",
            "session_status": session_tip,
            "workflow": _VARIABLE_MYSTERY_WORKFLOW,
            "tips": [
                "Use debug_variables(frame_index=1) to check caller's context",
                "Use debug_eval(expression='obj.__dict__') to see all attributes",
            ],
        }

    if situation == "choose_approach":
        return {
            "situation": "choose_approach",
            "session_status": session_tip,
            "use_dap_debugger_when": [
                "You need to step through code line by line",
                "You need to evaluate expressions at runtime",
                "The bug involves complex state interactions",
                "The program's language has a registered DAP adapter (Python via debugpy, Go via dlv)",
            ],
            "use_source_injection_when": [
                "You want to add logging and run the script normally",
                "Working with JavaScript/TypeScript",
                "You want traces that persist across runs",
                "Quick check — just need to see a variable value",
            ],
        }

    if situation == "troubleshooting":
        return {
            "situation": "troubleshooting",
            "session_status": session_tip,
            "workflow": _TROUBLESHOOTING_WORKFLOW,
            "common_issues": [
                "debugpy not installed → pip install debugpy",
                "Wrong file path → must be absolute or relative to cwd",
                "Syntax error → pre-launch validation catches this automatically",
                "Breakpoint never hit → check breakpoint_summary after termination",
                "Port conflict → launch uses auto-allocated ports; attach requires manual port",
            ],
            "error_categories": _error_category_help(),
        }

    # situation == "tool_help"
    return {
        "situation": "tool_help",
        "session_status": session_tip,
        "workflow_tip": (
            "To set breakpoints by function name: "
            "1. find_identifier(root_directory, identifier='func_name') to get the line number, "
            "2. debug_breakpoint(action='set', file_path='...', lines=[line]) to set it. "
            "On termination, breakpoint_summary shows which breakpoints were hit/missed."
        ),
        "dap_tools": {
            "find_identifier": "Resolve function/class name to file + line number (use before debug_breakpoint)",
            "debug_guide": "Get debugging guidance — which tools to use and in what order (this tool)",
            "debug_session": "Launch/attach/stop/status of debug session (Python via debugpy is the primary tested adapter)",
            "debug_action": "Step through code (step_over/step_into/step_out/continue/pause) — auto-returns context",
            "debug_eval": "Evaluate an expression in the adapter's language (Python for debugpy)",
            "debug_variables": "Get all variables in a stack frame (+ expand nested objects via variables_reference)",
            "debug_stack": "Get call stack (thread-aware via thread_id parameter)",
            "debug_threads": "List all threads in the debug session",
            "debug_breakpoint": "Set/list/clear/add/remove/save/load breakpoints + set_exception for exception breakpoints",
        },
        "source_injection_tools": {
            "source_add_breakpoint": "Inject breakpoint()/debugger; into source code (Python/JS/TS)",
            "source_inject_trace": "Inject print/console.log to log variable values",
            "source_list_injections": "List all injected breakpoints and traces",
            "source_remove_injections": "Remove all injected code from source files",
        },
        "other": {
            "debug_stacktrace": "Parse any language stacktrace and map to local code",
            "debug_server": "Start debugpy server for IDE attachment (VS Code, PyCharm)",
        },
        "error_categories": _error_category_help(),
    }


def _error_category_help() -> dict[str, str]:
    """Flatten ErrorCategory + _CATEGORY_RECOVERY for LLM consumption.

    Every error envelope now carries `category: <str>`; this block tells
    the LLM what each value means and the first thing to try. Keys are the
    string values that land on the wire (matches ErrorCategory.value).
    """
    return {cat.value: _CATEGORY_RECOVERY[cat] for cat in ErrorCategory}
