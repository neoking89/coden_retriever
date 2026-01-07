"""
Simplified Debugging Tools for MCP.

Consolidates granular DAP operations into 3 high-level tools:
- debug_session: Manage lifecycle (launch, stop, status)
- debug_action: Handle execution flow (step, continue) with auto-context
- debug_state: Deep inspection (eval, variables, stack) when needed

Design goal: Reduce cognitive load for LLM agents by auto-returning
rich context after each action, eliminating the need for follow-up calls.
"""
import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from .dap_client import get_dap_client, get_or_reset_dap_client, reset_dap_client

logger = logging.getLogger(__name__)

# Maximum values to prevent context window bloat
MAX_VARIABLE_VALUE_LENGTH = 200
MAX_VARIABLES_PER_SCOPE = 20
MAX_CODE_SNIPPET_LINES = 11  # 5 before, current, 5 after
MAX_OUTPUT_LINES = 20  # Recent program output lines to include
MAX_STACK_SUMMARY_DEPTH = 5  # Stack frames in summary


def _truncate_value(value: str, max_length: int = MAX_VARIABLE_VALUE_LENGTH) -> str:
    """Truncate long values to prevent context bloat."""
    if len(value) <= max_length:
        return value
    return value[:max_length - 3] + "..."


def _read_code_snippet(file_path: str, current_line: int, context_lines: int = 5) -> list[dict[str, Any]]:
    """Read a code snippet around the current line.

    Returns list of {line_number, code, is_current} dicts.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return []

        lines = path.read_text(encoding="utf-8").splitlines()

        start = max(0, current_line - context_lines - 1)
        end = min(len(lines), current_line + context_lines)

        snippet = []
        for i in range(start, end):
            snippet.append({
                "line_number": i + 1,
                "code": lines[i],
                "is_current": (i + 1) == current_line,
            })

        return snippet
    except Exception as e:
        logger.debug(f"Failed to read code snippet: {e}")
        return []


def _format_code_snippet(snippet: list[dict[str, Any]]) -> str:
    """Format code snippet as a readable string with line numbers."""
    if not snippet:
        return ""

    lines = []
    for item in snippet:
        marker = ">>>" if item["is_current"] else "   "
        lines.append(f"{marker} {item['line_number']:4d} | {item['code']}")

    return "\n".join(lines)


async def _get_rich_debug_context(client, include_code: bool = True) -> dict[str, Any]:
    """Gather Stack + Variables + Code snippet in one call.

    This is the core helper that makes debug_action useful -
    it auto-fetches everything the LLM needs without follow-up calls.
    """
    result: dict[str, Any] = {
        "status": "stopped",
        "reason": client.state.stopped_reason,
    }

    # Check for program termination
    if client.state.program_terminated:
        return {
            "status": "terminated",
            "message": "Program has finished executing",
            "output": client.state.program_output[-MAX_OUTPUT_LINES:] if client.state.program_output else [],
        }

    # 1. Get Stack Trace (limited depth for token efficiency)
    stack_res = await client.get_stack_trace(levels=5)
    frames = stack_res.get("frames", [])

    if not frames:
        result["error"] = "No stack frames available"
        return result

    top_frame = frames[0]

    # 2. Build location info
    result["location"] = {
        "file": top_frame.get("file"),
        "line": top_frame.get("line"),
        "function": top_frame.get("name"),
    }

    # 3. Get Variables for top frame (flattened for easy reading)
    if top_frame.get("id"):
        vars_res = await client.get_variables(frame_id=top_frame["id"])
        if vars_res.get("status") == "success":
            variables = {}
            for scope_name, scope_vars in vars_res.get("variables", {}).items():
                # Only include Locals and limited count to prevent bloat
                if scope_name == "Locals":
                    for var in scope_vars[:MAX_VARIABLES_PER_SCOPE]:
                        variables[var["name"]] = _truncate_value(var["value"])
            result["variables"] = variables

    # 4. Get Code Snippet
    if include_code and top_frame.get("file") and top_frame.get("line"):
        snippet = _read_code_snippet(
            top_frame["file"],
            top_frame["line"],
            context_lines=5,
        )
        if snippet:
            result["code_snippet"] = _format_code_snippet(snippet)

    # 5. Short stack summary
    result["stack_summary"] = [
        f"{f.get('name', '?')} ({Path(f.get('file', '?')).name}:{f.get('line', '?')})"
        for f in frames[:MAX_STACK_SUMMARY_DEPTH]
    ]

    # 6. Guidance for LLM
    result["next_action_hint"] = (
        "Analyze the code and variables. "
        "Use debug_action to step/continue, or debug_state to eval expressions."
    )

    # 7. Include any program output
    if client.state.program_output:
        result["recent_output"] = client.state.program_output[-5:]

    return result


async def debug_session(
    action: Annotated[
        Literal["launch", "stop", "status"],
        Field(
            description=(
                "'launch': Start debugging a Python script; "
                "'stop': End the debug session; "
                "'status': Check current session state"
            )
        ),
    ],
    program: Annotated[
        str | None,
        Field(description="Path to Python script (required for 'launch')"),
    ] = None,
    args: Annotated[
        list[str] | None,
        Field(description="Command line arguments for the script"),
    ] = None,
    cwd: Annotated[
        str | None,
        Field(description="Working directory for the script"),
    ] = None,
    stop_on_entry: Annotated[
        bool,
        Field(description="Pause at first line when launching"),
    ] = True,
) -> dict[str, Any]:
    """Manage debug session lifecycle - launch, stop, or check status.

    WHEN TO USE DEBUGGING (vs just reading code):
    - Runtime errors: TypeError, ValueError, KeyError - you need to see actual values
    - Logic bugs: Code runs but produces wrong output - step through to find where
    - State mysteries: "Why is this variable None here?" - inspect at runtime
    - Complex control flow: Loops, recursion, callbacks - trace actual execution path
    - Integration issues: Data from external sources behaves unexpectedly

    WHEN NOT TO USE DEBUGGING:
    - Syntax errors (won't run at all)
    - Import errors (fix imports first)
    - Simple bugs obvious from reading code
    - Performance issues (use profiling instead)

    WORKFLOW:
    1. debug_session(action='launch', program='script.py') - Start debugging
    2. Use debug_action to step through code (returns context automatically)
    3. Use debug_state for deeper inspection when needed
    4. debug_session(action='stop') - End session

    Returns rich context (stack, variables, code) when launching with stop_on_entry=True.
    """
    try:
        if action == "status":
            client = get_dap_client()
            status = client.get_status()

            # Enhance with current location if stopped
            if status.get("is_stopped") and status.get("connected"):
                context = await _get_rich_debug_context(client, include_code=True)
                status.update({
                    "location": context.get("location"),
                    "variables": context.get("variables"),
                    "code_snippet": context.get("code_snippet"),
                })

            return {"status": "success", **status}

        if action == "stop":
            client = get_dap_client()
            await client.stop()
            # Also reset the global client to ensure clean state for next launch
            await reset_dap_client()
            return {
                "status": "stopped",
                "message": "Debug session ended",
            }

        # action == "launch"
        if not program:
            return {
                "status": "error",
                "error_type": "missing_parameter",
                "message": "Program path required for launch",
                "suggested_action": "Provide 'program' parameter with path to Python script",
            }

        # Use get_or_reset to ensure clean state before launch
        client = await get_or_reset_dap_client()

        # Clean up any existing session
        if client.is_connected:
            await client.stop()

        # Launch the program (client.launch already has internal timeout handling)
        result = await client.launch(
            program=program,
            args=args,
            cwd=cwd,
            stop_on_entry=stop_on_entry,
            timeout=30.0,
        )

        if "error" in result:
            return {
                "status": "error",
                "error_type": "launch_failed",
                "message": result["error"],
                "suggested_action": "Check that the file exists and debugpy is installed",
            }

        # If stopped on entry, return rich context immediately
        if stop_on_entry and client.state.is_stopped:
            context = await _get_rich_debug_context(client, include_code=True)
            # Spread context first, then override status to "launched"
            return {
                **context,
                "status": "launched",
                "program": program,
            }

        return {
            "status": "launched",
            "program": program,
            "message": "Program is running. Use debug_action to pause.",
        }

    except ImportError:
        return {
            "status": "error",
            "error_type": "dependency_missing",
            "message": "debugpy not installed",
            "suggested_action": "Install with: pip install debugpy",
        }
    except Exception as e:
        logger.exception(f"debug_session failed: {e}")
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e),
        }


async def debug_action(
    action: Annotated[
        Literal["step_over", "step_into", "step_out", "continue"],
        Field(
            description=(
                "'step_over': Execute current line, skip function calls; "
                "'step_into': Step into function calls; "
                "'step_out': Finish current function; "
                "'continue': Run to next breakpoint"
            )
        ),
    ],
    timeout: Annotated[
        float,
        Field(description="Max seconds to wait for program to stop", ge=1, le=300),
    ] = 60.0,
) -> dict[str, Any]:
    """Control execution flow with automatic context return.

    RETURNS RICH CONTEXT after each action:
    - Current location (file, line, function)
    - Code snippet around current line (>>> marks current line)
    - Local variables (automatically fetched)
    - Stack summary

    HOW TO INTERPRET THE CONTEXT:

    1. FINDING THE BUG - Look for mismatches:
       - Variable has unexpected value? Found the corruption point
       - Variable is None when it should have data? Trace back where it was set
       - Wrong type? (e.g., str instead of int) - find the source

    2. COMMON PATTERNS:
       - "TypeError: 'NoneType'" → Step back to find where None came from
       - "KeyError: 'x'" → Check dict contents with debug_state(action='eval', expression='dict.keys()')
       - "IndexError" → Eval 'len(list)' to see actual size vs expected
       - Wrong output → Step through loop iterations, check accumulator variables

    3. STEPPING STRATEGY:
       - step_over: When you trust a function works correctly
       - step_into: When bug might be inside the function call
       - step_out: When you've seen enough of current function
       - continue: Jump to next breakpoint (set breakpoints at suspicious lines)

    4. EXAMPLE DEBUG SESSION:
       Bug: "calculate_total returns 0 instead of expected sum"
       → Set breakpoint at return statement
       → continue to breakpoint
       → Check 'total' variable - is it 0? Why?
       → Step back through loop - is it even executing?
       → Eval 'len(items)' - empty list? That's the bug!
    """
    try:
        client = get_dap_client()

        if not client.is_connected:
            return {
                "status": "error",
                "error_type": "no_session",
                "message": "No active debug session",
                "suggested_action": "Use debug_session(action='launch', program='...') first",
            }

        # Execute the requested action
        result = {}

        if action == "step_over":
            result = await client.step_over()
        elif action == "step_into":
            result = await client.step_into()
        elif action == "step_out":
            result = await client.step_out()
        elif action == "continue":
            result = await client.continue_execution(timeout=timeout)

        # Check for errors
        if "error" in result:
            return {
                "status": "error",
                "error_type": "action_failed",
                "message": result["error"],
            }

        # Handle termination
        if result.get("status") == "terminated" or client.state.program_terminated:
            return {
                "status": "terminated",
                "message": "Program has finished executing",
                "output": client.state.program_output[-20:] if client.state.program_output else [],
            }

        # Handle timeout
        if result.get("status") == "timeout":
            return {
                "status": "timeout",
                "message": f"Program did not stop within {timeout} seconds",
                "suggestion": "Program may be waiting for input or in infinite loop",
            }

        # Success - return rich context
        if result.get("status") == "stopped" or client.state.is_stopped:
            return await _get_rich_debug_context(client)

        return result

    except ImportError:
        return {
            "status": "error",
            "error_type": "dependency_missing",
            "message": "debugpy not installed",
            "suggested_action": "Install with: pip install debugpy",
        }
    except Exception as e:
        logger.exception(f"debug_action failed: {e}")
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e),
        }


async def debug_state(
    action: Annotated[
        Literal["eval", "variables", "stack", "breakpoints", "set_breakpoint", "clear_breakpoints"],
        Field(
            description=(
                "'eval': Evaluate a Python expression; "
                "'variables': Get detailed variables for a specific frame; "
                "'stack': Get full stack trace; "
                "'breakpoints': List current breakpoints; "
                "'set_breakpoint': Set a line breakpoint; "
                "'clear_breakpoints': Remove breakpoints from a file"
            )
        ),
    ],
    expression: Annotated[
        str | None,
        Field(description="Python expression to evaluate (for 'eval' action)"),
    ] = None,
    frame_index: Annotated[
        int,
        Field(description="Stack frame index, 0=current (for 'variables' action)", ge=0),
    ] = 0,
    file_path: Annotated[
        str | None,
        Field(description="File path (for 'set_breakpoint' and 'clear_breakpoints')"),
    ] = None,
    lines: Annotated[
        list[int] | None,
        Field(description="Line numbers for breakpoints (for 'set_breakpoint')"),
    ] = None,
    condition: Annotated[
        str | None,
        Field(description="Breakpoint condition applied to all lines (for 'set_breakpoint')"),
    ] = None,
    conditions: Annotated[
        dict[int, str] | None,
        Field(description="Per-line conditions: {line_number: 'condition'} (overrides 'condition')"),
    ] = None,
) -> dict[str, Any]:
    """Deep state inspection when debug_action's auto-context isn't enough.

    BREAKPOINT STRATEGY - Where to set breakpoints:

    1. AT ERROR LOCATION:
       - If you have a traceback, set breakpoint at the line BEFORE the error
       - This lets you inspect state just before the crash

    2. AT FUNCTION ENTRY:
       - First line of a suspicious function
       - Check: Are the input parameters what you expected?

    3. AT FUNCTION EXIT:
       - Line with 'return' statement
       - Check: Is the return value correct?

    4. AT DECISION POINTS:
       - Before 'if' statements - which branch will execute?
       - Before loop - will it execute at all?

    5. AT DATA TRANSFORMATIONS:
       - After parsing, API calls, or data processing
       - Check: Did the transformation produce expected result?

    CONDITIONAL BREAKPOINTS:
       Use 'condition' parameter to only break when something is wrong:
       - condition="len(items) == 0"  → Break only when list is empty
       - condition="user is None"     → Break only when user lookup failed
       - condition="i > 100"          → Break after 100 iterations

    EVAL EXPRESSION TIPS:
       - Check object attributes: "obj.__dict__"
       - Check dict keys: "list(data.keys())"
       - Check list length: "len(items)"
       - Check type: "type(variable).__name__"
       - Call methods: "obj.validate()" (careful with side effects!)
       - Complex expressions: "sum(x for x in items if x > 0)"

    USE CASES:
    - eval: Test expressions, call methods, check computed values
    - variables: Get variables from a specific stack frame (caller's context)
    - stack: See the full call chain (who called this function?)
    - set_breakpoint: Set breakpoints at specific lines
    - breakpoints: List current breakpoints
    - clear_breakpoints: Remove breakpoints from a file
    """
    try:
        client = get_dap_client()

        if not client.is_connected:
            return {
                "status": "error",
                "error_type": "no_session",
                "message": "No active debug session",
                "suggested_action": "Use debug_session(action='launch', program='...') first",
            }

        if action == "eval":
            if not expression:
                return {
                    "status": "error",
                    "error_type": "missing_parameter",
                    "message": "Expression required for eval",
                }

            # Get the frame ID for the requested index
            frame_id = None
            if frame_index > 0:
                stack = await client.get_stack_trace(levels=frame_index + 1)
                frames = stack.get("frames", [])
                if frame_index < len(frames):
                    frame_id = frames[frame_index].get("id")

            result = await client.evaluate(expression=expression, frame_id=frame_id)

            if "error" in result:
                return {
                    "status": "error",
                    "error_type": "eval_failed",
                    "message": result["error"],
                    "expression": expression,
                }

            return {
                "status": "success",
                "expression": expression,
                "result": result.get("result"),
                "type": result.get("type"),
            }

        if action == "variables":
            # Get the frame ID for the requested index
            stack = await client.get_stack_trace(levels=frame_index + 1)
            frames = stack.get("frames", [])

            if frame_index >= len(frames):
                return {
                    "status": "error",
                    "error_type": "invalid_frame",
                    "message": f"Frame index {frame_index} out of range (max: {len(frames) - 1})",
                }

            frame = frames[frame_index]
            result = await client.get_variables(frame_id=frame.get("id"))

            return {
                "status": "success",
                "frame": {
                    "index": frame_index,
                    "function": frame.get("name"),
                    "file": frame.get("file"),
                    "line": frame.get("line"),
                },
                "variables": result.get("variables", {}),
            }

        if action == "stack":
            result = await client.get_stack_trace(levels=50)
            frames = result.get("frames", [])

            return {
                "status": "success",
                "total_frames": len(frames),
                "frames": [
                    {
                        "index": i,
                        "function": f.get("name"),
                        "file": f.get("file"),
                        "line": f.get("line"),
                    }
                    for i, f in enumerate(frames)
                ],
            }

        if action == "set_breakpoint":
            if not file_path:
                return {
                    "status": "error",
                    "error_type": "missing_parameter",
                    "message": "file_path required for set_breakpoint",
                }
            if not lines:
                return {
                    "status": "error",
                    "error_type": "missing_parameter",
                    "message": "lines required for set_breakpoint",
                }

            # Build conditions map: per-line conditions override global condition
            bp_conditions: dict[int, str] = {}
            if conditions:
                # Per-line conditions take precedence
                bp_conditions = {int(k): v for k, v in conditions.items()}
            elif condition:
                # Apply global condition to all lines
                for line in lines:
                    bp_conditions[line] = condition

            result = await client.set_breakpoints(
                file=file_path,
                lines=lines,
                conditions=bp_conditions if bp_conditions else None,
            )

            return result

        if action == "breakpoints":
            # Return currently tracked breakpoints
            breakpoints = []
            for bp_file, bps in client._breakpoints.items():
                for bp in bps:
                    breakpoints.append({
                        "file": bp.file,
                        "line": bp.line,
                        "verified": bp.verified,
                        "condition": bp.condition,
                    })

            return {
                "status": "success",
                "breakpoints": breakpoints,
                "count": len(breakpoints),
            }

        if action == "clear_breakpoints":
            if not file_path:
                return {
                    "status": "error",
                    "error_type": "missing_parameter",
                    "message": "file_path required for clear_breakpoints",
                }

            # Set empty breakpoints to clear
            result = await client.set_breakpoints(file=file_path, lines=[])
            return {
                "status": "success",
                "message": f"Cleared breakpoints from {file_path}",
            }

        return {
            "status": "error",
            "error_type": "invalid_action",
            "message": f"Unknown action: {action}",
        }

    except ImportError:
        return {
            "status": "error",
            "error_type": "dependency_missing",
            "message": "debugpy not installed",
            "suggested_action": "Install with: pip install debugpy",
        }
    except Exception as e:
        logger.exception(f"debug_state failed: {e}")
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e),
        }


