"""
Simplified Debugging Tools for MCP.

Consolidates granular DAP operations into 3 high-level tools:
- debug_session: Manage lifecycle (launch, stop, status)
- debug_action: Handle execution flow (step, continue) with auto-context
- debug_eval/debug_variables/debug_stack/debug_breakpoint: Focused inspection (see debug_inspect.py)

Design goal: Reduce cognitive load for LLM agents by auto-returning
rich context after each action, eliminating the need for follow-up calls.
"""
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from ..constants import DEFAULT_DEBUG_PORT
from .adapters.base import DebugAdapter, LaunchConfig
from .adapters.registry import REGISTRY
from .dap_client import DAPClient, get_dap_client, get_or_reset_dap_client, reset_dap_client
from .debug_breakpoint_store import get_breakpoint_store
from .debug_errors import (
    adapter_handshake_failed,
    adapter_unsupported_in_production_error,
    debug_error,
    dependency_missing_error,
    dependency_missing_for_active_adapter,
    missing_param_error,
    no_session_error,
)
from .debug_inspect import _filter_locals_concise
from .debug_recovery import (
    debug_server_ever_started,
    suggest_script_alternatives,
    validate_script_syntax,
)
from .debug_trace import cleanup_all_source_injections

logger = logging.getLogger(__name__)

# Why: 5 lines above + 5 below gives enough surrounding context to tell a
# caller what block they're stopped in without ballooning the response body.
CODE_SNIPPET_CONTEXT_LINES = 5
# Derived from CONTEXT_LINES (before + current + after) so widening the
# window only requires touching a single constant.
MAX_CODE_SNIPPET_LINES = 2 * CODE_SNIPPET_CONTEXT_LINES + 1
MAX_OUTPUT_LINES = 20  # Recent program output lines to include
MAX_STACK_SUMMARY_DEPTH = 5  # Stack frames in summary
MAX_RECENT_OUTPUT_LINES = 5  # Recent output lines in rich context

# WHY 30s: debugpy cold-start on Windows with a slow antivirus scan has been
# observed up to ~18s; we allow ~2x headroom. Attach is typically <1s when the
# adapter is already listening, so the ceiling only matters for flaky CI.
_DEFAULT_ATTACH_TIMEOUT_SECONDS = 30.0


def _build_terminated_response(client: DAPClient) -> dict[str, Any]:
    """Build a termination response with optional breakpoint summary."""
    result: dict[str, Any] = {
        "status": "terminated",
        "message": "Program has finished executing",
        "output": client.state.program_output[-MAX_OUTPUT_LINES:] if client.state.program_output else [],
    }
    bp_summary = client.get_breakpoint_summary()
    if bp_summary:
        result["breakpoint_summary"] = bp_summary
    return result


def _read_code_snippet(
    file_path: str,
    current_line: int,
    context_lines: int = CODE_SNIPPET_CONTEXT_LINES,
) -> list[dict[str, Any]]:
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
    except OSError as e:
        logger.debug("Failed to read code snippet: %s", e)
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


async def _get_rich_debug_context(client: DAPClient, include_code: bool = True) -> dict[str, Any]:
    """Gather Stack + Variables + Code snippet in one call.

    This is the core helper that makes debug_action useful -
    it auto-fetches everything the LLM needs without follow-up calls.
    """
    result: dict[str, Any] = {
        "status": "stopped",
        "reason": client.state.stopped_reason,
    }
    if client.state.stopped_description:
        result["exception"] = client.state.stopped_description

    if client.state.program_terminated:
        return _build_terminated_response(client)

    stack_res = await client.get_stack_trace(levels=MAX_STACK_SUMMARY_DEPTH)
    frames = stack_res.get("frames", [])

    if not frames:
        result["error"] = "No stack frames available"
        return result

    top_frame = frames[0]

    result["location"] = {
        "file": top_frame.get("file"),
        "line": top_frame.get("line"),
        "function": top_frame.get("name"),
    }

    if top_frame.get("id"):
        vars_res = await client.get_variables(frame_id=top_frame["id"])
        if vars_res.get("status") == "success":
            # Reuses debug_inspect._filter_locals_concise so the "flat dict of
            # truncated Locals" shape stays identical between debug_action's
            # auto-context and debug_variables(detail='concise').
            result["variables"] = _filter_locals_concise(
                vars_res.get("variables", {}),
            )

    if include_code and top_frame.get("file") and top_frame.get("line"):
        snippet = _read_code_snippet(
            top_frame["file"], top_frame["line"], context_lines=CODE_SNIPPET_CONTEXT_LINES,
        )
        if snippet:
            result["code_snippet"] = _format_code_snippet(snippet)

    result["stack_summary"] = [
        f"{f.get('name', '?')} ({Path(f.get('file') or '?').name}:{f.get('line', '?')})"
        for f in frames[:MAX_STACK_SUMMARY_DEPTH]
    ]

    result["next_action_hint"] = (
        "Analyze the code and variables. "
        "Use debug_action to step/continue, debug_eval to test expressions, "
        "or debug_breakpoint to set breakpoints."
    )

    if client.state.program_output:
        result["recent_output"] = client.state.program_output[-MAX_RECENT_OUTPUT_LINES:]

    return result


# --- Session action handlers (extracted from debug_session) ---

SessionAction = Literal["launch", "attach", "stop", "status"]


async def _handle_session_status() -> dict[str, Any]:
    """Handle debug_session(action='status')."""
    client = get_dap_client()
    status = client.get_status()

    if status.get("is_stopped") and status.get("connected"):
        context = await _get_rich_debug_context(client, include_code=True)
        status.update({
            "location": context.get("location"),
            "variables": context.get("variables"),
            "code_snippet": context.get("code_snippet"),
        })

    return {"status": "success", **status}


async def _handle_session_stop(cleanup_injections: bool = True) -> dict[str, Any]:
    """Handle debug_session(action='stop').

    When cleanup_injections=True (default), removes any source-injected
    breakpoints/traces left in files by source_add_breakpoint/source_inject_trace.
    """
    client = get_dap_client()
    if client.breakpoints.by_file:
        store = get_breakpoint_store()
        await store.save_auto_restore(client.breakpoints.by_file, program=client.state.program)

    cleanup_summary: dict[str, Any] | None = None
    cleanup_errors: list[str] | None = None
    if cleanup_injections:
        cleanup_result = await cleanup_all_source_injections()
        if cleanup_result.get("status") == "success":
            cleanup_summary = {
                "breakpoints": cleanup_result.get("removed_breakpoints", 0),
                "traces": cleanup_result.get("removed_traces", 0),
            }
            if cleanup_result.get("errors"):
                cleanup_errors = cleanup_result["errors"]

    await client.stop()
    await reset_dap_client()

    response: dict[str, Any] = {"status": "stopped", "message": "Debug session ended"}
    if cleanup_summary is not None:
        response["injections_cleaned"] = cleanup_summary
    if cleanup_errors:
        response["cleanup_errors"] = cleanup_errors
    return response


def _resolve_adapter(
    language: str | None, program: str | None,
) -> tuple[DebugAdapter | None, dict[str, Any] | None]:
    """Resolve a DebugAdapter from an explicit language or a program extension.

    Precedence: explicit `language` → registry by name/alias → `program`
    suffix → registry by extension. Returns (adapter, None) on success or
    (None, error_dict) on failure. No silent fallback — an unregistered
    language/extension returns a structured error immediately.

    Adapters carrying `production_supported=False` short-circuit here so
    callers get `adapter_unsupported_in_production` instead of a known-
    broken launch. No adapters currently declare this — cpp + rust were
    the last holdouts and migrated to CodeLLDB in W7 of `prod-ready-
    exit.md`. The gate stays as a structural safety net for future
    regressions.
    """
    if language:
        adapter = REGISTRY.get_by_name(language)
        if adapter is None:
            return None, debug_error(
                "adapter_resolution_failed",
                f"No adapter registered for language '{language}'",
                f"Known languages: {', '.join(REGISTRY.names()) or '(none)'}",
            )
    elif program:
        suffix = Path(program).suffix
        adapter = REGISTRY.get_by_extension(suffix)
        if adapter is None:
            return None, debug_error(
                "adapter_resolution_failed",
                f"No adapter registered for file extension '{suffix}'",
                "Pass language='<name>' explicitly or use a supported file type",
            )
    else:
        return None, debug_error(
            "adapter_resolution_failed",
            "Cannot resolve adapter: no language or program provided",
            "Specify language='<name>' or a program path with a known extension",
        )
    if not adapter.production_supported:
        return None, adapter_unsupported_in_production_error(
            adapter.name, adapter.production_unsupported_reason,
        )
    return adapter, None


async def _handle_session_attach(
    host: str | None, port: int | None, language: str | None,
) -> dict[str, Any]:
    """Handle debug_session(action='attach'). Requires language= since attach
    has no program path to infer from.
    """
    adapter, err = _resolve_adapter(language, None)
    if err is not None:
        return err
    # `assert` would be stripped by `python -O`, leaving the subsequent
    # `adapter.name` access to raise a cryptic AttributeError. Explicit raise
    # preserves the err-gate invariant under every interpreter mode.
    if adapter is None:
        raise RuntimeError(
            "adapter not bound — err-gate invariant violated in _resolve_adapter",
        )
    client = await get_or_reset_dap_client()
    if client.is_connected:
        await client.stop()
    result = await client.attach(
        adapter,
        host=host or "127.0.0.1",
        port=port or DEFAULT_DEBUG_PORT,
        timeout=_DEFAULT_ATTACH_TIMEOUT_SECONDS,
    )
    if result.get("handshake_failed"):
        return adapter_handshake_failed(result["adapter_name"], result["error"])
    if "error" in result:
        return debug_error(
            "attach_failed",
            result["error"],
            f"Ensure the {adapter.name} adapter is listening on {host or '127.0.0.1'}:{port or DEFAULT_DEBUG_PORT}",
        )
    if client.state.is_stopped:
        context = await _get_rich_debug_context(client, include_code=True)
        context.pop("status", None)
        return {"status": "attached", **result, **context}
    return {"status": "attached", **result}


def _with_breakpoint_restore_errors(
    response: dict[str, Any], restore_errors: list[str] | None,
) -> dict[str, Any]:
    """Attach any auto-restore failures to an otherwise successful launch."""
    if restore_errors:
        response["breakpoint_restore_errors"] = restore_errors
    return response


async def _restore_breakpoints(client: DAPClient) -> list[str] | None:
    """Auto-restore breakpoints from previous session for the current program."""
    errors: list[str] = []
    try:
        store = get_breakpoint_store()
        saved = await store.get_auto_restore(program=client.state.program)
        if saved:
            by_file: dict[str, list[int]] = defaultdict(list)
            conditions: dict[str, dict[int, str]] = defaultdict(dict)
            log_messages: dict[str, dict[int, str]] = defaultdict(dict)
            for cfg in saved:
                by_file[cfg.file].append(cfg.line)
                if cfg.condition:
                    conditions[cfg.file][cfg.line] = cfg.condition
                if cfg.log_message:
                    log_messages[cfg.file][cfg.line] = cfg.log_message
            for file, lines in by_file.items():
                result = await client.set_breakpoints(
                    file=file, lines=lines,
                    conditions=conditions.get(file),
                    log_messages=log_messages.get(file),
                )
                if "error" in result:
                    errors.append(f"{file}: {result['error']}")
    except Exception as exc:
        logger.debug("Failed to auto-restore breakpoints", exc_info=True)
        errors.append(f"Auto-restore failed: {exc}")
    return errors or None


async def _handle_session_launch(
    program: str | None,
    args: list[str] | None,
    cwd: str | None,
    stop_on_entry: bool,
    language: str | None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Handle debug_session(action='launch')."""
    if not program:
        return missing_param_error("program", "debug_session(action='launch')")

    adapter, err = _resolve_adapter(language, program)
    if err is not None:
        return err
    # See _handle_session_attach for why we raise instead of asserting.
    if adapter is None:
        raise RuntimeError(
            "adapter not bound — err-gate invariant violated in _resolve_adapter",
        )

    ok, install_hint = adapter.detect_installed()
    if not ok:
        return dependency_missing_error(adapter.name, install_hint)

    # Python-specific pre-launch validation (syntax check via ast.parse).
    if adapter.name == "python":
        syntax_err = await validate_script_syntax(program)
        if syntax_err:
            return syntax_err

    # Path-existence guard is skipped for adapters whose `program` is a
    # language-level symbol (e.g., a JVM fully-qualified main class), not a
    # filesystem path.
    if not adapter.program_is_class_name and not Path(program).exists():
        suggestions = await suggest_script_alternatives(program)
        result = debug_error(
            "file_not_found",
            f"Script not found: {program}",
            "Check the file path (must be absolute or relative to cwd)",
        )
        if suggestions:
            result["suggestions"] = suggestions
        return result

    client = await get_or_reset_dap_client()
    if client.is_connected:
        await client.stop()

    # Safety net: debugpy's in-process listener (started by debug_server)
    # leaves non-recoverable state behind; if a prior stop couldn't join
    # the reader thread cleanly, a fresh launch is likely to hit a cryptic
    # "configurationDone" protocol error. Emit a targeted hint instead.
    if debug_server_ever_started() and client.last_reset_dirty:
        return debug_error(
            "dirty_state",
            "Leftover debugpy state from a previous debug_server session",
            "Please restart the MCP server to launch a new debug session",
        )

    cfg = LaunchConfig(
        program=program,
        args=tuple(args or ()),
        cwd=cwd,
        stop_on_entry=stop_on_entry,
        extras=dict(extras) if extras else {},
    )
    result = await client.launch(cfg, adapter, timeout=adapter.launch_timeout_seconds)

    if result.get("handshake_failed"):
        return adapter_handshake_failed(result["adapter_name"], result["error"])
    if "error" in result:
        return debug_error(
            "launch_failed",
            result["error"],
            f"Check that the program exists and the {adapter.name} adapter is installed",
        )

    restore_errors = await _restore_breakpoints(client)

    if stop_on_entry and client.state.is_stopped:
        context = await _get_rich_debug_context(client, include_code=True)
        # `stopped=True` mirrors DAPClient.launch()'s DebugResult.launched so
        # callers (matrix harness, MCP agent) can branch on "already paused"
        # without a follow-up status call. Without this flag the harness
        # issues an unneeded `continue` past the pre-launch bp, which on
        # short-lived programs (lua, jsdebug, php) ends the session before
        # any `stopped`-dependent tool can run.
        return _with_breakpoint_restore_errors(
            {**context, "status": "launched", "program": program, "stopped": True},
            restore_errors,
        )

    return _with_breakpoint_restore_errors(
        {
            "status": "launched",
            "program": program,
            "message": "Program is running. Use debug_action to pause.",
        },
        restore_errors,
    )


SessionHandler = Callable[[], Awaitable[dict[str, Any]]]

# Only no-arg handlers go here; "stop" needs cleanup_injections, "attach" needs host/port,
# "launch" needs program/args/cwd/stop_on_entry — those are dispatched explicitly.
_NULLARY_SESSION_HANDLERS: dict[str, SessionHandler] = {
    "status": _handle_session_status,
}


async def debug_session(
    action: Annotated[
        SessionAction,
        Field(
            description=(
                "'launch': Start debugging a program; "
                "'attach': Connect to an already-running DAP adapter server; "
                "'stop': End the debug session; "
                "'status': Check current session state"
            )
        ),
    ],
    program: Annotated[
        str | None,
        Field(description=(
            "Path to the program to debug. Required for 'launch'. "
            "Language is inferred from extension, or pass language= explicitly."
        )),
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
    host: Annotated[
        str | None,
        Field(description="Host for attach (default: 127.0.0.1)"),
    ] = None,
    port: Annotated[
        int | None,
        Field(description="Port for attach (default: 5678)", ge=1024, le=65535),
    ] = None,
    cleanup_injections: Annotated[
        bool,
        Field(description="On stop: auto-remove source-injected breakpoints/traces (default True)"),
    ] = True,
    language: Annotated[
        str | None,
        Field(
            description=(
                "DAP adapter language ('python', etc.). Optional for launch "
                "(inferred from program extension if omitted), required for "
                "attach. An unregistered language returns adapter_resolution_failed."
            ),
        ),
    ] = None,
    extras: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Adapter-specific launch options forwarded to the adapter via "
                "LaunchConfig.extras. Examples: Java accepts classPaths, "
                "modulePaths, vmArgs, projectName, jdtls_workspace, "
                "lsp_startup_timeout_seconds; Kotlin accepts classPath, "
                "vmArguments, modulePaths; JS/TS accepts source_maps, "
                "sourceMapPathOverrides, runtimeExecutable, runtimeArgs, env, "
                "outFiles. Unknown keys are dropped by the adapter."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """Manage debug session lifecycle - launch, stop, or check status.

    Language is inferred from the program's extension, or pass language=
    explicitly (e.g., language='python'). An unregistered language
    returns adapter_resolution_failed.

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

    WORKFLOW (launch):
    1. debug_session(action='launch', program='script.py') - Start debugging
    2. Use debug_action to step through code (returns context automatically)
    3. Use debug_eval / debug_variables for deeper inspection when needed
    4. debug_session(action='stop') - End session

    WORKFLOW (attach to external adapter — Python example):
    1. User starts: python -m debugpy --listen 5678 --wait-for-client script.py
    2. debug_session(action='attach', port=5678, language='python') - Connect to it
    3. Use debug_action / debug_eval as normal
    4. debug_session(action='stop') - Disconnect (does NOT kill the external process)

    Returns rich context (stack, variables, code) when launching with stop_on_entry=True.
    """
    try:
        if action in _NULLARY_SESSION_HANDLERS:
            return await _NULLARY_SESSION_HANDLERS[action]()
        if action == "stop":
            return await _handle_session_stop(cleanup_injections=cleanup_injections)
        if action == "attach":
            return await _handle_session_attach(host, port, language)
        if action == "launch":
            return await _handle_session_launch(program, args, cwd, stop_on_entry, language, extras)
        return debug_error("invalid_parameter", f"Unknown action: {action!r}")

    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_session failed: %s", e)
        return debug_error("unexpected_error", str(e))


# --- Action dispatch ---

DebugActionType = Literal["step_over", "step_into", "step_out", "continue", "pause"]


async def _execute_debug_action(client: DAPClient, action: DebugActionType, timeout: float) -> dict[str, Any]:
    """Dispatch and execute a debug action on the client."""
    step_actions: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
        "step_over": client.step_over,
        "step_into": client.step_into,
        "step_out": client.step_out,
    }
    if action == "continue":
        return await client.continue_execution(timeout=timeout)
    if action == "pause":
        return await client.pause()
    return await step_actions[action]()


async def _interpret_action_result(client: DAPClient, result: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Interpret the result of a debug action and return rich context."""
    if "error" in result:
        return debug_error("action_failed", result["error"])

    if result.get("status") == "terminated" or client.state.program_terminated:
        return _build_terminated_response(client)

    if result.get("status") == "timeout":
        return {
            "status": "timeout",
            "message": f"Program did not stop within {timeout} seconds",
            "suggestion": "Program may be waiting for input or in infinite loop",
        }

    if result.get("status") == "stopped" or client.state.is_stopped:
        return await _get_rich_debug_context(client)

    return result


async def debug_action(
    action: Annotated[
        DebugActionType,
        Field(
            description=(
                "'step_over': Execute current line, skip function calls; "
                "'step_into': Step into function calls; "
                "'step_out': Finish current function; "
                "'continue': Run to next breakpoint; "
                "'pause': Pause a running program"
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
       - "TypeError: 'NoneType'" -> Step back to find where None came from
       - "KeyError: 'x'" -> Check dict contents with debug_eval(expression='list(dict.keys())')
       - "IndexError" -> Eval 'len(list)' to see actual size vs expected
       - Wrong output -> Step through loop iterations, check accumulator variables

    3. STEPPING STRATEGY:
       - step_over: When you trust a function works correctly
       - step_into: When bug might be inside the function call
       - step_out: When you've seen enough of current function
       - continue: Jump to next breakpoint (set breakpoints at suspicious lines)

    4. EXAMPLE DEBUG SESSION:
       Bug: "calculate_total returns 0 instead of expected sum"
       -> Set breakpoint at return statement
       -> continue to breakpoint
       -> Check 'total' variable - is it 0? Why?
       -> Step back through loop - is it even executing?
       -> Eval 'len(items)' - empty list? That's the bug!
    """
    try:
        client = get_dap_client()

        if not client.is_connected:
            return no_session_error()

        result = await _execute_debug_action(client, action, timeout)
        return await _interpret_action_result(client, result, timeout)

    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_action failed: %s", e)
        return debug_error("unexpected_error", str(e))
