"""Focused debug inspection tools — split from the former debug_state mega-tool."""
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from .dap_client import get_dap_client
from .debug_breakpoint_store import BreakpointConfig, get_breakpoint_store
from .debug_errors import (
    CAPABILITY_REQUIRED,
    MAX_FRAMES_FULL_DUMP,
    MAX_STACK_TRACE_LEVELS,
    MAX_VARIABLES_PER_SCOPE,
    debug_error,
    dependency_missing_for_active_adapter,
    missing_param_error,
    no_session_error,
    session_not_paused_error,
    truncate_value,
    unsupported_capability,
)

if TYPE_CHECKING:
    from .dap_client import DAPClient

logger = logging.getLogger(__name__)


def _breakpoint_update_failed(action: str, message: str) -> dict[str, Any]:
    """Standardize tool-layer failures from DAP set_breakpoints calls."""
    return debug_error(
        "action_failed",
        f"Failed to {action} breakpoints: {message}",
        "Check that the source files exist in the active debuggee and retry.",
    )


def _get_connected_client() -> "DAPClient | None":
    """Get DAP client or None if not connected."""
    client = get_dap_client()
    return client if client.is_connected else None


def _require_paused_client() -> tuple["DAPClient | None", dict[str, Any] | None]:
    """Return (client, None) when connected AND paused; (None, error) otherwise.

    Helper owns error construction so callsites are a clean early-return:
        client, err = _require_paused_client()
        if err is not None:
            return err
    """
    client = get_dap_client()
    if not client.is_connected:
        return None, no_session_error()
    if not client.state.is_stopped:
        return None, session_not_paused_error()
    return client, None


async def _get_frame(client: "DAPClient", frame_index: int) -> tuple[dict | None, dict[str, Any] | None]:
    """Get a stack frame by index. Returns (frame, error_response)."""
    stack = await client.get_stack_trace(levels=frame_index + 1)
    frames = stack.get("frames", [])
    if frame_index >= len(frames):
        return None, debug_error(
            "invalid_frame",
            f"Frame index {frame_index} out of range (max: {len(frames) - 1})",
        )
    return frames[frame_index], None


async def debug_eval(
    expression: Annotated[str, Field(description="Expression to evaluate in the adapter's language (Python example: 'len(items)', 'type(x).__name__')")],
    frame_index: Annotated[int, Field(description="Stack frame: 0=current, 1=caller, 2=caller's caller", ge=0)] = 0,
) -> dict[str, Any]:
    """Evaluate an expression in the live debug context to test hypotheses
    about program state.

    USE THIS WHEN:
    - You want to check a computed value without re-running (e.g. `len(items)`)
    - You need to test a condition at the current frame (e.g. `x is None`)
    - You want to call a method and see its return value
    - You need type introspection (e.g. `type(x).__name__`)

    Expression syntax is whatever the active adapter understands (Python
    expressions for debugpy, Go expressions for delve, etc.).

    REQUIRES: Active debug session paused at a frame.
    """
    try:
        client, err = _require_paused_client()
        if err is not None:
            return err

        frame_id = None
        if frame_index > 0:
            frame, err = await _get_frame(client, frame_index)
            if err:
                return err
            frame_id = frame.get("id") if frame else None

        result = await client.evaluate(expression=expression, frame_id=frame_id)

        if "error" in result:
            err = debug_error("eval_failed", result["error"],
                              "Check expression syntax and variable names")
            err["expression"] = expression
            return err

        return {
            "status": "success", "expression": expression,
            "result": result.get("result"), "type": result.get("type"),
        }

    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_eval failed")
        return debug_error("unexpected_error", str(e))


def _filter_locals_concise(scope_vars_by_scope: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Pull only Locals values, truncating each to fit context budget."""
    flat: dict[str, str] = {}
    for scope_name, scope_vars in scope_vars_by_scope.items():
        if scope_name == "Locals":
            for var in scope_vars[:MAX_VARIABLES_PER_SCOPE]:
                flat[var["name"]] = truncate_value(var["value"])
    return flat


async def _dump_all_frames(client: "DAPClient", detail: Literal["concise", "full"]) -> dict[str, Any]:
    """Aggregate variables across all stack frames (capped at MAX_FRAMES_FULL_DUMP)."""
    stack = await client.get_stack_trace(levels=MAX_FRAMES_FULL_DUMP)
    frames = stack.get("frames", [])
    dumped: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        frame_id = frame.get("id")
        if frame_id is None:
            continue
        vars_res = await client.get_variables(frame_id=frame_id)
        scope_vars = vars_res.get("variables", {}) if vars_res.get("status") == "success" else {}
        variables = _filter_locals_concise(scope_vars) if detail == "concise" else scope_vars
        dumped.append({
            "index": i,
            "function": frame.get("name"),
            "file": frame.get("file"),
            "line": frame.get("line"),
            "variables": variables,
        })
    return {"status": "success", "frames": dumped, "frame_count": len(dumped)}


VariablesAction = Literal["get", "set"]


async def _resolve_locals_reference(
    client: "DAPClient", frame_id: int,
) -> int | None:
    """Find the DAP `Locals` scope's variablesReference for a frame.

    Returns None when the scopes response lacks a "Locals" entry — some
    adapters name their primary scope differently (e.g., "Local", "Scope")
    and we prefer explicit failure to silent misselection.
    """
    scopes_res = await client.get_scopes(frame_id)
    if "error" in scopes_res:
        return None
    for scope in scopes_res.get("scopes", []):
        if scope.get("name") == "Locals":
            ref = scope.get("variablesReference")
            return int(ref) if ref else None
    return None


async def _variables_set(
    client: "DAPClient",
    variable_name: str | None,
    new_value: str | None,
    variables_reference: int | None,
    frame_index: int,
) -> dict[str, Any]:
    """Handle debug_variables(action='set') — capability gate, scope
    resolution, and the `setVariable` wire call.

    Extracted so the top-level `debug_variables` stays ≤ 100 LOC per
    CLAUDE.md and the get/set branches read cleanly.
    """
    if not variable_name:
        return missing_param_error("variable_name", "debug_variables(action='set')")
    if new_value is None:
        return missing_param_error("new_value", "debug_variables(action='set')")

    flag = CAPABILITY_REQUIRED["set_variable"]
    if not client.capability(flag):
        return unsupported_capability(
            action="set_variable",
            capability_flag=flag,
            adapter_name=client.adapter_name,
        )

    if variables_reference is None:
        # Default scope: current frame's Locals. Resolve via DAP scopes.
        frame, err = await _get_frame(client, frame_index)
        if err:
            return err
        if frame is None:
            return debug_error("unexpected_error", "Frame was None despite no error")
        frame_id = frame.get("id")
        if frame_id is None:
            return debug_error(
                "invalid_frame",
                "Active frame has no DAP id; cannot resolve Locals scope",
            )
        variables_reference = await _resolve_locals_reference(client, frame_id)
        if variables_reference is None:
            return debug_error(
                "scope_resolution_failed",
                "Could not resolve a 'Locals' scope for this frame",
                "Pass an explicit variables_reference (from debug_variables detail='full')",
            )

    return await client.set_variable(
        variables_reference=variables_reference,
        name=variable_name,
        value=new_value,
    )


async def debug_variables(
    action: Annotated[
        VariablesAction,
        Field(description="'get': read variables (default); 'set': assign a new value"),
    ] = "get",
    frame_index: Annotated[int, Field(description="Stack frame: 0=current, 1=caller, 2=caller's caller", ge=0)] = 0,
    detail: Annotated[Literal["concise", "full"], Field(description="'concise': flat name=value; 'full': grouped by scope")] = "concise",
    variables_reference: Annotated[int | None, Field(description="Leave unset (null) to read locals for frame_index. Only set when expanding a nested object using a reference ID returned by detail='full'. DAP uses 0 as a 'no children' sentinel, so values must be >= 1.", ge=1)] = None,
    all_frames: Annotated[bool, Field(description="If True, dump variables from every frame (capped at 10) instead of one")] = False,
    variable_name: Annotated[str | None, Field(description="Variable name (required for action='set')")] = None,
    new_value: Annotated[str | None, Field(description="New value as string per DAP spec (required for action='set')")] = None,
) -> dict[str, Any]:
    """Get or set variables in a specific stack frame.

    USE THIS WHEN debug_action's auto-context isn't enough, or you need
    variables from a caller's frame (frame_index=1, 2, ...). Set
    all_frames=True to dump every frame's variables in one call.

    action='set' assigns a new value to a single variable (DAP
    `setVariable`). The adapter must advertise `supportsSetVariable`
    in its initialize response — call without `action='set'` first to
    confirm capability, or watch for the `unsupported_capability` error.

    REQUIRES: Active debug session paused at a breakpoint or step.
    """
    try:
        client, err = _require_paused_client()
        if err is not None:
            return err

        # DAP uses variablesReference=0 as the "no children" sentinel — it is
        # never a valid reference to dereference. The schema constraint (ge=1)
        # already rejects this at MCP dispatch; this guard covers direct
        # callers that bypass Pydantic validation.
        if variables_reference is not None and variables_reference <= 0:
            return debug_error(
                "invalid_parameter",
                f"variables_reference must be >= 1, got {variables_reference} (DAP reserves 0 as the 'no children' sentinel)",
                "Omit the parameter to read locals for frame_index, or pass a reference ID returned by debug_variables(detail='full')",
            )

        if action == "set":
            return await _variables_set(
                client, variable_name, new_value, variables_reference, frame_index,
            )

        if all_frames and variables_reference is not None:
            return debug_error(
                "conflicting_parameters",
                "all_frames and variables_reference are mutually exclusive",
                "Use variables_reference to expand a single nested object, or all_frames for a per-frame dump",
            )

        if all_frames:
            return await _dump_all_frames(client, detail)

        if variables_reference is not None:
            return await client.get_variable_children(variables_reference)

        frame, err = await _get_frame(client, frame_index)
        if err:
            return err
        if frame is None:
            return debug_error("unexpected_error", "Frame was None despite no error")

        frame_id = frame.get("id") if frame_index > 0 else None
        result = await client.get_variables(frame_id=frame_id)

        frame_info = {
            "index": frame_index, "function": frame.get("name"),
            "file": frame.get("file"), "line": frame.get("line"),
        }

        if detail == "concise":
            return {
                "status": "success", "frame": frame_info,
                "variables": _filter_locals_concise(result.get("variables", {})),
            }

        return {"status": "success", "frame": frame_info, "variables": result.get("variables", {})}

    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_variables failed")
        return debug_error("unexpected_error", str(e))


async def debug_stack(
    detail: Annotated[Literal["concise", "full"], Field(description="'concise': function+location; 'full': includes frame IDs")] = "concise",
    thread_id: Annotated[int | None, Field(description="Thread ID (from debug_threads). Defaults to stopped thread.")] = None,
) -> dict[str, Any]:
    """Get the full call stack to see how execution reached the current line.

    USE THIS WHEN:
    - You hit a breakpoint and need to know which caller invoked this function
    - You want to trace the entry point of a code path
    - You're debugging recursion and need to see the stack depth
    - You need a frame_index to pass into debug_variables(frame_index=N)

    REQUIRES: Active debug session paused at a breakpoint or step.
    """
    try:
        client, err = _require_paused_client()
        if err is not None:
            return err

        result = await client.get_stack_trace(levels=MAX_STACK_TRACE_LEVELS, thread_id=thread_id)
        frames = result.get("frames", [])
        include_id = detail == "full"

        formatted = []
        for i, f in enumerate(frames):
            entry: dict[str, Any] = {
                "index": i, "function": f.get("name"),
                "file": f.get("file"), "line": f.get("line"),
            }
            if include_id:
                entry["id"] = f.get("id")
            formatted.append(entry)

        return {"status": "success", "total_frames": len(frames), "frames": formatted}

    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_stack failed")
        return debug_error("unexpected_error", str(e))


async def debug_threads() -> dict[str, Any]:
    """List all threads in the debug session.

    WHEN TO USE:
    - Debugging multithreaded or async applications
    - Identifying which thread hit a breakpoint
    - Before using thread_id parameter in debug_stack/debug_variables

    REQUIRES: Active debug session paused at a breakpoint or step.
    """
    try:
        client, err = _require_paused_client()
        if err is not None:
            return err
        return await client.get_threads()
    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_threads failed")
        return debug_error("unexpected_error", str(e))


# --- Breakpoint action handlers (extracted from debug_breakpoint) ---

BreakpointAction = Literal["set", "list", "clear", "set_exception", "add", "remove", "save", "load"]

# Handler signature: receives all params, uses only what it needs
BpHandler = Callable[..., Awaitable[dict[str, Any]]]


async def _bp_set_exception(client: "DAPClient", exception_filter: str | None, **_: Any) -> dict[str, Any]:
    """Handle debug_breakpoint(action='set_exception')."""
    filter_value = exception_filter or "uncaught"
    return await client.set_exception_breakpoints(filters=[filter_value])


async def _bp_set(
    client: "DAPClient",
    file_path: str | None,
    lines: list[int] | None,
    condition: str | None,
    log_message: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Handle debug_breakpoint(action='set')."""
    if not file_path:
        return missing_param_error("file_path", "debug_breakpoint(action='set')")
    if not lines:
        return missing_param_error("lines", "debug_breakpoint(action='set')")
    bp_conditions = {line: condition for line in lines} if condition else None
    bp_log_messages = {line: log_message for line in lines} if log_message else None
    return await client.set_breakpoints(
        file=file_path, lines=lines,
        conditions=bp_conditions, log_messages=bp_log_messages,
    )


def _extract_bp_metadata(
    bps: list,
    line_filter: set[int] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Build (conditions, log_messages) line→value dicts from tracker entries.

    When `line_filter` is provided, only entries whose `bp.line` is in the
    filter contribute — used by `_bp_remove` to ignore lines being stripped.
    """
    conditions: dict[int, str] = {}
    log_messages: dict[int, str] = {}
    for bp in bps:
        if line_filter is not None and bp.line not in line_filter:
            continue
        if bp.condition:
            conditions[bp.line] = bp.condition
        if bp.log_message:
            log_messages[bp.line] = bp.log_message
    return conditions, log_messages


async def _bp_add(
    client: "DAPClient",
    file_path: str | None,
    lines: list[int] | None,
    condition: str | None,
    log_message: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Handle debug_breakpoint(action='add') — merge new breakpoints with existing.

    Preserves conditions / log_messages on existing breakpoints; applies the
    new ones (if any) to the lines being added. Without this merge,
    set_breakpoints would replace existing bps and silently drop their
    conditions or logpoint messages.
    """
    if not file_path:
        return missing_param_error("file_path", "debug_breakpoint(action='add')")
    if not lines:
        return missing_param_error("lines", "debug_breakpoint(action='add')")
    resolved = str(Path(file_path).resolve())
    existing_bps = client.breakpoints.by_file.get(resolved, [])
    bp_conditions, bp_log_messages = _extract_bp_metadata(existing_bps)
    if condition:
        for ln in lines:
            bp_conditions[ln] = condition
    if log_message:
        for ln in lines:
            bp_log_messages[ln] = log_message
    merged = sorted({bp.line for bp in existing_bps} | set(lines))
    return await client.set_breakpoints(
        file=file_path, lines=merged,
        conditions=bp_conditions or None,
        log_messages=bp_log_messages or None,
    )


async def _bp_remove(client: "DAPClient", file_path: str | None, lines: list[int] | None, **_: Any) -> dict[str, Any]:
    """Handle debug_breakpoint(action='remove') — remove specific lines.

    Must forward surviving breakpoints' conditions and log_messages to
    set_breakpoints; otherwise the wire-level replace semantics strip those
    fields from survivors. (Latent since Phase 3 for conditions; extended to
    log_messages by Phase 4 C8 — Phase 5 C2 fixes both.)
    """
    if not file_path:
        return missing_param_error("file_path", "debug_breakpoint(action='remove')")
    if not lines:
        return missing_param_error("lines", "debug_breakpoint(action='remove')")
    resolved = str(Path(file_path).resolve())
    existing_bps = client.breakpoints.by_file.get(resolved, [])
    remaining = sorted({bp.line for bp in existing_bps} - set(lines))
    bp_conditions, bp_log_messages = _extract_bp_metadata(
        existing_bps, line_filter=set(remaining),
    )
    return await client.set_breakpoints(
        file=file_path, lines=remaining,
        conditions=bp_conditions or None,
        log_messages=bp_log_messages or None,
    )


async def _bp_save(
    client: "DAPClient",
    preset_name: str | None,
    file_path: str | None,
    **_: Any,
) -> dict[str, Any]:
    """Handle debug_breakpoint(action='save') — save preset to disk.

    When file_path is provided, save only bps for that file. Otherwise save all
    currently active bps in the session (bps added this session AND any restored
    from previous sessions). Use file_path to avoid leaking auto-restored bps
    from unrelated programs into a new preset.
    """
    store = get_breakpoint_store()
    name = preset_name or "default"
    if file_path:
        resolved = str(Path(file_path).resolve())
        source_bps = client.breakpoints.by_file.get(resolved, [])
    else:
        source_bps = [bp for bps in client.breakpoints.by_file.values() for bp in bps]
    configs = [
        BreakpointConfig(
            file=bp.file, line=bp.line,
            condition=bp.condition, log_message=bp.log_message,
        )
        for bp in source_bps
    ]
    return await store.save_preset(name, configs)


async def _bp_load(client: "DAPClient", preset_name: str | None, **_: Any) -> dict[str, Any]:
    """Handle debug_breakpoint(action='load') — restore preset from disk."""
    store = get_breakpoint_store()
    name = preset_name or "default"
    configs = await store.load_preset(name)
    if not configs:
        return debug_error("preset_not_found", f"No preset named '{name}'")
    if any(cfg.condition for cfg in configs):
        flag = CAPABILITY_REQUIRED["conditional_breakpoint"]
        if not client.capability(flag):
            return unsupported_capability(
                action="conditional_breakpoint",
                capability_flag=flag,
                adapter_name=client.adapter_name,
            )
    if any(cfg.log_message for cfg in configs):
        flag = CAPABILITY_REQUIRED["log_points"]
        if not client.capability(flag):
            return unsupported_capability(
                action="log_points",
                capability_flag=flag,
                adapter_name=client.adapter_name,
            )
    by_file: dict[str, list[int]] = defaultdict(list)
    conds: dict[str, dict[int, str]] = defaultdict(dict)
    logs: dict[str, dict[int, str]] = defaultdict(dict)
    for cfg in configs:
        by_file[cfg.file].append(cfg.line)
        if cfg.condition:
            conds[cfg.file][cfg.line] = cfg.condition
        if cfg.log_message:
            logs[cfg.file][cfg.line] = cfg.log_message
    restored = 0
    for f, ln in by_file.items():
        result = await client.set_breakpoints(
            file=f, lines=ln,
            conditions=conds.get(f), log_messages=logs.get(f),
        )
        if "error" in result:
            return _breakpoint_update_failed("load", result["error"])
        restored += len(ln)
    return {"status": "success", "restored": restored, "preset": name}


async def _bp_list(client: "DAPClient", **_: Any) -> dict[str, Any]:
    """Handle debug_breakpoint(action='list')."""
    breakpoints = [
        bp.to_dict()
        for bps in client.breakpoints.by_file.values()
        for bp in bps
    ]
    return {"status": "success", "breakpoints": breakpoints, "count": len(breakpoints)}


async def _bp_clear(client: "DAPClient", file_path: str | None, **_: Any) -> dict[str, Any]:
    """Handle debug_breakpoint(action='clear')."""
    if not file_path:
        return missing_param_error("file_path", "debug_breakpoint(action='clear')")
    result = await client.set_breakpoints(file=file_path, lines=[])
    if "error" in result:
        return _breakpoint_update_failed("clear", result["error"])
    return {"status": "success", "message": f"Cleared breakpoints from {file_path}"}


_BP_HANDLERS: dict[BreakpointAction, BpHandler] = {
    "set_exception": _bp_set_exception,
    "set": _bp_set,
    "add": _bp_add,
    "remove": _bp_remove,
    "save": _bp_save,
    "load": _bp_load,
    "list": _bp_list,
    "clear": _bp_clear,
}


async def debug_breakpoint(
    action: Annotated[
        BreakpointAction,
        Field(description=(
            "'set': replace breakpoints in file; 'add': merge new breakpoints; "
            "'remove': remove specific lines; 'list': show current; "
            "'clear': remove all from file; 'set_exception': break on exceptions; "
            "'save': save preset to disk; 'load': restore preset from disk"
        )),
    ],
    file_path: Annotated[str | None, Field(description="File path (REQUIRED for set/add/remove/clear; OPTIONAL for save to restrict preset to one file)")] = None,
    lines: Annotated[list[int] | None, Field(description="Line numbers (REQUIRED for set/add/remove)")] = None,
    condition: Annotated[str | None, Field(description="Break only when true (e.g., 'x > 10')")] = None,
    exception_filter: Annotated[
        Literal["uncaught", "raised", "userUnhandled"] | None,
        Field(description="Exception filter for 'set_exception': 'uncaught' (default), 'raised' (all), 'userUnhandled'"),
    ] = None,
    preset_name: Annotated[str | None, Field(description="Preset name for save/load (default: 'default')")] = None,
    log_message: Annotated[
        str | None,
        Field(description="DAP logpoint: emit this message instead of stopping. Adapter must support logPoints."),
    ] = None,
) -> dict[str, Any]:
    """Manage DAP breakpoints in an active debug session.

    For source code injection (no session needed), use source_add_breakpoint.
    REQUIRES: Active debug session (except save/load which only need stored presets).

    `log_message` turns a breakpoint into a DAP logpoint — execution
    continues and the adapter emits the message via an `output` event.
    Adapter must advertise `supportsLogPoints`; otherwise the call
    returns `unsupported_capability`.
    """
    try:
        client = _get_connected_client()
        if not client:
            return no_session_error()

        if condition is not None and action in ("set", "add"):
            flag = CAPABILITY_REQUIRED["conditional_breakpoint"]
            if not client.capability(flag):
                return unsupported_capability(
                    action="conditional_breakpoint",
                    capability_flag=flag,
                    adapter_name=client.adapter_name,
                )

        if log_message is not None and action in ("set", "add"):
            flag = CAPABILITY_REQUIRED["log_points"]
            if not client.capability(flag):
                return unsupported_capability(
                    action="log_points",
                    capability_flag=flag,
                    adapter_name=client.adapter_name,
                )

        handler = _BP_HANDLERS[action]
        return await handler(
            client=client,
            file_path=file_path,
            lines=lines,
            condition=condition,
            exception_filter=exception_filter,
            preset_name=preset_name,
            log_message=log_message,
        )

    except ImportError:
        return dependency_missing_for_active_adapter(get_dap_client().adapter)
    except Exception as e:
        logger.exception("debug_breakpoint failed")
        return debug_error("unexpected_error", str(e))
