"""Shared error response helpers and constants for all debug MCP tools.

Ensures consistent error format:
{status, error_type, category, message, suggested_action}.
Also provides shared constants to avoid duplication across debug modules.
"""
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from .adapters.base import DebugAdapter


class ErrorCategory(str, Enum):
    """Coarse-grained classification for debug errors, surfaced to the LLM.

    Five buckets chosen so a small-context caller can pick its next action
    without reading the full message: install something, fix its own call,
    fix the user program, switch adapter, or escalate an internal bug.
    """

    INSTALL_MISSING = "install_missing"  # Toolchain binary / runtime dep absent
    CONFIG_ERROR = "config_error"         # Session shape wrong (no session, bad host/port, language unsupported)
    PROGRAM_ERROR = "program_error"       # User program broke (syntax, launch, eval at breakpoint)
    ADAPTER_INTERNAL = "adapter_internal" # Adapter/DAP wire issue; callers can't fix
    TOOL_MISUSE = "tool_misuse"           # Caller passed bad args (missing param, file-not-found, conflicting flags)


# Single source of truth: every error_type literal that reaches the LLM must be
# categorised here. Callers of `debug_error()` don't pass `category` — it is
# looked up on the wire. A `ValueError` at dispatch time (below) forces any new
# error_type to pick a category instead of quietly arriving uncategorised.
_CATEGORY_BY_ERROR_TYPE: dict[str, ErrorCategory] = {
    # install_missing
    "dependency_missing": ErrorCategory.INSTALL_MISSING,
    # config_error
    "no_session": ErrorCategory.CONFIG_ERROR,
    "session_not_paused": ErrorCategory.CONFIG_ERROR,
    "attach_failed": ErrorCategory.CONFIG_ERROR,
    "adapter_resolution_failed": ErrorCategory.CONFIG_ERROR,
    "dap_adapter_not_registered": ErrorCategory.CONFIG_ERROR,
    "dap_not_supported": ErrorCategory.CONFIG_ERROR,
    "no_native_breakpoint": ErrorCategory.CONFIG_ERROR,
    "adapter_unsupported_in_production": ErrorCategory.CONFIG_ERROR,
    # program_error
    "syntax_error": ErrorCategory.PROGRAM_ERROR,
    "launch_failed": ErrorCategory.PROGRAM_ERROR,
    "eval_failed": ErrorCategory.PROGRAM_ERROR,
    # adapter_internal
    "unsupported_capability": ErrorCategory.ADAPTER_INTERNAL,
    "adapter_handshake_failed": ErrorCategory.ADAPTER_INTERNAL,
    "action_failed": ErrorCategory.ADAPTER_INTERNAL,
    "debug_server_failed": ErrorCategory.ADAPTER_INTERNAL,
    "invalid_frame": ErrorCategory.ADAPTER_INTERNAL,
    "scope_resolution_failed": ErrorCategory.ADAPTER_INTERNAL,
    "dirty_state": ErrorCategory.ADAPTER_INTERNAL,
    "unexpected_error": ErrorCategory.ADAPTER_INTERNAL,
    # tool_misuse
    "missing_parameter": ErrorCategory.TOOL_MISUSE,
    "invalid_parameter": ErrorCategory.TOOL_MISUSE,
    "file_not_found": ErrorCategory.TOOL_MISUSE,
    "not_a_file": ErrorCategory.TOOL_MISUSE,
    "unsupported_extension": ErrorCategory.TOOL_MISUSE,
    "line_out_of_range": ErrorCategory.TOOL_MISUSE,
    "preset_not_found": ErrorCategory.TOOL_MISUSE,
    "conflicting_parameters": ErrorCategory.TOOL_MISUSE,
}


class DebugErrorResponse(TypedDict):
    """Standardized error response from debug tools."""

    status: str
    error_type: str
    category: str
    message: str
    suggested_action: NotRequired[str]


class DebugLocationInfo(TypedDict):
    """Location in source code returned by debug context."""

    file: str | None
    line: int | None
    function: str | None

# Truncation limits shared across debug_inspect and debug_simplified
MAX_VARIABLE_VALUE_LENGTH = 200  # Prevents context bloat from large variable repr
MAX_VARIABLES_PER_SCOPE = 20  # Caps per-frame dump to protect LLM token budget; typical frames fit well under this
MAX_STACK_TRACE_LEVELS = 50  # Generous upper bound; real stacks rarely exceed 20
MAX_FRAMES_FULL_DUMP = 10  # Cap for debug_variables(all_frames=True) to avoid token bloat

# Suggestions for breakpoints that were never hit during execution
SUGGESTION_CONDITION_NEVER_TRUE = (
    "Condition was never true. Try without condition to verify the line is reached."
)
SUGGESTION_LINE_NEVER_REACHED = (
    "This line was never reached. Verify the code path is executed."
)

# Valid exception breakpoint filters for debugpy (DAP standard)
VALID_EXCEPTION_FILTERS = frozenset({"uncaught", "raised", "userUnhandled"})

# Single source of truth for capability-flag literals used to gate MCP tools.
# Action name (internal) -> DAP capability flag advertised by the adapter in
# its `initialize` response. Forward-looking entries (set_variable, etc.)
# have no in-tree caller yet — listed so future tools can gate without
# re-introducing hardcoded literals.
CAPABILITY_REQUIRED: dict[str, str] = {
    "conditional_breakpoint": "supportsConditionalBreakpoints",
    "set_variable": "supportsSetVariable",
    "hover_evaluate": "supportsEvaluateForHovers",
    "log_points": "supportsLogPoints",
}


def truncate_value(value: str, max_length: int = MAX_VARIABLE_VALUE_LENGTH) -> str:
    """Truncate long values to prevent context bloat."""
    if len(value) <= max_length:
        return value
    return value[:max_length - 3] + "..."


def debug_error(
    error_type: str,
    message: str,
    suggested_action: str | None = None,
    category: ErrorCategory | None = None,
) -> dict[str, Any]:
    """Create a standardized DebugErrorResponse as dict[str, Any].

    `category` is auto-resolved from `_CATEGORY_BY_ERROR_TYPE` when not
    passed explicitly. Unknown error_types raise `ValueError` so any new
    error literal has to declare its category instead of quietly shipping
    uncategorised (phase-7 invariant #7 — "error categorization exhaustive").

    Shape matches DebugErrorResponse TypedDict:
    {status, error_type, category, message, suggested_action?}.
    """
    resolved = category or _CATEGORY_BY_ERROR_TYPE.get(error_type)
    if resolved is None:
        raise ValueError(
            f"error_type {error_type!r} has no entry in _CATEGORY_BY_ERROR_TYPE; "
            "add it to debug_errors.py or pass category=... explicitly."
        )
    result: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "category": resolved.value,
        "message": message,
    }
    if suggested_action:
        result["suggested_action"] = suggested_action
    return result


def no_session_error() -> dict[str, Any]:
    """No active debug session."""
    return debug_error(
        error_type="no_session",
        message="No active debug session",
        suggested_action="Use debug_session(action='launch', program='...') first",
    )


def session_not_paused_error() -> dict[str, Any]:
    """Session is connected but not paused at an inspectable frame.

    Query tools (debug_eval, debug_variables, debug_stack, debug_threads)
    require a paused frame to read state. Management tools (debug_breakpoint)
    do not gate on this.
    """
    return debug_error(
        error_type="session_not_paused",
        message=(
            "Debug session is connected but not paused at an inspectable frame. "
            "Wait for a breakpoint to hit, or call debug_action(action='pause') "
            "on a running session before inspecting state."
        ),
        suggested_action="debug_action(action='pause') or wait for a breakpoint",
    )


def unsupported_capability(
    action: str,
    capability_flag: str,
    adapter_name: str,
) -> dict[str, Any]:
    """Adapter does not advertise a DAP capability required for this action."""
    return debug_error(
        error_type="unsupported_capability",
        message=(
            f"The active debug adapter ({adapter_name!r}) does not "
            f"advertise {capability_flag}; '{action}' is unavailable."
        ),
        suggested_action=(
            f"Retry without this option, or switch to an adapter that "
            f"supports {capability_flag}."
        ),
    )


def missing_param_error(param: str, tool: str) -> dict[str, Any]:
    """Required parameter not provided."""
    return debug_error(
        error_type="missing_parameter",
        message=f"'{param}' is required for {tool}",
        suggested_action=f"Provide the '{param}' parameter",
    )


def dependency_missing_error(dependency_name: str, install_hint: str) -> dict[str, Any]:
    """Adapter runtime dependency not installed.

    `install_hint` is the exact install command (e.g. 'pip install debugpy'
    or 'go install github.com/go-delve/delve/cmd/dlv@latest'). It comes from
    each adapter's `detect_installed()` so the error surface stays
    adapter-agnostic.
    """
    return debug_error(
        error_type="dependency_missing",
        message=f"{dependency_name} is not installed",
        suggested_action=f"Install with: {install_hint}",
    )


def dependency_missing_for_active_adapter(
    adapter: "DebugAdapter | None",
) -> dict[str, Any]:
    """Error envelope for late ImportError when the active adapter's dep is missing.

    `adapter.detect_installed()` gates known hard-deps before launch, so this
    catch-all only fires when a late import inside a handler raises
    ImportError. The caller passes the active DAPClient's adapter so the
    message names the right dependency, not a hardcoded debugpy fallback.
    """
    if adapter is None:
        # No session attached yet — fall back to debugpy, the only adapter
        # whose Python-level modules can be imported outside launch().
        return dependency_missing_error("debugpy", "pip install debugpy")
    _ok, hint = adapter.detect_installed()
    if not hint:
        hint = f"reinstall the {adapter.name} adapter"
    return dependency_missing_error(adapter.name, hint)


def adapter_unsupported_in_production_error(
    adapter_name: str, reason: str = "",
) -> dict[str, Any]:
    """Caller resolved to an adapter that's declared unsupported in production.

    Fires from the MCP resolver when `adapter.production_supported is False`.
    The adapter class is still registered (so byte-parity contract tests keep
    running); the tool layer refuses to launch so callers don't walk into a
    known-broken adapter path. `reason` comes from
    `DebugAdapter.production_unsupported_reason` — kept short; the exit doc
    carries the full context.
    """
    base_message = f"The {adapter_name!r} adapter is declared unsupported in production"
    message = f"{base_message}: {reason}" if reason else f"{base_message}."
    return debug_error(
        error_type="adapter_unsupported_in_production",
        message=message,
        suggested_action=(
            "Use a supported adapter instead; see "
            "archive/debugger_mcp_docs/polyglot_debugger/work/prod-ready-exit.md for the current matrix."
        ),
    )


def adapter_handshake_failed(adapter_name: str, detail: str) -> dict[str, Any]:
    """Adapter's post_initialize hook raised; launch was aborted.

    Surfaced when an adapter implements `post_initialize()` to issue custom
    DAP requests (e.g. PSES's `powerShell/getVersion`) and that handshake
    fails. `detail` is the string form of the underlying exception.
    """
    return debug_error(
        error_type="adapter_handshake_failed",
        message=f"{adapter_name!r} handshake failed: {detail}",
        suggested_action=(
            f"Check that the {adapter_name} adapter dependencies match the "
            "expected version; see archive/debugger_mcp_docs/debug-adapters.md."
        ),
    )


def file_not_found_error(path: str) -> dict[str, Any]:
    """File does not exist."""
    return debug_error(
        error_type="file_not_found",
        message=f"File not found: {path}",
        suggested_action="Check the file path and try again",
    )


def unsupported_extension_error(ext: str, supported: str) -> dict[str, Any]:
    """File extension not supported for debug injection."""
    return debug_error(
        error_type="unsupported_extension",
        message=f"Unsupported file extension: {ext}. Supported: {supported}",
        suggested_action="Use a supported file type",
    )


def line_out_of_range_error(line: int, total: int) -> dict[str, Any]:
    """Line number exceeds file length."""
    return debug_error(
        error_type="line_out_of_range",
        message=f"Line {line} exceeds file length ({total} lines)",
        suggested_action=f"Use a line number between 1 and {total}",
    )
