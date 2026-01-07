"""
Debug Trace Tools for MCP.

Provides tools for adding breakpoints and trace statements to source code.
Supports Python, JavaScript, and TypeScript files.
Supports both DAP (Debug Adapter Protocol) integration and source code injection.
Works with VS Code, PyCharm, and other DAP-compatible IDEs.
"""
import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Language-agnostic marker tag for injected trace statements
# The comment prefix (# or //) is added per-language
_TRACE_MARKER = "[DEBUG_TRACE]"


@dataclass
class LanguageConfig:
    """Language-specific syntax for debugging."""

    name: str
    comment_prefix: str
    breakpoint_stmt: str


# Supported file extensions mapped to their language configurations
SUPPORTED_EXTENSIONS: dict[str, LanguageConfig] = {
    ".py": LanguageConfig(
        name="python",
        comment_prefix="#",
        breakpoint_stmt="breakpoint()",
    ),
    ".js": LanguageConfig(
        name="javascript",
        comment_prefix="//",
        breakpoint_stmt="debugger;",
    ),
    ".jsx": LanguageConfig(
        name="javascript",
        comment_prefix="//",
        breakpoint_stmt="debugger;",
    ),
    ".ts": LanguageConfig(
        name="typescript",
        comment_prefix="//",
        breakpoint_stmt="debugger;",
    ),
    ".tsx": LanguageConfig(
        name="typescript",
        comment_prefix="//",
        breakpoint_stmt="debugger;",
    ),
    ".mjs": LanguageConfig(
        name="javascript",
        comment_prefix="//",
        breakpoint_stmt="debugger;",
    ),
    ".cjs": LanguageConfig(
        name="javascript",
        comment_prefix="//",
        breakpoint_stmt="debugger;",
    ),
}


def _get_language_config(file_path: Path) -> LanguageConfig | None:
    """Get language configuration based on file extension."""
    return SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())


@dataclass
class Breakpoint:
    """Represents a breakpoint set in source code."""

    id: str
    file_path: str
    line_number: int
    mode: Literal["dap", "source"]
    condition: str | None = None
    log_message: str | None = None
    original_line: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True


@dataclass
class InjectedTrace:
    """Represents an injected trace statement."""

    id: str
    file_path: str
    line_number: int
    trace_code: str
    original_line: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DebugSessionManager:
    """Manages debug sessions, breakpoints, and traces with thread-safe operations."""

    def __init__(self):
        """Initialize the Debug Session Manager."""
        self._breakpoints: dict[str, Breakpoint] = {}
        self._traces: dict[str, InjectedTrace] = {}
        self._write_lock = asyncio.Lock()
        self._dap_port: int | None = None
        self._dap_started: bool = False
        self._state_loaded: bool = False

    def _get_state_file(self) -> Path:
        """Get the path to the debug state file."""
        state_dir = Path.home() / ".coden-retriever"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "debug_state.json"

    async def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self._state_loaded:
            return

        def _load_sync() -> None:
            state_file = self._get_state_file()
            if not state_file.exists():
                return

            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))

                for bp_data in data.get("breakpoints", []):
                    bp = Breakpoint(**bp_data)
                    self._breakpoints[bp.id] = bp

                for trace_data in data.get("traces", []):
                    trace = InjectedTrace(**trace_data)
                    self._traces[trace.id] = trace

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"Failed to load debug state: {e}")

        await asyncio.to_thread(_load_sync)
        self._state_loaded = True

    async def _save_state(self) -> None:
        """Persist state to disk."""

        def _save_sync() -> None:
            state_file = self._get_state_file()
            data = {
                "breakpoints": [asdict(bp) for bp in self._breakpoints.values()],
                "traces": [asdict(t) for t in self._traces.values()],
            }
            state_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )

        await asyncio.to_thread(_save_sync)

    async def add_breakpoint(self, bp: Breakpoint) -> None:
        """Add a breakpoint to state."""
        async with self._write_lock:
            await self._load_state()
            self._breakpoints[bp.id] = bp
            await self._save_state()

    async def remove_breakpoint(self, bp_id: str) -> Breakpoint | None:
        """Remove a breakpoint from state."""
        async with self._write_lock:
            await self._load_state()
            bp = self._breakpoints.pop(bp_id, None)
            if bp:
                await self._save_state()
            return bp

    async def get_breakpoints(
        self, file_path: str | None = None
    ) -> list[Breakpoint]:
        """Get breakpoints, optionally filtered by file."""
        await self._load_state()
        if file_path:
            normalized = str(Path(file_path).resolve())
            return [
                bp
                for bp in self._breakpoints.values()
                if str(Path(bp.file_path).resolve()) == normalized
            ]
        return list(self._breakpoints.values())

    async def add_trace(self, trace: InjectedTrace) -> None:
        """Add a trace to state."""
        async with self._write_lock:
            await self._load_state()
            self._traces[trace.id] = trace
            await self._save_state()

    async def remove_trace(self, trace_id: str) -> InjectedTrace | None:
        """Remove a trace from state."""
        async with self._write_lock:
            await self._load_state()
            trace = self._traces.pop(trace_id, None)
            if trace:
                await self._save_state()
            return trace

    async def get_traces(self, file_path: str | None = None) -> list[InjectedTrace]:
        """Get traces, optionally filtered by file."""
        await self._load_state()
        if file_path:
            normalized = str(Path(file_path).resolve())
            return [
                t
                for t in self._traces.values()
                if str(Path(t.file_path).resolve()) == normalized
            ]
        return list(self._traces.values())

    async def clear_all(self) -> tuple[int, int]:
        """Clear all breakpoints and traces from state."""
        async with self._write_lock:
            await self._load_state()
            bp_count = len(self._breakpoints)
            trace_count = len(self._traces)
            self._breakpoints.clear()
            self._traces.clear()
            await self._save_state()
            return bp_count, trace_count


# Global singleton instance
_manager = DebugSessionManager()


def _generate_id(prefix: str = "dbg") -> str:
    """Generate a unique ID for breakpoints/traces."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _create_backup(file_path: Path) -> Path:
    """Create a backup of the file before modification."""

    def _backup_sync() -> Path:
        backup_path = file_path.with_suffix(file_path.suffix + ".debugbak")
        content = file_path.read_text(encoding="utf-8")
        backup_path.write_text(content, encoding="utf-8")
        return backup_path

    return await asyncio.to_thread(_backup_sync)


async def _read_file_lines(file_path: Path) -> list[str]:
    """Read file and return lines with preserved endings."""

    def _read_sync() -> list[str]:
        content = file_path.read_text(encoding="utf-8")
        return content.splitlines(keepends=True)

    return await asyncio.to_thread(_read_sync)


async def _write_file_lines(file_path: Path, lines: list[str]) -> None:
    """Write lines back to file."""

    def _write_sync() -> None:
        content = "".join(lines)
        file_path.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_write_sync)


def _get_indentation(line: str) -> str:
    """Extract the indentation from a line."""
    stripped = line.lstrip()
    if not stripped:
        return ""
    return line[: len(line) - len(stripped)]


def _escape_for_python_fstring(s: str) -> str:
    """Escape a string for safe inclusion in a Python f-string with double quotes.

    Escapes backslashes, double quotes, and curly braces to prevent syntax errors
    and unintended format string interpolation.
    """
    # Escape backslashes first, then double quotes, then curly braces
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("{", "{{").replace("}", "}}")


def _escape_for_js_template_literal(s: str) -> str:
    """Escape a string for safe inclusion in a JavaScript template literal.

    Escapes backslashes, backticks, and template literal interpolation markers
    to prevent syntax errors and unintended interpolation.
    """
    # Escape backslashes first, then backticks, then ${
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def _generate_trace_statement(
    config: LanguageConfig,
    variables: list[str] | None,
    message: str | None,
    file_path: str,
    line_number: int,
    include_timestamp: bool,
    include_location: bool,
) -> str:
    """Generate a trace print/console.log statement based on language."""
    marker = f"{config.comment_prefix} {_TRACE_MARKER}"
    file_name = Path(file_path).name

    if config.name == "python":
        # Python: use f-string with print()
        parts = []

        if include_timestamp:
            parts.append("{__import__('datetime').datetime.now().isoformat()}")

        if include_location:
            parts.append(f"{file_name}:{line_number}")

        if message:
            # Escape the message for safe inclusion in Python f-string
            parts.append(_escape_for_python_fstring(message))

        var_parts = []
        if variables:
            for var in variables:
                var_parts.append(f"{var}={{{var}!r}}")

        if var_parts:
            parts.append(f"| {', '.join(var_parts)}")

        prefix = "[TRACE] " if parts else "[TRACE]"
        inner = " ".join(parts)

        return f'print(f"{prefix}{inner}")  {marker}'
    else:
        # JavaScript/TypeScript: use template literal with console.log()
        parts = []

        if include_timestamp:
            parts.append("${new Date().toISOString()}")

        if include_location:
            parts.append(f"{file_name}:{line_number}")

        if message:
            # Escape the message for safe inclusion in JS template literal
            parts.append(_escape_for_js_template_literal(message))

        var_parts = []
        if variables:
            for var in variables:
                var_parts.append(f"{var}=${{JSON.stringify({var})}}")

        if var_parts:
            parts.append(f"| {', '.join(var_parts)}")

        prefix = "[TRACE] " if parts else "[TRACE]"
        inner = " ".join(parts)

        return f"console.log(`{prefix}{inner}`);  {marker}"


def _generate_breakpoint_code(
    config: LanguageConfig,
    condition: str | None = None,
    log_message: str | None = None,
) -> str:
    """Generate breakpoint code based on language."""
    marker = f"{config.comment_prefix} {_TRACE_MARKER}"

    if log_message:
        # Logpoint: print/console.log instead of breaking
        if config.name == "python":
            escaped_msg = _escape_for_python_fstring(log_message)
            return f'print(f"[LOGPOINT] {escaped_msg}")  {marker}'
        else:
            escaped_msg = _escape_for_js_template_literal(log_message)
            return f"console.log(`[LOGPOINT] {escaped_msg}`);  {marker}"

    if condition:
        # Conditional breakpoint
        if config.name == "python":
            return f"if {condition}: {config.breakpoint_stmt}  {marker}"
        else:
            return f"if ({condition}) {{ {config.breakpoint_stmt} }}  {marker}"

    # Simple breakpoint
    return f"{config.breakpoint_stmt}  {marker}"


async def add_breakpoint(
    file_path: Annotated[
        str,
        Field(description="Absolute path to the source file (.py, .js, .ts, .jsx, .tsx)"),
    ],
    line_number: Annotated[
        int,
        Field(description="Line number (1-based) where to set the breakpoint", ge=1),
    ],
    mode: Annotated[
        Literal["dap", "source", "auto"],
        Field(
            description=(
                "'source' injects breakpoint()/debugger; into code, "
                "'dap' uses Debug Adapter Protocol (Python only, requires active session), "
                "'auto' tries DAP first then falls back to source"
            )
        ),
    ] = "auto",
    condition: Annotated[
        str | None,
        Field(description="Optional condition expression (e.g., 'x > 10')"),
    ] = None,
    log_message: Annotated[
        str | None,
        Field(
            description=(
                "If provided, creates a logpoint that logs this message "
                "instead of breaking execution"
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Inject a breakpoint/debugger statement into source code (modifies the file).

    Supports Python (.py), JavaScript (.js, .jsx, .mjs, .cjs), and TypeScript (.ts, .tsx).

    WHEN TO USE:
    - When you want to add a persistent breakpoint that triggers when the script runs
    - When using an external debugger (VS Code, PyCharm, browser DevTools)
    - For conditional breakpoints or logpoints
    - For JS/TS: injects `debugger;` statement

    WHEN NOT TO USE:
    - For interactive Python debugging sessions - use debug_launch instead

    NOTE: This modifies the source file. Use remove_injections to clean up.
    """
    try:
        path = Path(file_path).resolve()

        # Validate file exists
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        if not path.is_file():
            return {"error": f"Not a file: {file_path}"}

        # Get language configuration
        config = _get_language_config(path)
        if not config:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            return {"error": f"Unsupported file extension: {path.suffix}. Supported: {supported}"}

        # Read file
        lines = await _read_file_lines(path)
        if line_number > len(lines):
            return {
                "error": f"Line {line_number} exceeds file length ({len(lines)} lines)"
            }

        # Determine actual mode
        actual_mode: Literal["dap", "source"] = "source"
        if mode == "dap":
            # DAP only supports Python
            if config.name != "python":
                return {
                    "error": f"DAP mode only supports Python files. Use mode='source' for {config.name}."
                }
            # Check if DAP session is active
            if not _manager._dap_started:
                return {
                    "error": "No active DAP session. Use start_debug_session first, or use mode='source'"
                }
            actual_mode = "dap"
        elif mode == "auto":
            # Try DAP if available and Python, otherwise source
            if _manager._dap_started and config.name == "python":
                actual_mode = "dap"
            else:
                actual_mode = "source"

        bp_id = _generate_id("bp")

        if actual_mode == "source":
            # Create backup
            await _create_backup(path)

            # Get the target line and its indentation
            target_line = lines[line_number - 1]
            indent = _get_indentation(target_line)

            # Generate breakpoint code for the language
            bp_code = _generate_breakpoint_code(config, condition, log_message)

            # Insert breakpoint before the target line
            new_line = f"{indent}{bp_code}\n"
            lines.insert(line_number - 1, new_line)

            # Write back
            await _write_file_lines(path, lines)

            # Store breakpoint
            bp = Breakpoint(
                id=bp_id,
                file_path=str(path),
                line_number=line_number,
                mode="source",
                condition=condition,
                log_message=log_message,
                original_line=target_line.rstrip("\n\r"),
            )
            await _manager.add_breakpoint(bp)

            return {
                "status": "success",
                "breakpoint_id": bp_id,
                "mode": "source",
                "language": config.name,
                "file": str(path),
                "line": line_number,
                "injected_code": bp_code,
                "message": f"Breakpoint added at {path.name}:{line_number}",
            }

        else:
            # DAP mode - set breakpoint via debugpy
            try:
                import debugpy  # type: ignore

                # debugpy.breakpoint() only works at call site
                # For line-specific breakpoints, we still need source injection
                # but we can notify the DAP client
                return {
                    "error": (
                        "DAP line-specific breakpoints require source injection. "
                        "Use mode='source' or debugpy.breakpoint() in your code."
                    )
                }
            except ImportError:
                return {
                    "error": "debugpy not installed. Install with: pip install debugpy"
                }

    except Exception as e:
        logger.exception(f"Failed to add breakpoint: {e}")
        return {"error": f"Failed to add breakpoint: {e}"}


async def remove_injections(
    injection_type: Annotated[
        Literal["breakpoint", "trace", "all"],
        Field(
            description=(
                "'breakpoint': remove injected breakpoints; "
                "'trace': remove injected trace statements; "
                "'all': remove both breakpoints and traces"
            )
        ),
    ] = "all",
    injection_id: Annotated[
        str | None,
        Field(description="Specific injection ID to remove (breakpoint or trace ID)"),
    ] = None,
    file_path: Annotated[
        str | None,
        Field(description="Remove all injections from this file"),
    ] = None,
    remove_all: Annotated[
        bool,
        Field(description="Remove all injections of the specified type"),
    ] = False,
) -> dict[str, Any]:
    """Remove injected breakpoints and/or trace statements from source files.

    Works for Python (.py), JavaScript (.js, .jsx, .mjs, .cjs), and TypeScript (.ts, .tsx).
    Identifies injected lines by the [DEBUG_TRACE] marker regardless of comment style.

    WHEN TO USE:
    - To clean up after using add_breakpoint or inject_trace
    - To remove injected code from source files
    - Use injection_type='all' to clean up everything at once

    NOTE: This is for source-injected code. DAP breakpoints (from debug_set_breakpoint)
    are automatically cleared when the debug session ends.
    """
    try:
        if not injection_id and not file_path and not remove_all:
            return {
                "error": "Specify injection_id, file_path, or remove_all=True"
            }

        removed_breakpoints = []
        removed_traces = []
        errors = []

        # Determine which types to process
        process_breakpoints = injection_type in ("breakpoint", "all")
        process_traces = injection_type in ("trace", "all")

        # Collect items to remove
        breakpoints_to_remove: list[Breakpoint] = []
        traces_to_remove: list[InjectedTrace] = []

        if process_breakpoints:
            if remove_all:
                breakpoints_to_remove = await _manager.get_breakpoints()
            elif file_path:
                breakpoints_to_remove = await _manager.get_breakpoints(file_path)
            elif injection_id and injection_id.startswith("bp-"):
                for b in await _manager.get_breakpoints():
                    if b.id == injection_id:
                        breakpoints_to_remove = [b]
                        break

        if process_traces:
            if remove_all:
                traces_to_remove = await _manager.get_traces()
            elif file_path:
                traces_to_remove = await _manager.get_traces(file_path)
            elif injection_id and injection_id.startswith("trace-"):
                for t in await _manager.get_traces():
                    if t.id == injection_id:
                        traces_to_remove = [t]
                        break

        if not breakpoints_to_remove and not traces_to_remove:
            return {"status": "success", "removed": 0, "message": "No injections found"}

        # Group all items by file for efficient processing
        files_to_process: dict[str, dict[str, list]] = {}

        for bp in breakpoints_to_remove:
            if bp.file_path not in files_to_process:
                files_to_process[bp.file_path] = {"breakpoints": [], "traces": []}
            files_to_process[bp.file_path]["breakpoints"].append(bp)

        for trace in traces_to_remove:
            if trace.file_path not in files_to_process:
                files_to_process[trace.file_path] = {"breakpoints": [], "traces": []}
            files_to_process[trace.file_path]["traces"].append(trace)

        # Process each file
        for fpath, items in files_to_process.items():
            path = Path(fpath)

            # Handle missing files
            if not path.exists():
                for bp in items["breakpoints"]:
                    await _manager.remove_breakpoint(bp.id)
                    removed_breakpoints.append(bp.id)
                for trace in items["traces"]:
                    await _manager.remove_trace(trace.id)
                    removed_traces.append(trace.id)
                continue

            try:
                lines = await _read_file_lines(path)

                # Remove all lines containing our trace marker (from bottom up)
                indices_to_remove = []
                for i, line in enumerate(lines):
                    if _TRACE_MARKER in line:
                        indices_to_remove.append(i)

                for idx in reversed(indices_to_remove):
                    lines.pop(idx)

                await _write_file_lines(path, lines)

                # Update state
                for bp in items["breakpoints"]:
                    await _manager.remove_breakpoint(bp.id)
                    removed_breakpoints.append(bp.id)
                for trace in items["traces"]:
                    await _manager.remove_trace(trace.id)
                    removed_traces.append(trace.id)

            except Exception as e:
                errors.append(f"{fpath}: {e}")

        result: dict[str, Any] = {
            "status": "success",
            "removed_breakpoints": len(removed_breakpoints),
            "removed_traces": len(removed_traces),
            "total_removed": len(removed_breakpoints) + len(removed_traces),
            "removed_ids": removed_breakpoints + removed_traces,
        }
        if errors:
            result["errors"] = errors

        return result

    except Exception as e:
        logger.exception(f"Failed to remove injections: {e}")
        return {"error": f"Failed to remove injections: {e}"}


async def list_injections(
    injection_type: Annotated[
        Literal["breakpoint", "trace", "all"],
        Field(
            description=(
                "'breakpoint': list only breakpoints; "
                "'trace': list only traces; "
                "'all': list both"
            )
        ),
    ] = "all",
    file_path: Annotated[
        str | None,
        Field(description="Filter injections by file path"),
    ] = None,
) -> dict[str, Any]:
    """List all source-injected breakpoints and/or traces.

    WHEN TO USE:
    - To see what is injected in source files
    - To get IDs for remove_injections
    """
    try:
        # Always include all keys for consistent response structure
        result: dict[str, Any] = {
            "status": "success",
            "dap_session_active": _manager._dap_started,
            "breakpoints": [],
            "breakpoint_count": 0,
            "traces": [],
            "trace_count": 0,
        }

        if injection_type in ("breakpoint", "all"):
            breakpoints = await _manager.get_breakpoints(file_path)
            result["breakpoints"] = [
                {
                    "id": bp.id,
                    "file": bp.file_path,
                    "line": bp.line_number,
                    "mode": bp.mode,
                    "condition": bp.condition,
                    "log_message": bp.log_message,
                    "created_at": bp.created_at,
                }
                for bp in breakpoints
            ]
            result["breakpoint_count"] = len(breakpoints)

        if injection_type in ("trace", "all"):
            traces = await _manager.get_traces(file_path)
            result["traces"] = [
                {
                    "id": t.id,
                    "file": t.file_path,
                    "line": t.line_number,
                    "trace_code": t.trace_code,
                    "created_at": t.created_at,
                }
                for t in traces
            ]
            result["trace_count"] = len(traces)

        return result

    except Exception as e:
        logger.exception(f"Failed to list injections: {e}")
        return {"error": f"Failed to list injections: {e}"}


async def inject_trace(
    file_path: Annotated[
        str,
        Field(description="Absolute path to the source file (.py, .js, .ts, .jsx, .tsx)"),
    ],
    line_number: Annotated[
        int,
        Field(
            description="Line number (1-based) after which to inject the trace",
            ge=1,
        ),
    ],
    variables: Annotated[
        list[str] | None,
        Field(description="List of variable names to log (e.g., ['x', 'y', 'result'])"),
    ] = None,
    message: Annotated[
        str | None,
        Field(description="Custom message to include in the trace output"),
    ] = None,
    include_timestamp: Annotated[
        bool,
        Field(description="Include timestamp in trace output"),
    ] = True,
    include_location: Annotated[
        bool,
        Field(description="Include file:line in trace output"),
    ] = True,
) -> dict[str, Any]:
    """Inject a print/console.log statement into source code to log variable values (modifies the file).

    Supports Python (.py), JavaScript (.js, .jsx, .mjs, .cjs), and TypeScript (.ts, .tsx).

    WHEN TO USE:
    - When you want to see variable values by running the script normally (not in debugger)
    - When you want to trace execution flow with print/console.log statements
    - Quick alternative to full debugging

    WHEN NOT TO USE:
    - For interactive Python debugging - use debug_launch + debug_get_variables instead

    NOTE: This modifies the source file. Use remove_injections to clean up.
    Output format:
    - Python: [TRACE] timestamp file:line message | var1=repr(value1)
    - JS/TS:  [TRACE] timestamp file:line message | var1=JSON.stringify(value1)
    """
    try:
        path = Path(file_path).resolve()

        # Validate file exists
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        if not path.is_file():
            return {"error": f"Not a file: {file_path}"}

        # Get language configuration
        config = _get_language_config(path)
        if not config:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            return {"error": f"Unsupported file extension: {path.suffix}. Supported: {supported}"}

        # Read file
        lines = await _read_file_lines(path)
        if line_number > len(lines):
            return {
                "error": f"Line {line_number} exceeds file length ({len(lines)} lines)"
            }

        # Create backup
        await _create_backup(path)

        # Get the target line and its indentation
        target_line = lines[line_number - 1]
        indent = _get_indentation(target_line)

        # Generate trace statement for the language
        trace_code = _generate_trace_statement(
            config=config,
            variables=variables,
            message=message,
            file_path=str(path),
            line_number=line_number,
            include_timestamp=include_timestamp,
            include_location=include_location,
        )

        # Insert trace after the target line
        new_line = f"{indent}{trace_code}\n"
        lines.insert(line_number, new_line)

        # Write back
        await _write_file_lines(path, lines)

        # Store trace
        trace_id = _generate_id("trace")
        trace = InjectedTrace(
            id=trace_id,
            file_path=str(path),
            line_number=line_number + 1,  # Actual line where trace was inserted
            trace_code=trace_code,
            original_line=target_line.rstrip("\n\r"),
        )
        await _manager.add_trace(trace)

        return {
            "status": "success",
            "trace_id": trace_id,
            "language": config.name,
            "file": str(path),
            "line": line_number + 1,
            "injected_code": trace_code,
            "message": f"Trace added after {path.name}:{line_number}",
        }

    except Exception as e:
        logger.exception(f"Failed to inject trace: {e}")
        return {"error": f"Failed to inject trace: {e}"}


async def debug_server(
    action: Annotated[
        Literal["start", "stop", "status"],
        Field(
            description=(
                "'start': start debug server for IDE attachment; "
                "'stop': stop the debug server; "
                "'status': check if server is running"
            )
        ),
    ],
    port: Annotated[
        int,
        Field(description="Port for debugpy to listen on (only used with action='start')", ge=1024, le=65535),
    ] = 5678,
    wait_for_client: Annotated[
        bool,
        Field(description="Block until a debugger client connects (only used with action='start')"),
    ] = False,
) -> dict[str, Any]:
    """Control a debug server for external IDE attachment (VS Code, PyCharm).

    WHEN TO USE:
    - 'start': when user wants to debug with their IDE
    - 'stop': to stop the debug server
    - 'status': to check if server is running

    WHEN NOT TO USE:
    - For interactive debugging controlled by this tool - use debug_launch instead

    After starting, connect IDE debugger to localhost:<port>.
    """
    try:
        if action == "status":
            return {
                "status": "running" if _manager._dap_started else "not_running",
                "port": _manager._dap_port,
            }

        if action == "stop":
            if not _manager._dap_started:
                return {"status": "not_running", "message": "No debug session is active"}

            _manager._dap_started = False
            _manager._dap_port = None
            return {
                "status": "stopped",
                "message": "Debug session stopped. Note: debugpy may still be loaded in the process.",
            }

        # action == "start"
        try:
            import debugpy  # type: ignore
        except ImportError:
            return {
                "error": (
                    "debugpy not installed. Install with: pip install debugpy "
                    "or pip install coden-retriever[debug]"
                )
            }

        if _manager._dap_started:
            return {
                "status": "already_running",
                "port": _manager._dap_port,
                "message": f"Debug session already active on port {_manager._dap_port}",
            }

        # Bind to localhost only to prevent network exposure
        debugpy.listen(("127.0.0.1", port))
        _manager._dap_port = port
        _manager._dap_started = True

        if wait_for_client:
            await asyncio.to_thread(debugpy.wait_for_client)
            return {
                "status": "connected",
                "port": port,
                "message": f"Debugger connected on port {port}",
            }

        return {
            "status": "listening",
            "port": port,
            "message": f"Debug server listening on port {port}. Connect your IDE debugger to localhost:{port}",
            "vscode_launch_config": {
                "name": "Python: Attach",
                "type": "debugpy",
                "request": "attach",
                "connect": {"host": "localhost", "port": port},
            },
        }

    except Exception as e:
        logger.exception(f"Failed to {action} debug server: {e}")
        return {"error": f"Failed to {action} debug server: {e}"}




def register_debug_tools(mcp: "FastMCP", disabled_tools: set[str] | None = None) -> None:
    """Register debug trace MCP tools.

    Registers two sets of debugging tools:

    1. SIMPLIFIED DAP TOOLS - 3 high-level tools that auto-return rich context:
       - debug_session: lifecycle management (launch, stop, status)
       - debug_action: execution flow (step, continue) with auto-context
       - debug_state: deep inspection (eval, variables, stack, breakpoints)

    2. SOURCE INJECTION TOOLS - Simple tools that modify source code directly.
       These work without a DAP session for quick debugging.
    """
    disabled = disabled_tools or set()

    from .debug_simplified import (
        debug_session,
        debug_action,
        debug_state,
    )

    simplified_tools = [
        debug_session,
        debug_action,
        debug_state,
    ]

    # Source injection tools (simple, no debugpy needed)
    source_tools = [
        add_breakpoint,
        remove_injections,
        list_injections,
        inject_trace,
        debug_server,
    ]

    # Register simplified DAP tools
    for func in simplified_tools:
        if func.__name__ not in disabled:
            mcp.tool()(func)

    # Register source injection tools
    for func in source_tools:
        if func.__name__ not in disabled:
            mcp.tool()(func)
