"""Persistent storage for source-injected breakpoints and traces + DAP server
process handle.

The `DebugSessionManager` owns two concerns:
1. The on-disk catalog of injected source breakpoints / trace statements
   (survives MCP restarts so `source_remove_injections` can clean up).
2. The handle to the out-of-process debugpy helper used by `debug_server`.

Split out of `debug_trace.py` so the module matches the pattern of
`debug_breakpoint_store.py` (storage classes live in `_store.py` files) and
so `debug_trace.py` can stay focused on source-injection tool logic.
"""
import asyncio
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


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
    """Represents an injected trace statement.

    `group_id` ties together the start/end markers of a region trace
    (`source_inject_region_trace`). Single-line traces leave it None.
    """

    id: str
    file_path: str
    line_number: int
    trace_code: str
    original_line: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    group_id: str | None = None


# ruff: noqa: PLR0904  (class owns both the injection catalog AND the DAP
# server handle — splitting further would force the debug_server tool to
# coordinate across two singletons and broke the _write_lock invariant when
# tried. Sanctioned class-level exemption per CLAUDE.md.)
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
        self._dap_server_process: subprocess.Popen[bytes] | None = None

    # --- DAP server lifecycle (encapsulates _dap_* fields) ---

    def start_server(
        self, process: subprocess.Popen[bytes], port: int,
    ) -> None:
        """Record a running DAP helper subprocess.

        Replaces direct mutation of `_dap_server_process`, `_dap_port`, and
        `_dap_started` from module-level functions. Called exactly once per
        successful `debug_server(action='start')`.
        """
        self._dap_server_process = process
        self._dap_port = port
        self._dap_started = True

    def stop_server(self) -> subprocess.Popen[bytes] | None:
        """Clear DAP server state and return the previously-held process handle.

        Returning the handle lets the caller terminate/wait on it outside the
        manager so this method stays synchronous and does not need an event
        loop. The manager's state is cleared even when no process was held
        (idempotent) so a zombie-subprocess refresh can reuse this path.
        """
        proc = self._dap_server_process
        self._dap_server_process = None
        self._dap_started = False
        self._dap_port = None
        return proc

    def is_dap_active(self) -> bool:
        """Whether `debug_server` has an active helper subprocess."""
        return self._dap_started

    def dap_port(self) -> int | None:
        """Port the helper subprocess is listening on, or None when inactive."""
        return self._dap_port

    def dap_server_process(self) -> subprocess.Popen[bytes] | None:
        """Raw subprocess handle — callers that terminate/wait need this."""
        return self._dap_server_process

    # --- On-disk catalog of injected source breakpoints + traces ---

    def _get_state_file(self) -> Path:
        """Get the path to the debug state file."""
        state_dir = Path.home() / ".coden-retriever"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "debug_state.json"

    def _load_state_sync_locked(self) -> None:
        """Perform the actual disk read. MUST be called with _write_lock held
        (or in a context that doesn't need the lock, like inside _load_state
        after its own lock acquisition).
        """
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

    async def _load_state(self) -> None:
        """Load persisted state from disk (read-safe for callers outside _write_lock).

        Uses a double-checked locking pattern: the fast path avoids acquiring
        `_write_lock` when state is already loaded (the common case), while
        the slow path re-checks under the lock so two concurrent callers
        cannot race on the disk read and double-populate the in-memory maps.
        Callers already holding `_write_lock` must use `_load_state_locked`
        instead to avoid deadlocking on the non-reentrant asyncio.Lock.
        """
        if self._state_loaded:
            return

        async with self._write_lock:
            await self._load_state_locked()

    async def _load_state_locked(self) -> None:
        """Load state when `_write_lock` is already held by the caller.

        WHY separate from `_load_state`: asyncio.Lock is not reentrant, so
        methods like `add_breakpoint` that already hold the lock must skip
        re-acquiring it. This variant performs the same idempotent load.
        """
        if self._state_loaded:
            return
        await asyncio.to_thread(self._load_state_sync_locked)
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
            await self._load_state_locked()
            self._breakpoints[bp.id] = bp
            await self._save_state()

    async def remove_breakpoint(self, bp_id: str) -> Breakpoint | None:
        """Remove a breakpoint from state."""
        async with self._write_lock:
            await self._load_state_locked()
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
            await self._load_state_locked()
            self._traces[trace.id] = trace
            await self._save_state()

    async def remove_trace(self, trace_id: str) -> InjectedTrace | None:
        """Remove a trace from state."""
        async with self._write_lock:
            await self._load_state_locked()
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

    # WHY KEPT: Only bulk-reset for debug state; completes the CRUD interface
    # and is the only way to clear all breakpoints/traces without restarting.
    async def clear_all(self) -> tuple[int, int]:
        """Bulk-clear all breakpoints and traces. Completes the CRUD interface for debug session management."""
        async with self._write_lock:
            await self._load_state_locked()
            bp_count = len(self._breakpoints)
            trace_count = len(self._traces)
            self._breakpoints.clear()
            self._traces.clear()
            await self._save_state()
            return bp_count, trace_count

    async def prune_missing(self) -> tuple[int, int]:
        """Remove breakpoints and traces whose files no longer exist."""
        async with self._write_lock:
            await self._load_state_locked()
            bp_removed = [
                bp_id for bp_id, bp in self._breakpoints.items()
                if not Path(bp.file_path).exists()
            ]
            trace_removed = [
                t_id for t_id, t in self._traces.items()
                if not Path(t.file_path).exists()
            ]
            for bp_id in bp_removed:
                del self._breakpoints[bp_id]
            for t_id in trace_removed:
                del self._traces[t_id]
            if bp_removed or trace_removed:
                await self._save_state()
            return len(bp_removed), len(trace_removed)


# Global singleton instance used by the source-injection MCP tools.
_manager = DebugSessionManager()


def get_debug_session_manager() -> DebugSessionManager:
    """Accessor for the process-wide DebugSessionManager singleton."""
    return _manager
