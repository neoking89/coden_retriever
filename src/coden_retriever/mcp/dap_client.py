"""
DAP (Debug Adapter Protocol) Client for debugpy.

Enables programmatic debugging through the Debug Adapter Protocol.
The model can set breakpoints, step through code, inspect variables, and more.

Uses debugpy in server mode (--listen --wait-for-client) with socket connection.
"""
import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..constants import DEFAULT_DEBUG_PORT
from .adapters.base import DebugAdapter, IdentityPathMapper, LaunchConfig, PathMapper
from .dap_breakpoint_ops import DAPBreakpointOps
from .dap_breakpoint_tracker import BreakpointTracker, DebugBreakpoint
from .dap_constants import (
    DAP_MSG_POLL_INTERVAL,
    DAP_STACK_PAUSE_TIMEOUT,
)
from .dap_inspection import DAPInspection
from .dap_lifecycle import DAPLifecycle
from .dap_protocol import DAPProtocol
from .dap_status import DebugResult
from .dap_transport import DAPTransport, SocketTransport

# Re-export DebugBreakpoint for back-compat with callers that import it from
# this module's old location (e.g. tests/mcp/test_dap_client_fixes.py).
__all__ = ["DAPClient", "DebugBreakpoint", "PROGRAM_OUTPUT_BUFFER_CAP"]

logger = logging.getLogger(__name__)

# Keep program_output bounded — a chatty test can otherwise grow the list
# unboundedly over a long debug session. 100 lines is enough context for
# the termination summary; older output is trimmed FIFO.
PROGRAM_OUTPUT_BUFFER_CAP = 100


@dataclass
class StackFrame:
    """A stack frame from the debugger."""
    id: int
    name: str
    file: str | None
    line: int
    column: int = 0


@dataclass
class Variable:
    """A variable from the debugger."""
    name: str
    value: str
    type: str | None = None
    variables_reference: int = 0  # Non-zero means it has children


@dataclass
class DebugState:
    """Current state of the debug session."""
    is_running: bool = False
    is_stopped: bool = False
    stopped_reason: str | None = None
    stopped_file: str | None = None
    stopped_line: int | None = None
    stopped_description: str | None = None  # Exception text from DAP stopped event
    thread_id: int | None = None
    current_frame_id: int | None = None
    program: str | None = None
    program_output: list[str] = field(default_factory=list)
    program_terminated: bool = False


class DAPClient:
    """
    Debug Adapter Protocol client for communicating with DAP servers.

    Per-adapter transport: socket (debugpy) or stdio (everything else) via
    the `DebugAdapter` abstraction.

    Note on public method count: this class has ~22 public methods,
    exceeding the project 7-method limit (CLAUDE.md). Exempt by design —
    each public method maps 1:1 to a DAP protocol command, and the class
    is a thin wire-level façade. Splitting into sub-objects (`.wire`,
    `.introspection`) would fragment the protocol surface without adding
    value. Invariant traded: MCP envelopes stay OUT of this class — no
    error-dict shapes, no capability checks here. Those live one layer
    up in debug_inspect.py / debug_simplified.py.
    """

    DEFAULT_PORT = DEFAULT_DEBUG_PORT

    def __init__(self) -> None:
        # Default to SocketTransport; launch() swaps in StdioTransport when
        # the adapter declares transport_type="stdio".
        self.transport: DAPTransport = SocketTransport()
        self.protocol = DAPProtocol(self.transport)
        self._adapter: DebugAdapter | None = None
        self._path_mapper: PathMapper = IdentityPathMapper()
        # Port obtained from `adapter.prepare_launch(cfg)` — when non-None AND
        # the adapter uses socket transport AND its argv is empty, `_spawn_adapter_process`
        # bypasses the subprocess spawn and dials this port directly. Reset in stop().
        self._prepared_port: int | None = None
        self._message_processor_task: asyncio.Task | None = None
        self._state = DebugState()
        self.breakpoints = BreakpointTracker()
        self._stop_event = asyncio.Event()
        # Per DAP spec, the adapter emits 'initialized' after the initialize
        # response to signal it's ready for configuration (setBreakpoints,
        # configurationDone). Sending those before the event causes debugpy
        # to reject configurationDone with 'only allowed during handling of
        # a launch or an attach request'.
        self._initialized_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self.last_reset_dirty: bool = False  # True when stop() couldn't join the reader thread cleanly
        self._event_handlers: dict[str, Callable[[dict], None]] = {
            "initialized": self._on_initialized,
            "stopped": self._on_stopped,
            "terminated": self._on_terminated,
            "exited": self._on_exited,
            "thread": self._on_thread,
            "output": self._on_output,
            "continued": self._on_continued,
        }
        # `_append_program_output` is invoked from BOTH the asyncio event
        # loop (via `_on_output`) AND the stderr drain `threading.Thread`.
        # Without a mutex the non-atomic append + slice-trim can drop
        # lines or raise IndexError under load. A `threading.Lock` (not
        # asyncio.Lock) is required because one caller is a pure thread.
        self._program_output_lock = threading.Lock()
        self._inspection = DAPInspection(self)
        self._bp_ops = DAPBreakpointOps(self)
        self._lifecycle = DAPLifecycle(self)

    @property
    def state(self) -> DebugState:
        """Get current debug state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to debug adapter."""
        return self.transport.is_connected and self._running

    def capability(self, name: str) -> bool:
        """Read a single capability reported by the adapter's initialize response.

        Used by MCP tools to gate capability-dependent requests (e.g., conditional
        breakpoints) before they reach the adapter. `.get(..., False)` is
        deliberately conservative: an adapter that omits a flag is treated as
        not supporting it.
        """
        return bool(self.protocol.capabilities.get(name))

    @property
    def adapter_name(self) -> str:
        """Registered adapter name for the active session, or a placeholder if none."""
        return self._adapter.name if self._adapter else "<unknown>"

    @property
    def adapter(self) -> DebugAdapter | None:
        """Adapter instance for the active session, or None before launch/attach."""
        return self._adapter

    async def start(self, python_path: str | None = None) -> dict[str, Any]:
        """
        Start connection (placeholder - actual connection happens during launch).

        Args:
            python_path: Path to Python interpreter. Uses sys.executable if not specified.

        Returns:
            Status dict.
        """
        return DebugResult.ready("Use debug_launch to start debugging a script").to_dict()

    async def stop(self) -> dict[str, Any]:
        """Stop the debug session and clean up."""
        return await self._lifecycle.stop()

    async def attach(
        self,
        adapter: DebugAdapter,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_DEBUG_PORT,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Attach to an already-running DAP adapter on `host:port`."""
        return await self._lifecycle.attach(
            adapter, host=host, port=port, timeout=timeout,
        )

    async def launch(
        self,
        cfg: LaunchConfig,
        adapter: DebugAdapter,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Spawn the adapter for `cfg.program` and drive the DAP handshake."""
        return await self._lifecycle.launch(cfg, adapter, timeout=timeout)

    async def _process_messages(self) -> None:
        """Drain the transport's queue into the message handler."""
        while self._running:
            try:
                msg = self.transport.message_queue.get_nowait()
                await self._handle_message(msg)
            except queue.Empty:
                await asyncio.sleep(DAP_MSG_POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Message processor error: {e}")

    async def _handle_message(self, message: dict) -> None:
        """Handle a message from the adapter (response, event, or reverse request)."""
        msg_type = message.get("type")
        if msg_type == "response":
            self.protocol.route_response(message)
            return
        if msg_type == "request":
            await self._handle_reverse_request(message)
            return
        if msg_type != "event":
            return
        event_name = message.get("event", "")
        body = message.get("body", {})
        handler = self._event_handlers.get(event_name)
        if handler is not None:
            handler(body)
        # Adapter-level hook runs for every event, built-in or not — lets
        # adapters like R service vscDebugger's `custom/writeToStdin` events
        # that the built-in handlers don't cover.
        if self._adapter is not None:
            await self._adapter.handle_adapter_event(self, event_name, body)

    async def _handle_reverse_request(self, message: dict) -> None:
        """Route a server→client DAP `request` through the active adapter.

        Every request must get a response per DAP spec, even on error — we
        never drop the message. Adapter exceptions become a well-formed
        failure response rather than propagating up into the message loop.
        """
        command = message.get("command", "")
        arguments = message.get("arguments", {}) or {}
        request_seq = message.get("seq", 0)
        if self._adapter is None:
            success, body = False, {"error": "no adapter bound to this session"}
        else:
            try:
                success, body = await self._adapter.handle_reverse_request(
                    command, arguments,
                )
            except Exception as exc:
                success, body = False, {
                    "error": f"reverse-request handler raised: {exc}",
                }
        await self.protocol.send_response(request_seq, command, success, body)

    def _on_initialized(self, body: dict) -> None:
        self._initialized_event.set()

    def _clear_stop_state(self) -> None:
        """Clear stop-related state when execution resumes or the program exits.

        Without this, get_status() reports a stale stopped_reason (e.g.
        "breakpoint") even while is_stopped=False, confusing callers.
        """
        self._state.stopped_reason = None
        self._state.stopped_description = None
        self._state.stopped_file = None
        self._state.stopped_line = None

    def _on_stopped(self, body: dict) -> None:
        self._state.is_stopped = True
        self._state.stopped_reason = body.get("reason", "unknown")
        self._state.thread_id = body.get("threadId")
        if body.get("reason") == "exception":
            self._state.stopped_description = body.get("text") or body.get("description")
        else:
            self._state.stopped_description = None
        self.breakpoints.track_hit_event(body)
        self._stop_event.set()
        # Adapter-level side-effect hook — R injects a `.vsc.listenForDAP()`
        # kickoff into stdin when paused on exception so the DAP channel
        # stays reachable. Default no-op on every other adapter.
        if self._adapter is not None:
            self._adapter.on_stopped(self, body)

    def _on_terminated(self, body: dict) -> None:
        self._state.is_running = False
        self._state.is_stopped = False
        self._state.program_terminated = True
        self._clear_stop_state()
        self._stop_event.set()

    def _on_exited(self, body: dict) -> None:
        exit_code = body.get("exitCode", 0)
        self._state.is_running = False
        self._state.program_terminated = True
        self._state.program_output.append(f"[Process exited with code {exit_code}]")
        self._clear_stop_state()
        self._stop_event.set()

    def _on_thread(self, body: dict) -> None:
        logger.debug(
            f"Thread event: {body.get('reason', 'unknown')} (id={body.get('threadId')})"
        )

    def _on_continued(self, body: dict) -> None:
        self._state.is_stopped = False
        self._clear_stop_state()

    def _on_output(self, body: dict) -> None:
        category = body.get("category", "console")
        output = body.get("output", "")
        if output.strip() and category not in ("telemetry",):
            self._append_program_output(output.rstrip())
        if category == "telemetry":
            logger.debug(f"[{category}] {output.strip()}")

    def _append_program_output(self, line: str) -> None:
        """Append a single line to program_output, capped at PROGRAM_OUTPUT_BUFFER_CAP.

        Used by DAP `output` events and by the stderr drain thread (SC-003).
        The append + trim runs under `_program_output_lock` because the two
        call sites live in different concurrency domains (asyncio loop vs
        OS thread) and the list mutation is not atomic.
        """
        if not line:
            return
        with self._program_output_lock:
            self._state.program_output.append(line)
            if len(self._state.program_output) > PROGRAM_OUTPUT_BUFFER_CAP:
                self._state.program_output = self._state.program_output[-PROGRAM_OUTPUT_BUFFER_CAP:]

    async def _wait_for_stop(self, timeout: float = 30.0) -> bool:
        """Wait for the debugger to stop or terminate."""
        # Don't clear here - caller should clear before starting operation
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Check if we're already stopped or terminated (race condition)
            if not (self._state.is_stopped or self._state.program_terminated):
                return False

        # Auto-populate frame_id and location when stopped
        if self._state.is_stopped and not self._state.program_terminated:
            await self._inspection.refresh_frame_context()

        return True

    def _get_stop_info(self) -> dict[str, Any]:
        """Get info about current stop."""
        if self._state.is_stopped:
            return DebugResult.stop_info(
                reason=self._state.stopped_reason,
                file=self._state.stopped_file,
                line=self._state.stopped_line,
                exception=self._state.stopped_description,
            ).to_dict()
        if self._state.program_terminated:
            output = self._state.program_output[-10:] if self._state.program_output else []
            return DebugResult.terminated(output=output).to_dict()
        if not self._state.is_running:
            return DebugResult.not_running().to_dict()
        return DebugResult.running().to_dict()

    async def set_breakpoints(
        self,
        file: str,
        lines: list[int],
        conditions: dict[int, str] | None = None,
        log_messages: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Set breakpoints in a file."""
        return await self._bp_ops.set_breakpoints(
            file, lines, conditions=conditions, log_messages=log_messages,
        )

    async def set_exception_breakpoints(self, filters: list[str]) -> dict[str, Any]:
        """Set exception breakpoints (filters: 'raised', 'uncaught', 'userUnhandled')."""
        return await self._bp_ops.set_exception_breakpoints(filters)

    async def continue_execution(self, timeout: float = 60.0) -> dict[str, Any]:
        """Continue execution until next breakpoint or program end."""
        if not self.is_connected:
            return {"error": "Not connected"}
        # `is None` (not falsy) so threadId=0 is honoured — js-debug's child
        # session uses 0 as its primary thread id.
        if self._state.thread_id is None:
            return {"error": "No active thread"}

        # Clear state BEFORE clearing event to avoid race condition
        self._state.is_stopped = False
        self._clear_stop_state()
        self._stop_event.clear()

        # singleThread tells debugpy to resume only the reporting thread; others
        # stay paused so their queued 'stopped' events surface on the next wait.
        # Without this, debugpy coalesces sibling breakpoint hits into one event.
        response = await self.protocol.send_request("continue", {
            "threadId": self._state.thread_id,
            "singleThread": True,
        })

        if response.get("success"):
            stopped = await self._wait_for_stop(timeout=timeout)
            if not stopped and not self._state.program_terminated:
                return DebugResult.timeout(
                    f"Program did not stop within {timeout} seconds."
                ).to_dict()
            return self._get_stop_info()
        else:
            return {"error": response.get("message", "Continue failed")}

    async def step_over(self) -> dict[str, Any]:
        """Step over to the next line."""
        return await self._step("next")

    async def step_into(self) -> dict[str, Any]:
        """Step into a function call."""
        return await self._step("stepIn")

    async def step_out(self) -> dict[str, Any]:
        """Step out of the current function."""
        return await self._step("stepOut")

    async def pause(self, thread_id: int | None = None) -> dict[str, Any]:
        """Pause execution of a running thread.

        If no thread_id is known yet (no prior stop event), queries the adapter,
        picks MainThread (or the first thread if MainThread is absent), and uses that.
        """
        if not self.is_connected:
            return {"error": "Not connected"}
        tid = thread_id if thread_id is not None else self._state.thread_id
        if tid is None:
            tid = await self._resolve_pause_thread()
            if tid is None:
                return {"error": "No threads available to pause"}
            self._state.thread_id = tid

        self._stop_event.clear()
        response = await self.protocol.send_request("pause", {"threadId": tid})
        if response.get("success"):
            await self._wait_for_stop(timeout=DAP_STACK_PAUSE_TIMEOUT)
            return self._get_stop_info()
        return {"error": response.get("message", "Pause failed")}

    async def _resolve_pause_thread(self) -> int | None:
        """Query adapter for threads and pick MainThread or the first available."""
        threads_result = await self.get_threads()
        threads = threads_result.get("threads", [])
        if not threads:
            return None
        for t in threads:
            if t.get("name") == "MainThread":
                return t["id"]
        return threads[0]["id"]

    async def get_threads(self) -> dict[str, Any]:
        """Get all threads in the debug session."""
        return await self._inspection.get_threads()

    async def _step(self, command: str, timeout: float = 30.0) -> dict[str, Any]:
        """Execute a step command."""
        if not self.is_connected:
            return {"error": "Not connected"}
        if self._state.thread_id is None:
            return {"error": "No active thread"}
        if not self._state.is_stopped:
            return {"error": "Program is running. Wait for it to stop."}

        # Clear state BEFORE clearing event to avoid race condition
        self._state.is_stopped = False
        self._clear_stop_state()
        self._stop_event.clear()

        response = await self.protocol.send_request(command, {
            "threadId": self._state.thread_id,
            "granularity": "statement",
            "singleThread": True,
        })

        if response.get("success"):
            stopped = await self._wait_for_stop(timeout=timeout)
            if not stopped and not self._state.program_terminated:
                return DebugResult.timeout(
                    f"Step did not complete within {timeout} seconds."
                ).to_dict()
            return self._get_stop_info()
        else:
            return {"error": response.get("message", f"{command} failed")}

    async def get_stack_trace(
        self, levels: int = 20, thread_id: int | None = None,
    ) -> dict[str, Any]:
        """Get the stack trace for a thread (defaults to active thread)."""
        return await self._inspection.get_stack_trace(levels=levels, thread_id=thread_id)

    async def get_variables(self, frame_id: int | None = None) -> dict[str, Any]:
        """Get variables in the current scope, grouped by scope name."""
        return await self._inspection.get_variables(frame_id=frame_id)

    async def get_scopes(self, frame_id: int) -> dict[str, Any]:
        """Return the DAP `scopes` response body for a given frame."""
        return await self._inspection.get_scopes(frame_id)

    async def set_variable(
        self,
        variables_reference: int,
        name: str,
        value: str,
    ) -> dict[str, Any]:
        """DAP `setVariable` request — wire-level only."""
        return await self._inspection.set_variable(variables_reference, name, value)

    async def get_variable_children(self, variables_reference: int) -> dict[str, Any]:
        """Expand a variable to see its children (object properties, list items, etc.)."""
        return await self._inspection.get_variable_children(variables_reference)

    async def evaluate(
        self, expression: str, frame_id: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate an expression in the current context."""
        return await self._inspection.evaluate(expression, frame_id=frame_id)

    def get_breakpoint_summary(self) -> dict[str, Any] | None:
        """Summarize breakpoint hit/miss status. Returns None if no breakpoints were set."""
        return self.breakpoints.get_summary()

    def get_status(self) -> dict[str, Any]:
        """Get the current debug session status."""
        return {
            "connected": self.is_connected,
            "program": self._state.program,
            "is_running": self._state.is_running,
            "is_stopped": self._state.is_stopped,
            "stopped_reason": self._state.stopped_reason,
            "stopped_file": self._state.stopped_file,
            "stopped_line": self._state.stopped_line,
            "thread_id": self._state.thread_id,
            "program_terminated": self._state.program_terminated,
            "recent_output": self._state.program_output[-10:] if self._state.program_output else [],
        }


# Global singleton
_dap_client: DAPClient | None = None
_dap_lock: asyncio.Lock | None = None
# Guards the lazy init of `_dap_lock`. Without this, two coroutines can
# each observe `_dap_lock is None` and construct their own `asyncio.Lock`,
# leaving later callers acquiring *different* objects — zero mutual
# exclusion. `threading.Lock` is intentional: coroutines interleave at
# every `await`, so the check-and-set must run under a non-async
# synchronisation primitive.
_dap_lock_init_guard = threading.Lock()


def _get_lock() -> asyncio.Lock:
    """Get or create the global lock (lazy initialization for event loop compatibility)."""
    global _dap_lock
    with _dap_lock_init_guard:
        if _dap_lock is None:
            _dap_lock = asyncio.Lock()
        return _dap_lock


def get_dap_client() -> DAPClient:
    """Get or create the global DAP client.

    Note: The client may need cleanup via reset_dap_client() if in a bad state.
    The debug_session() function handles this automatically before launch.
    """
    global _dap_client
    if _dap_client is None:
        _dap_client = DAPClient()
    return _dap_client


async def reset_dap_client() -> None:
    """Reset the global DAP client (for testing or cleanup)."""
    global _dap_client, _dap_lock
    lock = _get_lock()
    async with lock:
        if _dap_client is not None:
            try:
                await asyncio.wait_for(_dap_client.stop(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Force cleanup even if stop fails
            _dap_client = None
    # Also reset the lock for next event loop. Mutate under the same
    # init guard used by `_get_lock()` so a concurrent lazy-init sees
    # a consistent view.
    with _dap_lock_init_guard:
        _dap_lock = None


async def get_or_reset_dap_client() -> DAPClient:
    """Get the global DAP client, resetting if in a bad state.

    This is safer for the simplified API as it ensures clean state.
    """
    global _dap_client
    lock = _get_lock()
    async with lock:
        if _dap_client is not None:
            # Check if client is in a potentially bad state
            if _dap_client.transport.process is not None:
                poll_result = _dap_client.transport.process.poll()
                if poll_result is not None:
                    # Process has exited - reset the client
                    try:
                        await asyncio.wait_for(_dap_client.stop(), timeout=2.0)
                    except (asyncio.TimeoutError, Exception):
                        pass
                    _dap_client = None

        if _dap_client is None:
            _dap_client = DAPClient()
        return _dap_client
