"""
DAP (Debug Adapter Protocol) Client for debugpy.

Enables programmatic debugging through the Debug Adapter Protocol.
The model can set breakpoints, step through code, inspect variables, and more.

Uses debugpy in server mode (--listen --wait-for-client) with socket connection.
"""
import asyncio
import json
import logging
import platform
import socket
import subprocess
import sys
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..constants import DEFAULT_DEBUG_PORT

logger = logging.getLogger(__name__)


@dataclass
class DebugBreakpoint:
    """A verified breakpoint."""
    id: int
    file: str
    line: int
    verified: bool = True
    condition: str | None = None


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
    thread_id: int | None = None
    current_frame_id: int | None = None
    program: str | None = None
    program_output: list[str] = field(default_factory=list)
    program_terminated: bool = False


class DAPClient:
    """
    Debug Adapter Protocol client for communicating with debugpy.

    Uses debugpy in server mode: starts the script with debugpy --listen --wait-for-client,
    then connects via socket for DAP communication.
    """

    DEFAULT_PORT = DEFAULT_DEBUG_PORT

    def __init__(self):
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._seq = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._reader_thread: threading.Thread | None = None
        self._message_processor_task: asyncio.Task | None = None
        self._state = DebugState()
        self._breakpoints: dict[str, list[DebugBreakpoint]] = {}  # file -> breakpoints
        self._capabilities: dict[str, Any] = {}
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._message_queue: queue.Queue = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._port = self.DEFAULT_PORT

    @property
    def state(self) -> DebugState:
        """Get current debug state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to debug adapter."""
        return self._socket is not None and self._running

    async def start(self, python_path: str | None = None) -> dict[str, Any]:
        """
        Start connection (placeholder - actual connection happens during launch).

        Args:
            python_path: Path to Python interpreter. Uses sys.executable if not specified.

        Returns:
            Status dict.
        """
        # Connection happens during launch, just return ready status
        return {"status": "ready", "message": "Use debug_launch to start debugging a script"}

    async def stop(self) -> dict[str, Any]:
        """Stop the debug session and clean up."""
        self._running = False

        # Cancel message processor task
        if self._message_processor_task and not self._message_processor_task.done():
            self._message_processor_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._message_processor_task),
                    timeout=1.0
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._message_processor_task = None

        if self._socket:
            try:
                # Try graceful disconnect with short timeout
                await asyncio.wait_for(
                    self._send_request("disconnect", {"terminateDebuggee": True}),
                    timeout=2.0
                )
            except Exception:
                pass
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        # Wait for reader thread to finish
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        self._state = DebugState()
        self._breakpoints.clear()
        self._pending_requests.clear()
        self._seq = 0

        return {"status": "stopped"}

    async def launch(
        self,
        program: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stop_on_entry: bool = False,
        timeout: float = 30.0,
        python_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Launch a Python script for debugging.

        Args:
            program: Path to the Python script to debug.
            args: Command line arguments for the script.
            cwd: Working directory.
            env: Environment variables.
            stop_on_entry: If True, break at the first line.
            timeout: Maximum time to wait for launch (seconds).
            python_path: Path to Python interpreter.

        Returns:
            Launch result.
        """
        # Clean up any existing session
        if self.is_connected:
            await self.stop()

        program_path = Path(program).resolve()
        if not program_path.exists():
            return {"error": f"Program not found: {program}"}

        python = python_path or sys.executable
        self._port = self._find_free_port()

        # Build debugpy command
        cmd = [
            python, "-m", "debugpy",
            "--listen", f"127.0.0.1:{self._port}",
            "--wait-for-client",
            str(program_path),
        ]

        if args:
            cmd.extend(args)

        # Set up working directory and environment
        proc_cwd = cwd or str(program_path.parent)
        proc_env = None
        if env:
            import os
            proc_env = os.environ.copy()
            proc_env.update(env)

        # Start the debug server
        # Use DEVNULL and CREATE_NEW_PROCESS_GROUP on Windows to avoid
        # pipe deadlock and handle inheritance issues in nested subprocess contexts
        try:
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._process = subprocess.Popen(
                cmd,
                cwd=proc_cwd,
                env=proc_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
        except Exception as e:
            return {"error": f"Failed to start debugpy: {e}"}

        # Wait for debugpy to start
        await asyncio.sleep(0.5)

        # Check if process is still running
        if self._process.poll() is not None:
            return {"error": f"debugpy exited immediately with code {self._process.returncode}"}

        # Connect to the debug server
        try:
            await self._connect(timeout=timeout)
        except Exception as e:
            self._process.terminate()
            self._process = None
            return {"error": f"Failed to connect to debugpy: {e}"}

        # Initialize the adapter
        try:
            response = await self._send_request("initialize", {
                "clientID": "mcp-debugger",
                "clientName": "MCP Debug Client",
                "adapterID": "debugpy",
                "pathFormat": "path",
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "supportsVariableType": True,
            }, timeout=timeout)
        except asyncio.TimeoutError:
            await self.stop()
            return {"error": "Initialize timed out"}

        if not response.get("success"):
            await self.stop()
            return {"error": response.get("message", "Initialize failed")}

        self._capabilities = response.get("body", {})

        # Use attach (not launch) because debugpy is already running
        # Send attach request - response may come after other events
        await self._send_request("attach", {"justMyCode": False}, timeout=timeout)

        # Set breakpoint at first executable line if stop_on_entry
        # Find first non-docstring/non-comment line
        first_line = self._find_first_code_line(program_path) if stop_on_entry else None
        if first_line:
            await self._send_request("setBreakpoints", {
                "source": {"path": str(program_path)},
                "breakpoints": [{"line": first_line}],
            })

        # Send configurationDone to start execution
        try:
            cfg_response = await self._send_request("configurationDone", {}, timeout=timeout)
        except asyncio.TimeoutError:
            await self.stop()
            return {"error": "configurationDone timed out"}

        if not cfg_response.get("success"):
            await self.stop()
            return {"error": cfg_response.get("message", "configurationDone failed")}

        self._state.is_running = True
        self._state.is_stopped = False
        self._state.program = str(program_path)
        self._state.program_output = []
        self._state.program_terminated = False

        # Wait for stop if stop_on_entry
        if stop_on_entry:
            stopped = await self._wait_for_stop(timeout=timeout)
            if stopped:
                return {
                    "status": "launched",
                    "program": str(program_path),
                    "stopped": True,
                    "reason": self._state.stopped_reason,
                    "file": self._state.stopped_file,
                    "line": self._state.stopped_line,
                }

        return {
            "status": "launched",
            "program": str(program_path),
            "stopped": self._state.is_stopped,
        }

    def _find_free_port(self) -> int:
        """Find a free port for debugpy."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    def _find_first_code_line(self, program_path: Path) -> int | None:
        """Find the first executable line in a Python file."""
        try:
            content = program_path.read_text()
            lines = content.split("\n")
            in_docstring = False
            docstring_char = None

            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                # Skip empty lines
                if not stripped:
                    continue

                # Skip comments
                if stripped.startswith("#"):
                    continue

                # Handle docstrings
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    quote = stripped[:3]
                    if in_docstring and docstring_char == quote:
                        in_docstring = False
                        docstring_char = None
                    elif not in_docstring:
                        # Check if docstring ends on same line
                        if stripped.count(quote) >= 2:
                            continue
                        in_docstring = True
                        docstring_char = quote
                    continue

                if in_docstring:
                    if docstring_char and docstring_char in stripped:
                        in_docstring = False
                        docstring_char = None
                    continue

                # This is a code line
                return i

            return 1  # Fallback to line 1
        except Exception:
            return 1

    async def _connect(self, timeout: float = 10.0) -> None:
        """Connect to the debugpy server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)

        # Retry connection a few times
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                self._socket.connect(('127.0.0.1', self._port))
                break
            except (ConnectionRefusedError, socket.timeout):
                await asyncio.sleep(0.2)
        else:
            raise ConnectionError(f"Could not connect to debugpy on port {self._port}")

        self._socket.setblocking(False)
        self._running = True
        self._loop = asyncio.get_event_loop()

        # Start reader thread
        self._message_queue = queue.Queue()
        self._reader_thread = threading.Thread(target=self._reader_thread_func, daemon=True)
        self._reader_thread.start()

        # Start message processor
        self._message_processor_task = asyncio.create_task(self._process_messages())

    def _reader_thread_func(self) -> None:
        """Read messages from socket in a separate thread."""
        if not self._socket:
            return

        buffer = b""
        self._socket.setblocking(True)
        self._socket.settimeout(0.5)

        while self._running:
            try:
                chunk = self._socket.recv(4096)
                if not chunk:
                    if self._running:
                        logger.debug("Socket closed by server")
                    break

                buffer += chunk

                # Parse complete messages
                while True:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        break

                    header = buffer[:header_end].decode("utf-8")
                    content_length = 0
                    for line in header.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            content_length = int(line.split(":")[1].strip())
                            break

                    if content_length == 0:
                        break

                    msg_start = header_end + 4
                    msg_end = msg_start + content_length

                    if len(buffer) < msg_end:
                        break

                    content = buffer[msg_start:msg_end].decode("utf-8")
                    buffer = buffer[msg_end:]

                    try:
                        msg = json.loads(content)
                        self._message_queue.put(msg)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {content[:100]}")

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.debug(f"Reader thread error: {e}")
                break

    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                msg = self._message_queue.get_nowait()
                await self._handle_message(msg)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Message processor error: {e}")

    async def _send_request(self, command: str, arguments: dict, timeout: float = 30.0) -> dict:
        """Send a DAP request and wait for response."""
        async with self._lock:
            self._seq += 1
            seq = self._seq

            message = {
                "seq": seq,
                "type": "request",
                "command": command,
                "arguments": arguments,
            }

            future: asyncio.Future = asyncio.Future()
            self._pending_requests[seq] = future

            await self._write_message(message)

            try:
                response = await asyncio.wait_for(future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self._pending_requests.pop(seq, None)
                return {"success": False, "message": f"Timeout waiting for {command} response"}

    async def _write_message(self, message: dict) -> None:
        """Write a DAP message to the socket."""
        if not self._socket:
            return

        content = json.dumps(message)
        data = f"Content-Length: {len(content)}\r\n\r\n{content}"

        # Use run_in_executor for the blocking socket send
        if self._socket is None:
            raise ConnectionError("Socket not connected")
        sock = self._socket  # Capture for lambda
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: sock.sendall(data.encode("utf-8")))

    async def _handle_message(self, message: dict) -> None:
        """Handle a message from the adapter."""
        msg_type = message.get("type")

        if msg_type == "response":
            request_seq = message.get("request_seq")
            if request_seq in self._pending_requests:
                future = self._pending_requests.pop(request_seq)
                if not future.done():
                    future.set_result(message)

        elif msg_type == "event":
            event = message.get("event")
            body = message.get("body", {})

            if event == "stopped":
                self._state.is_stopped = True
                self._state.stopped_reason = body.get("reason", "unknown")
                self._state.thread_id = body.get("threadId")
                self._stop_event.set()

            elif event == "terminated":
                self._state.is_running = False
                self._state.is_stopped = False
                self._state.program_terminated = True
                self._stop_event.set()

            elif event == "exited":
                exit_code = body.get("exitCode", 0)
                self._state.is_running = False
                self._state.program_terminated = True
                self._state.program_output.append(f"[Process exited with code {exit_code}]")
                self._stop_event.set()

            elif event == "output":
                category = body.get("category", "console")
                output = body.get("output", "")
                if output.strip() and category not in ("telemetry",):
                    self._state.program_output.append(output.rstrip())
                    if len(self._state.program_output) > 100:
                        self._state.program_output = self._state.program_output[-100:]
                if category == "telemetry":
                    logger.debug(f"[{category}] {output.strip()}")

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
            await self._refresh_frame_context()

        return True

    async def _refresh_frame_context(self) -> None:
        """Refresh frame context (frame_id, file, line) after stopping.

        This is called automatically after any stop to ensure evaluate() and
        get_variables() work correctly with the current frame.
        """
        if not self._state.thread_id:
            return

        try:
            response = await self._send_request("stackTrace", {
                "threadId": self._state.thread_id,
                "startFrame": 0,
                "levels": 1,
            }, timeout=5.0)

            if response.get("success"):
                frames = response.get("body", {}).get("stackFrames", [])
                if frames:
                    top_frame = frames[0]
                    self._state.current_frame_id = top_frame.get("id")
                    source = top_frame.get("source", {})
                    self._state.stopped_file = source.get("path")
                    self._state.stopped_line = top_frame.get("line", 0)
        except Exception as e:
            logger.debug(f"Failed to refresh frame context: {e}")

    def _get_stop_info(self) -> dict[str, Any]:
        """Get info about current stop."""
        if self._state.is_stopped:
            return {
                "status": "stopped",
                "reason": self._state.stopped_reason,
                "file": self._state.stopped_file,
                "line": self._state.stopped_line,
            }
        elif self._state.program_terminated:
            return {
                "status": "terminated",
                "output": self._state.program_output[-10:] if self._state.program_output else [],
            }
        elif not self._state.is_running:
            return {"status": "not_running"}
        else:
            return {"status": "running"}

    async def set_breakpoints(
        self,
        file: str,
        lines: list[int],
        conditions: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Set breakpoints in a file."""
        if not self.is_connected:
            return {"error": "Not connected. Launch a program first."}

        file_path = Path(file).resolve()
        if not file_path.exists():
            return {"error": f"File not found: {file}"}

        conditions = conditions or {}
        breakpoints: list[dict[str, Any]] = []
        for line in lines:
            bp: dict[str, Any] = {"line": line}
            if line in conditions:
                bp["condition"] = conditions[line]
            breakpoints.append(bp)

        response = await self._send_request("setBreakpoints", {
            "source": {"path": str(file_path)},
            "breakpoints": breakpoints,
        })

        if response.get("success"):
            verified = []
            for bp in response.get("body", {}).get("breakpoints", []):
                dbp = DebugBreakpoint(
                    id=bp.get("id", 0),
                    file=str(file_path),
                    line=bp.get("line", 0),
                    verified=bp.get("verified", False),
                )
                verified.append(dbp)

            self._breakpoints[str(file_path)] = verified

            return {
                "status": "success",
                "breakpoints": [
                    {"id": bp.id, "file": bp.file, "line": bp.line, "verified": bp.verified}
                    for bp in verified
                ],
            }
        else:
            return {"error": response.get("message", "Failed to set breakpoints")}

    async def continue_execution(self, timeout: float = 60.0) -> dict[str, Any]:
        """Continue execution until next breakpoint or program end."""
        if not self.is_connected:
            return {"error": "Not connected"}
        if not self._state.thread_id:
            return {"error": "No active thread"}

        # Clear state BEFORE clearing event to avoid race condition
        self._state.is_stopped = False
        self._stop_event.clear()

        response = await self._send_request("continue", {
            "threadId": self._state.thread_id,
        })

        if response.get("success"):
            stopped = await self._wait_for_stop(timeout=timeout)
            if not stopped and not self._state.program_terminated:
                return {
                    "status": "timeout",
                    "message": f"Program did not stop within {timeout} seconds.",
                }
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

    async def _step(self, command: str, timeout: float = 30.0) -> dict[str, Any]:
        """Execute a step command."""
        if not self.is_connected:
            return {"error": "Not connected"}
        if not self._state.thread_id:
            return {"error": "No active thread"}
        if not self._state.is_stopped:
            return {"error": "Program is running. Wait for it to stop."}

        # Clear state BEFORE clearing event to avoid race condition
        self._state.is_stopped = False
        self._stop_event.clear()

        response = await self._send_request(command, {
            "threadId": self._state.thread_id,
            "granularity": "statement",
        })

        if response.get("success"):
            stopped = await self._wait_for_stop(timeout=timeout)
            if not stopped and not self._state.program_terminated:
                return {
                    "status": "timeout",
                    "message": f"Step did not complete within {timeout} seconds.",
                }
            return self._get_stop_info()
        else:
            return {"error": response.get("message", f"{command} failed")}

    async def get_stack_trace(self, levels: int = 20) -> dict[str, Any]:
        """Get the current stack trace."""
        if not self.is_connected:
            return {"error": "Not connected"}
        if not self._state.thread_id:
            return {"error": "No active thread"}
        if not self._state.is_stopped:
            return {"error": "Program is running"}

        response = await self._send_request("stackTrace", {
            "threadId": self._state.thread_id,
            "startFrame": 0,
            "levels": levels,
        })

        if response.get("success"):
            frames = []
            for frame in response.get("body", {}).get("stackFrames", []):
                source = frame.get("source", {})
                sf = StackFrame(
                    id=frame.get("id", 0),
                    name=frame.get("name", ""),
                    file=source.get("path"),
                    line=frame.get("line", 0),
                    column=frame.get("column", 0),
                )
                frames.append(sf)

                # Update state with current location
                if not self._state.stopped_file and source.get("path"):
                    self._state.stopped_file = source.get("path")
                    self._state.stopped_line = frame.get("line", 0)

            if frames:
                self._state.current_frame_id = frames[0].id

            return {
                "status": "success",
                "frames": [
                    {"id": f.id, "name": f.name, "file": f.file, "line": f.line}
                    for f in frames
                ],
            }
        else:
            return {"error": response.get("message", "Failed to get stack trace")}

    async def get_variables(self, frame_id: int | None = None) -> dict[str, Any]:
        """Get variables in the current scope."""
        if not self.is_connected:
            return {"error": "Not connected"}
        if not self._state.is_stopped:
            return {"error": "Program is running"}

        frame = frame_id or self._state.current_frame_id
        if not frame:
            return {"error": "No frame selected. Get stack trace first."}

        scopes_response = await self._send_request("scopes", {"frameId": frame})

        if not scopes_response.get("success"):
            return {"error": scopes_response.get("message", "Failed to get scopes")}

        all_variables = {}

        for scope in scopes_response.get("body", {}).get("scopes", []):
            scope_name = scope.get("name", "Unknown")
            var_ref = scope.get("variablesReference", 0)

            if var_ref > 0:
                vars_response = await self._send_request("variables", {
                    "variablesReference": var_ref,
                })

                if vars_response.get("success"):
                    scope_vars = []
                    for var in vars_response.get("body", {}).get("variables", []):
                        scope_vars.append({
                            "name": var.get("name", ""),
                            "value": var.get("value", ""),
                            "type": var.get("type"),
                        })
                    all_variables[scope_name] = scope_vars

        return {"status": "success", "variables": all_variables}

    async def evaluate(self, expression: str, frame_id: int | None = None) -> dict[str, Any]:
        """Evaluate an expression in the current context."""
        if not self.is_connected:
            return {"error": "Not connected"}
        if not self._state.is_stopped:
            return {"error": "Program is running"}

        frame = frame_id or self._state.current_frame_id

        args: dict[str, Any] = {
            "expression": expression,
            "context": "repl",
        }
        if frame:
            args["frameId"] = frame

        response = await self._send_request("evaluate", args)

        if response.get("success"):
            body = response.get("body", {})
            return {
                "status": "success",
                "result": body.get("result", ""),
                "type": body.get("type"),
            }
        else:
            return {"error": response.get("message", "Evaluation failed")}

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


def _get_lock() -> asyncio.Lock:
    """Get or create the global lock (lazy initialization for event loop compatibility)."""
    global _dap_lock
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
    # Also reset the lock for next event loop
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
            if _dap_client._process is not None:
                poll_result = _dap_client._process.poll()
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
