"""DAP session lifecycle — launch / attach / stop + the spawn + handshake pipeline.

Component of `DAPClient` (composition pattern). Owns all transport spawning,
TCP/stdio connection, DAP `initialize` / `launch|attach` / `configurationDone`
handshake, and entry-stop interpretation. The facade delegates `launch()`,
`attach()`, and `stop()` here as one-line passthroughs; its remaining surface
is event dispatch + session control + state helpers.

State accessed on the back-ref `client`: `transport`, `protocol`, `_adapter`,
`_path_mapper`, `_prepared_port`, `_running`, `_loop`, `_message_processor_task`,
`_state`, `_initialized_event`, `_stop_event`, `breakpoints`, `last_reset_dirty`,
plus the program-output append helper. The wire boundary is `client.protocol.send_request`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import (
    DAP_ADAPTER_READY_WAIT_SECONDS,
    DEFAULT_DEBUG_PORT,
    LAUNCH_REQUEST_SHORT_TIMEOUT_SECONDS,
)
from .adapters.base import DebugAdapter, IdentityPathMapper, LaunchConfig
from .dap_constants import (
    DAP_CONNECT_TIMEOUT,
    DAP_STDERR_LINE_PREFIX,
    DAP_TASK_CANCEL_TIMEOUT,
)
from .dap_framing import find_free_port
from .dap_protocol import DAPProtocol
from .dap_status import DebugResult
from .dap_transport import SocketTransport
from .stdio_transport import StdioTransport

if TYPE_CHECKING:
    from .dap_client import DAPClient

logger = logging.getLogger(__name__)


class DAPLifecycle:
    """Owns the launch / attach / stop pipeline + spawn + handshake helpers."""

    def __init__(self, client: DAPClient) -> None:
        self._client = client

    # --------------------------------------------------------------------
    # Public lifecycle
    # --------------------------------------------------------------------

    async def launch(
        self,
        cfg: LaunchConfig,
        adapter: DebugAdapter,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Spawn the adapter for `cfg.program` and drive the DAP handshake."""
        await self._reset_if_connected()
        program = await self._resolve_program(cfg, adapter)
        if isinstance(program, dict):
            return program
        program_path, program_repr = program

        await self._bind_adapter_for_launch(cfg, adapter)
        spawn_err = await self._spawn_adapter_process(adapter, cfg, program_path)
        if spawn_err is not None:
            return spawn_err

        handshake_err = await self._launch_handshake(adapter, cfg, timeout)
        if handshake_err is not None:
            return handshake_err

        await self._client._bp_ops.set_entry_breakpoint_if_requested(cfg, program_path)
        finalize_err = await self._finalize_configuration_done(timeout)
        if finalize_err is not None:
            return finalize_err

        self._init_running_state(program_repr)
        return await self._build_launch_result(cfg, adapter, program_repr, timeout)

    async def attach(
        self,
        adapter: DebugAdapter,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_DEBUG_PORT,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Attach to an already-running DAP adapter on `host:port`.

        Use when the adapter is started externally. Unlike `launch()`, this
        does NOT spawn a subprocess.
        """
        await self._reset_if_connected()
        self._client._adapter = adapter
        self._client._path_mapper = adapter.path_mapper
        self._ensure_socket_transport_for_attach(host, port)

        try:
            await self._connect(timeout=timeout)
        except ConnectionError:
            return {"error": f"Could not connect to {adapter.name} adapter on {host}:{port}."}
        self._client.transport.is_attached = True

        cfg = LaunchConfig()
        handshake_err = await self._attach_handshake(adapter, cfg, timeout)
        if handshake_err is not None:
            return handshake_err

        finalize_err = await self._finalize_configuration_done(timeout)
        if finalize_err is not None:
            return finalize_err

        self._client.transport.is_attached = True
        self._client._state.is_running = True
        self._client._state.is_stopped = False
        return DebugResult.attached(
            host=host, port=port, stopped=self._client._state.is_stopped,
        ).to_dict()

    async def stop(self) -> dict[str, Any]:
        """Stop the debug session and clean up."""
        client = self._client
        client._running = False

        await self._cancel_message_processor()
        await self._send_disconnect_if_connected()

        # Reader thread join — longer timeout for platforms where shutdown is no-op.
        client.last_reset_dirty = not client.transport.join_reader(timeout=3.0)
        client.transport.drain_queue()
        client.transport.terminate_process()  # no-op when attached, not launched

        from .dap_client import DebugState  # local import: avoid cycle at module load
        client._state = DebugState()
        client.breakpoints.clear()
        client.protocol.reset()
        if hasattr(client.transport, "is_attached"):
            client.transport.is_attached = False
        client._adapter = None
        client._path_mapper = IdentityPathMapper()
        client._prepared_port = None
        # Reassign so any dangling wait_for cancels deterministically.
        client._initialized_event = asyncio.Event()
        return DebugResult.session_stopped().to_dict()

    async def _reset_if_connected(self) -> None:
        if self._client.is_connected:
            await self.stop()

    async def _cancel_message_processor(self) -> None:
        client = self._client
        task = client._message_processor_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=DAP_TASK_CANCEL_TIMEOUT)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        client._message_processor_task = None

    async def _send_disconnect_if_connected(self) -> None:
        client = self._client
        if not client.transport.is_connected:
            return
        try:
            terminate = not getattr(client.transport, "is_attached", False)
            await asyncio.wait_for(
                client.protocol.send_request(
                    "disconnect", {"terminateDebuggee": terminate},
                ),
                timeout=2.0,
            )
        except Exception:
            pass
        client.transport.shutdown_channel()

    # --------------------------------------------------------------------
    # Launch helpers
    # --------------------------------------------------------------------

    async def _resolve_program(
        self, cfg: LaunchConfig, adapter: DebugAdapter,
    ) -> tuple[Path | None, str] | dict[str, Any]:
        """Validate `cfg.program` against the adapter conventions.

        Returns `(program_path, program_repr)` on success, an error dict otherwise.
        For class-name adapters (`adapter.program_is_class_name=True`), `program_path`
        is None and `program_repr` is the class string.
        """
        if adapter.program_is_class_name:
            return None, cfg.program
        program_path = Path(cfg.program).resolve()
        if not program_path.exists():
            return {"error": f"Program not found: {cfg.program}"}
        return program_path, str(program_path)

    async def _bind_adapter_for_launch(
        self, cfg: LaunchConfig, adapter: DebugAdapter,
    ) -> None:
        """Wire `adapter` into the client + obtain its prepared port + swap transport."""
        client = self._client
        client._adapter = adapter
        adapter._client = client  # type: ignore[attr-defined]
        client._path_mapper = adapter.path_mapper
        client._prepared_port = await adapter.prepare_launch(cfg)
        self._prepare_transport(adapter)

    async def _launch_handshake(
        self, adapter: DebugAdapter, cfg: LaunchConfig, timeout: float,
    ) -> dict[str, Any] | None:
        """Connect + initialize + send launch + wait initialized."""
        connect_err = await self._connect_to_adapter_or_terminate(adapter, timeout)
        if connect_err is not None:
            return connect_err
        return await self._attach_handshake(adapter, cfg, timeout)

    async def _attach_handshake(
        self, adapter: DebugAdapter, cfg: LaunchConfig, timeout: float,
    ) -> dict[str, Any] | None:
        """initialize + attach/launch request + wait initialized.

        Shared tail of `_launch_handshake` (which prepends a connect step).
        Named for the attach path because that path uses it without any
        prefix; launch reuses it after spawning + connecting.
        """
        init_err = await self._do_initialize(adapter, cfg, timeout)
        if init_err is not None:
            return init_err
        launch_err = await self._send_launch_request(adapter, cfg, timeout)
        if launch_err is not None:
            return launch_err
        if not await self._wait_for_initialized(timeout=timeout):
            await self.stop()
            return {"error": "Adapter never signalled 'initialized'"}
        return None

    async def _connect_to_adapter_or_terminate(
        self, adapter: DebugAdapter, timeout: float,
    ) -> dict[str, Any] | None:
        try:
            await self._connect_after_spawn(adapter, timeout)
            return None
        except Exception as e:
            self._client.transport.terminate_process()
            return {"error": f"Failed to connect to {adapter.name} adapter: {e}"}

    def _init_running_state(self, program_repr: str) -> None:
        """Mark session running. Note: is_stopped is intentionally NOT reset.

        Fast adapters (dlv, kotlin-debug-adapter, …) emit their first `stopped`
        event during configurationDone, which sets is_stopped=True before this
        line. A blind reset would race with that event.
        """
        state = self._client._state
        state.is_running = True
        state.program = program_repr
        state.program_output = []
        state.program_terminated = False

    async def _build_launch_result(
        self,
        cfg: LaunchConfig,
        adapter: DebugAdapter,
        program_repr: str,
        timeout: float,
    ) -> dict[str, Any]:
        """Wait for the optional entry stop and return the launched envelope."""
        if not cfg.stop_on_entry:
            return self._launched_envelope(program_repr, with_stop_info=False)

        stopped = await self._client._wait_for_stop(timeout=timeout)
        if not stopped:
            return self._launched_envelope(program_repr, with_stop_info=False)

        await self._maybe_auto_continue_past_entry(cfg, adapter, timeout)
        return self._launched_envelope(program_repr, with_stop_info=True)

    def _launched_envelope(
        self, program_repr: str, *, with_stop_info: bool,
    ) -> dict[str, Any]:
        state = self._client._state
        if not with_stop_info:
            return DebugResult.launched(
                program=program_repr, stopped=state.is_stopped,
            ).to_dict()
        return DebugResult.launched(
            program=program_repr,
            stopped=state.is_stopped,
            reason=state.stopped_reason,
            file=state.stopped_file,
            line=state.stopped_line,
        ).to_dict()

    async def _maybe_auto_continue_past_entry(
        self, cfg: LaunchConfig, adapter: DebugAdapter, timeout: float,
    ) -> None:
        """CDP-adapter opt-in: rebind source bps + auto-continue past the bootstrap entry.

        js-debug binds source-file bps only after the user script parses, which
        completes shortly after the entry pause. Re-issuing setBreakpoints here
        forces the binding synchronously.
        """
        pre_bps = (cfg.extras or {}).get("pre_launch_breakpoints")
        client = self._client
        if not (
            adapter.skip_entry_stop_when_pre_launch_bp
            and pre_bps
            and client._state.stopped_reason == "entry"
            and client._state.thread_id is not None
        ):
            return
        await client._bp_ops.rebind_pre_launch_breakpoints(pre_bps, timeout=2.0)
        client._state.is_stopped = False
        client._clear_stop_state()
        client._stop_event.clear()
        await client.protocol.send_request("continue", {
            "threadId": client._state.thread_id,
            "singleThread": True,
        })
        await client._wait_for_stop(timeout=timeout)

    # --------------------------------------------------------------------
    # Spawn pipeline (split from the original D-grade _spawn_adapter_process)
    # --------------------------------------------------------------------

    def _prepare_transport(self, adapter: DebugAdapter) -> None:
        """Swap in the right transport type for the adapter (stdio vs socket)."""
        client = self._client
        want_stdio = adapter.transport_type == "stdio"
        have_stdio = isinstance(client.transport, StdioTransport)
        if want_stdio and not have_stdio:
            client.transport = StdioTransport()
            client.protocol = DAPProtocol(client.transport)
        elif not want_stdio and not isinstance(client.transport, SocketTransport):
            client.transport = SocketTransport()
            client.protocol = DAPProtocol(client.transport)

    def _ensure_socket_transport_for_attach(self, host: str, port: int) -> None:
        """Attach is socket-only; install a fresh SocketTransport pointing at host:port."""
        client = self._client
        if not isinstance(client.transport, SocketTransport):
            client.transport = SocketTransport()
            client.protocol = DAPProtocol(client.transport)
        client.transport.host = host
        client.transport.port = port

    async def _spawn_adapter_process(
        self,
        adapter: DebugAdapter,
        cfg: LaunchConfig,
        program_path: Path | None,
    ) -> dict[str, Any] | None:
        """Build argv and spawn the subprocess. Returns error dict or None."""
        if self._socket_only_launch_bypass(adapter, cfg):
            return None
        argv, proc_cwd, proc_env, port = self._resolve_spawn_argv(
            adapter, cfg, program_path,
        )
        if adapter.transport_type == "stdio":
            return await self._spawn_stdio_adapter(argv, proc_cwd, proc_env, adapter)
        spawn_err = self._spawn_socket_adapter(argv, proc_cwd, proc_env, adapter)
        if spawn_err is not None:
            return spawn_err
        bootstrap_err = await self._post_socket_spawn_bootstrap(adapter, cfg, port)
        if bootstrap_err is not None:
            return bootstrap_err
        return self._check_adapter_ready_or_error(adapter)

    def _socket_only_launch_bypass(
        self, adapter: DebugAdapter, cfg: LaunchConfig,
    ) -> bool:
        """When prepare_launch supplied a port and adapter.argv is empty, no spawn.

        java-debug lives inside an external JDTLS — port already bound, so we
        just point our socket transport at it and skip Popen.
        """
        client = self._client
        if (
            client._prepared_port is not None
            and adapter.transport_type == "socket"
            and not adapter.build_launch_argv(cfg, port=client._prepared_port)
        ):
            client.transport.port = client._prepared_port  # type: ignore[attr-defined]
            return True
        return False

    def _resolve_spawn_argv(
        self,
        adapter: DebugAdapter,
        cfg: LaunchConfig,
        program_path: Path | None,
    ) -> tuple[list[str], str, dict[str, str] | None, int | None]:
        """Allocate port (if socket), build argv, cwd, env. Mutates `transport.port`."""
        client = self._client
        port: int | None = None
        if adapter.transport_type == "socket":
            port = find_free_port()
            client.transport.port = port  # type: ignore[attr-defined]
        argv = adapter.build_launch_argv(cfg, port=port)
        proc_cwd = cfg.cwd or (
            str(program_path.parent) if program_path else os.getcwd()
        )
        proc_env = self._build_proc_env(adapter, cfg)
        return argv, proc_cwd, proc_env, port

    @staticmethod
    def _build_proc_env(
        adapter: DebugAdapter, cfg: LaunchConfig,
    ) -> dict[str, str] | None:
        """Compose the subprocess env: os env + adapter defaults + caller cfg.env (last wins)."""
        adapter_env = adapter.build_launch_env(cfg)
        if not (adapter_env or cfg.env):
            return None
        proc_env = os.environ.copy()
        if adapter_env:
            proc_env.update(adapter_env)
        if cfg.env:
            proc_env.update(cfg.env)  # caller wins over adapter defaults
        return proc_env

    async def _spawn_stdio_adapter(
        self,
        argv: list[str],
        proc_cwd: str,
        proc_env: dict[str, str] | None,
        adapter: DebugAdapter,
    ) -> dict[str, Any] | None:
        """Spawn the stdio adapter process via the transport's own helper."""
        client = self._client
        try:
            await client.transport.spawn(argv, cwd=proc_cwd, env=proc_env)  # type: ignore[attr-defined]
        except Exception as e:
            return {"error": f"Failed to start {adapter.name} adapter: {e}"}
        client.transport.start_stderr_drain(client._append_program_output)
        return None

    def _spawn_socket_adapter(
        self,
        argv: list[str],
        proc_cwd: str,
        proc_env: dict[str, str] | None,
        adapter: DebugAdapter,
    ) -> dict[str, Any] | None:
        """`Popen` for socket adapters. stderr=PIPE so startup errors surface fast."""
        # stdin is normally DEVNULL; adapters that need a live stdin pipe
        # (R/vscDebugger's `.vsc.preBreakpoint` writeToStdin events) opt in.
        stdin_mode = (
            subprocess.PIPE if adapter.wants_stdin_pipe else subprocess.DEVNULL
        )
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            if platform.system() == "Windows" else 0
        )
        try:
            self._client.transport.process = subprocess.Popen(
                argv,
                cwd=proc_cwd,
                env=proc_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=stdin_mode,
                creationflags=creation_flags,
            )
        except Exception as e:
            return {"error": f"Failed to start {adapter.name} adapter: {e}"}
        return None

    async def _post_socket_spawn_bootstrap(
        self,
        adapter: DebugAdapter,
        cfg: LaunchConfig,
        port: int | None,
    ) -> dict[str, Any] | None:
        """Wire stderr drain, optionally bootstrap stdin, then sleep for adapter readiness."""
        client = self._client
        client.transport.start_stderr_drain(client._append_program_output)
        if adapter.wants_stdin_pipe:
            stdin_err = await self._write_stdin_bootstrap(adapter, cfg, port)
            if stdin_err is not None:
                return stdin_err
        await asyncio.sleep(DAP_ADAPTER_READY_WAIT_SECONDS)
        return None

    async def _write_stdin_bootstrap(
        self,
        adapter: DebugAdapter,
        cfg: LaunchConfig,
        port: int | None,
    ) -> dict[str, Any] | None:
        """Write the adapter's bootstrap payload into the live stdin pipe."""
        bootstrap = await adapter.bootstrap_stdin(cfg, port)
        process = self._client.transport.process
        if not bootstrap or process is None or process.stdin is None:
            return None
        try:
            process.stdin.write(bootstrap)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return {"error": f"Failed to write bootstrap to {adapter.name} stdin: {e}"}
        return None

    def _check_adapter_ready_or_error(
        self, adapter: DebugAdapter,
    ) -> dict[str, Any] | None:
        """If the spawned process already exited, drain stderr and surface an error dict."""
        client = self._client
        process = client.transport.process
        if process is None or process.poll() is None:
            return None
        client.transport.join_stderr_drain()
        stderr_tail = client.transport.drain_stderr_remaining()
        for line in stderr_tail:
            client._append_program_output(f"{DAP_STDERR_LINE_PREFIX}{line}")
        return {
            "error": f"{adapter.name} adapter exited immediately with code {process.returncode}",
            "stderr": stderr_tail,
        }

    # --------------------------------------------------------------------
    # Connect + handshake
    # --------------------------------------------------------------------

    async def _connect(self, timeout: float = DAP_CONNECT_TIMEOUT) -> None:
        """Open transport, start the reader thread, and the async message loop."""
        client = self._client
        await client.transport.connect(timeout=timeout)
        client._running = True
        client._loop = asyncio.get_running_loop()
        client.transport.start_reader(lambda: client._running)
        client._message_processor_task = asyncio.create_task(client._process_messages())

    async def _connect_after_spawn(
        self, adapter: DebugAdapter, timeout: float,
    ) -> None:
        """Socket adapters need a TCP connect; stdio are already connected via pipes."""
        if adapter.transport_type == "socket":
            await self._connect(timeout=timeout)
            return
        # Stdio: wire up reader + message loop without a socket handshake.
        client = self._client
        client._running = True
        client._loop = asyncio.get_running_loop()
        client.transport.start_reader(lambda: client._running)
        client._message_processor_task = asyncio.create_task(client._process_messages())

    async def _do_initialize(
        self,
        adapter: DebugAdapter,
        cfg: LaunchConfig,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Send DAP `initialize`, store capabilities, fire post-initialize hook."""
        client = self._client
        try:
            response = await client.protocol.send_request(
                "initialize", adapter.initialize_args(cfg), timeout=timeout,
            )
        except asyncio.TimeoutError:
            await self.stop()
            return {"error": "Initialize timed out"}

        if not response.get("success"):
            await self.stop()
            return {"error": response.get("message", "Initialize failed")}

        client.protocol.capabilities = response.get("body", {})
        try:
            await adapter.post_initialize(client)
        except Exception as exc:
            await self.stop()
            return {
                "error": str(exc),
                "handshake_failed": True,
                "adapter_name": adapter.name,
            }
        return None

    async def _send_launch_request(
        self,
        adapter: DebugAdapter,
        cfg: LaunchConfig,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Send `adapter.launch_request_command` with adapter-supplied body.

        Some adapters (debugpy attach, kotlin launch) defer the request response
        until `configurationDone`. Those surface as a synthetic timeout response
        from `DAPProtocol.send_request()` and must NOT be treated as fatal here.
        """
        launch_send_timeout = min(timeout, LAUNCH_REQUEST_SHORT_TIMEOUT_SECONDS)
        try:
            response = await self._client.protocol.send_request(
                adapter.launch_request_command,
                adapter.launch_request_args(cfg),
                timeout=launch_send_timeout,
            )
        except asyncio.TimeoutError:
            return None
        if response.get("success"):
            return None
        timeout_message = f"Timeout waiting for {adapter.launch_request_command} response"
        if response.get("message") == timeout_message:
            return None
        await self.stop()
        return {
            "error": response.get("message", f"{adapter.launch_request_command} failed"),
        }

    async def _wait_for_initialized(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(
                self._client._initialized_event.wait(), timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def _finalize_configuration_done(
        self, timeout: float,
    ) -> dict[str, Any] | None:
        """Send `configurationDone` — the real launch/attach success gate.

        Skipped when the adapter sets `supportsConfigurationDoneRequest=False`.
        Sent fire-and-forget when the adapter opts into
        `configuration_done_fire_and_forget` (e.g. fwcd/kotlin-debug-adapter).
        """
        client = self._client
        if client.protocol.capabilities.get("supportsConfigurationDoneRequest") is False:
            return None
        if client._adapter is not None and client._adapter.configuration_done_fire_and_forget:
            await self._fire_and_forget_configuration_done()
            return None
        try:
            cfg_response = await client.protocol.send_request(
                "configurationDone", {}, timeout=timeout,
            )
        except asyncio.TimeoutError:
            await self.stop()
            return {"error": "configurationDone timed out"}
        if not cfg_response.get("success"):
            await self.stop()
            return {
                "error": cfg_response.get("message", "configurationDone failed"),
            }
        return None

    async def _fire_and_forget_configuration_done(self) -> None:
        """Reserve seq under protocol lock and send configurationDone without awaiting a response."""
        client = self._client
        seq = await client.protocol.allocate_seq()
        await client.transport.send_message({
            "seq": seq, "type": "request",
            "command": "configurationDone", "arguments": {},
        })
