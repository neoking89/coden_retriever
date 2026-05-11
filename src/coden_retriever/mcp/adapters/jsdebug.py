"""JS/TS adapter via `@vscode/js-debug` (multi-session child proxy).

js-debug is a "session leader": the TCP connection we speak on is the
"bootstrap" session, and the actual debuggee runs in a *child* session
that js-debug negotiates via a server→client `startDebugging` reverse
request (see dapDebugServer.js, `acquireDap`/`handleConnection`).

Acknowledging `startDebugging` with an empty body does NOT collapse
children onto the main pipe — js-debug's `acquireDap` strictly waits
for a NEW DAP connection that carries `__pendingTargetId` in its
launch params, and errors otherwise. Until that child connection
lands, provisional breakpoints never bind and no `stopped` events
arrive, so the matrix's pause-at-bp flow times out.

The proxy here opens that child connection on demand: when js-debug
asks for `startDebugging` we dial a second socket to the same port,
drive a minimal `initialize` + `launch` (echoing the parent's
initialize args and the supplied child `configuration`), wire the
child's reader into the main DAPClient message queue, and swap the
transport's socket so outgoing requests (setBreakpoints, continue,
stack, eval, …) reach the live runtime instead of the bootstrap
session.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import socket as _socket
from typing import Any, Mapping

from ._node_bridge import NODE_BINARY, resolve_bridge_script
from .availability import (
    DebugDependencyStatus,
    binary_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

logger = logging.getLogger(__name__)

_BRIDGE_ENV_VAR = "VSCODE_JS_DEBUG"
_BRIDGE_RELATIVE_PATHS = (
    "node_modules/@vscode/js-debug/src/dapDebugServer.js",
    "lib/node_modules/@vscode/js-debug/src/dapDebugServer.js",
)
_INSTALL_HINT = (
    "Install Node.js 18+ AND: npm install -g @vscode/js-debug "
    "(or set VSCODE_JS_DEBUG to an absolute dapDebugServer.js path)."
)
_NODE_RUNTIME_HINT = "Install Node.js 18+ and ensure `node` is on PATH"
# js-debug wire ID for Node targets; also accepts pwa-chrome / pwa-extensionHost
# for other runtimes, but Phase 6 only supports the Node launch mode.
_ADAPTER_ID = "pwa-node"


def _log_child_launch_failure(task: asyncio.Task[Any]) -> None:
    """Surface exceptions from the fire-and-forget child `launch` send.

    Attached as a done-callback so a failed launch (e.g. js-debug rejects the
    __pendingTargetId, or the child pipe breaks) isn't silently discarded the
    way a bare `asyncio.create_task(...)` would swallow it.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("jsdebug child launch request failed", exc_info=exc)


class JSDebugAdapter(DebugAdapter):
    """@vscode/js-debug Node-bridge DAP adapter for JavaScript / TypeScript."""

    name = "jsdebug"
    file_extensions = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
    # dapDebugServer.js is a TCP server: `node dapDebugServer.js <port> [host]`.
    # It does not speak stdio, so the adapter must dial 127.0.0.1:<port> after
    # spawning. DAPClient drives the port through build_launch_argv(..., port=).
    transport_type = "socket"
    # node's source-file bps bind only after script parse — the entry stop
    # is the window where binding completes, so it must be auto-continued.
    skip_entry_stop_when_pre_launch_bp = True

    def __init__(self) -> None:
        super().__init__()
        self._port: int | None = None
        self._child_initialized_event: asyncio.Event | None = None
        self._child_initialize_args: dict[str, Any] | None = None
        # When True, the first `reason=entry` stop is swallowed (auto-continue)
        # so the matrix-level `stopped` reflects the real pre-launch bp hit.
        self._auto_skip_entry: bool = False
        # REGISTRY.register keeps a single adapter instance alive across sessions,
        # so session-scoped fields above must be wiped at the start of each new
        # launch or stale state (old port, old __pendingTargetId gate, stale
        # auto-skip) bleeds from session N into session N+1.
        self._pending_child_launch_task: asyncio.Task[Any] | None = None

    def _reset_session_state(self) -> None:
        self._port = None
        self._child_initialized_event = None
        self._child_initialize_args = None
        self._auto_skip_entry = False
        self._pending_child_launch_task = None

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(NODE_BINARY) is None:
            return (False, _INSTALL_HINT)
        if resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=NODE_BINARY,
                kind="runtime",
                name="Node.js runtime",
                install_hint=_NODE_RUNTIME_HINT,
            ),
            resolver_dependency_status(
                kind="debugger",
                name="@vscode/js-debug",
                install_hint=_INSTALL_HINT,
                resolver=lambda: resolve_bridge_script(
                    _BRIDGE_ENV_VAR,
                    _BRIDGE_RELATIVE_PATHS,
                ),
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        # First adapter method DAPClient invokes per launch — the right spot to
        # wipe per-session state on this shared singleton instance.
        self._reset_session_state()
        bridge = resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS)
        if bridge is None:
            raise RuntimeError(_INSTALL_HINT)
        if port is None:
            raise RuntimeError("jsdebug requires a TCP port; transport_type='socket'")
        # Remember the port so handle_reverse_request can open the child
        # session socket on the same js-debug listener.
        self._port = port
        return [NODE_BINARY, bridge, str(port), "127.0.0.1"]

    def initialize_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        args = {
            "clientID": "mcp-debugger",
            "clientName": "MCP Debug Client",
            "adapterID": _ADAPTER_ID,
            "pathFormat": "path",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "supportsVariableType": True,
        }
        # Echo the bootstrap session's initialize args on the child session
        # so js-debug's `handleConnection` copies consistent caps across.
        self._child_initialize_args = dict(args)
        return args

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        # stopOnEntry is always requested. Two-phase reasoning:
        #   * node's CDP source-file breakpoints only bind after the script
        #     is parsed. Without an entry pause, short fixtures run to
        #     completion before the breakpoint manager binds them.
        #   * With stopOnEntry=true js-debug pauses immediately after parsing
        #     (frame: [<bootstrap>] — no user code), giving the bp time to
        #     bind. Then `on_stopped` below auto-continues past the
        #     bootstrap-frame pause so callers observing the matrix-level
        #     `stopped` see the real line-8 bp pause with user frames.
        has_pre_launch_bp = bool((cfg.extras or {}).get("pre_launch_breakpoints"))
        self._auto_skip_entry = has_pre_launch_bp
        body: dict[str, Any] = {
            "type": _ADAPTER_ID,
            "program": cfg.program,
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "stopOnEntry": cfg.stop_on_entry or has_pre_launch_bp,
            # Source maps default ON so breakpoints on .ts files map to .js.
            # Disable via cfg.extras["source_maps"] = False.
            "sourceMaps": cfg.extras.get("source_maps", True),
        }
        overrides = cfg.extras.get("source_map_path_overrides")
        if overrides:
            body["sourceMapPathOverrides"] = dict(overrides)
        for key in ("runtimeExecutable", "runtimeArgs", "env", "outFiles"):
            value = cfg.extras.get(key)
            if value is not None:
                body[key] = value
        return body

    async def handle_reverse_request(
        self,
        command: str,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        if command == "runInTerminal":
            # console="internalConsole" keeps spawn inside js-debug.
            return (True, {})
        if command == "startDebugging":
            try:
                await self._open_child_session(arguments)
            except Exception as exc:
                return (False, {"error": f"child-session attach failed: {exc}"})
            return (True, {})
        return await super().handle_reverse_request(command, arguments)


    async def _open_child_session(self, arguments: Mapping[str, Any]) -> None:
        """Dial a second DAP connection to js-debug for the real runtime.

        Called from `handle_reverse_request` when js-debug asks the client to
        start a child session. We:

          1. Open a new TCP socket to the same js-debug port.
          2. Hand the socket to `transport.swap_socket()` which closes the
             bootstrap socket, joins its reader, and starts a new registered
             reader on the child socket feeding the same `message_queue`.
             Subsequent requests (setBreakpoints, continue, stack, eval, …)
             land on the runtime session rather than the idle bootstrap.
          3. Send `initialize` (child connections need their own handshake)
             then `launch` with `__pendingTargetId` — js-debug's
             `handleConnection` validates this cookie and routes the new
             connection to the matching pending target.
        """
        client = getattr(self, "_client", None)
        if client is None or self._port is None:
            raise RuntimeError("jsdebug: client/port not bound yet")
        transport = client.transport
        configuration = dict(arguments.get("configuration") or {})
        if not configuration.get("__pendingTargetId"):
            raise RuntimeError("startDebugging missing __pendingTargetId")
        request_verb = arguments.get("request", "launch")

        child = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        # Non-blocking connect on the running loop: a blocking socket.connect
        # here would stall the event loop (and every DAP message it drains)
        # for the full TCP handshake round-trip. swap_socket's reader thread
        # re-enables blocking mode on its side, so we flip it back afterwards.
        child.setblocking(False)
        loop = asyncio.get_running_loop()
        await loop.sock_connect(child, ("127.0.0.1", self._port))
        child.setblocking(True)

        # Transport-level swap: closes the bootstrap socket, joins its reader,
        # and spawns a new registered reader on `child` feeding the existing
        # message_queue. Avoids the prior leaky dance that left the bootstrap
        # fd pinned open and an unregistered thread draining into the queue.
        transport.swap_socket(child)

        # Each DAP connection requires its own initialize handshake before
        # anything else; re-send the bootstrap's args over the child socket.
        init_args = self._child_initialize_args or self.initialize_args(
            LaunchConfig(program="")
        )
        await client.protocol.send_request("initialize", init_args, timeout=10.0)
        # js-debug's per-session `launch` handler awaits `configurationDone`
        # before it proceeds (see dapDebugServer.js `SJ` → `u(l)`), so the
        # launch request below never gets a response unless we also send
        # configurationDone. Order between them doesn't matter; the handler
        # ANDs them. Fire-and-forget because the launch response is deferred
        # until the debuggee is running and the handshake is the real gate.
        launch_task = asyncio.create_task(
            client.protocol.send_request(request_verb, configuration, timeout=30.0)
        )
        # Keep a reference so the task isn't GC'd mid-flight, and surface any
        # launch-side failure via the logger — without this a blown send_request
        # would vanish into the void (no await, no callback).
        self._pending_child_launch_task = launch_task
        launch_task.add_done_callback(_log_child_launch_failure)
        await client.protocol.send_request(
            "configurationDone", {}, timeout=10.0,
        )


REGISTRY.register(JSDebugAdapter())
