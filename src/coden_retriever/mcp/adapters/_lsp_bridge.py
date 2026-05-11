"""Minimal LSP JSON-RPC client for java-debug port discovery.

`microsoft/java-debug` is hosted inside Eclipse JDT LSP, not a standalone
process. The single `workspace/executeCommand("vscode.java.startDebugSession")`
call that returns java-debug's socket port is all we need — a full LSP
client (pygls, etc.) is over-engineered for one method call.

Content-Length framing is byte-identical between LSP and DAP, so we import
`encode_framed_message` + `drain_framed_messages` from `dap_framing.py`
(one source of truth; no duplication).

Transport resolution precedence:
1. `JDTLS_SOCKET=host:port` — dial an already-running JDTLS.
2. `JDTLS_COMMAND=<path>` — spawn the named binary (argv split on whitespace).
3. `shutil.which("jdtls")` — standard PATH lookup; spawn with no extra args.

Any failure (missing binary, timeout, malformed response) raises
`RuntimeError`; the caller (`JavaAdapter.prepare_launch`) surfaces via
`dependency_missing_for_active_adapter`.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..dap_framing import drain_framed_messages, encode_framed_message

logger = logging.getLogger(__name__)

_JDTLS_SOCKET_ENV = "JDTLS_SOCKET"
_JDTLS_COMMAND_ENV = "JDTLS_COMMAND"
_JDTLS_DEBUG_BUNDLES_ENV = "JDTLS_DEBUG_BUNDLES"
_DEFAULT_BINARY = "jdtls"
_JAVA_DEBUG_COMMAND = "vscode.java.startDebugSession"
_READ_CHUNK_BYTES = 4096
# Cap on captured JDTLS stderr tail. Startup errors (JDK version mismatch,
# missing bundle, workspace lock) fit well inside 4 KB; larger tails just
# bloat error envelopes without adding diagnostic value.
_STDERR_TAIL_BYTES = 4096

# java-debug is an OSGi bundle hosted INSIDE the JDTLS process, not a separate
# binary. Terminating JDTLS after the `vscode.java.startDebugSession` RPC also
# tears down the DAP server listening on the port it just returned, so the
# subsequent dial fails (observed on Linux: "Could not connect to DAP adapter"
# after JDTLS returned port 40017). The spawned JDTLS is therefore held at
# module scope and reaped only at interpreter exit. Attaching via JDTLS_SOCKET
# to a pre-running JDTLS is unaffected — that path only closes our own writer.
_held_jdtls_processes: list[asyncio.subprocess.Process] = []


@atexit.register
def _reap_held_jdtls() -> None:
    for proc in _held_jdtls_processes:
        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass


async def request_debug_port(workspace: str, timeout: float) -> int:
    """Open LSP → initialize → executeCommand → return java-debug port.

    Wraps the whole handshake in `asyncio.wait_for(..., timeout=timeout)`.
    Raises RuntimeError on any failure.
    """
    try:
        return await asyncio.wait_for(_run_lsp_flow(workspace), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"jdtls LSP handshake timed out after {timeout}s",
        ) from exc


async def _run_lsp_flow(workspace: str) -> int:
    """Drive the full LSP handshake; return the java-debug port.

    Shared buffer + inbox across the whole handshake — `drain_framed_messages`
    can parse multiple messages per chunk, and we must not drop the ones we
    don't immediately need (e.g. the initialize response vs. log/notification
    traffic that arrives before executeCommand completes).
    """
    reader, writer, cleanup, stderr_tail = await _open_channel()
    state = _LspChannelState()
    state.stderr_tail = stderr_tail
    kept_alive = False
    try:
        _send(writer, _initialize_request(workspace))
        await _await_response(reader, state, request_id=1)
        _send(writer, _initialized_notification())
        _send(writer, _execute_command_request())
        response = await _await_response(reader, state, request_id=2)
        port = _parse_port(response)
        # JDTLS must outlive this coroutine (java-debug lives inside it).
        # Spawn a background drainer so subsequent JDTLS stdout (log spam,
        # notifications) doesn't block when the 64 KB pipe buffer fills —
        # a full pipe stalls JDTLS, which indirectly starves java-debug.
        asyncio.create_task(_drain_forever(reader))
        kept_alive = True
        return port
    finally:
        if not kept_alive:
            await cleanup()


async def _drain_forever(reader: asyncio.StreamReader) -> None:
    try:
        while True:
            chunk = await reader.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
    except Exception:
        return


class _LspChannelState:
    """Persistent read-buffer + already-parsed-but-unmatched message inbox."""

    def __init__(self) -> None:
        self.buffer = b""
        self.inbox: list[dict[str, Any]] = []
        # Populated by a background task started from `_open_subprocess_channel`.
        # When the LSP stream closes unexpectedly, the tail is appended to the
        # RuntimeError so callers see WHY JDTLS died (JDK version, missing
        # bundle, etc.) instead of an opaque "stream closed before response".
        self.stderr_tail: bytearray | None = None


async def _open_channel() -> tuple[
    asyncio.StreamReader, asyncio.StreamWriter, Any, bytearray | None,
]:
    """Resolve LSP transport → (reader, writer, async cleanup, stderr tail).

    Fourth tuple element is a live bytearray being appended to by a background
    drainer (subprocess path only); None for attach-to-running sockets.
    """
    socket_env = os.environ.get(_JDTLS_SOCKET_ENV)
    if socket_env:
        reader, writer, cleanup = await _open_socket_channel(socket_env)
        return reader, writer, cleanup, None
    command_env = os.environ.get(_JDTLS_COMMAND_ENV)
    if command_env:
        argv = shlex.split(command_env)
    else:
        binary = shutil.which(_DEFAULT_BINARY)
        if binary is None:
            raise RuntimeError(
                f"{_DEFAULT_BINARY} not on PATH; set {_JDTLS_SOCKET_ENV} or {_JDTLS_COMMAND_ENV}",
            )
        argv = [binary]
    return await _open_subprocess_channel(argv)


async def _open_socket_channel(
    host_port: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, Any]:
    if ":" not in host_port:
        raise RuntimeError(
            f"{_JDTLS_SOCKET_ENV}={host_port!r} must be host:port",
        )
    host, _, port_str = host_port.rpartition(":")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise RuntimeError(
            f"{_JDTLS_SOCKET_ENV} port {port_str!r} is not an integer",
        ) from exc
    reader, writer = await asyncio.open_connection(host or "127.0.0.1", port)

    async def cleanup() -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    return reader, writer, cleanup


async def _open_subprocess_channel(
    argv: list[str],
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, Any, bytearray]:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # PIPE (not DEVNULL) so the background drainer can capture JDTLS's
            # last-words on stderr — surfaced in the error envelope when the
            # LSP stream EOFs before an initialize response.
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"failed to spawn jdtls: {exc}") from exc
    assert (
        process.stdout is not None
        and process.stdin is not None
        and process.stderr is not None
    )
    _held_jdtls_processes.append(process)
    stderr_tail = bytearray()
    asyncio.create_task(_drain_stderr_tail(process.stderr, stderr_tail))

    async def cleanup() -> None:
        # Deliberate no-op: java-debug lives inside this JDTLS process and the
        # caller is about to dial the port JDTLS just returned. Closing stdin
        # or terminating now takes the DAP server down with it. Reaped at
        # interpreter exit by `_reap_held_jdtls`.
        return None

    return process.stdout, process.stdin, cleanup, stderr_tail


async def _drain_stderr_tail(
    reader: asyncio.StreamReader, tail: bytearray,
) -> None:
    """Append stderr chunks into `tail`, keeping only the last _STDERR_TAIL_BYTES.

    Ring-buffer semantics: older bytes are dropped once the cap is exceeded.
    Swallows all exceptions — this is a diagnostic best-effort, never load-bearing.
    """
    try:
        while True:
            chunk = await reader.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            tail.extend(chunk)
            overflow = len(tail) - _STDERR_TAIL_BYTES
            if overflow > 0:
                del tail[:overflow]
    except Exception:
        return


def _send(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write(encode_framed_message(message))


async def _await_response(
    reader: asyncio.StreamReader,
    state: _LspChannelState,
    *,
    request_id: int,
) -> dict[str, Any]:
    """Consume an already-parsed response with matching id, else read more."""
    while True:
        for i, msg in enumerate(state.inbox):
            if msg.get("id") == request_id:
                del state.inbox[i]
                if "error" in msg:
                    raise RuntimeError(f"LSP error: {msg['error']}")
                return msg
        chunk = await reader.read(_READ_CHUNK_BYTES)
        if not chunk:
            raise RuntimeError(_format_eof_error(state))
        state.buffer += chunk
        state.buffer = drain_framed_messages(state.buffer, state.inbox.append)


def _format_eof_error(state: _LspChannelState) -> str:
    """Build a diagnostic error message, embedding the JDTLS stderr tail."""
    base = "jdtls LSP stream closed before response"
    if not state.stderr_tail:
        return base
    tail_text = bytes(state.stderr_tail).decode("utf-8", errors="replace").strip()
    if not tail_text:
        return base
    return f"{base}; stderr tail: {tail_text}"


def _initialize_request(workspace: str) -> dict[str, Any]:
    workspace_path = Path(workspace).resolve()
    root_uri = f"file:///{quote(str(workspace_path).replace(chr(92), '/').lstrip('/'))}"
    params: dict[str, Any] = {
        "processId": os.getpid(),
        "rootUri": root_uri,
        "capabilities": {},
        "workspaceFolders": [{"uri": root_uri, "name": workspace_path.name}],
    }
    bundles = _resolve_debug_bundles()
    if bundles:
        # JDTLS loads dynamic OSGi bundles (e.g. `com.microsoft.java.debug.plugin`)
        # only when their jar paths are passed via `initializationOptions.bundles`
        # in the LSP initialize request. Dropping the jar into JDTLS's plugins/
        # dir on its own does NOT register the `vscode.java.startDebugSession`
        # delegate command — the bundle has to be contributed at startup.
        params["initializationOptions"] = {"bundles": bundles}
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": params,
    }


def _resolve_debug_bundles() -> list[str]:
    """Return absolute paths to java-debug plugin jars from env var.

    `JDTLS_DEBUG_BUNDLES` is an `os.pathsep`-separated list. Empty / unset →
    empty list (caller uses bundle-less initialize). Paths that are empty or
    do not exist on disk are dropped with a WARNING so misconfigured env vars
    surface in logs instead of failing opaquely at JDTLS initialize-time.
    """
    raw = os.environ.get(_JDTLS_DEBUG_BUNDLES_ENV, "")
    if not raw:
        return []
    resolved: list[str] = []
    for part in raw.split(os.pathsep):
        if not part:
            logger.warning(
                "ignored bundle path %r: empty entry in %s",
                part, _JDTLS_DEBUG_BUNDLES_ENV,
            )
            continue
        candidate = Path(part)
        if not candidate.exists():
            logger.warning(
                "ignored bundle path %s: file does not exist", part,
            )
            continue
        resolved.append(str(candidate.resolve()))
    return resolved


def _initialized_notification() -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "initialized", "params": {}}


def _execute_command_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "workspace/executeCommand",
        "params": {"command": _JAVA_DEBUG_COMMAND, "arguments": []},
    }


def _parse_port(response: dict[str, Any]) -> int:
    result = response.get("result")
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        port = result.get("port")
        if isinstance(port, int):
            return port
    raise RuntimeError(
        f"java-debug port missing from LSP response: {response!r}",
    )
