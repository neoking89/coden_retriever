"""Socket-backed DAP transport, the shared Protocol, and the shared base.

`SocketTransport` owns a TCP connection to a DAP adapter plus the launched
subprocess handle (when `DAPClient.launch()` spawned it) and a blocking
reader thread that drains the socket into a queue consumed by the async
message loop.

The `DAPTransport` Protocol captures the common surface so `DAPClient`
treats `SocketTransport` and `StdioTransport` interchangeably. Framing is
shared via `dap_framing.drain_framed_messages` so both transports parse
the same way. `_BaseTransport` holds the channel-agnostic process /
reader-thread / stderr-drain machinery both concrete transports inherit.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import socket
import subprocess
import threading
from typing import Any, Callable, Protocol, runtime_checkable

from ..constants import (
    DAP_CONNECT_RETRY_SLEEP_SECONDS,
    DAP_PROCESS_TERMINATE_TIMEOUT,
    DAP_READ_CHUNK_BYTES,
    DAP_READ_TIMEOUT_SECONDS,
    DAP_STDERR_DRAIN_JOIN_TIMEOUT,
    DEFAULT_DEBUG_PORT,
)
from .dap_constants import DAP_CONNECT_TIMEOUT, DAP_STDERR_LINE_PREFIX
from .dap_framing import drain_framed_messages, encode_framed_message

logger = logging.getLogger(__name__)
# Bounds the wait for the old reader thread after a mid-session swap_socket().
# The reader's recv() is on a DAP_READ_TIMEOUT_SECONDS (0.5s) cadence, so 3s
# (= 6 ticks) is well above the natural exit latency while still failing loud
# if the reader is genuinely wedged.
_SWAP_READER_JOIN_TIMEOUT: float = 3.0


@runtime_checkable
class DAPTransport(Protocol):
    """Shared surface both socket- and stdio-backed transports satisfy.

    Declared as a `typing.Protocol` (not ABC) per refined plan decision #1 —
    structural typing keeps the two concrete transports independent while
    `DAPClient` binds to the interface.
    """

    process: subprocess.Popen | None
    message_queue: queue.Queue[dict[str, Any]]

    @property
    def is_connected(self) -> bool: ...

    async def send_message(self, message: dict[str, Any]) -> None: ...

    def start_reader(self, is_running: Callable[[], bool]) -> None: ...

    def shutdown_channel(self) -> None: ...

    def join_reader(self, timeout: float = 3.0) -> bool: ...

    def drain_queue(self) -> None: ...

    def terminate_process(self) -> None: ...

    def start_stderr_drain(self, sink: Callable[[str], None]) -> None: ...

    def drain_stderr_remaining(self) -> list[str]: ...

    def join_stderr_drain(self) -> bool: ...


class _BaseTransport:
    """Shared subprocess + reader-thread + stderr-drain machinery.

    Holds the channel-agnostic state both `SocketTransport` and
    `StdioTransport` need (`process`, `reader_thread`, `stderr_thread`,
    `message_queue`, `_stderr_sink`) plus the lifecycle methods that operate
    on it. Subclasses contribute the channel-specific pieces: `is_connected`,
    `start_reader`/`_reader_loop`, `send_message`, `shutdown_channel`.

    `is_attached` lives here for Protocol parity — `StdioTransport` never
    flips it, but `SocketTransport.attach()` does and the surface stays
    uniform from `DAPClient`'s perspective.
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.reader_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.is_attached: bool = False
        self._stderr_sink: Callable[[str], None] | None = None

    def join_reader(self, timeout: float = 3.0) -> bool:
        """Wait for reader thread to finish. Returns True on clean join."""
        thread = self.reader_thread
        if thread is None or not thread.is_alive():
            self.reader_thread = None
            return True
        thread.join(timeout=timeout)
        alive = thread.is_alive()
        self.reader_thread = None
        return not alive

    def drain_queue(self) -> None:
        """Discard any stale messages still in the queue."""
        while True:
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                return

    def terminate_process(self) -> None:
        """Terminate the launched adapter subprocess; no-op in attach mode."""
        proc = self.process
        self.process = None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=DAP_PROCESS_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
        self.join_stderr_drain()

    def start_stderr_drain(self, sink: Callable[[str], None]) -> None:
        """Drain `process.stderr` line-by-line, feeding each line to `sink`.

        Required when launching with `stderr=PIPE`: the OS pipe buffer can fill
        and block the child process if nothing reads from it.
        """
        proc = self.process
        if proc is None or proc.stderr is None:
            return
        self._stderr_sink = sink
        self.stderr_thread = threading.Thread(
            target=self._stderr_drain_loop, args=(proc,), daemon=True,
        )
        self.stderr_thread.start()

    def _stderr_drain_loop(self, proc: subprocess.Popen) -> None:
        stream = proc.stderr
        if stream is None:
            return
        sink = self._stderr_sink
        try:
            for raw in iter(stream.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line and sink is not None:
                    sink(f"{DAP_STDERR_LINE_PREFIX}{line}")
        except (ValueError, OSError) as exc:
            # Mirrors `_reader_loop`'s exit-on-closed-pipe log — staying silent
            # here hides the case where the adapter process was killed mid-line.
            logger.debug("stderr drain exiting on closed pipe: %s", exc)

    def drain_stderr_remaining(self) -> list[str]:
        """Read and return whatever stderr is still buffered (used on early exit)."""
        proc = self.process
        if proc is None or proc.stderr is None:
            return []
        try:
            data = proc.stderr.read() or b""
        except (ValueError, OSError) as exc:
            logger.debug("stderr tail read failed: %s", exc)
            return []
        if not data:
            return []
        return [line for line in data.decode("utf-8", errors="replace").splitlines() if line]

    def join_stderr_drain(self) -> bool:
        """Wait briefly for the stderr drain thread to finish.

        Returns ``True`` when the thread has exited cleanly (or never started)
        and ``False`` when the join timed out — mirrors ``join_reader`` so
        callers can treat both drain-join results uniformly.
        """
        thread = self.stderr_thread
        self.stderr_thread = None
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=DAP_STDERR_DRAIN_JOIN_TIMEOUT)
        return not thread.is_alive()


class SocketTransport(_BaseTransport):
    """Socket + subprocess + reader-thread owner for a single DAP session."""

    def __init__(self) -> None:
        super().__init__()
        self.socket: socket.socket | None = None
        self.host: str = "127.0.0.1"
        self.port: int = DEFAULT_DEBUG_PORT
        # Captured in start_reader so swap_socket() can spawn a replacement
        # reader with the identical stop predicate without plumbing it back
        # through the caller.
        self._is_running: Callable[[], bool] | None = None
        # Sockets orphaned by swap_socket(): kept open for the rest of the
        # session because js-debug treats a bootstrap-socket close as session
        # termination and cascades it to the just-opened child session.
        # Closed atomically in shutdown_channel.
        self._orphaned_sockets: list[socket.socket] = []

    @property
    def is_connected(self) -> bool:
        return self.socket is not None

    async def connect(self, timeout: float = DAP_CONNECT_TIMEOUT) -> None:
        """Open and retry-connect the socket to (host, port).

        A fresh socket is created per attempt. Reusing a single socket across
        retries is a Linux kernel bug trap: after a failed connect() returns
        ECONNREFUSED, a second connect() on the same socket can return
        ECONNABORTED (errno 103, "Software caused connection abort") rather
        than retrying cleanly. Windows tolerates reuse by accident.
        """
        loop = asyncio.get_running_loop()
        start = loop.time()
        sock: socket.socket | None = None
        while loop.time() - start < timeout:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect((self.host, self.port))
                break
            except (ConnectionRefusedError, socket.timeout, OSError):
                sock.close()
                sock = None
                await asyncio.sleep(DAP_CONNECT_RETRY_SLEEP_SECONDS)
        if sock is None:
            raise ConnectionError(
                f"Could not connect to DAP adapter on {self.host}:{self.port}"
            )
        sock.setblocking(False)
        self.socket = sock

    def start_reader(self, is_running: Callable[[], bool]) -> None:
        """Spawn the blocking reader thread. `is_running` is the stop predicate."""
        self.message_queue = queue.Queue()
        self._is_running = is_running
        self._spawn_reader_thread(is_running)

    def _spawn_reader_thread(self, is_running: Callable[[], bool]) -> None:
        """Start a reader thread bound to the current `self.socket`.

        Extracted from `start_reader` so `swap_socket` can launch a replacement
        reader on a new socket without clobbering `self.message_queue` (callers
        waiting on the queue must keep their reference valid across the swap).
        """
        self.reader_thread = threading.Thread(
            target=self._reader_loop, args=(is_running,), daemon=True,
        )
        self.reader_thread.start()

    def swap_socket(self, new_socket: socket.socket) -> None:
        """Atomically replace the active socket with `new_socket`.

        Used by multi-session adapters (e.g. js-debug) that receive a reverse
        `startDebugging` request and must route subsequent DAP traffic over a
        freshly-dialed child connection. Without this primitive an adapter
        would have to reach into transport internals and could leak the old
        socket + reader thread (bug_008).

        Invariants:
          * The old reader exits before the new one starts (no double-drain
            into `message_queue`).
          * The old socket is fully closed (fd released, OS-level teardown).
          * `message_queue` is preserved — events already queued remain
            visible to the consumer.
          * `reader_thread` post-call points at the *new* reader so
            `join_reader()` during `stop()` waits on the live thread.
        """
        if self._is_running is None:
            raise RuntimeError(
                "swap_socket requires start_reader() to have been called first",
            )
        # Orphan the bootstrap socket rather than closing it: js-debug
        # interprets a bootstrap close as full session termination and will
        # tear down the just-opened child session, causing subsequent DAP
        # writes to fail with broken-pipe. Orphaned sockets are closed in
        # shutdown_channel at session end. The old reader thread remains
        # alive but idle — its blocking recv wakes only when shutdown_channel
        # closes the orphaned socket, at which point it exits cleanly.
        old_sock = self.socket
        if old_sock is not None:
            self._orphaned_sockets.append(old_sock)
        self.socket = new_socket
        self._spawn_reader_thread(self._is_running)

    @staticmethod
    def _close_socket(sock: socket.socket | None) -> None:
        """Shutdown + close `sock`, swallowing the usual post-teardown errors.

        Shared between `shutdown_channel` (end-of-session) and `swap_socket`
        (mid-session fd replacement). Both paths want the same tolerant close:
        calling shutdown() on an already-closed fd raises on some platforms
        and the caller has nothing useful to do with that error.
        """
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, Exception):
            pass
        try:
            sock.close()
        except Exception:
            pass

    def _reader_loop(self, is_running: Callable[[], bool]) -> None:
        sock = self.socket
        if sock is None:
            return
        buffer = b""
        # setblocking/settimeout can fail with OSError if the socket was
        # closed between start_reader() and the thread actually running
        # (notably during a swap_socket shutdown race). Treat that as a
        # clean exit rather than an unhandled daemon-thread exception.
        try:
            sock.setblocking(True)
            sock.settimeout(DAP_READ_TIMEOUT_SECONDS)
        except OSError:
            return

        while is_running():
            try:
                chunk = sock.recv(DAP_READ_CHUNK_BYTES)
                if not chunk:
                    if is_running():
                        logger.debug("Socket closed by server")
                    break
                buffer = drain_framed_messages(buffer + chunk, self.message_queue.put)
            except socket.timeout:
                continue
            except Exception as e:
                if is_running():
                    logger.debug(f"Reader thread error: {e}")
                break

    async def send_message(self, message: dict[str, Any]) -> None:
        """Encode and write a DAP message (Content-Length framed JSON)."""
        sock = self.socket
        if sock is None:
            raise ConnectionError("Socket not connected")
        data = encode_framed_message(message)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, sock.sendall, data)

    def shutdown_channel(self) -> None:
        """Unblock reader's recv and close the socket(s)."""
        sock = self.socket
        self.socket = None
        self._close_socket(sock)
        while self._orphaned_sockets:
            self._close_socket(self._orphaned_sockets.pop())
