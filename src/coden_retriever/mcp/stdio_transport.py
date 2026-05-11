"""Stdio-backed DAP transport.

For DAP adapters that speak Content-Length-framed JSON over the spawned
process's stdin/stdout (dlv dap, lldb-dap, PowerShell Editor Services, etc.)
rather than a TCP socket. Same reader-thread/queue pattern as
`SocketTransport`; the I/O channels are the process pipes instead of a
socket.

The transport owns the `Popen` handle — the process IS its I/O, so they
can't be split cleanly. Process / reader-thread / stderr-drain machinery
is inherited from `_BaseTransport`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import subprocess
import threading
from typing import Any, Callable

from ..constants import DAP_READ_CHUNK_BYTES
from .dap_framing import drain_framed_messages, encode_framed_message
from .dap_transport import _BaseTransport

logger = logging.getLogger(__name__)


class StdioTransport(_BaseTransport):
    """Subprocess stdin/stdout DAP transport."""

    @property
    def is_connected(self) -> bool:
        proc = self.process
        return proc is not None and proc.poll() is None

    async def spawn(
        self,
        argv: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Spawn the adapter subprocess with stdin/stdout/stderr piped."""
        proc_env = None
        if env:
            proc_env = os.environ.copy()
            proc_env.update(env)

        self.process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=proc_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def start_reader(self, is_running: Callable[[], bool]) -> None:
        """Spawn the blocking reader thread. `is_running` is the stop predicate."""
        self.message_queue = queue.Queue()
        self.reader_thread = threading.Thread(
            target=self._reader_loop, args=(is_running,), daemon=True,
        )
        self.reader_thread.start()

    def _reader_loop(self, is_running: Callable[[], bool]) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        buffer = b""
        stream = proc.stdout
        while is_running():
            try:
                chunk = stream.read1(DAP_READ_CHUNK_BYTES)
            except (ValueError, OSError) as e:
                if is_running():
                    logger.debug(f"Stdio reader error: {e}")
                break
            if not chunk:
                if is_running():
                    logger.debug("Adapter stdout closed")
                break
            buffer = drain_framed_messages(buffer + chunk, self.message_queue.put)

    async def send_message(self, message: dict[str, Any]) -> None:
        """Encode and write a DAP message to the adapter's stdin."""
        proc = self.process
        if proc is None or proc.stdin is None:
            raise ConnectionError("Stdio transport not connected")
        data = encode_framed_message(message)
        loop = asyncio.get_running_loop()
        stdin = proc.stdin

        def _write() -> None:
            stdin.write(data)
            stdin.flush()

        await loop.run_in_executor(None, _write)

    def shutdown_channel(self) -> None:
        """Close stdin so the adapter exits cleanly; reader sees EOF on stdout."""
        proc = self.process
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass

