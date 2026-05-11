"""DAP request/response sequencing.

Owns the monotonic `seq` counter, the `pending_requests` future map,
`capabilities` advertised by the adapter, and the per-send lock. Given a
`DAPTransport`, it turns (command, args) into a sent request and awaits the
matching response. Event handling lives in `DAPClient._handle_message`.
"""
import asyncio
import logging
from typing import Any

from .dap_constants import DAP_DEFAULT_REQUEST_TIMEOUT
from .dap_transport import DAPTransport

logger = logging.getLogger(__name__)


class DAPProtocol:
    """DAP wire-level request/response coordination."""

    def __init__(self, transport: DAPTransport) -> None:
        self._transport = transport
        self._seq = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self.capabilities: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def send_request(
        self,
        command: str,
        arguments: dict[str, Any],
        timeout: float = DAP_DEFAULT_REQUEST_TIMEOUT,
    ) -> dict[str, Any]:
        """Send a DAP request and await the matching response.

        The lock only covers seq allocation + send so that concurrent callers
        don't interleave their writes. The response wait runs outside the lock:
        debugpy sometimes sends the configurationDone response before the attach
        response (out-of-order), so we must allow configurationDone to be sent
        (and its future registered) while the attach response is still pending.
        """
        future: asyncio.Future = asyncio.Future()
        async with self._lock:
            self._seq += 1
            seq = self._seq
            message = {
                "seq": seq,
                "type": "request",
                "command": command,
                "arguments": arguments,
            }
            self._pending_requests[seq] = future
            await self._transport.send_message(message)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(seq, None)
            return {
                "success": False,
                "message": f"Timeout waiting for {command} response",
            }

    def route_response(self, message: dict[str, Any]) -> None:
        """Resolve the pending future for an incoming response message."""
        request_seq = message.get("request_seq")
        if request_seq not in self._pending_requests:
            return
        future = self._pending_requests.pop(request_seq)
        if not future.done():
            future.set_result(message)

    async def send_response(
        self,
        request_seq: int,
        command: str,
        success: bool,
        body: dict[str, Any] | None = None,
    ) -> None:
        """Send a DAP `response` for a server-initiated `request`.

        Fire-and-forget per DAP spec — no future to await. The lock covers
        seq allocation + send so a concurrent `send_request` can't interleave.
        """
        async with self._lock:
            self._seq += 1
            message: dict[str, Any] = {
                "seq": self._seq,
                "type": "response",
                "request_seq": request_seq,
                "command": command,
                "success": success,
                "body": body or {},
            }
            await self._transport.send_message(message)

    def reset(self) -> None:
        """Clear per-session state (called during DAPClient.stop).

        Cancel any in-flight request futures BEFORE clearing the map —
        callers are blocked in `asyncio.wait_for(future, ...)` and would
        otherwise hang until their timeout fires. `cancel()` raises
        CancelledError in the awaiter, which callers already handle.
        """
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        self.capabilities = {}
        self._seq = 0

    async def allocate_seq(self) -> int:
        """Allocate a new monotonic sequence number under the protocol lock.

        Exists so callers that must send a raw message outside `send_request`
        (e.g. fire-and-forget configurationDone) can still reserve a `seq`
        without racing concurrent `send_request` allocations.
        """
        async with self._lock:
            self._seq += 1
            return self._seq
