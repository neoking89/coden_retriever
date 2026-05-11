"""DAP wire-format framing and socket utilities shared by all transports.

`Content-Length`-framed JSON is the DAP wire format; both socket- and
stdio-transport adapters need the same parser. Keeping it as free functions
(not on a transport class) lets transports compose it without inheritance.

`find_free_port` is here because ephemeral-port allocation is a
socket-transport concern that the DAP client needs before spawning a
socket-adapter subprocess.
"""
from __future__ import annotations

import json
import logging
import socket
from typing import Any, Callable

logger = logging.getLogger(__name__)

_HEADER_DELIM = b"\r\n\r\n"

OnMessage = Callable[[dict[str, Any]], None]


def drain_framed_messages(buffer: bytes, on_message: OnMessage) -> bytes:
    """Extract every complete Content-Length message from `buffer`.

    Calls `on_message(msg_dict)` for each fully-parsed message. Returns the
    tail of `buffer` that has not yet formed a complete message so the
    caller can concatenate with the next chunk and re-invoke.

    Malformed headers (missing Content-Length) are skipped rather than
    re-triggering on every subsequent chunk — otherwise the reader would
    stall silently on a bad byte.
    """
    while True:
        header_end = buffer.find(_HEADER_DELIM)
        if header_end == -1:
            return buffer
        header = buffer[:header_end].decode("utf-8")
        content_length = _parse_content_length(header)
        if content_length == 0:
            logger.warning(f"No Content-Length in header, skipping: {header[:100]!r}")
            buffer = buffer[header_end + len(_HEADER_DELIM):]
            continue
        msg_start = header_end + len(_HEADER_DELIM)
        msg_end = msg_start + content_length
        if len(buffer) < msg_end:
            return buffer
        content = buffer[msg_start:msg_end].decode("utf-8")
        buffer = buffer[msg_end:]
        try:
            on_message(json.loads(content))
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {content[:100]}")


def encode_framed_message(message: dict[str, Any]) -> bytes:
    """Encode a DAP message as Content-Length-framed UTF-8 bytes."""
    content = json.dumps(message)
    return f"Content-Length: {len(content)}\r\n\r\n{content}".encode("utf-8")


def find_free_port() -> int:
    """Allocate an ephemeral TCP port by binding to :0 on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _parse_content_length(header: str) -> int:
    for line in header.split("\r\n"):
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return 0
