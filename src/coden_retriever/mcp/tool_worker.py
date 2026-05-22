"""Kill-on-timeout worker subprocess for memory-heavy stateless MCP tools.

A timeout must actually *free memory* at the deadline — not merely make the agent
give up. Python cannot kill a thread, so the old anyio-thread-abandon scheme left a
timed-out tool's ONNX model + whole-repo cache resident. Only a killable **process**
delivers reclamation, so heavy stateless tools (those marked ``@worker_safe``) run in
a persistent warm worker subprocess. On timeout the server ``kill()``s the worker — the
OS reclaims its RSS — and respawns a fresh one for the next call.

Two roles live here:

* **Server side** — :class:`ToolWorker` (singleton via :func:`get_worker`) spawns the
  worker with :class:`subprocess.Popen`, dispatches a tool by ``module:qualname`` + JSON
  kwargs, and enforces the deadline with ``asyncio.wait_for(asyncio.to_thread(read))``.
  ``Popen`` (not ``asyncio.create_subprocess_exec``) is deliberate: ``Popen.kill()`` is
  synchronous and loop-independent, sidestepping the Windows Proactor-vs-Selector
  event-loop-policy question for child subprocesses and making the ``atexit`` cleanup safe.
* **Worker side** — ``python -I -OO -m coden_retriever.mcp.tool_worker`` runs
  :func:`_serve`: it reassigns fd-1 to stderr so no native-library write can corrupt the
  length-prefixed frame stream (the frame channel is a private ``dup`` of the original
  stdout), emits a ``ready`` handshake, then loops dispatching requests.

The frame protocol is a 4-byte big-endian length header followed by a UTF-8 JSON body,
in both directions.
"""
from __future__ import annotations

import asyncio
import atexit
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

# Frame = 4-byte big-endian length prefix + UTF-8 JSON body. 4 bytes (max ~4 GiB)
# is far above any tool result; whole-repo dicts are MBs at most.
_LENGTH_HEADER_BYTES = 4

# Time allowed for a freshly spawned worker to import this module and emit its
# ``ready`` frame. Generous because a loaded box may be slow to start a process;
# import of the worker entrypoint itself is light (tool modules load lazily per call).
_HANDSHAKE_TIMEOUT_S = 30.0

# Bound on reaping a killed worker so a wedged kill never blocks the server loop.
_KILL_WAIT_S = 5.0


class ToolWorkerError(RuntimeError):
    """The worker failed to start, crashed, or its pipe broke — not a tool error."""


class ToolWorkerTimeout(RuntimeError):
    """A tool exceeded its deadline; the worker was killed to reclaim its memory."""


# ---------------------------------------------------------------------------
# Frame protocol (shared by both roles)
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars/arrays so a tool result survives the JSON boundary.

    Heavy tools return numpy values (e.g. ``code_search`` similarity scores from
    cosine fusion). Detected by duck-typing rather than importing numpy — that import
    must never be pulled into the server process, only the worker.
    """
    if hasattr(obj, "item") and hasattr(obj, "dtype") and getattr(obj, "ndim", None) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _encode_frame(payload: dict[str, Any]) -> bytes:
    """Serialize a payload dict to a length-prefixed JSON frame."""
    body = json.dumps(payload, default=_json_default).encode("utf-8")
    return len(body).to_bytes(_LENGTH_HEADER_BYTES, "big") + body


def _read_exactly(stream: BinaryIO, count: int) -> bytes | None:
    """Read exactly ``count`` bytes, looping over short reads; ``None`` on EOF.

    A single ``read(n)`` on a Windows pipe returns short, so a large result body
    must be reassembled across multiple reads.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    """Read one length-prefixed JSON frame; ``None`` if the stream is at EOF."""
    header = _read_exactly(stream, _LENGTH_HEADER_BYTES)
    if header is None:
        return None
    body = _read_exactly(stream, int.from_bytes(header, "big"))
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def _write_frame_to_fd(fd: int, payload: dict[str, Any]) -> None:
    """Write a frame to a raw fd, looping over partial ``os.write`` results."""
    view = memoryview(_encode_frame(payload))
    while view:
        view = view[os.write(fd, view):]


# ---------------------------------------------------------------------------
# Worker side — runs in the spawned subprocess
# ---------------------------------------------------------------------------


def _handle_request(module: str, qualname: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``module:qualname``, call it, and wrap the outcome as a response dict.

    Async tools run via ``asyncio.run``; sync tools are called directly. A raised
    exception becomes ``{"ok": False, ...}`` so it can be re-raised server-side.
    """
    try:
        target: Any = importlib.import_module(module)
        for part in qualname.split("."):
            target = getattr(target, part)
        if inspect.iscoroutinefunction(target):
            result = asyncio.run(target(**kwargs))
        else:
            result = target(**kwargs)
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 — relayed to the server as a structured response
        return {"ok": False, "exc_type": type(exc).__name__, "error": str(exc)}


def _serve() -> None:
    """Worker entrypoint: protect the frame channel, handshake, then dispatch.

    fd-1 is reassigned to stderr before any tool import so a native-library or stray
    ``print`` write cannot corrupt the frame stream; frames go only to a private
    ``dup`` of the original stdout.
    """
    frame_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    _write_frame_to_fd(frame_fd, {"ready": True})

    stdin = sys.stdin.buffer
    while True:
        request = _read_frame(stdin)
        if request is None:
            return
        response = _handle_request(
            request["module"], request["qualname"], request.get("kwargs", {})
        )
        try:
            frame = _encode_frame(response)
        except (TypeError, ValueError) as exc:
            frame = _encode_frame(
                {"ok": False, "exc_type": "TypeError", "error": f"result not JSON-serializable: {exc}"}
            )
        view = memoryview(frame)
        while view:
            view = view[os.write(frame_fd, view):]


# ---------------------------------------------------------------------------
# Server side — ToolWorker manager
# ---------------------------------------------------------------------------


def _default_spawn_command() -> list[str]:
    """Argv for the production worker: isolated (-I) + optimized (-OO)."""
    return [sys.executable, "-I", "-OO", "-m", "coden_retriever.mcp.tool_worker"]


class ToolWorker:
    """Owns one persistent warm worker subprocess, serialized across calls.

    A single warm worker keeps resident footprint minimal on a memory-bound box
    (no N× warm ONNX/cache copies). The ``asyncio.Lock`` serializes marked-tool
    calls through it; in-process (unmarked) tools are unaffected.
    """

    def __init__(self, spawn_command: list[str] | None = None) -> None:
        self._spawn_command = spawn_command or _default_spawn_command()
        self._lock = asyncio.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        atexit.register(self.shutdown)

    async def call(
        self,
        module: str,
        qualname: str,
        kwargs: dict[str, Any],
        timeout_s: float,
        name: str,
    ) -> Any:
        """Dispatch a tool in the worker, enforcing the kill-on-timeout deadline.

        Returns the tool's result. Raises :class:`ToolWorkerTimeout` (after killing
        the worker to free its memory), :class:`ToolWorkerError` (worker crash / pipe
        failure), or re-raises a tool's own exception as ``RuntimeError``.
        """
        async with self._lock:
            await self._ensure_ready()
            proc = self._proc
            assert proc is not None and proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(_encode_frame({"module": module, "qualname": qualname, "kwargs": kwargs}))
                proc.stdin.flush()
            except OSError as exc:
                self._kill()
                raise ToolWorkerError(f"worker pipe broke writing {name!r}: {exc}") from exc

            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(_read_frame, proc.stdout), timeout_s
                )
            except asyncio.TimeoutError as exc:
                pid = proc.pid
                self._kill()
                logger.warning(
                    "tool %s exceeded %ss; killed worker pid=%d to reclaim memory",
                    name, timeout_s, pid,
                )
                raise ToolWorkerTimeout(name) from exc

            if response is None:
                self._kill()
                raise ToolWorkerError(f"worker exited before responding to {name!r} (crash/OOM?)")
            if response.get("ok"):
                return response["result"]
            raise RuntimeError(f"{response.get('exc_type', 'Error')}: {response.get('error', '')}")

    async def _ensure_ready(self) -> None:
        """Spawn the worker if needed and consume its ``ready`` handshake.

        The handshake turns stdout-corruption or an import failure into a prompt,
        deterministic error instead of a hang on the first request.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        self._spawn()
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            ready = await asyncio.wait_for(
                asyncio.to_thread(_read_frame, proc.stdout), _HANDSHAKE_TIMEOUT_S
            )
        except asyncio.TimeoutError as exc:
            self._kill()
            raise ToolWorkerError("worker did not handshake within timeout") from exc
        if not (ready and ready.get("ready")):
            self._kill()
            raise ToolWorkerError(f"worker sent a bad handshake: {ready!r}")

    def _spawn(self) -> None:
        """Start a fresh worker. stderr inherits the server's (no PIPE to deadlock on)."""
        self._proc = subprocess.Popen(
            self._spawn_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def _kill(self) -> None:
        """Kill and reap the worker so the OS reclaims its memory; clear the handle."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=_KILL_WAIT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("reaping killed worker failed: %s", exc)

    def shutdown(self) -> None:
        """atexit hook: best-effort kill so the worker can't outlive the server.

        An orphaned worker holding the ONNX model is exactly the leak being fixed.
        Runs at interpreter teardown where logging/loops may be gone, so a failing
        kill is swallowed — there is nothing further to do.
        """
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


_worker: ToolWorker | None = None


def get_worker() -> ToolWorker:
    """Return the process-wide :class:`ToolWorker` singleton, creating it on first use."""
    global _worker
    if _worker is None:
        _worker = ToolWorker()
    return _worker


if __name__ == "__main__":
    _serve()
