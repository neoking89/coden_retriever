"""Per-call timeout wrapper applied globally to every MCP tool.

A single chokepoint in ``server_factory.create_mcp_server_with_config`` wraps
``mcp.tool`` so that every registered tool — built-in and dynamic — is bounded
by ``wrap_with_timeout``. The tool body runs entirely on a **worker thread**
(async tools via ``asyncio.run`` in that thread) while the event loop only waits
on a deadline; when it fires the worker is *abandoned* (the thread keeps running
until it finishes, since Python cannot kill it) and a structured error is
returned at once.

Why a worker thread rather than a same-loop timeout: coden's tools run large
**synchronous** sections directly on the calling coroutine — a
``daemon_enabled()`` connect attempt, then the CPU analysis itself — and they
yield to the loop only at sparse ``await`` points. A same-loop scheme
(``anyio.fail_after``, ``asyncio.wait`` on the coroutine, FastMCP's native
``timeout=``, an ``on_call_tool`` middleware) cannot fire while that synchronous
code holds the loop thread — verified live as ~60s against a 1s limit. Running
the whole tool off-loop is the only scheme that bounds it regardless of how the
tool is written internally.
"""
import asyncio
import functools
import inspect
import logging
from typing import Any, Callable

from anyio.to_thread import run_sync as _to_thread_run_sync

logger = logging.getLogger(__name__)

# Attribute marking a function already wrapped, so the chokepoint never
# double-wraps (e.g. a re-registration path, or an incomplete edit).
_WRAPPED_SENTINEL = "__coden_timeout_wrapped__"


def timeout_error_payload(name: str, timeout_s: float) -> dict[str, str]:
    """Structured error payload returned when a tool exceeds its timeout.

    Shape matches the error-dict convention the agent already parses
    (``react_loop.extract_tool_results`` treats a dict with ``error`` as a
    failed tool call).
    """
    return {
        "error": f"Tool '{name}' exceeded {timeout_s}s timeout",
        "type": "TimeoutError",
    }


def _discard_abandoned_task(task: "asyncio.Task[Any]") -> None:
    """Retrieve the result/exception of an abandoned task so asyncio stays quiet.

    A timed-out task is cancelled but not awaited; when it eventually finishes
    we must consume its outcome or asyncio logs "Task exception was never
    retrieved" / "Task was destroyed but it is pending".
    """
    if not task.cancelled():
        task.exception()


def wrap_with_timeout(func: Callable[..., Any], timeout_s: float) -> Callable[..., Any]:
    """Wrap a tool with a per-call timeout that surfaces a structured error.

    The whole tool runs on a worker thread (async tools via ``asyncio.run`` in
    that thread, sync tools called directly); the event loop only does
    ``asyncio.wait(timeout=timeout_s)``. The instant the deadline passes the
    worker task is *abandoned* (cancelled but not awaited) and the structured
    error is returned. Running off-loop is what lets the timeout fire even while
    the tool holds a thread on synchronous work — see the module docstring for
    why a same-loop timeout cannot (verified live: ~60s instead of 1s).

    Note: Python cannot kill an OS thread, so an abandoned tool keeps running
    until it finishes on its own — the agent just gets its clean structured
    error at ``timeout_s`` instead of waiting. For tools that spawn
    subprocesses, prefer the cancellable async helper at
    ``coden_retriever.agent.shell_exec.execute_shell``.

    Preserves __name__, __doc__, __annotations__, __signature__ so FastMCP's
    schema generator continues to inspect the wrapper as if it were the
    original function. Idempotent: re-wrapping an already-wrapped func is a
    no-op (guards the single-chokepoint design against double-wrap).
    """
    if getattr(func, _WRAPPED_SENTINEL, False):
        return func

    name = func.__name__
    is_async = inspect.iscoroutinefunction(func)

    def run_tool(*args: Any, **kwargs: Any) -> Any:
        # Runs on the worker thread. Async tools get their own short-lived event
        # loop here so their internal awaits never touch the server's loop.
        if is_async:
            return asyncio.run(func(*args, **kwargs))
        return func(*args, **kwargs)

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # abandon_on_cancel=True lets the worker thread be abandoned cleanly once
        # we stop awaiting it; the thread itself is not killable.
        task = asyncio.ensure_future(
            _to_thread_run_sync(
                functools.partial(run_tool, *args, **kwargs),
                abandon_on_cancel=True,
            )
        )
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if task not in done:
            task.cancel()
            task.add_done_callback(_discard_abandoned_task)
            logger.warning("tool %s exceeded %ss timeout", name, timeout_s)
            return timeout_error_payload(name, timeout_s)
        return task.result()

    # functools.wraps copies __wrapped__ but not __signature__ when the original
    # was inspected via typing — set it explicitly so FastMCP's ParsedFunction
    # sees the real signature.
    wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    setattr(wrapper, _WRAPPED_SENTINEL, True)
    return wrapper
