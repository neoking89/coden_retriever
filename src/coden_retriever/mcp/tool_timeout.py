"""Worker-dispatch wrapper for memory-heavy stateless MCP tools.

A tool opts in to the kill-on-timeout worker by carrying the :data:`WORKER_SAFE_ATTR`
marker (set by the :func:`worker_safe` decorator at its definition site). The single
chokepoint in ``server_factory._install_tool_timeout`` wraps **only** marked tools with
:func:`wrap_with_timeout`; unmarked tools register unchanged and run in-process, bounded
solely by the client-side ``read_timeout`` give-up.

``wrap_with_timeout`` does not run the tool in this process at all — it forwards the call
by ``module:qualname`` + JSON kwargs to the warm worker subprocess (see
``coden_retriever.mcp.tool_worker``), which is killed on timeout so its ONNX model and
whole-repo caches are reclaimed at the deadline. The structured error returned on a kill
is :func:`timeout_error_payload`, the same shape the agent already parses.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from .tool_worker import ToolWorkerError, ToolWorkerTimeout, get_worker

# Attribute marking a function as safe to run in the kill-on-timeout worker
# (stateless: loads caches from disk, returns a JSON-ish dict, holds no live
# session/registry/request-context state). Set declaratively via @worker_safe.
WORKER_SAFE_ATTR = "__coden_worker_safe__"

# Attribute marking a function already wrapped, so the chokepoint never
# double-wraps (e.g. a re-registration path, or an incomplete edit).
_WRAPPED_SENTINEL = "__coden_timeout_wrapped__"


def worker_safe(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a stateless tool as safe to run in the kill-on-timeout worker subprocess.

    Declarative opt-in at the definition site — no central name list. The function is
    returned unchanged; the chokepoint reads :data:`WORKER_SAFE_ATTR` to route it.
    """
    setattr(func, WORKER_SAFE_ATTR, True)
    return func


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


def wrap_with_timeout(func: Callable[..., Any], timeout_s: float) -> Callable[..., Any]:
    """Wrap a worker-safe tool so its call runs in the kill-on-timeout worker.

    The wrapper never invokes ``func`` in this process; it forwards
    ``func.__module__:func.__qualname__`` plus JSON kwargs to the worker. On timeout the
    worker is killed (freeing its memory) and :func:`timeout_error_payload` is returned;
    on a worker crash a structured error dict is returned; a tool's own exception
    propagates. Incoming positional/keyword args are bound to the signature with
    ``apply_defaults()`` so the worker runs with the same effective defaults as in-process.

    Preserves ``__name__``/``__doc__``/``__signature__`` so FastMCP's schema generator
    still inspects the wrapper as the original function. Idempotent: re-wrapping an
    already-wrapped func is a no-op (guards the single-chokepoint design).
    """
    if getattr(func, _WRAPPED_SENTINEL, False):
        return func

    name = func.__name__
    module = func.__module__
    qualname = func.__qualname__
    signature = inspect.signature(func)

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call_kwargs = dict(bound.arguments)
        try:
            return await get_worker().call(module, qualname, call_kwargs, timeout_s, name)
        except ToolWorkerTimeout:
            return timeout_error_payload(name, timeout_s)
        except ToolWorkerError as exc:
            return {"error": f"Tool '{name}' worker failed: {exc}", "type": "ToolWorkerError"}

    # functools.wraps copies __wrapped__ but not __signature__ when the original
    # was inspected via typing — set it explicitly so FastMCP's ParsedFunction
    # sees the real signature.
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
    setattr(wrapper, _WRAPPED_SENTINEL, True)
    return wrapper
