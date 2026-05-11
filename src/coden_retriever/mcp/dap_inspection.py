"""DAP inspection ops — read-only queries against a paused session.

Owns the wire calls behind threads/stackTrace/scopes/variables/evaluate/setVariable.
Component of `DAPClient` (composition pattern); accesses shared session state via
`self._client._state`, `self._client._path_mapper`, `self._client._adapter`, and
`self._client.protocol` for the wire call. Does NOT mutate transport, breakpoint
tracker (except via `_refresh_frame_context` and `_scan_other_threads_bp_hits`),
or adapter binding — those belong to lifecycle/breakpoint ops.
"""
from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any

from .dap_constants import DAP_STACK_PAUSE_TIMEOUT
from .dap_status import success_with

if TYPE_CHECKING:
    from .dap_client import DAPClient

logger = logging.getLogger(__name__)


class DAPInspection:
    """Read-only DAP queries — threads, frames, scopes, variables, eval."""

    def __init__(self, client: DAPClient) -> None:
        self._client = client

    async def get_threads(self) -> dict[str, Any]:
        """Get all threads in the debug session."""
        if not self._client.is_connected:
            return {"error": "Not connected"}

        response = await self._client.protocol.send_request("threads", {})
        if response.get("success"):
            threads = [
                {"id": t["id"], "name": t["name"]}
                for t in response.get("body", {}).get("threads", [])
            ]
            return success_with(threads=threads)
        return {"error": response.get("message", "Failed to get threads")}

    async def get_stack_trace(
        self, levels: int = 20, thread_id: int | None = None,
    ) -> dict[str, Any]:
        """Get the stack trace for a thread (defaults to active thread)."""
        guard = self._guard_stopped_with_thread(thread_id)
        if isinstance(guard, dict):
            return guard
        tid = guard

        response = await self._client.protocol.send_request("stackTrace", {
            "threadId": tid,
            "startFrame": 0,
            "levels": levels,
        })
        if not response.get("success"):
            return {"error": response.get("message", "Failed to get stack trace")}

        frames = self._format_stack_frames(response.get("body", {}).get("stackFrames", []))
        return success_with(frames=frames)

    def _guard_stopped_with_thread(self, thread_id: int | None) -> int | dict[str, Any]:
        """Common guard for thread-scoped queries — returns tid or error dict."""
        if not self._client.is_connected:
            return {"error": "Not connected"}
        tid = thread_id if thread_id is not None else self._client._state.thread_id
        if tid is None:
            return {"error": "No active thread"}
        if not self._client._state.is_stopped:
            return {"error": "Program is running"}
        return tid

    def _format_stack_frames(
        self, raw_frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map adapter paths to client paths, set first-frame state, return dict list."""
        state = self._client._state
        path_mapper = self._client._path_mapper
        out: list[dict[str, Any]] = []
        for frame in raw_frames:
            source = frame.get("source", {})
            file = path_mapper.to_client(source.get("path"))
            line = frame.get("line", 0)
            out.append({
                "id": frame.get("id", 0),
                "name": frame.get("name", ""),
                "file": file,
                "line": line,
            })
            if not state.stopped_file and file:
                state.stopped_file = file
                state.stopped_line = line
        if out:
            state.current_frame_id = out[0]["id"]
        return out

    async def get_variables(self, frame_id: int | None = None) -> dict[str, Any]:
        """Get variables in the current scope, grouped by scope name."""
        guard = self._guard_stopped()
        if guard is not None:
            return guard
        frame = frame_id or self._client._state.current_frame_id
        if not frame:
            return {"error": "No frame selected. Get stack trace first."}

        scopes_response = await self._client.protocol.send_request(
            "scopes", {"frameId": frame},
        )
        if not scopes_response.get("success"):
            return {"error": scopes_response.get("message", "Failed to get scopes")}

        all_variables: dict[str, list[dict[str, Any]]] = {}
        for scope in scopes_response.get("body", {}).get("scopes", []):
            scope_vars = await self._fetch_scope_variables(scope)
            if scope_vars is not None:
                all_variables[scope.get("name", "Unknown")] = scope_vars
        return success_with(variables=all_variables)

    def _guard_stopped(self) -> dict[str, Any] | None:
        """Common guard — connected + stopped. Returns error dict or None."""
        if not self._client.is_connected:
            return {"error": "Not connected"}
        if not self._client._state.is_stopped:
            return {"error": "Program is running"}
        return None

    async def _fetch_scope_variables(
        self, scope: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Fetch+format the variables for one scope, or None if scope is unfetchable."""
        var_ref = scope.get("variablesReference", 0)
        if var_ref <= 0:
            return None
        vars_response = await self._client.protocol.send_request(
            "variables", {"variablesReference": var_ref},
        )
        if not vars_response.get("success"):
            return None
        return [
            self._format_variable(var)
            for var in vars_response.get("body", {}).get("variables", [])
        ]

    @staticmethod
    def _format_variable(var: dict[str, Any]) -> dict[str, Any]:
        """Normalize a DAP variable record to the public dict shape."""
        child_ref = var.get("variablesReference", 0)
        return {
            "name": var.get("name", ""),
            "value": var.get("value", ""),
            "type": var.get("type"),
            "expandable": child_ref > 0,
            "variables_reference": child_ref,
        }

    async def get_scopes(self, frame_id: int) -> dict[str, Any]:
        """Return the DAP `scopes` response body for a given frame.

        Used by `debug_variables(action='set')` to resolve the default
        `variablesReference` when the caller doesn't supply one — the
        "Locals" scope is the natural target for bare variable sets.
        """
        guard = self._guard_stopped()
        if guard is not None:
            return guard
        response = await self._client.protocol.send_request(
            "scopes", {"frameId": frame_id},
        )
        if not response.get("success"):
            return {"error": response.get("message", "Failed to get scopes")}
        return success_with(scopes=response.get("body", {}).get("scopes", []))

    async def set_variable(
        self,
        variables_reference: int,
        name: str,
        value: str,
    ) -> dict[str, Any]:
        """DAP `setVariable` request — wire-level only.

        Tool-layer callers (see `debug_variables(action='set')`) are
        responsible for capability gating via `CAPABILITY_REQUIRED[
        "set_variable"]` before calling. This method returns primitives
        only; no MCP envelope shaping here.
        """
        guard = self._guard_stopped()
        if guard is not None:
            return guard
        response = await self._client.protocol.send_request("setVariable", {
            "variablesReference": variables_reference,
            "name": name,
            "value": value,
        })
        if response.get("success"):
            body = response.get("body", {})
            return success_with(
                value=body.get("value", ""),
                type=body.get("type"),
                variables_reference=body.get("variablesReference", 0),
            )
        return {"error": response.get("message", "Failed to set variable")}

    async def get_variable_children(
        self, variables_reference: int,
    ) -> dict[str, Any]:
        """Expand a variable to see its children (object props, list items, etc.)."""
        guard = self._guard_stopped()
        if guard is not None:
            return guard

        response = await self._client.protocol.send_request("variables", {
            "variablesReference": variables_reference,
        })
        if not response.get("success"):
            return {"error": response.get("message", "Failed to expand variable")}
        children = [
            self._format_variable(var)
            for var in response.get("body", {}).get("variables", [])
        ]
        return success_with(variables=children)

    async def evaluate(
        self, expression: str, frame_id: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate an expression in the current context."""
        guard = self._guard_stopped()
        if guard is not None:
            return guard

        validation_err = self._validate_python_expression(expression)
        if validation_err is not None:
            return validation_err

        args = self._build_evaluate_args(expression, frame_id)
        response = await self._client.protocol.send_request("evaluate", args)
        if response.get("success"):
            body = response.get("body", {})
            return success_with(result=body.get("result", ""), type=body.get("type"))
        return {"error": response.get("message", "Evaluation failed")}

    def _validate_python_expression(self, expression: str) -> dict[str, Any] | None:
        """Reject statements (e.g. `import`, `a=1`) on Python sessions only.

        DAP `evaluate` accepts expressions only; statements silently return
        empty results otherwise. Other adapters see their own syntax intact.
        """
        adapter = self._client._adapter
        if adapter is not None and adapter.name != "python":
            return None
        try:
            ast.parse(expression, mode="eval")
            return None
        except SyntaxError as e:
            return {
                "error": (
                    f"Not a single expression (DAP evaluate only accepts "
                    f"expressions, not statements): {e.msg}"
                )
            }

    def _build_evaluate_args(
        self, expression: str, frame_id: int | None,
    ) -> dict[str, Any]:
        """Build the DAP `evaluate` request body, applying adapter expression transforms."""
        adapter = self._client._adapter
        wire_expression = (
            adapter.transform_eval_expression(expression)
            if adapter is not None else expression
        )
        args: dict[str, Any] = {
            "expression": wire_expression,
            "context": "repl",
        }
        frame = frame_id or self._client._state.current_frame_id
        if frame:
            args["frameId"] = frame
        return args

    async def refresh_frame_context(self) -> None:
        """Refresh frame context (frame_id, file, line) after stopping.

        Called automatically after every stop so evaluate() and get_variables()
        work with the current frame. Public to the facade — `_wait_for_stop`
        calls this instead of a previously-private helper.
        """
        if self._client._state.thread_id is None:
            return
        await self._populate_top_frame()
        if self._client._state.stopped_reason == "breakpoint":
            await self._scan_other_threads_bp_hits()

    async def _populate_top_frame(self) -> None:
        """Fetch the top frame and write file/line/current_frame_id to state."""
        try:
            response = await self._client.protocol.send_request(
                "stackTrace",
                {
                    "threadId": self._client._state.thread_id,
                    "startFrame": 0,
                    "levels": 1,
                },
                timeout=DAP_STACK_PAUSE_TIMEOUT,
            )
        except Exception as e:
            logger.debug(f"Failed to refresh frame context: {e}")
            return
        if not response.get("success"):
            return
        frames = response.get("body", {}).get("stackFrames", [])
        if not frames:
            return
        top = frames[0]
        state = self._client._state
        state.current_frame_id = top.get("id")
        source = top.get("source", {})
        state.stopped_file = self._client._path_mapper.to_client(source.get("path"))
        state.stopped_line = top.get("line", 0)
        self._client.breakpoints.resolve_location_match(
            state.stopped_file, state.stopped_line, state.thread_id,
        )

    async def _scan_other_threads_bp_hits(self) -> None:
        """Count concurrent bp hits on threads other than the primary stopped one.

        When debugpy emits a single stopped event for allThreadsStopped=True,
        sibling threads paused at the same bp line don't produce their own
        events. This scan fills that gap by inspecting each thread's top
        frame and delegating bookkeeping to BreakpointTracker.
        """
        primary_tid = self._client._state.thread_id
        tids = await self._fetch_thread_ids()
        if tids is None:
            return
        for tid in tids:
            if tid == primary_tid:
                continue
            await self._record_sibling_thread_state(tid)

    async def _fetch_thread_ids(self) -> list[int] | None:
        """Return all known thread IDs, or None if the threads request failed."""
        try:
            resp = await self._client.protocol.send_request(
                "threads", {}, timeout=DAP_STACK_PAUSE_TIMEOUT,
            )
        except Exception as e:
            logger.debug(f"Thread scan skipped: {e}")
            return None
        if not resp.get("success"):
            return None
        return [t["id"] for t in resp.get("body", {}).get("threads", [])]

    async def _record_sibling_thread_state(self, tid: int) -> None:
        """Inspect one sibling thread's top frame and update breakpoint marks."""
        try:
            st = await self._client.protocol.send_request(
                "stackTrace",
                {"threadId": tid, "startFrame": 0, "levels": 1},
                timeout=DAP_STACK_PAUSE_TIMEOUT,
            )
        except Exception:
            return
        if not st.get("success"):
            return
        frames = st.get("body", {}).get("stackFrames", [])
        if not frames:
            self._client.breakpoints.drop_thread(tid)
            return
        top = frames[0]
        file_path = self._client._path_mapper.to_client(
            (top.get("source") or {}).get("path"),
        )
        self._client.breakpoints.record_sibling_thread_at(
            tid, file_path, top.get("line"),
        )
