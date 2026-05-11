"""DAP breakpoint ops — owns setBreakpoints / setExceptionBreakpoints wire calls.

Component of `DAPClient` (composition pattern). Holds no state beyond a back-ref
to the client; reads/writes `client.breakpoints` (BreakpointTracker), uses
`client._path_mapper` for adapter-side path translation, and `client.protocol`
for the actual wire call. The first-code-line scanner lives here too because
its sole call site is the pre-launch-bp setup in `_set_entry_breakpoint_if_requested`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .adapters.base import LaunchConfig
from .dap_breakpoint_tracker import DebugBreakpoint
from .dap_constants import DAP_REBIND_POLL_INTERVAL
from .dap_status import success_with
from .debug_errors import VALID_EXCEPTION_FILTERS

if TYPE_CHECKING:
    from .dap_client import DAPClient

logger = logging.getLogger(__name__)


def _dap_source_dict(adapter_path: str | None) -> dict[str, str]:
    """Build a DAP `Source` object for setBreakpoints requests.

    The DAP spec marks `name` optional, but fwcd/kotlin-debug-adapter
    dereferences `source.name` unconditionally inside `DAPConverter.toInternalSource`
    and NPEs with "Internal error." when it's absent. Sending the basename is
    harmless for every other adapter (they either ignore it or echo it back),
    so we always include it.
    """
    path = adapter_path or ""
    return {"path": path, "name": os.path.basename(path)}


class DAPBreakpointOps:
    """Wire-level ownership of all setBreakpoints / setExceptionBreakpoints traffic."""

    def __init__(self, client: DAPClient) -> None:
        self._client = client

    async def set_breakpoints(
        self,
        file: str,
        lines: list[int],
        conditions: dict[int, str] | None = None,
        log_messages: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Set breakpoints in a file.

        `log_messages` turns a breakpoint into a DAP logpoint — the adapter
        prints the message instead of pausing execution. Only meaningful when
        the adapter advertises `supportsLogPoints`; the tool layer
        (`debug_inspect.debug_breakpoint`) gates on that.
        """
        guard = self._validate_set_breakpoints_request(file)
        if isinstance(guard, dict):
            return guard
        file_path = guard
        conditions = conditions or {}
        log_messages = log_messages or {}

        if self._bps_match_existing(
            str(file_path), lines, conditions, log_messages,
        ):
            return self._cached_breakpoints_response(str(file_path))

        return await self._send_and_record_breakpoints(
            file_path, lines, conditions, log_messages,
        )

    def _validate_set_breakpoints_request(
        self, file: str,
    ) -> Path | dict[str, Any]:
        """Common pre-flight: connection check + file existence. Returns Path or error dict."""
        if not self._client.is_connected:
            return {"error": "Not connected. Launch a program first."}
        file_path = Path(file).resolve()
        if not file_path.exists():
            return {"error": f"File not found: {file}"}
        return file_path

    def _cached_breakpoints_response(self, file_path: str) -> dict[str, Any]:
        """Return the success envelope for an idempotent re-register (no wire call)."""
        cached = self._client.breakpoints.by_file.get(file_path, [])
        return success_with(breakpoints=[bp.to_dict() for bp in cached])

    async def _send_and_record_breakpoints(
        self,
        file_path: Path,
        lines: list[int],
        conditions: dict[int, str],
        log_messages: dict[int, str],
    ) -> dict[str, Any]:
        """Wire-call + tracker update for a non-idempotent setBreakpoints."""
        bp_payload = self._build_dap_breakpoints(lines, conditions, log_messages)
        response = await self._client.protocol.send_request("setBreakpoints", {
            "source": _dap_source_dict(self._client._path_mapper.to_adapter(str(file_path))),
            "breakpoints": bp_payload,
        })
        if not response.get("success"):
            return {"error": response.get("message", "Failed to set breakpoints")}

        verified = self._format_response_breakpoints(
            response.get("body", {}).get("breakpoints", []),
            lines, conditions, log_messages, str(file_path),
        )
        self._client.breakpoints.store_for_file(str(file_path), verified)
        return success_with(breakpoints=[bp.to_dict() for bp in verified])

    @staticmethod
    def _build_dap_breakpoints(
        lines: list[int],
        conditions: dict[int, str],
        log_messages: dict[int, str],
    ) -> list[dict[str, Any]]:
        """Build the DAP `setBreakpoints.breakpoints` payload from request inputs."""
        out: list[dict[str, Any]] = []
        for line in lines:
            bp: dict[str, Any] = {"line": line}
            if line in conditions:
                bp["condition"] = conditions[line]
            if line in log_messages:
                bp["logMessage"] = log_messages[line]
            out.append(bp)
        return out

    @staticmethod
    def _format_response_breakpoints(
        response_bps: list[dict[str, Any]],
        request_lines: list[int],
        conditions: dict[int, str],
        log_messages: dict[int, str],
        file_path: str,
    ) -> list[DebugBreakpoint]:
        """Match adapter response bps back to the request and build DebugBreakpoint records."""
        out: list[DebugBreakpoint] = []
        for i, bp in enumerate(response_bps):
            requested_line = request_lines[i] if i < len(request_lines) else None
            out.append(DebugBreakpoint(
                id=bp.get("id", 0),
                file=file_path,
                line=bp.get("line", 0),
                verified=bp.get("verified", False),
                condition=conditions.get(requested_line) if requested_line else None,
                log_message=log_messages.get(requested_line) if requested_line else None,
            ))
        return out

    def _bps_match_existing(
        self,
        file_path: str,
        lines: list[int],
        conditions: dict[int, str],
        log_messages: dict[int, str],
    ) -> bool:
        """True when the requested bps equal what we already have stored."""
        existing = self._client.breakpoints.by_file.get(file_path, [])
        if not self._lines_match(existing, lines):
            return False
        existing_cond = {bp.line: bp.condition for bp in existing if bp.condition}
        existing_log = {bp.line: bp.log_message for bp in existing if bp.log_message}
        return existing_cond == conditions and existing_log == log_messages

    @staticmethod
    def _lines_match(existing: list[DebugBreakpoint], lines: list[int]) -> bool:
        """True when the existing breakpoint set covers exactly the requested line set."""
        if len(existing) != len(lines):
            return False
        return {bp.line for bp in existing} == set(lines)

    async def set_exception_breakpoints(self, filters: list[str]) -> dict[str, Any]:
        """Set exception breakpoints (filters: 'raised', 'uncaught', 'userUnhandled')."""
        if not self._client.is_connected:
            return {"error": "Not connected. Launch or attach first."}

        invalid = [f for f in filters if f not in VALID_EXCEPTION_FILTERS]
        if invalid:
            return {
                "error": f"Invalid filters: {invalid}. Valid: {sorted(VALID_EXCEPTION_FILTERS)}",
            }

        response = await self._client.protocol.send_request(
            "setExceptionBreakpoints", {"filters": filters},
        )
        if response.get("success"):
            self._client.breakpoints.set_exception_filters(filters)
            return success_with(active_filters=filters)
        return {"error": response.get("message", "Failed to set exception breakpoints")}

    async def set_entry_breakpoint_if_requested(
        self, cfg: LaunchConfig, program_path: Path | None,
    ) -> None:
        """Install pre-configurationDone breakpoints.

        Priority:
        1. Explicit `cfg.extras["pre_launch_breakpoints"]` — DAP-spec correct
           path: caller passes concrete `{source, line}` pairs; required for
           compiled adapters where stop-on-entry is unreliable (lldb-dap,
           netcoredbg).
        2. `stop_on_entry=True` with a program path — fall back to the
           first-code-line heuristic (debugpy's working path).
        """
        pre_bps = (cfg.extras or {}).get("pre_launch_breakpoints")
        if pre_bps:
            await self.install_pre_launch_breakpoints(pre_bps)
            return
        if not cfg.stop_on_entry or program_path is None:
            return
        adapter = self._client._adapter
        if adapter is not None and not adapter.supports_entry_line_autopause:
            return
        first_line = self._find_first_code_line(program_path)
        if not first_line:
            return
        mapped = self._client._path_mapper.to_adapter(str(program_path))
        await self._client.protocol.send_request("setBreakpoints", {
            "source": _dap_source_dict(mapped),
            "breakpoints": [{"line": first_line}],
        })

    async def install_pre_launch_breakpoints(
        self, bps: list[dict[str, Any]],
    ) -> None:
        """Send setBreakpoints once per distinct source path in `bps`.

        DAP `setBreakpoints` replaces *all* breakpoints for a source, so a
        single call per source is required to preserve grouping when multiple
        lines target the same file.
        """
        for source, lines in self._group_bps_by_source(bps).items():
            await self._install_one_source(source, lines)

    @staticmethod
    def _group_bps_by_source(
        bps: list[dict[str, Any]],
    ) -> dict[str, list[int]]:
        """Group bp entries (each with `source`/`file` and `line`) by source path."""
        by_source: dict[str, list[int]] = {}
        for bp in bps:
            source = bp.get("source") or bp.get("file")
            line = bp.get("line")
            if not source or not line:
                continue
            by_source.setdefault(str(source), []).append(int(line))
        return by_source

    async def _install_one_source(self, source: str, lines: list[int]) -> None:
        """Send setBreakpoints for one source; record the result in BreakpointTracker."""
        mapped = self._client._path_mapper.to_adapter(source)
        resp = await self._client.protocol.send_request("setBreakpoints", {
            "source": _dap_source_dict(mapped),
            "breakpoints": [{"line": ln} for ln in lines],
        })
        verified = self._verified_from_install_response(source, lines, resp)
        if verified:
            self._client.breakpoints.store_for_file(source, verified)

    @staticmethod
    def _verified_from_install_response(
        source: str, lines: list[int], resp: dict[str, Any],
    ) -> list[DebugBreakpoint]:
        """Build DebugBreakpoint records from the install-source response, with empty-success fallback."""
        body = resp.get("body") or {}
        response_bps = body.get("breakpoints", [])
        if response_bps:
            return [
                DebugBreakpoint(
                    id=bp.get("id", 0),
                    file=source,
                    line=bp.get("line", lines[i] if i < len(lines) else 0),
                    verified=bool(bp.get("verified", False)),
                )
                for i, bp in enumerate(response_bps)
            ]
        if resp.get("success"):
            return [
                DebugBreakpoint(id=0, file=source, line=line, verified=False)
                for line in lines
            ]
        return []

    async def rebind_pre_launch_breakpoints(
        self, bps: list[dict[str, Any]], *, timeout: float,
    ) -> bool:
        """Re-issue setBreakpoints and poll until every bp reports verified=true.

        Only needed for CDP-backed adapters where source-file bps bind after
        script parse. Returns True if all bps verified within `timeout`.
        """
        by_source = self._group_bps_by_source(bps)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self._all_sources_verified(by_source):
                return True
            await asyncio.sleep(DAP_REBIND_POLL_INTERVAL)
        return False

    async def _all_sources_verified(
        self, by_source: dict[str, list[int]],
    ) -> bool:
        """One pass: send setBreakpoints for every source; return True iff all verified."""
        all_verified = True
        for source, lines in by_source.items():
            mapped = self._client._path_mapper.to_adapter(source)
            resp = await self._client.protocol.send_request("setBreakpoints", {
                "source": _dap_source_dict(mapped),
                "breakpoints": [{"line": ln} for ln in lines],
            })
            body = resp.get("body") or {}
            verified_all = all(
                bp.get("verified") is True for bp in body.get("breakpoints", [])
            )
            if not verified_all:
                all_verified = False
        return all_verified

    def _find_first_code_line(self, program_path: Path) -> int | None:
        """Find the first executable line — language-agnostic line scanner.

        Called for EVERY adapter's pre-launch-bp path (not just debugpy) to
        install an auto-entry bp before configurationDone. `ast.parse` would
        narrow the helper to Python-only and SyntaxError on `.rb`/`.go` sources
        — the fallback-to-1 then hands rdbg et al. a line-1 comment bp they
        hang on. A simple comment/docstring-aware scan works across languages
        sharing Python-style leading comments (every matrix fixture).
        """
        try:
            # errors="replace" so a compiled-binary program_path (cpp / rust /
            # dotnet) does not raise UnicodeDecodeError on stray non-UTF-8
            # bytes. The scanner returns 1 on a binary anyway (no executable
            # source line to find); the alternative — letting decode raise —
            # surfaces as 'adapter_internal' launch failure.
            content = program_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return 1
        state: tuple[bool, str | None] = (False, None)
        for i, line in enumerate(content.split("\n"), 1):
            skip, state = self._classify_line(line.strip(), state)
            if not skip:
                return i
        return 1

    @staticmethod
    def _classify_line(
        stripped: str, state: tuple[bool, str | None],
    ) -> tuple[bool, tuple[bool, str | None]]:
        """One step of the docstring state machine.

        Returns (skip_this_line, new_state). `state` is `(in_docstring, quote_char)`.
        """
        if DAPBreakpointOps._is_skippable(stripped):
            return True, state
        in_docstring, _ = state
        if in_docstring:
            return DAPBreakpointOps._step_inside_docstring(stripped, state)
        return DAPBreakpointOps._step_outside_docstring(stripped, state)

    @staticmethod
    def _is_skippable(stripped: str) -> bool:
        """Empty lines and `#` comments are always skipped, in or out of docstrings."""
        return not stripped or stripped.startswith("#")

    @staticmethod
    def _step_inside_docstring(
        stripped: str, state: tuple[bool, str | None],
    ) -> tuple[bool, tuple[bool, str | None]]:
        """Inside a docstring: stay in until the matching quote appears."""
        _, docstring_char = state
        if docstring_char and docstring_char in stripped:
            return True, (False, None)
        return True, state

    @staticmethod
    def _step_outside_docstring(
        stripped: str, state: tuple[bool, str | None],
    ) -> tuple[bool, tuple[bool, str | None]]:
        """Outside a docstring: open one (single or multi-line) or report first code line."""
        if not (stripped.startswith('"""') or stripped.startswith("'''")):
            return False, state
        quote = stripped[:3]
        if stripped.count(quote) >= 2:
            return True, state
        return True, (True, quote)
