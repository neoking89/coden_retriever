"""Breakpoint bookkeeping — pure in-memory state, no DAP I/O.

`BreakpointTracker` owns every bp-related data structure that used to live on
`DAPClient`:
- `by_file` / `by_id` indexes
- `exception_filters` active on the session
- `thread_marks` — which (thread_id, bp_id) pairs are already counted, so
  allThreadsStopped re-exposures don't double-count siblings at a bp line
- `pending_location_match` — the location-based fallback flag when
  debugpy omits `hitBreakpointIds` on a stopped event

The tracker exposes helpers the orchestrator calls; all DAP requests still
live on `DAPClient`.
"""
from dataclasses import dataclass
from typing import Any

from .debug_errors import (
    SUGGESTION_CONDITION_NEVER_TRUE,
    SUGGESTION_LINE_NEVER_REACHED,
)


@dataclass
class DebugBreakpoint:
    """A verified breakpoint."""

    id: int
    file: str
    line: int
    verified: bool = True
    condition: str | None = None
    log_message: str | None = None
    hit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Canonical serialization — shared by set/add/list responses.

        `hit_count` is intentionally excluded here: hit counts belong in
        breakpoint_summary, not in the per-bp response shape.
        """
        return {
            "id": self.id,
            "file": self.file,
            "line": self.line,
            "verified": self.verified,
            "condition": self.condition,
            "log_message": self.log_message,
        }


class BreakpointTracker:
    """In-memory bookkeeping for a single debug session's breakpoints."""

    def __init__(self) -> None:
        self.by_file: dict[str, list[DebugBreakpoint]] = {}
        self.by_id: dict[int, DebugBreakpoint] = {}
        self.exception_filters: list[str] = []
        self.thread_marks: dict[int, set[int]] = {}
        self.pending_location_match: bool = False

    def clear(self) -> None:
        self.by_file.clear()
        self.by_id.clear()
        self.exception_filters = []
        self.thread_marks.clear()
        self.pending_location_match = False

    def store_for_file(self, file_path: str, bps: list[DebugBreakpoint]) -> None:
        """Replace the set of breakpoints for one file; refresh the id index.

        Preserves hit_count across re-issuance: DAP `setBreakpoints` is
        replace-all-in-file and debugpy assigns fresh ids on every call, so
        a BreakpointTracker keyed only on id would lose hit counts whenever
        a caller added/removed a line. We carry the count forward from any
        prior bp at the same line before overwriting.

        Also prunes stale ids from `by_id`: otherwise old ids accumulate
        forever across repeated setBreakpoints calls.
        """
        prior = self.by_file.get(file_path, [])
        prior_counts = {bp.line: bp.hit_count for bp in prior if bp.hit_count}
        for bp in bps:
            if bp.hit_count == 0 and bp.line in prior_counts:
                bp.hit_count = prior_counts[bp.line]
        for old_bp in prior:
            if old_bp.id > 0:
                self.by_id.pop(old_bp.id, None)
        self.by_file[file_path] = bps
        for bp in bps:
            if bp.id > 0:
                self.by_id[bp.id] = bp

    def set_exception_filters(self, filters: list[str]) -> None:
        self.exception_filters = list(filters)

    def track_hit_event(self, body: dict[str, Any]) -> None:
        """Update hit_count from a DAP 'stopped' event body.

        Prefers `hitBreakpointIds` (authoritative). Without it, arms
        `pending_location_match` so `resolve_location_match` can fill in once
        `_refresh_frame_context` has the stopped file/line.
        """
        hit_ids = body.get("hitBreakpointIds", [])
        tid = body.get("threadId")
        if hit_ids:
            for bp_id in hit_ids:
                bp = self.by_id.get(bp_id)
                if bp is None:
                    continue
                bp.hit_count += 1
                if tid is not None:
                    self.thread_marks.setdefault(tid, set()).add(bp_id)
            return
        if body.get("reason") == "breakpoint":
            self.pending_location_match = True

    def resolve_location_match(
        self,
        file: str | None,
        line: int | None,
        thread_id: int | None,
    ) -> None:
        """Consume `pending_location_match` by recording a hit at file:line."""
        if not self.pending_location_match:
            return
        self.pending_location_match = False
        if not file or not line:
            return
        for bp in self.by_file.get(file, []):
            if bp.line != line:
                continue
            bp.hit_count += 1
            if thread_id is not None:
                self.thread_marks.setdefault(thread_id, set()).add(bp.id)
            break

    def record_sibling_thread_at(
        self,
        thread_id: int,
        file: str | None,
        line: int | None,
    ) -> None:
        """Count a sibling thread parked at a bp line; prune stale marks."""
        marks = self.thread_marks.setdefault(thread_id, set())
        current_bp_ids: set[int] = set()
        if file and line:
            for bp in self.by_file.get(file, []):
                if bp.line != line:
                    continue
                current_bp_ids.add(bp.id)
                if bp.id in marks:
                    continue
                bp.hit_count += 1
                marks.add(bp.id)
        marks.intersection_update(current_bp_ids)
        if not marks:
            self.thread_marks.pop(thread_id, None)

    def drop_thread(self, thread_id: int) -> None:
        """Forget marks for a thread no longer parked at any bp."""
        self.thread_marks.pop(thread_id, None)

    def get_summary(self) -> dict[str, Any] | None:
        """Hit/miss summary across all known breakpoints, or None if none set."""
        all_bps = [bp for bps in self.by_file.values() for bp in bps]
        if not all_bps:
            return None

        hit: list[dict[str, Any]] = []
        never_hit: list[dict[str, Any]] = []
        for bp in all_bps:
            entry: dict[str, Any] = {"file": bp.file, "line": bp.line}
            if bp.hit_count > 0:
                entry["hit_count"] = bp.hit_count
                hit.append(entry)
                continue
            if bp.condition:
                entry["condition"] = bp.condition
                entry["suggestion"] = SUGGESTION_CONDITION_NEVER_TRUE
            else:
                entry["suggestion"] = SUGGESTION_LINE_NEVER_REACHED
            never_hit.append(entry)
        return {"total": len(all_bps), "hit": hit, "never_hit": never_hit}
