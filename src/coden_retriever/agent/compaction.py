"""Minimal token-threshold history compaction for the interactive agent.

Runs once per turn, after the fallback loop settles and before
``QueryExecutor._finalize_response`` is called. When retained-context tokens
cross ``cfg.compaction_token_threshold``, drop the oldest complete tool-call
groups — nothing is inserted in their place. Forks an ``/undo`` snapshot
holding the full pre-compaction list so the user can always recover.

The only structural constraint is: never split a ``ToolCallPart`` from its
``ToolReturnPart``. Every interaction-group body lives between two
``ModelResponse`` messages, so cutting at any ``ModelResponse`` boundary
always drops a whole number of groups.

No synthetic placeholder is injected into the history. An earlier iteration
spliced a ``UserPromptPart`` reading "[COMPACTION] ... re-execute any tool
call you need" between ``messages[0]`` and the preserved suffix; the model
treated that as a fresh user directive and reverted to misusing tools or
skipping work. Dropping the placeholder removes that prompt-injection vector
— the user-visible bottom-line notice still names what was elided, and
``/undo`` still recovers the full history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from rich.markup import escape as _rich_escape
from rich.panel import Panel
from rich.text import Text

from .rich_console import console
from .token_usage import estimate_history_tokens

if TYPE_CHECKING:
    from ..config_loader import AgentConfig
    from .debug_logger import DebugLogger
    from .undo.branch import ConversationTree

logger = logging.getLogger(__name__)

_EventKind = Literal["attempt", "success", "abort"]

# (log-level, title, border color, render-panel?) — single source of truth
# for all three lifecycle events. The success event only logs; its summary
# lands in the dim bottom-line notice below the agent answer, so we don't
# double up by rendering a green panel mid-stream on top of it.
_EVENT_STYLES: dict[_EventKind, tuple[int, str, str, bool]] = {
    "attempt": (logging.INFO, "COMPACTING", "yellow", True),
    "success": (logging.INFO, "COMPACTED", "green", False),
    "abort": (logging.WARNING, "COMPACTION ABORTED", "red", True),
}

# Cap on the number of elided tool calls listed in the success panel.
# Anything past this gets a "+N more" tail so the banner stays compact.
_MAX_TOOL_CALL_LINES = 3

# Truncation cap for a single tool-call arg value shown in the banner.
_MAX_ARG_VALUE_CHARS = 40

# Minimum cut index: we always preserve messages[0] (the original user prompt),
# so the earliest safe elision starts at messages[1], making k=2 the first candidate.
_MIN_CUT_INDEX = 2

# Multiplier for integer percentage (out of 100).
_PCT_SCALE = 100


@dataclass(frozen=True)
class _CompactionStats:
    """Computed metrics for a successful compaction, passed to the display helpers.

    Groups the 6-9 individually-computed numbers so neither _build_bottom_line
    nor _build_success_detail exceeds the 5-parameter limit.
    """

    tool_call_lines: list[str]
    tool_call_count: int
    n_elided: int
    n_preserved: int
    before: int
    after: int
    saved: int
    pct: int
    snapshot_id: str


@dataclass(frozen=True)
class CompactionOutcome:
    """Result of a compaction attempt.

    ``messages`` is either the original list (no-op / abort) or a new list
    holding ``[messages[0], *messages[cut_index:]]``. ``happened`` is True
    only on a successful rewrite. ``bottom_line`` is a one-line rich-markup
    string the hook site prints *after* ``_finalize_response`` renders the
    agent answer, so the user sees the compaction outcome without scrolling
    back past the agent panel.
    """

    messages: list[ModelMessage]
    happened: bool = False
    bottom_line: Optional[str] = None


async def maybe_compact_history(
    messages: list[ModelMessage],
    tree: "ConversationTree",
    cfg: "AgentConfig",
    debug_logger: Optional["DebugLogger"] = None,
) -> CompactionOutcome:
    """Try to compact ``messages``. Returns an outcome describing what happened.

    On no-op / abort the outcome carries the original ``messages`` list
    unchanged and ``happened=False`` — the tree is not mutated and no
    bottom-line notice is produced.
    """
    threshold = cfg.compaction_token_threshold
    if threshold <= 0:
        return CompactionOutcome(messages=messages)

    before = estimate_history_tokens(messages)
    if before < threshold:
        return CompactionOutcome(messages=messages)

    _emit("attempt", f"retained={before:,} \u2265 threshold={threshold:,}", debug_logger)

    cut_index = _find_cut_index(messages, threshold)
    if cut_index is None:
        _abort("no safe boundary below threshold", debug_logger)
        return CompactionOutcome(messages=messages)

    new_messages = [messages[0], *messages[cut_index:]]
    after = estimate_history_tokens(new_messages)
    if after >= before:
        _abort(f"no savings ({before:,} \u2192 {after:,})", debug_logger)
        return CompactionOutcome(messages=messages)

    snapshot_id = _fork_snapshot(messages, tree)
    tool_call_lines, tool_call_count = _summarize_tool_calls(messages[1:cut_index])
    saved = before - after
    stats = _CompactionStats(
        tool_call_lines=tool_call_lines,
        tool_call_count=tool_call_count,
        n_elided=cut_index - 1,
        n_preserved=len(messages) - cut_index,
        before=before, after=after, saved=saved,
        pct=saved * _PCT_SCALE // max(before, 1),
        snapshot_id=snapshot_id,
    )

    _emit("success", _build_success_detail(stats), debug_logger)
    return CompactionOutcome(
        messages=new_messages, happened=True, bottom_line=_build_bottom_line(stats),
    )


def _build_bottom_line(stats: _CompactionStats) -> str:
    """Two dim lines printed below the agent answer.

    Line 1 names *what* was elided so the user can tell whether anything
    important was lost. Line 2 carries the numbers + recovery handle.
    """
    if stats.tool_call_count:
        plural = "s" if stats.tool_call_count != 1 else ""
        calls = ", ".join(_rich_escape(line) for line in stats.tool_call_lines)
        head = (
            f"[dim]\u25CF compaction: dropped {stats.tool_call_count} tool result{plural} "
            f"\u00b7 {calls}[/dim]"
        )
    else:
        head = f"[dim]\u25CF compaction: elided {stats.n_elided} msgs[/dim]"
    tail = (
        f"[dim]  saved {stats.saved:,} tokens (-{stats.pct}%) \u00b7 "
        f"snapshot {stats.snapshot_id} \u00b7 /undo to recover[/dim]"
    )
    return f"{head}\n{tail}"


def _find_cut_index(messages: list[ModelMessage], threshold: int) -> Optional[int]:
    """Smallest safe boundary index K where a splice drops us below threshold.

    A safe K is one where ``messages[K]`` is a ``ModelResponse`` (or ``K ==
    len(messages)``). Every tool-call group body sits between two
    ``ModelResponse``s, so dropping ``messages[1:K]`` at such a K never
    leaves an orphan ``ToolReturnPart``.

    Returns ``None`` only when the whole message list is one unbreakable
    block (no ``ModelResponse`` past index 0) — nothing safe to cut.
    """
    n = len(messages)
    for k in range(_MIN_CUT_INDEX, n + 1):
        if k < n and not isinstance(messages[k], ModelResponse):
            continue
        candidate = [messages[0], *messages[k:]]
        if estimate_history_tokens(candidate) < threshold:
            return k
    return None


def _summarize_tool_calls(elided: list[ModelMessage]) -> tuple[list[str], int]:
    """One-line-per-tool-call summary + the true total count.

    The returned list is capped at ``_MAX_TOOL_CALL_LINES`` with a ``+N
    more`` tail so the banner stays readable; the integer is the *real*
    number of unique tool calls regardless of capping — callers use it to
    label the panel ("dropped N tool results") accurately. Duplicate
    ``tool_call_id``s are skipped (same call is never counted twice
    across RSP/REQ pairs).
    """
    summaries: list[str] = []
    seen_ids: set[str] = set()
    for msg in elided:
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_call_id in seen_ids:
                continue
            seen_ids.add(part.tool_call_id)
            summaries.append(f"{part.tool_name}({_short_args(part.args)})")

    total = len(summaries)
    if total > _MAX_TOOL_CALL_LINES:
        remaining = total - _MAX_TOOL_CALL_LINES
        summaries = summaries[:_MAX_TOOL_CALL_LINES] + [f"+{remaining} more"]
    return summaries, total


def _short_args(args: object) -> str:
    """Pick the first dict key/value as a ``key=value`` string, truncated.

    Providers serialize tool args as either a ``dict`` or a JSON string;
    both are handled. Non-dict scalar args render as a bare truncated
    string so the reader still sees *something*.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return _truncate(args, _MAX_ARG_VALUE_CHARS)
    if not isinstance(args, dict) or not args:
        return ""
    first_key, first_val = next(iter(args.items()))
    return f"{first_key}={_truncate(str(first_val), _MAX_ARG_VALUE_CHARS)}"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)] + "\u2026"


def _build_success_detail(stats: _CompactionStats) -> str:
    """Multi-line body logged on success (never rendered as a panel — show_panel=False).

    Structure is fixed so the debug log is predictable: tool-call list (capped),
    token delta, snapshot id.
    """
    if stats.tool_call_count:
        call_block = f"dropped {stats.tool_call_count} tool results:\n"
        call_block += "\n".join(f"  \u2022 {line}" for line in stats.tool_call_lines)
    else:
        call_block = f"dropped {stats.n_elided} messages (no tool calls in elided span)"

    totals = (
        f"tokens: {stats.before:,} \u2192 {stats.after:,} (saved {stats.saved:,}, -{stats.pct}%) "
        f"\u00b7 {stats.n_preserved} tool results preserved"
    )
    footer = f"snapshot: {stats.snapshot_id} \u2014 /undo to recover full history"
    return "\n".join([call_block, totals, footer])


def _fork_snapshot(
    messages: list[ModelMessage], tree: "ConversationTree",
) -> str:
    """Fork a snapshot carrying the full pre-compaction history.

    The tree's current branch still holds the *previous* turn's history at
    hook time — ``loop.update_history`` runs later in ``_finalize_response``.
    We push the live pre-compaction list in first so ``fork``'s slice at
    ``len(messages)`` is in range.
    """
    old_branch_id = tree.current_id
    tree.update_current(messages)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    snapshot_id = tree.fork(
        old_branch_id,
        at_message_index=len(messages),
        label_hint=f"pre-compaction @ {stamp}",
    )
    tree.switch(old_branch_id)
    return snapshot_id


def _emit(
    kind: _EventKind, detail: str, debug_logger: Optional["DebugLogger"],
) -> None:
    level, title, border, show_panel = _EVENT_STYLES[kind]
    logger.log(level, "[compaction] %s: %s", title, detail.replace("\n", " | "))
    if show_panel:
        body = Text(_rich_escape(detail), style="bold")
        console.print()
        console.print(Panel(
            body,
            title=f"[bold {border}]\u25A0 {title}[/bold {border}]",
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ))
        console.print()
    if debug_logger is not None:
        debug_logger.log_compaction_event(f"{title}: {detail}")


def _abort(reason: str, debug_logger: Optional["DebugLogger"]) -> None:
    _emit("abort", f"{reason} \u2014 history unchanged", debug_logger)
