"""Flatten a ConversationTree into picker rows (branch headers + tool calls)."""

from dataclasses import dataclass
from typing import Any, Union

from pydantic_ai.messages import ModelResponse, ToolCallPart

from .branch import ConversationBranch, ConversationTree
from .constants import MAX_ARGS_PREVIEW_LEN

# WHY 20: each arg value is capped before the whole preview is concatenated.
# Keeps one oversized arg from pushing shorter args off the visible line.
_MAX_ARG_VALUE_LEN: int = 20

# Ellipsis suffix used when truncating either a single arg value or the whole
# preview. Three dots is the universal "more here" signal in CLI output.
_TRUNCATION_SUFFIX: str = "..."


@dataclass(frozen=True)
class BranchHeaderRow:
    """A header row marking the start of a branch's tool calls in the picker."""

    branch_id: str
    is_current: bool
    label: str
    child_count: int


@dataclass(frozen=True)
class ToolCallRow:
    """A single tool call in the picker list. Selecting this forks the branch."""

    branch_id: str
    fork_message_index: int
    ordinal: int
    tool_call_id: str
    tool_name: str
    args_preview: str


PickerRow = Union[BranchHeaderRow, ToolCallRow]


def format_args_preview(args: dict[str, Any]) -> str:
    """Render a tool_input dict as a short one-line preview."""
    if not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        value_str = repr(value) if isinstance(value, str) else str(value)
        if len(value_str) > _MAX_ARG_VALUE_LEN:
            value_str = value_str[:_MAX_ARG_VALUE_LEN - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
        parts.append(f"{key}={value_str}")
    preview = ", ".join(parts)
    if len(preview) > MAX_ARGS_PREVIEW_LEN:
        preview = preview[:MAX_ARGS_PREVIEW_LEN - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
    return preview


def iter_tool_calls(branch: ConversationBranch) -> list[ToolCallRow]:
    """Walk a branch and emit one ToolCallRow per ToolCallPart encountered."""
    rows: list[ToolCallRow] = []
    ordinal = 0
    for msg_index, message in enumerate(branch.messages):
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if not isinstance(part, ToolCallPart):
                continue
            args = part.args if isinstance(part.args, dict) else {}
            rows.append(ToolCallRow(
                branch_id=branch.id,
                fork_message_index=msg_index,
                ordinal=ordinal,
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
                args_preview=format_args_preview(args),
            ))
            ordinal += 1
    return rows


def _ordered_branches(tree: ConversationTree) -> list[ConversationBranch]:
    """Return branches with current first, then others by creation time."""
    current = tree.current
    others = sorted(
        (b for b in tree.branches.values() if b.id != current.id),
        key=lambda b: b.created_at,
    )
    return [current, *others]


def build_rows(tree: ConversationTree) -> list[PickerRow]:
    """Flatten a tree into the picker's display order."""
    rows: list[PickerRow] = []
    for branch in _ordered_branches(tree):
        rows.append(BranchHeaderRow(
            branch_id=branch.id,
            is_current=(branch.id == tree.current_id),
            label=branch.label_hint,
            child_count=len(tree.children_of(branch.id)),
        ))
        rows.extend(iter_tool_calls(branch))
    return rows
