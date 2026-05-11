"""Conversation branch tree — the source of truth for `/undo`.

A `ConversationTree` holds N branches, each a list of pydantic-ai
`ModelMessage` objects. Fork slices a parent's prefix into a new branch
and switches HEAD; the parent is never mutated. Messages are pydantic
models (effectively immutable), so sharing references across branches
is safe and avoids deep copies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pydantic_ai.messages import ModelMessage

from .constants import BRANCH_ID_PREFIX, ROOT_BRANCH_ID


@dataclass
class ConversationBranch:
    """A single path through the conversation — ordered list of messages."""

    id: str
    parent_id: Optional[str]
    fork_message_index: Optional[int]
    messages: list[ModelMessage]
    created_at: datetime
    label_hint: str

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


@dataclass
class ConversationTree:
    """All branches in the session plus the HEAD pointer."""

    branches: dict[str, ConversationBranch] = field(default_factory=dict)
    current_id: str = ROOT_BRANCH_ID
    _next_id_counter: int = 1

    @classmethod
    def empty(cls) -> "ConversationTree":
        """Build a tree with a single empty root branch."""
        tree = cls()
        tree.branches[ROOT_BRANCH_ID] = ConversationBranch(
            id=ROOT_BRANCH_ID,
            parent_id=None,
            fork_message_index=None,
            messages=[],
            created_at=datetime.now(timezone.utc),
            label_hint="main",
        )
        return tree

    @property
    def current(self) -> ConversationBranch:
        return self.branches[self.current_id]

    def update_current(self, messages: list[ModelMessage]) -> None:
        """Replace the HEAD branch's messages. Called after every agent turn."""
        self.branches[self.current_id].messages = list(messages)

    def fork(
        self,
        source_branch_id: str,
        at_message_index: int,
        label_hint: str = "",
    ) -> str:
        """Create a new branch from `source.messages[:at_message_index]` and switch to it.

        The source branch is not mutated — pydantic-ai messages are frozen models,
        so sharing the sliced list is safe.
        """
        if source_branch_id not in self.branches:
            raise ValueError(f"Unknown source branch: {source_branch_id}")
        source = self.branches[source_branch_id]
        if at_message_index < 0 or at_message_index > len(source.messages):
            raise ValueError(
                f"Fork index {at_message_index} out of range for branch "
                f"{source_branch_id} with {len(source.messages)} messages",
            )

        new_id = f"{BRANCH_ID_PREFIX}{self._next_id_counter}"
        self._next_id_counter += 1

        self.branches[new_id] = ConversationBranch(
            id=new_id,
            parent_id=source_branch_id,
            fork_message_index=at_message_index,
            messages=source.messages[:at_message_index],
            created_at=datetime.now(timezone.utc),
            label_hint=label_hint or f"fork from {source_branch_id}@{at_message_index}",
        )
        self.current_id = new_id
        return new_id

    def switch(self, branch_id: str) -> None:
        """Move HEAD to an existing branch without mutating any messages."""
        if branch_id not in self.branches:
            raise ValueError(f"Unknown branch: {branch_id}")
        self.current_id = branch_id

    def clear(self) -> None:
        """Reset to a single empty root branch; all other branches are discarded."""
        self.branches.clear()
        self._next_id_counter = 1
        self.branches[ROOT_BRANCH_ID] = ConversationBranch(
            id=ROOT_BRANCH_ID,
            parent_id=None,
            fork_message_index=None,
            messages=[],
            created_at=datetime.now(timezone.utc),
            label_hint="main",
        )
        self.current_id = ROOT_BRANCH_ID

    def children_of(self, branch_id: str) -> list[ConversationBranch]:
        """Return direct children of a branch, ordered by creation time."""
        return sorted(
            (b for b in self.branches.values() if b.parent_id == branch_id),
            key=lambda b: b.created_at,
        )
