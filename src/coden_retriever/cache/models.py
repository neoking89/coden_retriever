"""
Cache data models.

Contains data structures for cache management.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from ..models import CodeEntity
    from ..search.bm25 import BM25Index


@dataclass
class ChangeSet:
    """Represents changes detected since last cache."""
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added or self.modified or self.deleted)


@dataclass
class CachedIndices:
    """Container for all cached search indices."""

    entities: dict[str, "CodeEntity"]
    embeddings: "np.ndarray | None"
    node_ids: list[str]
    bm25_index: "BM25Index"
    graph: "nx.DiGraph"
    pagerank: dict[str, float]
    betweenness: dict[str, float]

    # Lookup structures
    name_to_nodes: dict[str, list[str]]
    file_scopes: dict[str, list[tuple[int, int, str]]]
    file_to_entities: dict[str, list[str]]

    # Metadata
    source_dir: Path
    manifest: dict

    # Optional lookup structures (with defaults)
    qualified_name_to_nodes: dict[str, list[str]] = field(default_factory=dict)  # ClassName.method -> node_ids

    @property
    def entity_count(self) -> int:
        """Number of cached entities."""
        return len(self.entities)

    @property
    def has_semantic(self) -> bool:
        """Whether semantic embeddings are cached."""
        return self.embeddings is not None and len(self.node_ids) > 0
