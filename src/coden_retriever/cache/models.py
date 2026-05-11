"""
Cache data models.

Contains data structures for cache management.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..models.scores import CentralityCache

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from ..models import CodeEntity
    from ..search.bm25 import BM25Index
    from .embedding_cache import EmbeddingCache


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
    type_graph: "nx.DiGraph"
    centrality: CentralityCache
    lookups: "EntityLookups"

    # Metadata
    source_dir: Path
    manifest: dict

    # Embedding model identity, persisted so cache invalidation can detect a swap.
    model_path: str | None = None

    embedding_cache: "EmbeddingCache | None" = field(default=None)

    # Per-file map of identifiers appearing in non-definition contexts
    # (Vulture-style). Tracking by file is what lets the watcher *replace* a
    # file's contribution when it's re-parsed — a flat set would only ever
    # grow, leaving stale suppressions in place after a callback reference is
    # deleted. `used_names` is the union, recomputed on access.
    used_names_by_file: dict[str, set[str]] = field(default_factory=dict)

    @property
    def used_names(self) -> set[str]:
        """Union of per-file identifier-usage sets. Powers callback-dispatch
        suppression in dead-code detection."""
        return {
            name
            for names in self.used_names_by_file.values()
            for name in names
        }

    @property
    def entity_count(self) -> int:
        """Number of cached entities."""
        return len(self.entities)

    @property
    def has_semantic(self) -> bool:
        """Whether semantic embeddings are cached."""
        return self.embeddings is not None and len(self.node_ids) > 0


@dataclass
class LiteCachedIndices:
    """Container for the lite cache payload (parsed entities + commit counts).

    Path-encoding asymmetry: `entities` keys use native-separator relative
    paths (matches `_scan_source_files`); `change_count` keys use POSIX
    relative paths (matches `harvest_change_count`). `_simple_map_search`
    already converts via `to_repo_relative_posix` before crossing the two,
    so the two dicts can stay in their native formats here.
    """

    entities: dict[str, "CodeEntity"]
    change_count: dict[str, int]

    source_dir: Path
    manifest: dict


@dataclass
class ParsedFileBatch:
    """Outputs of `CacheManager._parse_all_files`.

    All five fields are produced together from one sweep over the source tree
    and travel as a unit into index construction. Per-file `used_names` are
    kept by file (not flattened) so the watcher can replace a file's
    contribution on re-parse instead of monotonically unioning forever.
    """

    entities: dict[str, "CodeEntity"]
    documents: dict[str, str]
    references: list[tuple[str, int, str, str, str | None]]
    file_metadata: dict[str, dict]
    used_names_by_file: dict[str, set[str]]


@dataclass
class BuiltIndices:
    """Outputs of `CacheManager._build_indices`.

    Holds the indices computed from a `ParsedFileBatch`: the dependency graph,
    the BM25 lexical index, the centrality snapshot, and (when semantic mode
    is enabled) the embeddings + ordered node-id list.
    """

    graph: "nx.DiGraph"
    type_graph: "nx.DiGraph"
    bm25_index: "BM25Index"
    centrality: CentralityCache
    embeddings: "np.ndarray | None"
    node_ids: list[str]


@dataclass
class EntityLookups:
    """Per-entity index structures derived from a `dict[node_id, CodeEntity]`.

    All four maps are produced together by `build_lookup_structures` from the
    same iteration over entities. They travel as a unit because every caller
    that needs name resolution also needs file-scope lookups (graph building,
    incremental updates, full rebuilds). `qualified_name_to_nodes` covers
    `ClassName.method` for receiver-typed call resolution.
    """

    name_to_nodes: dict[str, list[str]]
    file_scopes: dict[str, list[tuple[int, int, str]]]
    file_to_entities: dict[str, list[str]]
    qualified_name_to_nodes: dict[str, list[str]]
