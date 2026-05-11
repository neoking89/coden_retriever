"""
Incremental index updater for file changes.

Handles updating search indices when files change without
requiring a full rebuild.
"""

import logging
import math
import threading
import time
from collections import defaultdict
from pathlib import Path


from ..cache import CachedIndices, CacheManager
from ..config import Config
from ..cache.graph_building import (
    compute_centrality,
    build_lookup_structures,
)
from ..constants import AMBIGUOUS_METHOD_NAMES
from ..models import CodeEntity
from ..parsers import RepoParser
from ..search.bm25 import BM25Index
from ..semantic_config import SemanticConfig
from .debouncer import BatchedChanges

logger = logging.getLogger(__name__)


# Threshold for triggering full rebuild vs incremental update
INCREMENTAL_THRESHOLD = 20  # Files changed


class IncrementalUpdater:
    """
    Updates cached indices incrementally when files change.

    For small changes (< INCREMENTAL_THRESHOLD files):
    - Removes entities from changed/deleted files
    - Re-parses changed/created files
    - Updates BM25 index (recalculates IDF)
    - Updates graph edges
    - Marks semantic index as needing rebuild

    For large changes (>= INCREMENTAL_THRESHOLD files):
    - Triggers full rebuild via CacheManager
    """

    def __init__(
        self,
        source_dir: Path,
        indices: CachedIndices,
        semantic: SemanticConfig = SemanticConfig(),
    ):
        """
        Initialize the incremental updater.

        Args:
            source_dir: Root directory of the project
            indices: Current cached indices to update
            semantic: Semantic search configuration
        """
        self.source_dir = Path(source_dir).resolve()
        self.indices = indices
        self.enable_semantic = semantic.enabled
        self.model_path = semantic.model_path
        self._parser = RepoParser()
        self._lock = threading.Lock()
        self._semantic_dirty = False

    def apply_changes(self, changes: BatchedChanges) -> tuple[CachedIndices, bool]:
        """
        Apply file changes to the indices.

        Args:
            changes: Batched file changes to apply

        Returns:
            Tuple of (updated_indices, was_full_rebuild)
        """
        with self._lock:
            total_changes = changes.total_count

            if total_changes == 0:
                return self.indices, False

            logger.info(f"Applying changes: +{len(changes.created)}, ~{len(changes.modified)}, -{len(changes.deleted)}")

            # For large changes, trigger full rebuild
            if total_changes >= INCREMENTAL_THRESHOLD:
                logger.info(f"Large change detected ({total_changes} files), triggering full rebuild")
                return self._full_rebuild(), True

            # Incremental update
            start_time = time.time()
            try:
                self._apply_incremental(changes)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"Incremental update completed in {elapsed_ms:.0f}ms")
                return self.indices, False
            except Exception as e:
                logger.exception(f"Incremental update failed: {e}, triggering full rebuild")
                return self._full_rebuild(), True

    def _full_rebuild(self) -> CachedIndices:
        """Perform a full cache rebuild."""
        cache_manager = CacheManager(
            source_dir=self.source_dir,
            semantic=SemanticConfig(enabled=self.enable_semantic, model_path=self.model_path),
        )
        return cache_manager.load_or_rebuild()

    def _apply_incremental(self, changes: BatchedChanges) -> None:
        """
        Apply incremental changes to indices.

        Strategy:
        1. Collect all affected files (created + modified + deleted)
        2. Remove old entities for affected files
        3. Parse new/modified files
        4. Add new entities
        5. Rebuild BM25 index (fast, O(n) where n = total entities)
        6. Update graph edges
        7. Recalculate centrality metrics
        """
        # Step 1: Collect affected files as relative paths
        affected_files: set[str] = set()
        files_to_parse: list[Path] = []

        for path in changes.deleted:
            rel_path = self._get_rel_path(path)
            if rel_path:
                affected_files.add(rel_path)

        for path in changes.modified:
            rel_path = self._get_rel_path(path)
            if rel_path:
                affected_files.add(rel_path)
                files_to_parse.append(path.resolve())

        for path in changes.created:
            rel_path = self._get_rel_path(path)
            if rel_path:
                affected_files.add(rel_path)
                files_to_parse.append(path.resolve())

        logger.debug(f"Affected files: {affected_files}")

        # Step 2: Remove old entities AND drop each affected file's
        # used_names contribution. Without the drop, a deleted callback
        # reference would keep suppressing the target's dead-code flag.
        entities_removed = self._remove_entities_for_files(affected_files)
        logger.debug(f"Removed {entities_removed} entities")

        # Resolve before keying — the full build keys used_names_by_file
        # under (source_dir.resolve() / rel_path); the OS watcher can emit
        # non-canonical paths, so without this both the pop and the write
        # below would silently miss the existing entry.
        for path in (*changes.deleted, *changes.modified, *changes.created):
            self.indices.used_names_by_file.pop(str(path.resolve()), None)

        # Step 3: Parse new/modified files
        new_entities: dict[str, CodeEntity] = {}
        all_references: list[tuple[str, int, str, str, str | None]] = []

        for file_path in files_to_parse:
            if not file_path.exists():
                continue

            entities, refs, used_names = self._parse_file(file_path)
            for entity in entities:
                new_entities[entity.node_id] = entity

            for ref in refs:
                all_references.append((str(file_path), *ref))

            self.indices.used_names_by_file[str(file_path)] = used_names

        logger.debug(f"Parsed {len(new_entities)} new entities")

        # Step 4: Add new entities
        self.indices.entities.update(new_entities)

        # Step 5: Rebuild BM25 index
        self._rebuild_bm25_index()

        # Step 6: Update graph
        self._update_graph(new_entities, all_references, affected_files)

        # Step 7: Recalculate centrality
        self._recalculate_centrality()

        # Step 8: Update lookup structures
        self._rebuild_lookup_structures()

        # Step 9: Update manifest
        self._update_manifest(affected_files)

        # Mark semantic as dirty (will be rebuilt on next full rebuild)
        if self.enable_semantic and new_entities:
            self._semantic_dirty = True
            logger.debug("Semantic index marked as dirty")

    def _get_rel_path(self, path: Path) -> str | None:
        """Get relative path from source dir, or None if outside."""
        try:
            abs_path = path.resolve()
            return str(abs_path.relative_to(self.source_dir))
        except ValueError:
            return None

    def _remove_entities_for_files(self, affected_files: set[str]) -> int:
        """Remove entities belonging to affected files."""
        entities_to_remove = []

        for node_id, entity in self.indices.entities.items():
            entity_rel_path = self._get_rel_path(Path(entity.file_path))
            if entity_rel_path in affected_files:
                entities_to_remove.append(node_id)

        for node_id in entities_to_remove:
            del self.indices.entities[node_id]
            
            for graph in [self.indices.graph, self.indices.type_graph]:
                if graph.has_node(node_id):
                    graph.remove_node(node_id)

        return len(entities_to_remove)

    def _parse_file(
        self, file_path: Path
    ) -> tuple[list[CodeEntity], list[tuple[int, str, str, str | None]], set[str]]:
        """Parse a single file and return entities, references, and used identifiers.

        Returns:
            Tuple of (entities, references, used_names). references are
            (line, target_name, ref_type, receiver) tuples.
        """
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="ignore")
            return self._parser.parse_file(str(file_path), source_code)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return [], [], set()

    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from current entities."""
        documents = {
            node_id: entity.searchable_text
            for node_id, entity in self.indices.entities.items()
        }

        new_index = BM25Index()
        new_index.index(documents)
        self.indices.bm25_index = new_index

    def _update_graph(
        self,
        new_entities: dict[str, CodeEntity],
        new_references: list[tuple[str, int, str, str, str | None]],
        affected_files: set[str],
    ) -> None:
        """Update graph with new entities and references.

        Args:
            new_entities: Dict mapping node_id -> CodeEntity for new/modified entities
            new_references: List of (file_path, line, target_name, ref_type, receiver) tuples.
                receiver is the object name for method calls (e.g., 'cache' in 'cache.get()').
            affected_files: Set of relative file paths that were changed
        """
        graph = self.indices.graph
        type_graph = self.indices.type_graph

        # Add new entity nodes
        for node_id in new_entities:
            if not graph.has_node(node_id):
                graph.add_node(node_id)

        # Build name to nodes mapping from current entities
        name_to_nodes: dict[str, list[str]] = defaultdict(list)
        qualified_name_to_nodes: dict[str, list[str]] = defaultdict(list)
        for node_id, entity in self.indices.entities.items():
            name_to_nodes[entity.name].append(node_id)
            # Build qualified name index for methods (ClassName.method)
            if entity.parent_class and entity.entity_type in ("method", "function"):
                qualified_name = f"{entity.parent_class}.{entity.name}"
                qualified_name_to_nodes[qualified_name].append(node_id)

        # Build file scopes for finding containing scope
        file_scopes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for node_id, entity in self.indices.entities.items():
            file_scopes[entity.file_path].append(
                (entity.line_start, entity.line_end, node_id)
            )

        # Sort scopes by size (smallest first)
        for file_path in file_scopes:
            file_scopes[file_path].sort(key=lambda x: x[1] - x[0])

        def find_containing_scope(file_path: str, line: int) -> str | None:
            scopes = file_scopes.get(file_path, [])
            for start, end, node_id in scopes:
                if start <= line <= end:
                    return node_id
            return None

        # Add edges from new references
        for file_path, line, target_name, ref_type, receiver in new_references:
            source_id = find_containing_scope(file_path, line)
            if not source_id:
                continue

            # Try qualified lookup first if receiver is available
            targets = []
            if receiver and qualified_name_to_nodes:
                qualified_name = f"{receiver}.{target_name}"
                targets = qualified_name_to_nodes.get(qualified_name, [])

            # Fall back to simple name lookup, but skip ambiguous method names
            # when we have a receiver but couldn't resolve it (too many false positives)
            if not targets:
                if receiver and target_name in AMBIGUOUS_METHOD_NAMES:
                    # Skip: e.g., cache.get() where we don't know what class cache is
                    continue
                targets = name_to_nodes.get(target_name, [])

            if not targets:
                continue

            dilution = 1.0 / math.sqrt(len(targets))

            for target_id in targets:
                if source_id == target_id:
                    continue

                target = self.indices.entities.get(target_id)
                if not target:
                    continue

                weight = Config.EDGE_WEIGHTS.get(ref_type, 1.0) * dilution

                if target.is_utility:
                    weight *= Config.PENALTY_UTILITY
                if target.is_tiny:
                    weight *= Config.PENALTY_TINY_FUNC
                if target.is_test:
                    weight *= Config.PENALTY_TEST

                if graph.has_edge(source_id, target_id):
                    graph[source_id][target_id]["weight"] += weight
                    graph[source_id][target_id]["types"].add(ref_type)
                else:
                    graph.add_edge(source_id, target_id, weight=weight, types={ref_type})

                if ref_type == "type":
                    if type_graph.has_edge(source_id, target_id):
                        type_graph[source_id][target_id]["weight"] += weight
                        type_graph[source_id][target_id]["types"].add(ref_type)
                    else:
                        type_graph.add_edge(source_id, target_id, weight=weight, types={ref_type})

    def _recalculate_centrality(self) -> None:
        """Recalculate PageRank, betweenness, and type-graph PageRank."""
        self.indices.centrality = compute_centrality(
            self.indices.graph, self.indices.type_graph
        )

    def _rebuild_lookup_structures(self) -> None:
        self.indices.lookups = build_lookup_structures(self.indices.entities)

    def _update_manifest(self, affected_files: set[str]) -> None:
        """Update manifest with new file metadata."""
        manifest = self.indices.manifest
        files = manifest.get("files", {})

        # Remove deleted files
        for rel_path in list(files.keys()):
            if rel_path in affected_files:
                abs_path = self.source_dir / rel_path
                if not abs_path.exists():
                    del files[rel_path]

        # Update modified/created files
        for rel_path in affected_files:
            abs_path = self.source_dir / rel_path
            if abs_path.exists():
                try:
                    stat = abs_path.stat()
                    entity_ids = [
                        node_id for node_id, entity in self.indices.entities.items()
                        if self._get_rel_path(Path(entity.file_path)) == rel_path
                    ]
                    files[rel_path] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "entity_ids": entity_ids,
                    }
                except OSError:
                    pass

        manifest["files"] = files
        manifest["entity_count"] = len(self.indices.entities)
        manifest["file_count"] = len(files)

    @property
    def semantic_needs_rebuild(self) -> bool:
        """Check if semantic index needs rebuilding."""
        return self._semantic_dirty
