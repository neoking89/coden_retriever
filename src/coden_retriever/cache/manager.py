"""
Cache manager module.

Provides unified cache management for CLI and MCP with smart invalidation.
"""
import json
import logging
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Config, get_central_cache_root, get_project_cache_dir
from ..config_loader import get_semantic_model_path
from ..constants import SEMANTIC_INDEX_PROGRESS_LABEL
from ..semantic_config import SemanticConfig
from .graph_building import compute_centrality, build_lookup_structures, build_edges_from_references
from ..utils.source_walker import iter_source_files

from .embedding_cache import EmbeddingCache
from .layout import (
    LITE_CHANGE_COUNT_FILE,
    LITE_ENTITIES_FILE,
    STATIC_BM25_FILE,
    STATIC_CENTRALITY_FILE,
    STATIC_EMBEDDINGS_FILE,
    STATIC_ENTITIES_FILE,
    STATIC_GRAPH_FILE,
    STATIC_LAYOUT,
    STATIC_NODE_IDS_FILE,
    STATIC_TYPE_GRAPH_FILE,
    STATIC_USED_NAMES_BY_FILE_FILE,
    CacheLayout,
    all_paths,
    manifest_path,
)

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np

from ..git.process_metrics import git_head_sha, git_is_dirty, harvest_change_count
from ..models import CodeEntity
from ..parsers import RepoParser
from ..search.bm25 import BM25Index
from ..utils.progress import encoding_progress
from .models import BuiltIndices, CachedIndices, ChangeSet, LiteCachedIndices, ParsedFileBatch

logger = logging.getLogger(__name__)

# Five parallel pickle loads (entities, bm25, graph, centrality, used_names)
# match the five independent cache files; more workers would add scheduling
# overhead without parallelism gains because I/O is the bottleneck, not CPU.
_CACHE_LOAD_WORKERS = 5

# 30s is generous headroom over observed cold-cache loads (~2-5s on large repos
# with networked disks). A shorter timeout would spuriously fail on slow NFS;
# a longer one would mask a genuinely hung pickle load.
_CACHE_LOAD_TIMEOUT_SECONDS = 30

# Reported sizes in MB derive from bytes / (1024 * 1024); one named constant
# keeps the conversion identical at both call sites (cache status + list).
_BYTES_PER_MB = 1024 * 1024


class CacheManager:
    """
    Unified cache manager for CLI and MCP.

    Handles:
    - Cache validation (mtime/size checks)
    - Incremental updates
    - Full rebuilds
    - All index types

    Caches are stored centrally in ~/.coden-retriever/{project_key}/ for easy
    management across all projects.
    """

    LOGS_DIR = "logs"

    def __init__(
        self,
        source_dir: Path,
        semantic: SemanticConfig = SemanticConfig(),
        verbose: bool = False,
        layout: CacheLayout = STATIC_LAYOUT,
    ):
        self.source_dir = Path(source_dir).resolve()
        # Use central cache location instead of per-project .coden-retriever/
        self.cache_dir = get_project_cache_dir(self.source_dir)
        self.enable_semantic = semantic.enabled
        # Resolve here so manifest tracks the concrete path used for embeddings.
        # When semantic is disabled the path is meaningless — pinning it to None
        # at construction lets every consumer use self.model_path directly.
        self.model_path = (semantic.model_path or get_semantic_model_path()) if semantic.enabled else None
        self.verbose = verbose
        self._layout = layout

        self._manifest: dict | None = None
        self._parser = RepoParser()

    def load_or_rebuild(self) -> CachedIndices:
        """
        Main entry point. Returns ready-to-use indices.

        This is the primary method for obtaining search indices. It automatically
        handles cache validation and rebuilding as needed:

        1. Check cache validity (version, semantic mode)
        2. If valid and unchanged: load from cache (fast path, ~100ms)
        3. If changes detected: full rebuild (ensures graph consistency)
        4. If no cache exists: full rebuild

        Returns:
            CachedIndices containing all pre-computed search data structures.

        Example:
            >>> from pathlib import Path
            >>> from coden_retriever.cache import CacheManager
            >>> from coden_retriever.search import SearchEngine
            >>> from coden_retriever.semantic_config import SemanticConfig
            >>>
            >>> # Load or build cache for a repository
            >>> cache_mgr = CacheManager(
            ...     source_dir=Path("/path/to/repo"),
            ...     semantic=SemanticConfig(enabled=True),
            ...     verbose=True
            ... )
            >>> indices = cache_mgr.load_or_rebuild()
            >>> print(f"Loaded {len(indices.entities)} entities")
            Loaded 1247 entities
            >>>
            >>> # Create search engine from cached indices
            >>> engine = SearchEngine.from_cached_indices(indices)
            >>> results = engine.search("authentication")
        """
        start_time = time.time()

        # Try to load manifest
        manifest = self._load_manifest()

        if manifest is None:
            logger.info("No cache found, performing full rebuild...")
            return self._full_rebuild()

        # Check for version mismatch
        if manifest.get("version") != self._layout.version:
            logger.info("Cache version mismatch, performing full rebuild...")
            return self._full_rebuild()

        # Only rebuild when upgrading to semantic mode (need to build embeddings).
        # Downgrading (True→False) is fine: the extra semantic index is just unused.
        cached_semantic = manifest.get("enable_semantic", False)
        if self.enable_semantic and not cached_semantic:
            logger.info("Semantic mode enabled but cache lacks semantic index, performing full rebuild...")
            return self._full_rebuild()

        # Embedding model identity must match — different models produce
        # incompatible vectors, so the cache is invalid if the model changed.
        if self.enable_semantic and manifest.get("semantic_model_path") != self.model_path:
            logger.info("Semantic model path changed, performing full rebuild...")
            return self._full_rebuild()

        # Detect changes
        changes = self._detect_changes(manifest)

        if not changes.has_changes:
            # Fast path: load from cache
            logger.info("No changes detected, loading from cache...")
            indices = self._load_cached(manifest)
            if indices is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Cache loaded in {elapsed:.0f}ms")
                return indices
            else:
                logger.warning("Cache load failed, performing full rebuild...")
                return self._full_rebuild()

        # Any changes -> full rebuild (graph dependencies require complete data)
        return self._incremental_update(changes, manifest)

    def get_changes(self) -> ChangeSet:
        """Detect what files changed since last cache."""
        manifest = self._load_manifest()
        if manifest is None:
            return ChangeSet()
        return self._detect_changes(manifest)

    def invalidate(self) -> None:
        """Clear cache (for --refresh-cache or --clear-cache flags)."""
        self._clear_cache_preserve_logs()

    @staticmethod
    def _clear_single_cache_dir(cache_dir: Path) -> bool:
        """Clear a single cache directory but preserve the logs subdirectory.

        On Windows, log files may be locked by the debug logger. This method
        avoids WinError 32 by deleting only cache files, not the logs directory.

        Uses best-effort deletion: continues deleting other items if one fails.

        Args:
            cache_dir: The cache directory to clear

        Returns:
            True if cache existed (regardless of individual item failures),
            False if cache directory doesn't exist
        """
        if not cache_dir.exists():
            return False

        logs_dir = cache_dir / CacheManager.LOGS_DIR

        for item in cache_dir.iterdir():
            if item == logs_dir:
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {item}: {e}")

        return True

    def _clear_cache_preserve_logs(self) -> None:
        """Clear cache directory but preserve the logs subdirectory."""
        if CacheManager._clear_single_cache_dir(self.cache_dir):
            logger.info(f"Cache cleared: {self.cache_dir}")

    def _clear_layout(self, layout: CacheLayout) -> None:
        """Delete only the files declared by `layout`, leaving any coexisting
        flavor's cache and the logs subdirectory intact.

        Used when two flavors share the same cache directory and one needs to
        be invalidated without disturbing the other. The full-rebuild path
        still uses `_clear_cache_preserve_logs` (scorched-earth) because it
        rebuilds every artifact for its flavor anyway.
        """
        for path in all_paths(self.cache_dir, layout):
            if not path.exists():
                continue
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {path}: {e}")

    def get_cache_status(self) -> dict:
        """Get cache status information."""
        if not self.cache_dir.exists():
            return {"exists": False, "message": "No cache found"}

        manifest = self._load_manifest()
        if manifest is None:
            return {"exists": False, "message": "No valid manifest"}

        # Get file sizes for the artifacts owned by this manager's layout.
        # node_ids.json is excluded historically because it's a tiny mapping,
        # not user-facing index data; preserve that omission here.
        file_sizes = {}
        for name in self._layout.artifact_files:
            if name == STATIC_NODE_IDS_FILE:
                continue
            path = self.cache_dir / name
            if path.exists():
                size_mb = path.stat().st_size / _BYTES_PER_MB
                file_sizes[name] = f"{size_mb:.1f} MB"

        changes = self._detect_changes(manifest)

        return {
            "exists": True,
            "cache_dir": str(self.cache_dir),
            "created_at": manifest.get("created_at"),
            "updated_at": manifest.get("updated_at"),
            "file_count": manifest.get("file_count", 0),
            "entity_count": manifest.get("entity_count", 0),
            "enable_semantic": manifest.get("enable_semantic", False),
            "files": file_sizes,
            "changes": {
                "added": len(changes.added),
                "modified": len(changes.modified),
                "deleted": len(changes.deleted),
            },
            "recommended_action": self._recommend_action(changes),
        }

    def _recommend_action(self, changes: ChangeSet) -> str:
        """Recommend cache action based on changes."""
        if not changes.has_changes:
            return "Use cache (no changes)"
        return "Full rebuild (changes detected)"

    @staticmethod
    def list_all_caches() -> list[dict]:
        """List all cached projects in the central cache directory.

        Returns:
            List of dicts with cache info for each project:
            - cache_key: The directory name (e.g., "my_project_a1b2c3d4")
            - source_dir: Original project path (if available in manifest)
            - cache_dir: Full path to cache directory
            - created_at: Cache creation timestamp
            - updated_at: Last update timestamp
            - entity_count: Number of cached entities
            - size_mb: Total cache size in MB
        """
        cache_root = get_central_cache_root()
        if not cache_root.exists():
            return []

        caches = []
        for cache_dir in cache_root.iterdir():
            if not cache_dir.is_dir():
                continue

            # Static method enumerates the central cache root, where every project's
            # cache today is the static flavor — read its manifest filename directly.
            project_manifest = cache_dir / STATIC_LAYOUT.manifest_file
            if not project_manifest.exists():
                continue

            try:
                with open(project_manifest, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                # Calculate total cache size
                total_size = 0
                for item in cache_dir.iterdir():
                    if item.is_file():
                        total_size += item.stat().st_size

                caches.append({
                    "cache_key": cache_dir.name,
                    "source_dir": manifest.get("source_dir", "Unknown"),
                    "cache_dir": str(cache_dir),
                    "created_at": manifest.get("created_at"),
                    "updated_at": manifest.get("updated_at"),
                    "entity_count": manifest.get("entity_count", 0),
                    "file_count": manifest.get("file_count", 0),
                    "size_mb": round(total_size / _BYTES_PER_MB, 2),
                })
            except (json.JSONDecodeError, IOError):
                # Invalid manifest, include basic info
                caches.append({
                    "cache_key": cache_dir.name,
                    "source_dir": "Unknown (invalid manifest)",
                    "cache_dir": str(cache_dir),
                    "created_at": None,
                    "updated_at": None,
                    "entity_count": 0,
                    "file_count": 0,
                    "size_mb": 0,
                })

        # Sort by updated_at (most recent first)
        caches.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return caches

    @staticmethod
    def clear_all_caches() -> tuple[int, list[str]]:
        """Clear all project caches from the central cache directory.

        Preserves the logs subdirectory in each cache to avoid WinError 32
        on Windows when log files are open.

        Returns:
            Tuple of (count of cleared caches, list of error messages)
        """
        cache_root = get_central_cache_root()
        if not cache_root.exists():
            return 0, []

        cleared = 0
        errors = []

        for cache_dir in cache_root.iterdir():
            if not cache_dir.is_dir():
                continue

            if CacheManager._clear_single_cache_dir(cache_dir):
                cleared += 1
                logger.info(f"Cleared cache: {cache_dir.name}")
            else:
                errors.append(f"Failed to clear {cache_dir.name}")

        return cleared, errors

    @staticmethod
    def clear_cache_by_source_dir(source_dir: Path) -> bool:
        """Clear cache for a specific project by its source directory.

        Preserves the logs subdirectory to avoid WinError 32 on Windows
        when log files are open.

        Args:
            source_dir: The project source directory

        Returns:
            True if cache was cleared, False if not found or error
        """
        cache_dir = get_project_cache_dir(source_dir)
        if CacheManager._clear_single_cache_dir(cache_dir):
            logger.info(f"Cleared cache for: {source_dir}")
            return True
        return False


    def _load_manifest(self) -> dict | None:
        """Load manifest from cache."""
        path = manifest_path(self.cache_dir, self._layout)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load manifest: {e}")
            return None

    def _save_manifest(self, manifest: dict) -> None:
        """Save manifest to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = manifest_path(self.cache_dir, self._layout)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _create_manifest(
        self,
        files: dict[str, dict],
        entity_count: int,
        enable_semantic: bool
    ) -> dict:
        """Create a new manifest."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": self._layout.version,
            "source_dir": str(self.source_dir),  # Store source path for cache listing
            "created_at": now,
            "updated_at": now,
            "file_count": len(files),
            "entity_count": entity_count,
            "enable_semantic": enable_semantic,
            "semantic_model_path": self.model_path,
            "files": files,
        }


    def _detect_changes(self, manifest: dict) -> ChangeSet:
        """Detect what changed since last cache."""
        cached_files = manifest.get("files", {})
        current_files = self._scan_source_files()

        added = []
        modified = []
        deleted = []
        unchanged = []

        cached_paths = set(cached_files.keys())
        current_paths = set(current_files.keys())

        # Check for new files
        for rel_path in current_paths - cached_paths:
            added.append(rel_path)

        # Check for deleted files
        for rel_path in cached_paths - current_paths:
            deleted.append(rel_path)

        # Check for modified files
        for rel_path in cached_paths & current_paths:
            cached_info = cached_files[rel_path]
            current_info = current_files[rel_path]

            if (cached_info["mtime"] != current_info["mtime"] or
                    cached_info["size"] != current_info["size"]):
                modified.append(rel_path)
            else:
                unchanged.append(rel_path)

        return ChangeSet(added, modified, deleted, unchanged)

    def _scan_source_files(self) -> dict[str, dict]:
        """Scan source files and get their metadata."""
        files: dict[str, dict] = {}
        for path, stat in iter_source_files(self.source_dir):
            rel_path = str(path.relative_to(self.source_dir))
            files[rel_path] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        return files


    def _load_cached(self, manifest: dict) -> CachedIndices | None:
        """Load all indices from cache using parallel I/O."""
        try:
            # Load pickle files in parallel for faster I/O
            with ThreadPoolExecutor(max_workers=_CACHE_LOAD_WORKERS) as executor:
                futures = {
                    "entities": executor.submit(self._load_pickle, STATIC_ENTITIES_FILE),
                    "bm25": executor.submit(self._load_pickle, STATIC_BM25_FILE),
                    "graph": executor.submit(self._load_pickle, STATIC_GRAPH_FILE),
                    "type_graph": executor.submit(self._load_pickle, STATIC_TYPE_GRAPH_FILE),
                    "centrality": executor.submit(self._load_pickle, STATIC_CENTRALITY_FILE),
                    "used_names_by_file": executor.submit(self._load_pickle, STATIC_USED_NAMES_BY_FILE_FILE),
                }

                # Collect results with timeout
                results = {}
                for name, future in futures.items():
                    result = future.result(timeout=_CACHE_LOAD_TIMEOUT_SECONDS)
                    if result is None:
                        logger.warning(f"Failed to load {name} from cache")
                        return None
                    results[name] = result

            entities = results["entities"]
            bm25_index = results["bm25"]
            graph = results["graph"]
            type_graph = results["type_graph"]
            centrality = results["centrality"]
            used_names_by_file = results["used_names_by_file"]

            # Load embeddings (optional, already uses mmap for fast loading)
            embeddings = None
            node_ids = []
            if manifest.get("enable_semantic", False):
                embeddings = self._load_embeddings()
                node_ids = self._load_node_ids()

            embedding_cache = EmbeddingCache.load(self.cache_dir) or EmbeddingCache()

            lookups = build_lookup_structures(entities)

            return CachedIndices(
                entities=entities,
                embeddings=embeddings,
                node_ids=node_ids,
                bm25_index=bm25_index,
                graph=graph,
                type_graph=type_graph,
                centrality=centrality,
                lookups=lookups,
                source_dir=self.source_dir,
                manifest=manifest,
                model_path=manifest.get("semantic_model_path"),
                embedding_cache=embedding_cache,
                used_names_by_file=used_names_by_file,
            )

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _load_pickle(self, filename: str) -> Any:
        """Load a pickle file from cache.

        Returns the unpickled object or None on missing/failed load. `Any` is
        justified here: each cached file has a distinct payload shape
        (entities dict, BM25Index, nx.DiGraph, centrality dict) and the caller
        narrows the type at each use site.
        """
        path = self.cache_dir / filename
        if not path.exists():
            logger.warning(f"Cache file not found: {path}")
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            return None

    def _save_pickle(self, filename: str, data: Any) -> None:
        """Save data to a pickle file in cache.

        `Any` mirrors `_load_pickle`: the four cached payloads have unrelated
        types and pickle accepts arbitrary picklable objects.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / filename

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_embeddings(self) -> "np.ndarray | None":
        """Load embeddings from cache using mmap."""
        import numpy as np

        path = self.cache_dir / STATIC_EMBEDDINGS_FILE
        if not path.exists():
            return None

        try:
            # Use mmap for fast loading
            return np.load(path, mmap_mode='r')
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            return None

    def _save_embeddings(self, embeddings: "np.ndarray") -> None:
        """Save embeddings to cache."""
        import numpy as np

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / STATIC_EMBEDDINGS_FILE
        np.save(path, embeddings)

    def _load_node_ids(self) -> list[str]:
        """Load node IDs mapping from cache."""
        path = self.cache_dir / STATIC_NODE_IDS_FILE
        if not path.exists():
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load node IDs: {e}")
            return []

    def _save_node_ids(self, node_ids: list[str]) -> None:
        """Save node IDs mapping to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / STATIC_NODE_IDS_FILE

        with open(path, "w", encoding="utf-8") as f:
            json.dump(node_ids, f)


    def _parse_all_files(self, current_files: dict[str, dict]) -> ParsedFileBatch:
        """
        Parse all source files and collect entities, documents, references, and per-file used names.

        Args:
            current_files: Dict mapping relative path -> file metadata (mtime, size).
        """
        entities: dict[str, CodeEntity] = {}
        documents: dict[str, str] = {}
        all_references: list[tuple[str, int, str, str, str | None]] = []
        file_metadata: dict[str, dict] = {}
        used_names_by_file: dict[str, set[str]] = {}

        for rel_path, file_info in current_files.items():
            file_path = self.source_dir / rel_path
            success, file_entities, file_refs, file_used_names = self._parse_file(file_path)

            if success:
                entity_ids = []
                for entity in file_entities:
                    node_id = entity.node_id
                    entities[node_id] = entity
                    documents[node_id] = entity.searchable_text
                    entity_ids.append(node_id)

                for ref in file_refs:
                    all_references.append((str(file_path), *ref))

                used_names_by_file[str(file_path)] = file_used_names

                file_info["entity_ids"] = entity_ids
            else:
                file_info["entity_ids"] = []

            file_metadata[rel_path] = file_info

        return ParsedFileBatch(
            entities=entities,
            documents=documents,
            references=all_references,
            file_metadata=file_metadata,
            used_names_by_file=used_names_by_file,
        )

    def _build_indices(
        self,
        entities: dict[str, CodeEntity],
        documents: dict[str, str],
        all_references: list[tuple[str, int, str, str, str | None]],
    ) -> BuiltIndices:
        """
        Build all search indices from parsed entities.

        Args:
            entities: Dict mapping node_id -> CodeEntity
            documents: Dict mapping node_id -> searchable text
            all_references: List of (file_path, line, target_name, ref_type, receiver) tuples for graph building
        """
        graph, type_graph = self._build_graph(entities, all_references)
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        bm25_index = BM25Index()
        bm25_index.index(documents)
        logger.info("Built BM25 index")

        centrality = compute_centrality(graph, type_graph)
        logger.info("Computed centrality metrics")

        embeddings: "np.ndarray | None" = None
        node_ids: list[str] = []
        if self.enable_semantic:
            embeddings, node_ids = self._build_semantic_index(entities)
            logger.info(f"Built semantic index with {len(node_ids)} embeddings")

        return BuiltIndices(
            graph=graph,
            type_graph=type_graph,
            bm25_index=bm25_index,
            centrality=centrality,
            embeddings=embeddings,
            node_ids=node_ids,
        )

    def _full_rebuild(self) -> CachedIndices:
        """Rebuild all indices from scratch and return ready-to-use CachedIndices."""
        start_time = time.time()
        logger.info(f"Full rebuild starting for: {self.source_dir}")

        # Content-addressable entries remain valid after code changes — misses
        # are handled automatically, so we preserve the cache across rebuilds.
        embedding_cache = EmbeddingCache.load(self.cache_dir) or EmbeddingCache()

        # Surgical clear of just this flavor's files, leaving any coexisting
        # lite/static cache (and the embedding cache) intact.
        self._clear_layout(self._layout)

        current_files = self._scan_source_files()
        parsed = self._parse_all_files(current_files)
        logger.info(f"Parsed {len(parsed.entities)} entities from {len(current_files)} files")

        built = self._build_indices(parsed.entities, parsed.documents, parsed.references)

        lookups = build_lookup_structures(parsed.entities)

        manifest = self._create_manifest(
            files=parsed.file_metadata,
            entity_count=len(parsed.entities),
            enable_semantic=self.enable_semantic,
        )

        indices = CachedIndices(
            entities=parsed.entities,
            embeddings=built.embeddings,
            node_ids=built.node_ids,
            bm25_index=built.bm25_index,
            graph=built.graph,
            type_graph=built.type_graph,
            centrality=built.centrality,
            lookups=lookups,
            source_dir=self.source_dir,
            manifest=manifest,
            model_path=self.model_path,
            embedding_cache=embedding_cache,
            used_names_by_file=parsed.used_names_by_file,
        )

        self._save_cache(indices)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Full rebuild complete in {elapsed:.0f}ms")
        return indices


    def _incremental_update(self, changes: ChangeSet, manifest: dict) -> CachedIndices:
        """
        Update cache when files have changed.

        NOTE: Since graph/BM25 rebuild requires re-parsing all files for references anyway,
        we just do a full rebuild. The main value of caching is the "no changes" fast path.
        """
        logger.info(f"Changes detected: +{len(changes.added)}, ~{len(changes.modified)}, -{len(changes.deleted)}")
        logger.info("Performing full rebuild (graph dependencies require complete re-parse)")
        return self._full_rebuild()


    def _parse_file(
        self,
        file_path: Path
    ) -> tuple[bool, list[CodeEntity], list[tuple[int, str, str, str | None]], set[str]]:
        """Parse a single file."""
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="ignore")
            entities, references, used_names = self._parser.parse_file(
                str(file_path), source_code
            )
            return True, entities, references, used_names
        except Exception as e:
            if self.verbose:
                logger.debug(f"Failed to parse {file_path}: {e}")
            return False, [], [], set()

    def _build_graph(
        self,
        entities: dict[str, CodeEntity],
        references: list[tuple[str, int, str, str, str | None]]
    ) -> "tuple[nx.DiGraph, nx.DiGraph]":
        """Build the call graph and the parallel type-annotation subgraph.

        Args:
            entities: Dict mapping node_id -> CodeEntity
            references: List of (file_path, line, target_name, ref_type, receiver) tuples.
                receiver is the object name for method calls (e.g., 'cache' in 'cache.get()').

        Returns:
            (graph, type_graph). Type-annotation refs are added to BOTH graphs:
            the main graph at low weight (so type-only nodes accumulate some main
            PR), the type subgraph alone (so the type_ref signal sees a clean
            per-relation PR).
        """
        import networkx as nx  # lazy import: 140ms startup cost

        graph = nx.DiGraph()
        type_graph = nx.DiGraph()

        # Add all entities as nodes
        for node_id in entities:
            graph.add_node(node_id)

        lookups = build_lookup_structures(entities)
        build_edges_from_references(
            graph, entities, references,
            lookups.name_to_nodes, lookups.qualified_name_to_nodes,
            type_graph=type_graph,
        )

        return graph, type_graph

    def _build_semantic_index(
        self,
        entities: dict[str, CodeEntity]
    ) -> "tuple[np.ndarray | None, list[str]]":
        """Build semantic embeddings."""
        if not entities:
            return None, []

        try:
            from ..search.semantic import SemanticIndex

            model_path = self.model_path
            if model_path is None or not Path(model_path).exists():
                logger.warning("Semantic model not found, skipping semantic index")
                return None, []

            semantic_index = SemanticIndex(model_path)
            with encoding_progress(SEMANTIC_INDEX_PROGRESS_LABEL, len(entities)) as advance:
                semantic_index.index(entities, on_batch_done=advance)

            return semantic_index._embeddings, semantic_index._node_ids

        except ImportError as e:
            logger.warning(f"Cannot build semantic index: {e}")
            return None, []
        except Exception as e:
            logger.warning(f"Semantic indexing failed: {e}")
            return None, []

    def _save_cache(self, indices: CachedIndices) -> None:
        """Save all indices to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._save_pickle(STATIC_ENTITIES_FILE, indices.entities)
        self._save_pickle(STATIC_BM25_FILE, indices.bm25_index)
        self._save_pickle(STATIC_GRAPH_FILE, indices.graph)
        self._save_pickle(STATIC_TYPE_GRAPH_FILE, indices.type_graph)
        self._save_pickle(STATIC_CENTRALITY_FILE, indices.centrality)
        self._save_pickle(STATIC_USED_NAMES_BY_FILE_FILE, indices.used_names_by_file)

        if indices.embeddings is not None:
            self._save_embeddings(indices.embeddings)
            self._save_node_ids(indices.node_ids)

        if indices.embedding_cache is not None:
            indices.embedding_cache.save(self.cache_dir)

        self._save_manifest(indices.manifest)

        logger.info(f"Cache saved to: {self.cache_dir}")

    # ------------------------------------------------------------------ #
    # Lite cache flavor: parsing-skipping load path for `coden src --simple`.
    # Reuses _load_manifest, _save_manifest, _detect_changes,
    # _scan_source_files, _parse_all_files, _load_pickle, _save_pickle,
    # and _clear_layout — no plumbing duplication with the static path.
    # ------------------------------------------------------------------ #

    def load_or_rebuild_lite(self) -> LiteCachedIndices:
        """Load lite cache or rebuild whatever is invalid.

        Two-tier validity:

        - Entity validity: file mtime/size (reuses `_detect_changes`).
        - change_count validity: `git_head_sha` + `git_is_dirty`.

        On entity invalidation: full rebuild (re-parse + re-harvest).
        On git-only invalidation: keep entities, re-harvest change_count,
        refresh the manifest's git block.
        """
        start_time = time.time()

        manifest = self._load_manifest()
        if manifest is None:
            logger.info("No lite cache found, performing full rebuild...")
            return self._full_lite_rebuild()

        if manifest.get("version") != self._layout.version:
            logger.info("Lite cache version mismatch, performing full rebuild...")
            return self._full_lite_rebuild()

        changes = self._detect_changes(manifest)
        if changes.has_changes:
            logger.info(
                f"Lite cache: file changes detected (+{len(changes.added)}, "
                f"~{len(changes.modified)}, -{len(changes.deleted)}), full rebuild..."
            )
            return self._full_lite_rebuild()

        entities = self._load_pickle(LITE_ENTITIES_FILE)
        if entities is None:
            logger.info("Lite entities pickle missing or corrupt, full rebuild...")
            return self._full_lite_rebuild()

        cached_git = manifest.get("git") or {}
        cached_head = cached_git.get("head_sha")
        cached_dirty = cached_git.get("is_dirty", False)

        source_str = str(self.source_dir)
        current_head = git_head_sha(source_str)
        current_dirty = git_is_dirty(source_str)

        if cached_head == current_head and cached_dirty == current_dirty:
            change_count = self._load_pickle(LITE_CHANGE_COUNT_FILE)
            if change_count is None:
                logger.info("Lite change_count pickle missing, re-harvesting...")
                change_count = harvest_change_count(source_str)
                self._save_pickle(LITE_CHANGE_COUNT_FILE, change_count)
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Lite cache loaded in {elapsed:.0f}ms (warm)")
            return LiteCachedIndices(
                entities=entities,
                change_count=change_count,
                source_dir=self.source_dir,
                manifest=manifest,
            )

        logger.info(
            f"Lite cache: git state changed "
            f"(sha {cached_head}->{current_head}, dirty {cached_dirty}->{current_dirty}), "
            f"re-harvesting change_count..."
        )
        change_count = harvest_change_count(source_str)
        self._save_pickle(LITE_CHANGE_COUNT_FILE, change_count)

        manifest["git"] = {"head_sha": current_head, "is_dirty": current_dirty}
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Lite cache loaded in {elapsed:.0f}ms (entities warm, change_count refreshed)"
        )
        return LiteCachedIndices(
            entities=entities,
            change_count=change_count,
            source_dir=self.source_dir,
            manifest=manifest,
        )

    def _full_lite_rebuild(self) -> LiteCachedIndices:
        """Rebuild lite cache from scratch (parse + harvest).

        Uses surgical `_clear_layout(self._layout)` instead of the
        scorched-earth `_clear_cache_preserve_logs` so any coexisting static
        cache survives.
        """
        start_time = time.time()
        logger.info(f"Lite full rebuild starting for: {self.source_dir}")

        self._clear_layout(self._layout)

        current_files = self._scan_source_files()
        parsed = self._parse_all_files(current_files)
        logger.info(
            f"Lite: parsed {len(parsed.entities)} entities from {len(current_files)} files"
        )

        source_str = str(self.source_dir)
        change_count = harvest_change_count(source_str)
        head_sha = git_head_sha(source_str)
        is_dirty = git_is_dirty(source_str)

        manifest = self._create_lite_manifest(
            files=parsed.file_metadata,
            entity_count=len(parsed.entities),
            head_sha=head_sha,
            is_dirty=is_dirty,
        )

        self._save_pickle(LITE_ENTITIES_FILE, parsed.entities)
        self._save_pickle(LITE_CHANGE_COUNT_FILE, change_count)
        self._save_manifest(manifest)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Lite full rebuild complete in {elapsed:.0f}ms")

        return LiteCachedIndices(
            entities=parsed.entities,
            change_count=change_count,
            source_dir=self.source_dir,
            manifest=manifest,
        )

    def _create_lite_manifest(
        self,
        files: dict[str, dict],
        entity_count: int,
        head_sha: str | None,
        is_dirty: bool,
    ) -> dict:
        """Build a lite-cache manifest.

        `files` shape mirrors the static manifest so `_detect_changes` works
        unchanged. Top-level `git` block carries the keys consulted by
        `load_or_rebuild_lite` for the second validity tier.
        """
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": self._layout.version,
            "source_dir": str(self.source_dir),
            "created_at": now,
            "updated_at": now,
            "file_count": len(files),
            "entity_count": entity_count,
            "files": files,
            "git": {
                "head_sha": head_sha,
                "is_dirty": is_dirty,
            },
        }
