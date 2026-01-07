"""
Cache manager module.

Provides unified cache management for CLI and MCP with smart invalidation.
"""
import json
import logging
import os
import pickle
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config, get_central_cache_root, get_project_cache_dir
from .graph_building import compute_centrality, build_lookup_structures, build_edges_from_references
from ..language import LANGUAGE_MAP

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np

from ..models import CodeEntity
from ..parsers import RepoParser
from ..search.bm25 import BM25Index
from .models import CachedIndices, ChangeSet

logger = logging.getLogger(__name__)


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

    CACHE_VERSION = "1"
    LOGS_DIR = "logs"  # Preserved during cache clear (may have open handles)

    # Cache file names
    MANIFEST_FILE = "manifest.json"
    ENTITIES_FILE = "entities.pkl"
    EMBEDDINGS_FILE = "embeddings.npy"
    NODE_IDS_FILE = "node_ids.json"
    BM25_FILE = "bm25_index.pkl"
    GRAPH_FILE = "graph.pkl"
    CENTRALITY_FILE = "centrality.pkl"

    def __init__(
        self,
        source_dir: Path,
        enable_semantic: bool = False,
        model_path: str | None = None,
        verbose: bool = False
    ):
        self.source_dir = Path(source_dir).resolve()
        # Use central cache location instead of per-project .coden-retriever/
        self.cache_dir = get_project_cache_dir(self.source_dir)
        self.enable_semantic = enable_semantic
        self.model_path = model_path
        self.verbose = verbose

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
            >>>
            >>> # Load or build cache for a repository
            >>> cache_mgr = CacheManager(
            ...     source_dir=Path("/path/to/repo"),
            ...     enable_semantic=True,
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
        if manifest.get("version") != self.CACHE_VERSION:
            logger.info("Cache version mismatch, performing full rebuild...")
            return self._full_rebuild()

        # Check for source pattern mismatch (semantic mode change)
        cached_semantic = manifest.get("enable_semantic", False)
        if cached_semantic != self.enable_semantic:
            logger.info(f"Semantic mode changed ({cached_semantic} -> {self.enable_semantic}), performing full rebuild...")
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

        # Any changes → full rebuild (graph dependencies require complete data)
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

    def get_cache_status(self) -> dict:
        """Get cache status information."""
        if not self.cache_dir.exists():
            return {"exists": False, "message": "No cache found"}

        manifest = self._load_manifest()
        if manifest is None:
            return {"exists": False, "message": "No valid manifest"}

        # Get file sizes
        file_sizes = {}
        for name in [
            self.ENTITIES_FILE, self.EMBEDDINGS_FILE, self.BM25_FILE,
            self.GRAPH_FILE, self.CENTRALITY_FILE
        ]:
            path = self.cache_dir / name
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
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

            manifest_path = cache_dir / CacheManager.MANIFEST_FILE
            if not manifest_path.exists():
                continue

            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
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
                    "size_mb": round(total_size / (1024 * 1024), 2),
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
        manifest_path = self.cache_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load manifest: {e}")
            return None

    def _save_manifest(self, manifest: dict) -> None:
        """Save manifest to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / self.MANIFEST_FILE

        with open(manifest_path, "w", encoding="utf-8") as f:
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
            "version": self.CACHE_VERSION,
            "source_dir": str(self.source_dir),  # Store source path for cache listing
            "created_at": now,
            "updated_at": now,
            "file_count": len(files),
            "entity_count": entity_count,
            "enable_semantic": enable_semantic,
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
        files = {}
        skip_dirs = Config.SKIP_DIRS
        skip_files = Config.SKIP_FILES

        for root, dirs, filenames in os.walk(self.source_dir):
            # Prune directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]

            for name in filenames:
                if name in skip_files:
                    continue

                path = Path(root) / name
                if path.suffix.lower() not in LANGUAGE_MAP:
                    continue

                try:
                    stat = path.stat()
                    if stat.st_size > 1_000_000:
                        continue

                    rel_path = str(path.relative_to(self.source_dir))
                    files[rel_path] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                except Exception:
                    continue

        return files


    def _load_cached(self, manifest: dict) -> CachedIndices | None:
        """Load all indices from cache using parallel I/O."""
        try:
            # Load pickle files in parallel for faster I/O
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    "entities": executor.submit(self._load_pickle, self.ENTITIES_FILE),
                    "bm25": executor.submit(self._load_pickle, self.BM25_FILE),
                    "graph": executor.submit(self._load_pickle, self.GRAPH_FILE),
                    "centrality": executor.submit(self._load_pickle, self.CENTRALITY_FILE),
                }

                # Collect results with timeout
                results = {}
                for name, future in futures.items():
                    result = future.result(timeout=30)
                    if result is None:
                        logger.warning(f"Failed to load {name} from cache")
                        return None
                    results[name] = result

            entities = results["entities"]
            bm25_index = results["bm25"]
            graph = results["graph"]
            centrality = results["centrality"]

            # Load embeddings (optional, already uses mmap for fast loading)
            embeddings = None
            node_ids = []
            if manifest.get("enable_semantic", False):
                embeddings = self._load_embeddings()
                node_ids = self._load_node_ids()

            # Rebuild lookup structures from entities
            name_to_nodes, file_scopes, file_to_entities, qualified_name_to_nodes = build_lookup_structures(entities)

            return CachedIndices(
                entities=entities,
                embeddings=embeddings,
                node_ids=node_ids,
                bm25_index=bm25_index,
                graph=graph,
                pagerank=centrality.get("pagerank", {}),
                betweenness=centrality.get("betweenness", {}),
                name_to_nodes=name_to_nodes,
                file_scopes=file_scopes,
                file_to_entities=file_to_entities,
                qualified_name_to_nodes=qualified_name_to_nodes,
                source_dir=self.source_dir,
                manifest=manifest,
            )

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _load_pickle(self, filename: str):
        """Load a pickle file from cache."""
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

    def _save_pickle(self, filename: str, data) -> None:
        """Save data to a pickle file in cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / filename

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_embeddings(self) -> "np.ndarray | None":
        """Load embeddings from cache using mmap."""
        import numpy as np

        path = self.cache_dir / self.EMBEDDINGS_FILE
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
        path = self.cache_dir / self.EMBEDDINGS_FILE
        np.save(path, embeddings)

    def _load_node_ids(self) -> list[str]:
        """Load node IDs mapping from cache."""
        path = self.cache_dir / self.NODE_IDS_FILE
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
        path = self.cache_dir / self.NODE_IDS_FILE

        with open(path, "w", encoding="utf-8") as f:
            json.dump(node_ids, f)


    def _parse_all_files(
        self,
        current_files: dict[str, dict]
    ) -> tuple[dict[str, CodeEntity], dict[str, str], list[tuple[str, int, str, str, str | None]], dict[str, dict]]:
        """
        Parse all source files and collect entities, documents, and references.

        Args:
            current_files: Dict mapping relative path -> file metadata (mtime, size).

        Returns:
            Tuple of (entities, documents, all_references, file_metadata):
            - entities: Dict mapping node_id -> CodeEntity
            - documents: Dict mapping node_id -> searchable text (for BM25)
            - all_references: List of (file_path, line, target_name, ref_type, receiver) tuples
            - file_metadata: Updated file metadata with entity_ids added
        """
        entities: dict[str, CodeEntity] = {}
        documents: dict[str, str] = {}
        all_references: list[tuple[str, int, str, str, str | None]] = []
        file_metadata: dict[str, dict] = {}

        for rel_path, file_info in current_files.items():
            file_path = self.source_dir / rel_path
            success, file_entities, file_refs = self._parse_file(file_path)

            if success:
                entity_ids = []
                for entity in file_entities:
                    node_id = entity.node_id
                    entities[node_id] = entity
                    documents[node_id] = entity.searchable_text
                    entity_ids.append(node_id)

                for ref in file_refs:
                    all_references.append((str(file_path), *ref))

                file_info["entity_ids"] = entity_ids
            else:
                file_info["entity_ids"] = []

            file_metadata[rel_path] = file_info

        return entities, documents, all_references, file_metadata

    def _build_indices(
        self,
        entities: dict[str, CodeEntity],
        documents: dict[str, str],
        all_references: list[tuple[str, int, str, str, str | None]]
    ) -> "tuple[nx.DiGraph, BM25Index, dict[str, float], dict[str, float], np.ndarray | None, list[str]]":
        """
        Build all search indices from parsed entities.

        Args:
            entities: Dict mapping node_id -> CodeEntity
            documents: Dict mapping node_id -> searchable text
            all_references: List of (file_path, line, target_name, ref_type, receiver) tuples for graph building

        Returns:
            Tuple of (graph, bm25_index, pagerank, betweenness, embeddings, node_ids)
        """
        # Build dependency graph
        graph = self._build_graph(entities, all_references)
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Build BM25 index for lexical search
        bm25_index = BM25Index()
        bm25_index.index(documents)
        logger.info("Built BM25 index")

        # Compute centrality metrics for structural ranking
        pagerank, betweenness = compute_centrality(graph)
        logger.info("Computed centrality metrics")

        # Build semantic index if enabled
        embeddings = None
        node_ids: list[str] = []
        if self.enable_semantic:
            embeddings, node_ids = self._build_semantic_index(entities)
            logger.info(f"Built semantic index with {len(node_ids)} embeddings")

        return graph, bm25_index, pagerank, betweenness, embeddings, node_ids

    def _full_rebuild(self) -> CachedIndices:
        """
        Rebuild all indices from scratch.

        Orchestrates the full rebuild process:
        1. Clear existing cache
        2. Parse all source files
        3. Build search indices (graph, BM25, centrality, semantic)
        4. Build lookup structures
        5. Save everything to cache

        Returns:
            CachedIndices containing all pre-computed search data.
        """
        start_time = time.time()
        logger.info(f"Full rebuild starting for: {self.source_dir}")

        # Step 1: Clear existing cache (preserves logs directory)
        self._clear_cache_preserve_logs()

        # Step 2: Scan and parse all files
        current_files = self._scan_source_files()
        entities, documents, all_references, file_metadata = self._parse_all_files(current_files)
        logger.info(f"Parsed {len(entities)} entities from {len(current_files)} files")

        # Step 3: Build search indices
        graph, bm25_index, pagerank, betweenness, embeddings, node_ids = self._build_indices(
            entities, documents, all_references
        )

        # Step 4: Build lookup structures for fast access
        name_to_nodes, file_scopes, file_to_entities, qualified_name_to_nodes = build_lookup_structures(entities)

        # Step 5: Create manifest and save cache
        manifest = self._create_manifest(
            files=file_metadata,
            entity_count=len(entities),
            enable_semantic=self.enable_semantic,
        )

        self._save_cache(
            entities=entities,
            bm25_index=bm25_index,
            graph=graph,
            pagerank=pagerank,
            betweenness=betweenness,
            embeddings=embeddings,
            node_ids=node_ids,
            manifest=manifest,
        )

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Full rebuild complete in {elapsed:.0f}ms")

        return CachedIndices(
            entities=entities,
            embeddings=embeddings,
            node_ids=node_ids,
            bm25_index=bm25_index,
            graph=graph,
            pagerank=pagerank,
            betweenness=betweenness,
            name_to_nodes=name_to_nodes,
            file_scopes=file_scopes,
            file_to_entities=file_to_entities,
            qualified_name_to_nodes=qualified_name_to_nodes,
            source_dir=self.source_dir,
            manifest=manifest,
        )


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
    ) -> tuple[bool, list[CodeEntity], list[tuple[int, str, str, str | None]]]:
        """Parse a single file."""
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="ignore")
            entities, references = self._parser.parse_file(str(file_path), source_code)
            return True, entities, references
        except Exception as e:
            if self.verbose:
                logger.debug(f"Failed to parse {file_path}: {e}")
            return False, [], []

    def _build_graph(
        self,
        entities: dict[str, CodeEntity],
        references: list[tuple[str, int, str, str, str | None]]
    ) -> "nx.DiGraph":
        """Build the code graph from entities and references.
        
        Args:
            entities: Dict mapping node_id -> CodeEntity
            references: List of (file_path, line, target_name, ref_type, receiver) tuples.
                receiver is the object name for method calls (e.g., 'cache' in 'cache.get()').
        """
        import networkx as nx  # lazy import: 140ms startup cost

        graph = nx.DiGraph()

        # Add all entities as nodes
        for node_id in entities:
            graph.add_node(node_id)

        # Use shared lookup structures and edge building
        name_to_nodes, _, _, qualified_name_to_nodes = build_lookup_structures(entities)
        build_edges_from_references(
            graph, entities, references,
            name_to_nodes, qualified_name_to_nodes
        )

        return graph

    def _build_semantic_index(
        self,
        entities: dict[str, CodeEntity]
    ) -> "tuple[np.ndarray | None, list[str]]":
        """Build semantic embeddings."""
        if not entities:
            return None, []

        try:
            from ..search.semantic import SemanticIndex

            # Get model path
            model_path = Config.get_semantic_model_path(self.model_path)
            if model_path is None or not Path(model_path).exists():
                logger.warning("Semantic model not found, skipping semantic index")
                return None, []

            semantic_index = SemanticIndex(model_path)
            semantic_index.index(entities)

            return semantic_index._embeddings, semantic_index._node_ids

        except ImportError as e:
            logger.warning(f"Cannot build semantic index: {e}")
            return None, []
        except Exception as e:
            logger.warning(f"Semantic indexing failed: {e}")
            return None, []

    def _build_lookup_structures(
        self,
        entities: dict[str, CodeEntity]
    ) -> tuple[dict[str, list[str]], dict[str, list[tuple[int, int, str]]], dict[str, list[str]]]:
        """Build lookup structures from entities."""
        name_to_nodes: dict[str, list[str]] = defaultdict(list)
        file_scopes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        file_to_entities: dict[str, list[str]] = defaultdict(list)

        for node_id, entity in entities.items():
            name_to_nodes[entity.name].append(node_id)
            file_scopes[entity.file_path].append(
                (entity.line_start, entity.line_end, node_id)
            )
            file_to_entities[entity.file_path].append(node_id)

        # Sort file scopes by size
        for file_path in file_scopes:
            file_scopes[file_path].sort(key=lambda x: x[1] - x[0])

        return dict(name_to_nodes), dict(file_scopes), dict(file_to_entities)

    def _save_cache(
        self,
        entities: dict[str, CodeEntity],
        bm25_index: BM25Index,
        graph: "nx.DiGraph",
        pagerank: dict[str, float],
        betweenness: dict[str, float],
        embeddings: "np.ndarray | None",
        node_ids: list[str],
        manifest: dict,
    ) -> None:
        """Save all indices to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._save_pickle(self.ENTITIES_FILE, entities)
        self._save_pickle(self.BM25_FILE, bm25_index)
        self._save_pickle(self.GRAPH_FILE, graph)
        self._save_pickle(self.CENTRALITY_FILE, {
            "pagerank": pagerank,
            "betweenness": betweenness,
        })

        if embeddings is not None:
            self._save_embeddings(embeddings)
            self._save_node_ids(node_ids)

        self._save_manifest(manifest)

        logger.info(f"Cache saved to: {self.cache_dir}")
