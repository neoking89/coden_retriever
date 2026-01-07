"""
LRU cache for project indices with file watching support.

Provides thread-safe caching of project indices and search engines,
with automatic eviction of least-recently-used entries.
"""
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from ..cache import CachedIndices
from ..search import SearchEngine
from ..watcher import FileWatcher
from .protocol import MAX_PROJECTS_DEFAULT


logger = logging.getLogger(__name__)


@dataclass
class CachedProject:
    """A cached project with its indices, engine, and optional watcher."""
    indices: CachedIndices
    engine: SearchEngine
    last_access: float
    watcher: FileWatcher | None = None
    enable_semantic: bool = False
    model_path: str | None = None


class ProjectCache:
    """LRU cache for project indices with file watching support.

    Thread-safe cache that maintains a maximum number of projects in memory,
    evicting least-recently-used entries when capacity is reached.

    Attributes:
        max_projects: Maximum number of projects to keep in cache.
    """

    def __init__(
        self,
        max_projects: int = MAX_PROJECTS_DEFAULT,
        on_evict: Callable[[str, CachedProject], None] | None = None,
    ):
        """Initialize the project cache.

        Args:
            max_projects: Maximum number of projects to keep in memory.
            on_evict: Optional callback when a project is evicted.
        """
        self.max_projects = max_projects
        self._cache: OrderedDict[str, CachedProject] = OrderedDict()
        self._lock = threading.Lock()
        self._on_evict = on_evict

    def get(self, source_dir: str) -> CachedProject | None:
        """Get cached project, updating LRU order.

        Args:
            source_dir: Absolute path to the project directory.

        Returns:
            CachedProject if found, None otherwise.
        """
        with self._lock:
            if source_dir in self._cache:
                self._cache.move_to_end(source_dir)
                project = self._cache[source_dir]
                project.last_access = time.time()
                return project
            return None

    def put(
        self,
        source_dir: str,
        indices: CachedIndices,
        engine: SearchEngine,
        watcher: FileWatcher | None = None,
        enable_semantic: bool = False,
        model_path: str | None = None,
    ) -> None:
        """Add or update project in cache.

        If the cache is at capacity, the least-recently-used project
        is evicted before adding the new one.

        Args:
            source_dir: Absolute path to the project directory.
            indices: Cached indices for the project.
            engine: Search engine instance.
            watcher: Optional file watcher for live updates.
            enable_semantic: Whether semantic search is enabled.
            model_path: Path to semantic model if used.
        """
        with self._lock:
            # If updating existing, stop old watcher
            if source_dir in self._cache:
                old_project = self._cache[source_dir]
                if old_project.watcher:
                    old_project.watcher.stop()
                self._cache.move_to_end(source_dir)
            else:
                # Evict oldest if at capacity
                while len(self._cache) >= self.max_projects:
                    evicted_key, evicted_project = self._cache.popitem(last=False)
                    logger.info(f"Evicted project from cache: {evicted_key}")
                    if evicted_project.watcher:
                        evicted_project.watcher.stop()
                    if self._on_evict:
                        self._on_evict(evicted_key, evicted_project)

            self._cache[source_dir] = CachedProject(
                indices=indices,
                engine=engine,
                last_access=time.time(),
                watcher=watcher,
                enable_semantic=enable_semantic,
                model_path=model_path,
            )

    def update_indices(
        self,
        source_dir: str,
        indices: CachedIndices,
        engine: SearchEngine,
    ) -> None:
        """Update indices for an existing cached project (preserves watcher).

        Args:
            source_dir: Absolute path to the project directory.
            indices: New cached indices.
            engine: New search engine instance.
        """
        with self._lock:
            if source_dir in self._cache:
                project = self._cache[source_dir]
                project.indices = indices
                project.engine = engine
                project.last_access = time.time()

    def invalidate(self, source_dir: str | None = None) -> None:
        """Invalidate cache for a project or all projects.

        Args:
            source_dir: Project path to invalidate, or None for all.
        """
        with self._lock:
            if source_dir is None:
                # Stop all watchers
                for project in self._cache.values():
                    if project.watcher:
                        project.watcher.stop()
                self._cache.clear()
                logger.info("Invalidated all cached projects")
            elif source_dir in self._cache:
                project = self._cache[source_dir]
                if project.watcher:
                    project.watcher.stop()
                del self._cache[source_dir]
                logger.info(f"Invalidated cached project: {source_dir}")

    def status(self) -> dict:
        """Get cache status.

        Returns:
            Dictionary with cache statistics and project list.
        """
        with self._lock:
            projects = [
                {
                    "source_dir": source_dir,
                    "entity_count": len(project.indices.entities),
                    "has_semantic": project.indices.has_semantic,
                    "last_access": project.last_access,
                    "watching": project.watcher is not None and project.watcher.is_running(),
                }
                for source_dir, project in self._cache.items()
            ]
            return {
                "cached_projects": len(self._cache),
                "max_projects": self.max_projects,
                "projects": projects,
            }

    def stop_all_watchers(self) -> None:
        """Stop all file watchers (for shutdown)."""
        with self._lock:
            for project in self._cache.values():
                if project.watcher:
                    project.watcher.stop()
