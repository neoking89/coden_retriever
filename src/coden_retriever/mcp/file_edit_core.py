"""Core classes for file editing operations.

This module provides SOLID-compliant abstractions for:
- FileCache: Thread-safe LRU cache for tracking read files
- PathPermissions: Path boundary checking and permission management
- UndoManager: One-step undo buffer management

These classes follow the Single Responsibility Principle and can be tested independently.
"""
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache size for FileCache
DEFAULT_MAX_CACHE_SIZE = 1000


class FileCache:
    """Thread-safe LRU cache for tracking files that have been read.

    This class encapsulates the read-before-write verification system.
    Files must be read (and registered in the cache) before they can be
    modified, preventing accidental overwrites.

    Follows Single Responsibility Principle: only handles caching logic.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_CACHE_SIZE):
        """Initialize the file cache.

        Args:
            max_size: Maximum number of entries before LRU eviction.
        """
        self._max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_size(self) -> int:
        """Return the maximum cache size."""
        return self._max_size

    def mark_as_read(self, file_path: str, content: str) -> None:
        """Mark a file as having been read with its content.

        Uses LRU eviction when cache exceeds max_size.

        Args:
            file_path: Absolute path to the file.
            content: The content that was read.
        """
        abs_path = self._resolve_path(file_path)
        with self._lock:
            # Move to end if exists (LRU update)
            if abs_path in self._cache:
                self._cache.move_to_end(abs_path)
            self._cache[abs_path] = content

            # Evict oldest entries if over limit
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def was_read(self, file_path: str) -> bool:
        """Check if a file was previously read in this session.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file was previously read.
        """
        abs_path = self._resolve_path(file_path)
        with self._lock:
            return abs_path in self._cache

    def get_content(self, file_path: str) -> str | None:
        """Get cached content for a file if it was previously read.

        Args:
            file_path: Path to the file.

        Returns:
            The cached content, or None if not in cache.
        """
        abs_path = self._resolve_path(file_path)
        with self._lock:
            content = self._cache.get(abs_path)
            if content is not None:
                # Move to end on access (LRU)
                self._cache.move_to_end(abs_path)
            return content

    def remove(self, file_path: str) -> None:
        """Remove a file from the cache.

        Args:
            file_path: Path to the file to remove.
        """
        abs_path = self._resolve_path(file_path)
        with self._lock:
            self._cache.pop(abs_path, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve a path to absolute form."""
        return str(Path(path).resolve())


class PathPermissions:
    """Manages allowed paths for write operations.

    Provides path boundary protection by restricting file operations
    to specific directories. When allowed_paths is empty, all paths
    are permitted (default for trusted environments).

    Follows Single Responsibility Principle: only handles permission logic.
    """

    def __init__(self, allowed_paths: list[str] | None = None):
        """Initialize with optional allowed paths.

        Args:
            allowed_paths: List of directory paths that are allowed.
                          Empty or None means no restrictions.
        """
        self._allowed_paths: list[str] = []
        self._allowed_paths_cache: list[Path] = []
        self._lock = threading.Lock()

        if allowed_paths:
            self.set_allowed_paths(allowed_paths)

    @classmethod
    def from_env(cls) -> "PathPermissions":
        """Create a PathPermissions instance from environment variable.

        Reads CODEN_RETRIEVER_ALLOWED_PATHS using os.pathsep as separator.

        Returns:
            PathPermissions instance configured from environment.
        """
        env_paths = os.environ.get("CODEN_RETRIEVER_ALLOWED_PATHS", "")
        if env_paths:
            # Use os.pathsep for platform-appropriate path list separator
            # Windows: ; (semicolon) - avoids conflict with C:\path drive letters
            # Unix: : (colon)
            paths = [p.strip() for p in env_paths.split(os.pathsep) if p.strip()]
            return cls(allowed_paths=paths)
        return cls()

    def set_allowed_paths(self, paths: list[str]) -> None:
        """Configure allowed paths for write operations.

        Args:
            paths: List of directory paths. Only files under these directories
                   can be written. Empty list removes restrictions.
        """
        with self._lock:
            self._allowed_paths = [str(Path(p).resolve()) for p in paths]
            # Rebuild cache with resolved Path objects for faster checks
            self._allowed_paths_cache = [Path(p).resolve() for p in self._allowed_paths]

    def get_allowed_paths(self) -> list[str]:
        """Get the currently configured allowed paths.

        Returns:
            Copy of the allowed paths list.
        """
        with self._lock:
            return self._allowed_paths.copy()

    def is_path_allowed(self, abs_path: str) -> bool:
        """Check if a path is within the allowed paths for write operations.

        Returns True if:
        - allowed_paths is empty (no restrictions)
        - The path is under one of the allowed directories

        Args:
            abs_path: Absolute path to check.

        Returns:
            True if the path is allowed for write operations.
        """
        with self._lock:
            if not self._allowed_paths:
                return True

            # Use cached resolved paths for faster comparison
            path = Path(abs_path).resolve()
            for allowed_path in self._allowed_paths_cache:
                try:
                    path.relative_to(allowed_path)
                    return True
                except ValueError:
                    continue
            return False

    def validate_path_for_write(self, file_path: str) -> tuple[str, dict[str, Any] | None]:
        """Validate a path for write operations and resolve it.

        Args:
            file_path: The path to validate.

        Returns:
            Tuple of (resolved_absolute_path, error_dict_or_none).
            If error_dict is not None, the path is invalid.
        """
        # Block obvious path traversal attempts
        if '..' in file_path:
            return file_path, {"error": "Path traversal (..) not allowed"}

        abs_path = str(Path(file_path).resolve())

        # Validate it's an absolute path
        if not os.path.isabs(file_path) and not os.path.isabs(abs_path):
            return abs_path, {"error": f"Path must be absolute: {file_path}"}

        # Check path is within allowed boundaries (if configured)
        if not self.is_path_allowed(abs_path):
            return abs_path, {
                "error": f"Path not in allowed directories: {abs_path}",
                "allowed_paths": self.get_allowed_paths(),
                "hint": "Configure allowed paths with set_allowed_paths() or remove restrictions."
            }

        return abs_path, None


@dataclass
class UndoEntry:
    """Represents a single undo entry for a file."""
    previous_content: str | None
    was_new_file: bool


class UndoManager:
    """Manages one-step undo buffer for file operations.

    Stores the previous state of files for undo functionality.
    Only the most recent change per file is stored.

    Follows Single Responsibility Principle: only handles undo logic.
    """

    def __init__(self):
        """Initialize the undo manager."""
        self._buffer: dict[str, UndoEntry] = {}
        self._lock = threading.Lock()

    def save_for_undo(self, abs_path: str, was_new_file: bool = False) -> None:
        """Save the current state of a file to the undo buffer.

        Args:
            abs_path: Absolute path to the file.
            was_new_file: If True, the file didn't exist before this operation.
        """
        abs_path = str(Path(abs_path).resolve())

        if was_new_file:
            # File was newly created - undo means delete it
            with self._lock:
                self._buffer[abs_path] = UndoEntry(previous_content=None, was_new_file=True)
            return

        # Read file content OUTSIDE the lock to avoid blocking other threads
        # during potentially slow I/O operations
        content = None
        if os.path.exists(abs_path):
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                # Log warning but don't fail - undo just won't work for this file
                logger.warning(f"Could not save undo state for {abs_path}: {e}")
                return

        # Only acquire lock for the dictionary update
        if content is not None:
            with self._lock:
                self._buffer[abs_path] = UndoEntry(previous_content=content, was_new_file=False)

    def has_history(self, file_path: str) -> bool:
        """Check if a file has undo history available.

        Args:
            file_path: Path to the file.

        Returns:
            True if undo history exists for this file.
        """
        abs_path = str(Path(file_path).resolve())
        with self._lock:
            return abs_path in self._buffer

    def pop_history(self, file_path: str) -> UndoEntry | None:
        """Get and remove the undo entry for a file.

        Args:
            file_path: Path to the file.

        Returns:
            The UndoEntry if it exists, None otherwise.
        """
        abs_path = str(Path(file_path).resolve())
        with self._lock:
            return self._buffer.pop(abs_path, None)

    def clear(self) -> None:
        """Clear all undo history."""
        with self._lock:
            self._buffer.clear()


# Singleton instances used by the facade functions in file_edit.py.
_default_file_cache: FileCache | None = None
_default_path_permissions: PathPermissions | None = None
_default_undo_manager: UndoManager | None = None
_init_lock = threading.Lock()


def get_file_cache() -> FileCache:
    """Get the default FileCache singleton instance."""
    global _default_file_cache
    if _default_file_cache is None:
        with _init_lock:
            if _default_file_cache is None:
                _default_file_cache = FileCache()
    return _default_file_cache


def get_path_permissions() -> PathPermissions:
    """Get the default PathPermissions singleton instance."""
    global _default_path_permissions
    if _default_path_permissions is None:
        with _init_lock:
            if _default_path_permissions is None:
                _default_path_permissions = PathPermissions.from_env()
    return _default_path_permissions


def get_undo_manager() -> UndoManager:
    """Get the default UndoManager singleton instance."""
    global _default_undo_manager
    if _default_undo_manager is None:
        with _init_lock:
            if _default_undo_manager is None:
                _default_undo_manager = UndoManager()
    return _default_undo_manager


# WHY KEPT: Not called in production, but every test in test_file_edit_core.py
# depends on this via autouse fixture to isolate singleton state between tests.
def reset_singletons() -> None:
    """Reset all singleton instances. Used by test fixtures to isolate state between test cases."""
    global _default_file_cache, _default_path_permissions, _default_undo_manager
    with _init_lock:
        _default_file_cache = None
        _default_path_permissions = None
        _default_undo_manager = None
