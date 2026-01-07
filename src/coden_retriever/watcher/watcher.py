"""
File watcher using watchdog library.

Monitors source files for changes and triggers cache updates.
"""

import logging
import threading
from pathlib import Path
from typing import Protocol

from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from ..config import Config
from ..language import LANGUAGE_MAP
from .debouncer import BatchedChanges, ChangeDebouncer, ChangeType, FileChange

logger = logging.getLogger(__name__)


class WatcherCallback(Protocol):
    """Protocol for watcher callbacks."""

    def __call__(self, changes: BatchedChanges) -> None:
        """Called when file changes are detected and debounced."""
        ...


class CodeFileHandler(FileSystemEventHandler):
    """
    Handles file system events for code files.

    Filters events to only include relevant source files and
    forwards them to the debouncer for coalescing.
    """

    def __init__(
        self,
        source_dir: Path,
        debouncer: ChangeDebouncer,
    ):
        super().__init__()
        self.source_dir = source_dir.resolve()
        self.debouncer = debouncer

        # Build set of valid extensions from LANGUAGE_MAP
        self._valid_extensions = set(LANGUAGE_MAP.keys())
        self._skip_dirs = Config.SKIP_DIRS
        self._skip_files = Config.SKIP_FILES

    def _should_process(self, path: str) -> bool:
        """
        Check if a file path should be processed.

        Filters out:
        - Directories
        - Files in skip directories
        - Files with non-code extensions
        - Skip files (lock files, etc.)
        - Files larger than 1MB
        """
        file_path = Path(path)

        # Skip directories
        if file_path.is_dir():
            return False

        # Check extension
        if file_path.suffix.lower() not in self._valid_extensions:
            return False

        # Check skip files
        if file_path.name in self._skip_files:
            return False

        # Check skip directories in path
        for part in file_path.parts:
            if part in self._skip_dirs or part.startswith('.'):
                return False

        # Check file size (skip very large files)
        try:
            if file_path.exists() and file_path.stat().st_size > 1_000_000:
                return False
        except OSError:
            pass

        return True

    def _handle_event(self, event, change_type: ChangeType) -> None:
        """Handle a file system event."""
        if event.is_directory:
            return

        path = event.src_path
        if not self._should_process(path):
            return

        file_path = Path(path)
        change = FileChange(path=file_path, change_type=change_type)
        self.debouncer.add_change(change)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation."""
        self._handle_event(event, ChangeType.CREATED)

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification."""
        self._handle_event(event, ChangeType.MODIFIED)

    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion."""
        self._handle_event(event, ChangeType.DELETED)

    def on_moved(self, event: FileMovedEvent) -> None:
        """
        Handle file move/rename.

        Treated as delete of source + create of destination.
        """
        if event.is_directory:
            return

        # Handle source (deleted)
        if self._should_process(event.src_path):
            src_path = Path(event.src_path)
            self.debouncer.add_change(
                FileChange(path=src_path, change_type=ChangeType.DELETED)
            )

        # Handle destination (created)
        if self._should_process(event.dest_path):
            dest_path = Path(event.dest_path)
            self.debouncer.add_change(
                FileChange(path=dest_path, change_type=ChangeType.CREATED)
            )


class FileWatcher:
    """
    Watches a source directory for file changes.

    Uses watchdog for cross-platform file system monitoring with
    automatic backend selection (inotify, FSEvents, ReadDirectoryChangesW).

    Integrates with ChangeDebouncer to coalesce rapid changes and
    batch updates for efficient processing.
    """

    def __init__(
        self,
        source_dir: Path,
        callback: WatcherCallback,
        debounce_ms: int = 300,
        batch_window_ms: int = 500,
        use_polling: bool = False,
        polling_interval: float = 1.0,
    ):
        """
        Initialize the file watcher.

        Args:
            source_dir: Directory to watch recursively
            callback: Function called with batched changes
            debounce_ms: Milliseconds to wait for additional changes to same file
            batch_window_ms: Milliseconds to batch changes before processing
            use_polling: Force polling observer (for network filesystems)
            polling_interval: Polling interval in seconds (if use_polling=True)
        """
        self.source_dir = Path(source_dir).resolve()
        self.callback = callback
        self.debounce_ms = debounce_ms
        self.batch_window_ms = batch_window_ms
        self.use_polling = use_polling
        self.polling_interval = polling_interval

        self._observer: Observer | None = None
        self._debouncer: ChangeDebouncer | None = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """
        Start watching for file changes.

        The watcher runs in a background thread and calls the callback
        when changes are detected (after debouncing/batching).
        """
        with self._lock:
            if self._running:
                logger.warning("FileWatcher already running")
                return

            logger.info(f"Starting file watcher for: {self.source_dir}")

            # Create debouncer
            self._debouncer = ChangeDebouncer(
                callback=self.callback,
                debounce_ms=self.debounce_ms,
                batch_window_ms=self.batch_window_ms,
            )

            # Create event handler
            handler = CodeFileHandler(
                source_dir=self.source_dir,
                debouncer=self._debouncer,
            )

            # Create observer
            if self.use_polling:
                logger.info("Using polling observer (for network filesystem)")
                self._observer = PollingObserver(timeout=self.polling_interval)
            else:
                self._observer = Observer()

            # Schedule recursive watch
            self._observer.schedule(
                handler,
                str(self.source_dir),
                recursive=True,
            )

            # Start observer thread
            self._observer.start()
            self._running = True

            logger.info("File watcher started")

    def stop(self) -> None:
        """
        Stop watching for file changes.

        Stops the observer thread and cleans up resources.
        """
        with self._lock:
            if not self._running:
                return

            logger.info("Stopping file watcher")

            if self._debouncer:
                self._debouncer.shutdown()
                self._debouncer = None

            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None

            self._running = False
            logger.info("File watcher stopped")

    def is_running(self) -> bool:
        """Check if the watcher is running."""
        with self._lock:
            return self._running

    def flush(self) -> BatchedChanges | None:
        """
        Flush any pending changes synchronously.

        Useful for testing or when you need immediate processing.
        """
        with self._lock:
            if self._debouncer:
                return self._debouncer.flush_sync()
            return None

    def __enter__(self) -> "FileWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
