"""
Debouncing and batching for file system events.

Handles rapid file changes by coalescing events and batching updates.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of file change."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileChange:
    """Represents a single file change event."""
    path: Path
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)

    @property
    def rel_path(self) -> str:
        """Get path as string for use as key."""
        return str(self.path)


@dataclass
class BatchedChanges:
    """A batch of coalesced file changes ready for processing."""
    created: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Check if there are any changes."""
        return not (self.created or self.modified or self.deleted)

    @property
    def total_count(self) -> int:
        """Total number of changes."""
        return len(self.created) + len(self.modified) + len(self.deleted)

    def __repr__(self) -> str:
        return f"BatchedChanges(+{len(self.created)}, ~{len(self.modified)}, -{len(self.deleted)})"


class ChangeDebouncer:
    """
    Debounces and batches file change events.

    Uses a two-stage approach:
    1. Debouncing: Coalesces rapid changes to the same file (e.g., multiple saves)
    2. Batching: Groups changes within a time window for efficient processing

    This prevents excessive index updates during operations like:
    - IDE auto-save (triggers multiple events per file)
    - git operations (many files change at once)
    - npm install / package manager updates
    """

    def __init__(
        self,
        callback: Callable[[BatchedChanges], None],
        debounce_ms: int = 300,
        batch_window_ms: int = 500,
        max_batch_size: int = 100,
    ):
        """
        Initialize the debouncer.

        Args:
            callback: Function to call with batched changes
            debounce_ms: Time to wait for additional changes to same file
            batch_window_ms: Time window to collect changes before processing
            max_batch_size: Maximum changes before forcing a flush
        """
        self.callback = callback
        self.debounce_ms = debounce_ms
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        self._pending: dict[str, FileChange] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._last_event_time: float = 0
        self._shutdown = False
        self._force_flush = False

    def add_change(self, change: FileChange) -> None:
        """
        Add a file change event.

        The change will be debounced and batched before being sent to the callback.
        """
        if self._shutdown:
            return

        with self._lock:
            path_key = change.rel_path
            existing = self._pending.get(path_key)

            if existing:
                # Coalesce changes: newer change type takes precedence
                # Exception: created + deleted = no change (file came and went)
                if existing.change_type == ChangeType.CREATED and change.change_type == ChangeType.DELETED:
                    del self._pending[path_key]
                    logger.debug(f"Coalesced: {path_key} (created+deleted = removed)")
                else:
                    # Update to latest change type
                    self._pending[path_key] = change
                    logger.debug(f"Coalesced: {path_key} ({existing.change_type} -> {change.change_type})")
            else:
                self._pending[path_key] = change
                logger.debug(f"Queued: {path_key} ({change.change_type})")

            self._last_event_time = time.time()

            # Force flush if batch is too large
            if len(self._pending) >= self.max_batch_size:
                logger.info(f"Max batch size reached ({self.max_batch_size}), forcing flush")
                self._schedule_flush(immediate=True)
            else:
                self._schedule_flush()

    def _schedule_flush(self, immediate: bool = False) -> None:
        """Schedule a flush of pending changes."""
        if self._timer:
            self._timer.cancel()

        if immediate:
            self._force_flush = True

        delay = 0.001 if immediate else self.batch_window_ms / 1000.0
        self._timer = threading.Timer(delay, self._check_and_flush)
        self._timer.daemon = True
        self._timer.start()

    def _check_and_flush(self) -> None:
        """Check if debounce period passed and flush if so."""
        if self._shutdown:
            return

        with self._lock:
            if not self._pending:
                return

            # Skip debounce check if force flush is set
            if self._force_flush:
                self._force_flush = False
                self._flush()
                return

            # Check if we're still receiving events (debounce check)
            elapsed_ms = (time.time() - self._last_event_time) * 1000
            if elapsed_ms < self.debounce_ms:
                # Still receiving events, reschedule
                remaining_ms = self.debounce_ms - elapsed_ms
                self._timer = threading.Timer(remaining_ms / 1000.0, self._check_and_flush)
                self._timer.daemon = True
                self._timer.start()
                return

            # Debounce period passed, flush changes
            self._flush()

    def _flush(self) -> None:
        """Flush pending changes to callback."""
        if not self._pending:
            return

        # Build batched changes
        batch = BatchedChanges()
        for change in self._pending.values():
            if change.change_type == ChangeType.CREATED:
                batch.created.append(change.path)
            elif change.change_type == ChangeType.MODIFIED:
                batch.modified.append(change.path)
            elif change.change_type == ChangeType.DELETED:
                batch.deleted.append(change.path)

        # Clear pending
        self._pending.clear()
        self._timer = None

        logger.info(f"Flushing batch: {batch}")

        # Call callback outside lock
        try:
            self.callback(batch)
        except Exception as e:
            logger.exception(f"Error in change callback: {e}")

    def flush_sync(self) -> BatchedChanges | None:
        """
        Synchronously flush any pending changes.

        Returns the batched changes, or None if no pending changes.
        """
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

            if not self._pending:
                return None

            batch = BatchedChanges()
            for change in self._pending.values():
                if change.change_type == ChangeType.CREATED:
                    batch.created.append(change.path)
                elif change.change_type == ChangeType.MODIFIED:
                    batch.modified.append(change.path)
                elif change.change_type == ChangeType.DELETED:
                    batch.deleted.append(change.path)

            self._pending.clear()
            return batch

    def shutdown(self) -> None:
        """Shutdown the debouncer, flushing any pending changes."""
        self._shutdown = True
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            # Don't flush on shutdown - changes may be incomplete
            self._pending.clear()
