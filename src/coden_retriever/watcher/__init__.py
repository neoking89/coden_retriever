"""
File watching module for automatic cache updates.

Provides real-time file system monitoring with debouncing and
incremental index updates.

Lazy Import Pattern
-------------------
This module uses Python's module-level ``__getattr__`` (PEP 562) for lazy
imports. The ``IncrementalUpdater`` class depends on ``networkx``, which has
significant import overhead (~100ms). By deferring its import until first
access, we avoid this cost for CLI commands that don't need the updater.

Usage is transparent - just import normally:

    from coden_retriever.watcher import IncrementalUpdater

The import only happens when ``IncrementalUpdater`` is actually accessed.
"""

from .watcher import FileWatcher, WatcherCallback
from .debouncer import ChangeDebouncer, FileChange, ChangeType, BatchedChanges


def __getattr__(name: str):
    """Lazy import handler for heavy dependencies."""
    if name == "IncrementalUpdater":
        from .updater import IncrementalUpdater
        return IncrementalUpdater
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FileWatcher",
    "WatcherCallback",
    "ChangeDebouncer",
    "FileChange",
    "ChangeType",
    "BatchedChanges",
    "IncrementalUpdater",
]
