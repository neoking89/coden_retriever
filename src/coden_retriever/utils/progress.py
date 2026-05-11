"""Progress bar utilities for CLI embedding operations."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


# Advance-by-N callback used across all progress-aware encoding paths
ProgressCallback = Callable[[int], None]


@dataclass
class ProgressHandle:
    """Wraps advance + set_total so callers can defer the total until known."""

    advance: ProgressCallback
    set_total: Callable[[int], None]

    def __call__(self, n: int) -> None:
        """Allow using the handle directly as a ProgressCallback."""
        self.advance(n)


# Shared no-op handle reused by all non-interactive / zero-total early returns
_NOOP_HANDLE = ProgressHandle(advance=lambda _n: None, set_total=lambda _n: None)


def _is_interactive() -> bool:
    """True when stderr is a real terminal (not piped or redirected)."""
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


@contextmanager
def encoding_progress(
    description: str,
    total: int | None = None,
) -> Generator[ProgressHandle, None, None]:
    """Rich progress bar for embedding / file-processing operations.

    Yields a ``ProgressHandle`` with ``advance(n)`` and ``set_total(n)``
    callbacks. Writes to stderr so it never contaminates stdout output
    piped to other tools. Falls back to no-ops when stderr is not a TTY
    (MCP, daemon, CI).

    When *total* is None, the bar starts indeterminate; call
    ``handle.set_total(n)`` once the total is known.
    """
    if not _is_interactive():
        yield _NOOP_HANDLE
        return

    # total=0 means nothing to do; skip the bar entirely
    if total is not None and total <= 0:
        yield _NOOP_HANDLE
        return

    # Write to stderr so progress never contaminates piped stdout
    console = Console(stderr=True)

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=total)

        def advance(n: int) -> None:
            progress.advance(task_id, n)

        def set_total(n: int) -> None:
            progress.update(task_id, total=n)

        yield ProgressHandle(advance=advance, set_total=set_total)
