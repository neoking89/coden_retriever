"""Smart error recovery for debug tools.

Pre-launch validation and diagnostic helpers:
- Script syntax validation (ast.parse) — Python-specific
- Script-not-found suggestions (difflib)

Adapter installation is detected via `DebugAdapter.detect_installed()` on
each adapter subclass — not a helper here.
"""
import ast
import difflib
import logging
from pathlib import Path
from typing import Any

from .debug_errors import debug_error

logger = logging.getLogger(__name__)

# How many alternative file suggestions to return
MAX_SUGGESTIONS = 5

# difflib.get_close_matches cutoff ratio. WHY 0.4: empirically we want "ap.py"
# to surface "app.py" (ratio ~0.67) and "test_ap.py" (~0.5), but NOT wildly
# different names like "zzzzzzzzz.py" (~0.0). 0.4 sits below the realistic
# matches and above the noise floor; 0.6 (library default) misses "ap.py"
# against "test_app.py" which real typos commonly produce.
_CLOSE_MATCH_CUTOFF = 0.4


class _ServerStartTracker:
    """One-shot latch: flips to True the first time debug_server starts.

    WHY a class instead of a module-level bool: a plain global requires a
    `global` statement inside `mark_debug_server_started`, which both
    CLAUDE.md (no global mutation from inside functions beyond what's
    necessary) and the code-review findings flagged as an anti-pattern.
    Encapsulating the flag as an instance attribute keeps the "once True,
    stays True for the process lifetime" semantic while removing the
    `global` declaration.
    """

    def __init__(self) -> None:
        self._started: bool = False

    def mark_started(self) -> None:
        """Flip the latch. Idempotent — subsequent calls are no-ops."""
        self._started = True

    def ever_started(self) -> bool:
        """Has `mark_started` been called at any point in this process?"""
        return self._started


# Process-wide singleton. debugpy's in-process listener cannot be cleanly torn
# down, so a later DAP launch may inherit corrupted state. Stored here (not in
# debug_trace) to keep debug_simplified and debug_trace free of a cyclic import.
_server_start_tracker = _ServerStartTracker()


def mark_debug_server_started() -> None:
    """Record that `debug_server(action='start')` has succeeded once in this process."""
    _server_start_tracker.mark_started()


def debug_server_ever_started() -> bool:
    """True if `debug_server` has ever started a helper in this process."""
    return _server_start_tracker.ever_started()


async def validate_script_syntax(program: str) -> dict[str, Any] | None:
    """Parse-check a Python script before launch. Returns error dict if invalid, None if OK."""
    path = Path(program)
    if not path.exists():
        return None  # File-not-found is handled separately

    if path.suffix != ".py":
        return None  # Only validate Python files

    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        return None
    except SyntaxError as e:
        details = f"Line {e.lineno}: {e.msg}" if e.lineno else str(e.msg)
        return debug_error(
            "syntax_error",
            f"Syntax error in {path.name}: {details}",
            "Fix the syntax error before debugging",
        )


async def suggest_script_alternatives(program: str) -> list[str]:
    """When a script isn't found, suggest similar filenames in the same directory."""
    path = Path(program)
    parent = path.parent if path.parent.exists() else Path.cwd()

    candidates = [p.name for p in parent.glob("*.py")]
    if not candidates:
        return []

    matches = difflib.get_close_matches(
        path.name, candidates, n=MAX_SUGGESTIONS, cutoff=_CLOSE_MATCH_CUTOFF,
    )
    return [str(parent / m) for m in matches]
