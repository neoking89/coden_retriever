"""
Directory tree formatter for search results.

Generates a visual tree representation of search results, showing
only the directories and files that contain matching entities.

Features:
- Rich-based colors for score visualization
- Clickable hyperlinks to files and entities

Also provides a compact shallow tree generator for system prompt injection.
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config
from .terminal_style import get_terminal_style

if TYPE_CHECKING:
    from ..models.entities import CodeEntity
    from ..models import SearchResult


# Priority files to always show at root level (language-agnostic)
ROOT_PRIORITY_FILES = {
    # Python
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile",
    # JavaScript/TypeScript
    "package.json", "tsconfig.json", "deno.json",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    # Java/Kotlin/Scala
    "pom.xml", "build.gradle", "build.gradle.kts", "build.sbt",
    # Ruby
    "Gemfile",
    # PHP
    "composer.json",
    # .NET / C#
    "*.csproj", "*.fsproj", "*.sln",  # Note: these need glob matching
    # Elixir
    "mix.exs",
    # Swift
    "Package.swift",
    # Zig
    "build.zig",
    # Haskell
    "stack.yaml", "cabal.project",
    # Documentation
    "README.md", "README.rst", "README.txt", "README",
    "CHANGELOG.md", "CHANGELOG", "HISTORY.md",
    "LICENSE", "LICENSE.md", "LICENSE.txt",
    # Build/Config
    "Makefile", "CMakeLists.txt", "meson.build",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example",
}

# Files that indicate a nested/separate project
PROJECT_MARKERS = {
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml",
    "build.gradle", "Gemfile", "composer.json", "mix.exs", "Package.swift",
}

# Max number of non-priority root files to show (prevents token explosion)
MAX_ROOT_FILES_NON_PRIORITY = 15

# Directories to prioritize (source code) - lower number = higher priority
DIR_PRIORITY = {
    "src": 0, "lib": 1, "app": 2, "core": 3, "pkg": 4,
    "tests": 5, "test": 6, "spec": 7,
    "scripts": 10, "tools": 11, "bin": 12,
    "docs": 20, "doc": 21, "documentation": 22,
    "examples": 25, "example": 26, "samples": 27,
    "assets": 30, "static": 31, "images": 32, "img": 33,
    "archive": 40, "old": 41, "backup": 42,
}


# Tree drawing characters - ASCII-safe for shell compatibility
_BRANCH = "+-- "
_LAST = "`-- "
_PIPE = "|   "
_SPACE = "    "

# Sort priority for directories not in DIR_PRIORITY: between source code and docs
_DEFAULT_DIR_PRIORITY = 15


def _is_nested_project(dir_path: str) -> bool:
    """Check if directory is a nested project (has its own project file)."""
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.name in PROJECT_MARKERS and entry.is_file(follow_symlinks=False):
                    return True
    except OSError:
        pass
    return False


def _dir_sort_key(item: tuple[str, str]) -> tuple[int, str]:
    name = item[0].lower()
    return (DIR_PRIORITY.get(name, _DEFAULT_DIR_PRIORITY), name)


def _is_priority_file(name: str) -> bool:
    """Check if filename is a priority file (supports `*.ext` glob patterns)."""
    if name in ROOT_PRIORITY_FILES:
        return True
    for pattern in ROOT_PRIORITY_FILES:
        if pattern.startswith("*.") and name.endswith(pattern[1:]):
            return True
    return False


def _scan_dir(dir_path: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Fast directory scan via os.scandir(). Returns (dirs, files) sorted for display."""
    skip_dirs = Config.SKIP_DIRS
    skip_files = Config.SKIP_FILES
    dirs: list[tuple[str, str]] = []
    files: list[tuple[str, str]] = []
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                name = entry.name
                if name[0] == "." or name in skip_dirs or name in skip_files:
                    continue
                if name.endswith(".egg-info"):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    dirs.append((name, entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.append((name, entry.path))
    except OSError:
        pass
    dirs.sort(key=_dir_sort_key)
    files.sort(key=lambda x: (not _is_priority_file(x[0]), x[0].lower()))
    return dirs, files


def _select_collapsed_entries(
    dirs: list[tuple[str, str]],
    files: list[tuple[str, str]],
    collapse_threshold: int,
) -> tuple[list[tuple[str, str, bool]], int, int]:
    """When a dir has too many items, keep top-N dirs + priority files. Returns (entries, hidden_dirs, hidden_files)."""
    priority_only = [(n, p) for n, p in files if n in ROOT_PRIORITY_FILES]
    shown_dirs = dirs[:collapse_threshold]
    hidden_dirs = len(dirs) - len(shown_dirs)
    hidden_files = len(files) - len(priority_only)
    entries = (
        [(n, p, True) for n, p in shown_dirs]
        + [(n, p, False) for n, p in priority_only]
    )
    return entries, hidden_dirs, hidden_files


def _format_hidden_notice(hidden_dirs: int, hidden_files: int) -> str | None:
    """Format `... (N more dirs, M more files)` or None if nothing hidden."""
    if hidden_dirs <= 0 and hidden_files <= 0:
        return None
    parts = []
    if hidden_dirs > 0:
        parts.append(f"{hidden_dirs} more dirs")
    if hidden_files > 0:
        parts.append(f"{hidden_files} more files")
    return f"... ({', '.join(parts)})"


def generate_shallow_tree(
    root: Path,
    max_depth: int = 3,
    max_items_per_dir: int = 15,
    collapse_threshold: int = 8,
    max_lines: int = 60,
) -> str:
    """
    Generate a compact, shallow directory tree for system prompt injection.

    Optimized for minimal tokens while maximizing structural insight.
    Uses os.scandir() for fast directory traversal with cached stat info.

    Smart truncation features:
    - ALWAYS shows all root-level files (critical for LLM context)
    - Detects nested projects (dirs with pyproject.toml etc.) and collapses them
    - Prioritizes source directories (src/, lib/) over docs/examples
    - Limits total output lines with intelligent truncation (subdirs only)

    Args:
        root: Project root path
        max_depth: Maximum directory depth to traverse (default: 3)
        max_items_per_dir: Max items to show per directory before collapsing (default: 15)
        collapse_threshold: If more than this many items, show summary (default: 8)
        max_lines: Maximum total lines in output (default: 60)

    Returns:
        Compact tree string suitable for LLM context
    """
    root_path = str(Path(root).resolve())
    lines: list[str] = [f"{os.path.basename(root_path)}/"]
    truncated = False

    def _walk(dir_path: str, prefix: str, depth: int, lines_budget: int) -> int:
        nonlocal truncated
        if lines_budget <= 0:
            truncated = True
            return 0
        if depth > max_depth:
            return 0

        dirs, files = _scan_dir(dir_path)
        if len(dirs) + len(files) > max_items_per_dir:
            entries, hidden_dirs, hidden_files = _select_collapsed_entries(
                dirs, files, collapse_threshold
            )
            hidden_notice = _format_hidden_notice(hidden_dirs, hidden_files)
        else:
            entries = (
                [(n, p, True) for n, p in dirs]
                + [(n, p, False) for n, p in files]
            )
            hidden_notice = None

        last_idx = len(entries) - 1 if hidden_notice is None else -1
        lines_used = 0
        for i, (name, path, is_dir) in enumerate(entries):
            if lines_used >= lines_budget:
                truncated = True
                return lines_used
            is_last = i == last_idx
            conn = _LAST if is_last else _BRANCH

            if is_dir and _is_nested_project(path):
                lines.append(f"{prefix}{conn}{name}/ (nested project)")
                lines_used += 1
            elif is_dir:
                lines.append(f"{prefix}{conn}{name}/")
                lines_used += 1
                child_prefix = prefix + (_SPACE if is_last else _PIPE)
                lines_used += _walk(path, child_prefix, depth + 1, lines_budget - lines_used)
            else:
                lines.append(f"{prefix}{conn}{name}")
                lines_used += 1

        if hidden_notice and lines_used < lines_budget:
            lines.append(f"{prefix}{_LAST}{hidden_notice}")
            lines_used += 1
        return lines_used

    # Root level handling: priority files always shown, non-priority files capped.
    root_dirs, root_files = _scan_dir(root_path)
    priority_root_files = [(n, p) for n, p in root_files if _is_priority_file(n)]
    other_root_files = [(n, p) for n, p in root_files if not _is_priority_file(n)]

    hidden_root_files = max(0, len(other_root_files) - MAX_ROOT_FILES_NON_PRIORITY)
    if hidden_root_files > 0:
        other_root_files = other_root_files[:MAX_ROOT_FILES_NON_PRIORITY]
    shown_root_files = priority_root_files + other_root_files

    root_files_count = len(shown_root_files) + (1 if hidden_root_files > 0 else 0)
    dir_budget = max_lines - 1 - root_files_count - 1

    for i, (name, path) in enumerate(root_dirs):
        is_last_dir = (i == len(root_dirs) - 1) and not shown_root_files
        conn = _LAST if is_last_dir else _BRANCH

        remaining_budget = dir_budget - (len(lines) - 1)
        if remaining_budget <= 0:
            truncated = True
            break

        if _is_nested_project(path):
            lines.append(f"{conn}{name}/ (nested project)")
        else:
            lines.append(f"{conn}{name}/")
            child_prefix = _SPACE if is_last_dir else _PIPE
            _walk(path, child_prefix, 2, remaining_budget - 1)

    for i, (name, _path) in enumerate(shown_root_files):
        is_last = (i == len(shown_root_files) - 1) and hidden_root_files == 0
        conn = _LAST if is_last else _BRANCH
        lines.append(f"{conn}{name}")

    if hidden_root_files > 0:
        lines.append(f"{_LAST}... ({hidden_root_files} more files)")

    if truncated:
        lines.append("... (truncated)")

    return "\n".join(lines)


class DirectoryTreeFormatter:
    """Formats search results as a directory tree with Rich colors and clickable links."""

    def __init__(
        self,
        root: Path,
        entities: dict[str, "CodeEntity"],
        file_to_entities: dict[str, list[str]],
    ):
        self.root = root
        self._entities = entities
        self._file_to_entities = file_to_entities

    def format_tree(self, results: list["SearchResult"]) -> str:
        """Generate a colored, clickable directory tree."""
        if not results:
            return ""

        # Get Rich styling
        style = get_terminal_style()

        # Build score map for coloring
        node_id_to_score: dict[str, float] = {r.entity.node_id: r.score for r in results}
        max_score = max(r.score for r in results)

        # Identify relevant entities and paths
        relevant_node_ids = {r.entity.node_id for r in results}

        paths_to_show = set()
        for r in results:
            path = Path(r.entity.file_path).resolve()
            current = path
            while True:
                paths_to_show.add(str(current))
                if current == self.root or current in self.root.parents:
                    break
                current = current.parent

        output = ["Repository Structure (Top Matches)", "=" * 32]

        def _walk(current_path: Path, prefix: str = ""):
            entries = []
            try:
                for item in current_path.iterdir():
                    if item.name.startswith(".") and item.name != ".":
                        continue
                    if item.name in Config.SKIP_DIRS or item.name in Config.SKIP_FILES:
                        continue
                    if str(item.resolve()) not in paths_to_show:
                        continue
                    if item.is_dir() or item.is_file():
                        entries.append(item)
            except OSError:
                return

            entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            for i, entry in enumerate(entries):
                is_last = (i == len(entries) - 1)
                connector = "`-- " if is_last else "+-- "
                child_prefix = "    " if is_last else "|   "

                if entry.is_dir():
                    output.append(f"{prefix}{connector}📂 {entry.name}/")
                    _walk(entry, prefix + child_prefix)
                else:
                    abs_path = str(entry.resolve())
                    file_entities = []
                    if abs_path in self._file_to_entities:
                        node_ids = self._file_to_entities[abs_path]
                        file_entities = [
                            (self._entities[nid], node_id_to_score.get(nid, 0))
                            for nid in node_ids
                            if nid in relevant_node_ids
                        ]
                        file_entities.sort(key=lambda x: x[0].line_start)

                    # Get file score from its entities
                    file_score = max((s for _, s in file_entities), default=0)

                    # Colored, clickable file using Rich
                    file_display = style.format_tree_file(
                        name=entry.name,
                        file_path=abs_path,
                        score=file_score,
                        max_score=max_score,
                    )
                    output.append(f"{prefix}{connector}{file_display}")

                    if file_entities:
                        file_prefix = prefix + child_prefix
                        for j, (entity, score) in enumerate(file_entities):
                            ent_is_last = (j == len(file_entities) - 1)
                            ent_connector = "`-- " if ent_is_last else "+-- "
                            # Colored, clickable entity using Rich
                            ent_display = style.format_tree_entity(
                                name=entity.name,
                                entity_type=entity.entity_type,
                                file_path=entity.file_path,
                                line=entity.line_start,
                                score=score,
                                max_score=max_score,
                            )
                            output.append(f"{file_prefix}{ent_connector}{ent_display}")

        output.append(f"📦 {self.root.name}/")
        _walk(self.root)
        return "\n".join(output)


def format_directory_tree(
    results: list["SearchResult"],
    root: Path,
    entities: dict[str, "CodeEntity"],
    file_to_entities: dict[str, list[str]],
) -> str:
    """
    Convenience function to format results as a directory tree.

    Args:
        results: List of search results to display
        root: Repository root path
        entities: Mapping of node_id to CodeEntity
        file_to_entities: Mapping of file_path to list of node_ids

    Returns:
        Formatted tree string
    """
    formatter = DirectoryTreeFormatter(root, entities, file_to_entities)
    return formatter.format_tree(results)
