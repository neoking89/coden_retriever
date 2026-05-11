"""Expand @file references in user prompts.

Allows users to inline file contents into their messages using @path syntax,
similar to Claude Code's file reference feature. Supports:
- @path/to/file.py — relative to working directory
- @"path with spaces/file.py" — quoted paths for spaces

Files are read and inserted as formatted code blocks before sending to the LLM.
"""

import re
from pathlib import Path

# WHY 50,000: prevents accidentally inlining huge files (e.g., minified JS,
# large data files) that would blow up the context window
MAX_FILE_INCLUDE_CHARS = 50_000

# WHY 20: show enough of truncated files to be useful while keeping context small
TRUNCATED_PREVIEW_LINES = 20

# Matches @path or @"quoted path" preceded by whitespace or start-of-string.
# Group 1 captures either the quoted content or the unquoted path.
_FILE_REF_PATTERN = re.compile(
    r'(?:^|(?<=\s))@(?:"([^"]+)"|(\S+))'
)


def _resolve_path(ref: str, root_dir: str) -> Path | None:
    """Resolve a file reference to an absolute path.

    Args:
        ref: The path string from the @reference.
        root_dir: Working directory to resolve relative paths against.

    Returns:
        Resolved Path if valid file, None otherwise.
    """
    path = Path(ref)
    if not path.is_absolute():
        path = Path(root_dir) / path
    path = path.resolve()

    if not path.is_file():
        return None
    return path


def _read_file_content(path: Path) -> tuple[str, bool]:
    """Read file content with size guard.

    Args:
        path: Absolute path to the file.

    Returns:
        Tuple of (content, was_truncated).
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", False

    if len(content) > MAX_FILE_INCLUDE_CHARS:
        lines = content.splitlines()
        head = "\n".join(lines[:TRUNCATED_PREVIEW_LINES])
        tail = "\n".join(lines[-TRUNCATED_PREVIEW_LINES:])
        truncated = (
            f"{head}\n\n... ({len(lines)} lines total, truncated) ...\n\n{tail}"
        )
        return truncated, True
    return content, False


def _guess_language(path: Path) -> str:
    """Guess language identifier from file extension for code fence."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".rs": "rust", ".go": "go",
        ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c",
        ".cs": "csharp", ".rb": "ruby", ".sh": "bash", ".bash": "bash",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".md": "markdown", ".html": "html",
        ".css": "css", ".sql": "sql", ".xml": "xml",
    }
    return ext_map.get(path.suffix.lower(), "")


def expand_file_references(text: str, root_dir: str) -> str:
    """Replace @file references with formatted file contents.

    Args:
        text: User input text potentially containing @file references.
        root_dir: Working directory for resolving relative paths.

    Returns:
        Text with @references replaced by file content blocks.
    """
    def _replace_ref(match: re.Match) -> str:
        # Group 1 = quoted path, group 2 = unquoted path
        ref = match.group(1) or match.group(2)
        path = _resolve_path(ref, root_dir)

        if path is None:
            return match.group(0)  # Leave unresolved references as-is

        content, was_truncated = _read_file_content(path)
        if not content:
            return f"[Could not read: {ref}]"

        lang = _guess_language(path)
        relative = _make_relative(path, root_dir)
        truncation_note = " (truncated)" if was_truncated else ""

        return f"\n[Content of {relative}{truncation_note}]\n```{lang}\n{content}\n```\n"

    return _FILE_REF_PATTERN.sub(_replace_ref, text)


def find_file_references(text: str) -> list[str]:
    """Extract all @file reference paths from text.

    Args:
        text: User input text.

    Returns:
        List of path strings found in @references.
    """
    return [
        (m.group(1) or m.group(2))
        for m in _FILE_REF_PATTERN.finditer(text)
    ]


def _make_relative(path: Path, root_dir: str) -> str:
    """Make a path relative to root_dir for display."""
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)
