"""Split `! cmd @@ query` input into (command, optional query).

For bash-family shells the split uses tree-sitter-bash via the project's
LanguageLoader, so quoting / heredocs / command substitution are handled
correctly. For PowerShell and cmd, a small quote-aware state machine is
used instead.
"""

from typing import Optional

from ..constants import SHELL_QUERY_SEPARATOR
from ..language.loader import LanguageLoader
from .shell_exec import ShellKind

# tree_sitter is an optional heavy dependency. Import at module level so that
# CLAUDE.md's "all imports at top of file" rule is satisfied; guard against
# absence at runtime via None sentinel on _TreeSitterParser.
try:
    from tree_sitter import Parser as _TreeSitterParser
except ImportError:
    _TreeSitterParser = None  # type: ignore[assignment,misc]

_loader: Optional[LanguageLoader] = None

# Sentinel that distinguishes "parser loaded but @@ not found in AST" (a real
# negative) from "parser unavailable" (fall through to stringy heuristic).
_PARSER_UNAVAILABLE = object()


def _get_bash_parser() -> Optional["_TreeSitterParser"]:  # type: ignore[name-defined]
    """Lazily build a Parser for bash. Returns None when unavailable."""
    global _loader
    if _TreeSitterParser is None:
        return None
    if _loader is None:
        _loader = LanguageLoader()
    language = _loader.load("bash")
    if language is None:
        return None
    try:
        return _TreeSitterParser(language)
    except TypeError:
        parser = _TreeSitterParser()
        parser.set_language(language)
        return parser


def split_command_and_query(
    line: str, shell: ShellKind,
) -> tuple[str, Optional[str]]:
    """Split ``line`` at the last bare ``@@`` into (cmd, query).

    Returns ``(line.strip(), None)`` when no separator is present.
    A present separator with no trailing text returns ``(cmd, "")`` — the
    caller still pipes the output to the LLM, just without a question.
    """
    if SHELL_QUERY_SEPARATOR not in line:
        return line.strip(), None

    split_idx = _find_separator(line, shell)
    if split_idx is None:
        return line.strip(), None

    cmd = line[:split_idx].rstrip()
    query = line[split_idx + len(SHELL_QUERY_SEPARATOR):].strip()
    return cmd, query


def _find_separator(line: str, shell: ShellKind) -> Optional[int]:
    """Return the char index of the last bare ``@@`` separator, or None."""
    if shell is ShellKind.BASH_FAMILY:
        result = _find_separator_bash(line)
        if result is _PARSER_UNAVAILABLE:
            # tree-sitter could not load: fall back to heuristic.
            return _find_separator_stringy(line)
        # result is a char index (possibly None = no bare @@ found by AST).
        return result  # type: ignore[return-value]
    return _find_separator_stringy(line)


def _find_separator_bash(line: str) -> "int | None | object":
    """Find the last top-level bare ``@@`` using tree-sitter-bash.

    Returns:
        An int char offset when a bare separator is found.
        None when the parser ran successfully but found no bare separator.
        _PARSER_UNAVAILABLE sentinel when tree-sitter could not be loaded,
        so the caller knows to apply the stringy fallback.
    """
    parser = _get_bash_parser()
    if parser is None:
        return _PARSER_UNAVAILABLE
    try:
        # Parse as UTF-8 bytes so node byte offsets are consistent with the
        # encoded form. All subsequent lookups work in byte space.
        encoded = line.encode("utf-8")
        tree = parser.parse(encoded)
    except Exception:  # noqa: BLE001 — tree-sitter can raise broad errors
        return _PARSER_UNAVAILABLE

    sep_bytes = SHELL_QUERY_SEPARATOR.encode("utf-8")
    candidate_byte_offsets: list[int] = []

    def _walk(node: object) -> None:
        # tree_sitter.Node attributes accessed via duck typing (no stub).
        if getattr(node, "type", None) == "word" and getattr(node, "text", None) == sep_bytes:
            start: int = node.start_byte  # type: ignore[attr-defined]
            end: int = node.end_byte  # type: ignore[attr-defined]
            if _is_surrounded_by_whitespace_bytes(encoded, start, end):
                candidate_byte_offsets.append(start)
            return  # word nodes have no meaningful children for this check
        for child in getattr(node, "children", ()):
            _walk(child)

    _walk(tree.root_node)
    if not candidate_byte_offsets:
        return None

    # Convert the last matching byte offset to a char offset so callers can
    # slice the original str directly. This is correct for any Unicode content.
    best_byte = candidate_byte_offsets[-1]
    return len(encoded[:best_byte].decode("utf-8"))


def _find_separator_stringy(line: str) -> Optional[int]:
    """Quote-aware fallback: scan for ``@@`` tokens outside string literals."""
    sep = SHELL_QUERY_SEPARATOR
    in_single = False
    in_double = False
    last_idx: Optional[int] = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and not in_single and i + 1 < n:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if (
            not in_single and not in_double
            and line.startswith(sep, i)
            and _is_surrounded_by_whitespace(line, i, i + len(sep))
        ):
            last_idx = i
            i += len(sep)
            continue
        i += 1
    return last_idx


def _is_surrounded_by_whitespace(line: str, start: int, end: int) -> bool:
    """Check the separator has a whitespace boundary (or is at line edge).

    Operates on char (str) indices.
    """
    before_ok = start == 0 or line[start - 1].isspace()
    after_ok = end == len(line) or line[end].isspace()
    return before_ok and after_ok


def _is_surrounded_by_whitespace_bytes(encoded: bytes, start: int, end: int) -> bool:
    """Check the separator has a whitespace boundary (or is at document edge).

    Operates on byte indices for use with tree-sitter node offsets.
    """
    before_ok = start == 0 or chr(encoded[start - 1]).isspace()
    after_ok = end == len(encoded) or chr(encoded[end]).isspace()
    return before_ok and after_ok
