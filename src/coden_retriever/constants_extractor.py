"""Tree-sitter based constant literal extraction from source files.

Walks the full AST to find every numeric and string literal (including default
parameter values), filters out language keywords structurally via the AST,
and returns them grouped by value with counts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .language.definitions import LANGUAGE_MAP, language_for_path
from .language.literal_types import NUMERIC_LITERAL_TYPES, STRING_LITERAL_TYPES
from .language.loader import LanguageLoader
from .language.parser_utils import (
    get_or_create_parser,
    prepare_entity_source_for_reparse,
)

logger = logging.getLogger(__name__)


@dataclass
class ConstantOccurrence:
    """A single occurrence of a constant in source code."""
    file_path: str
    line: int
    node_type: str


@dataclass
class ConstantGroup:
    """All occurrences of a constant sharing the same value."""
    value: str
    count: int = 0
    occurrences: list[ConstantOccurrence] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST walking
# ---------------------------------------------------------------------------

def _node_text(node: Any) -> str:
    """Decode node text to str."""
    text = node.text
    return text.decode("utf-8") if isinstance(text, bytes) else text


def _is_keyword_node(node: Any) -> bool:
    """Detect language keywords structurally via the AST.

    In tree-sitter grammars, keyword literals (true, false, None, null, nil,
    undefined, ...) are leaf nodes whose node type matches their text content
    (case-insensitive).  Real user-authored literals never match: an ``integer``
    node has text ``"42"``, a ``string`` node has text ``'"hello"'``.

    This check is grammar-driven — no hardcoded keyword list required.
    """
    if node.child_count > 0:
        return False
    text = _node_text(node)
    return node.type.lower() == text.lower()


def _collect_constants(
    node: Any,
    lang: str,
    numeric_types: frozenset[str],
    string_types: frozenset[str],
    results: list[tuple[str, str, int]],
) -> None:
    """Recursively collect literal constant nodes from the AST.

    Skips language keywords (detected structurally via ``_is_keyword_node``).
    Appends (value, node_type, 0-based line) tuples to *results*.
    """
    ntype = node.type

    # Numeric literal — tree-sitter node types like "integer"/"float" never
    # have node.type == node.text, so _is_keyword_node is always False here.
    if ntype in numeric_types:
        results.append((_node_text(node), ntype, node.start_point[0]))
        return

    # String literal — same reasoning; capture the full quoted text as-is.
    if ntype in string_types:
        results.append((_node_text(node), ntype, node.start_point[0]))
        return

    # Any other leaf whose type matches its text is a keyword — skip subtree
    if _is_keyword_node(node):
        return

    for child in node.children:
        _collect_constants(child, lang, numeric_types, string_types, results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_constants_from_source(
    source: str,
    language: str,
    file_path: str,
    loader: LanguageLoader,
    parser_cache: dict[str, Any],
) -> list[tuple[str, str, int]]:
    """Parse *source* and return all constant literals.

    Returns list of (value, node_type, 1-based line) tuples.
    """
    numeric_types = NUMERIC_LITERAL_TYPES.get(language, frozenset())
    string_types = STRING_LITERAL_TYPES.get(language, frozenset())

    parser = get_or_create_parser(language, loader, parser_cache)
    if parser is None:
        return []

    try:
        prepared = prepare_entity_source_for_reparse(source, language)
        tree = parser.parse(prepared.encode("utf-8"))
    except Exception as e:
        logger.debug("Failed to parse %s: %s", file_path, e)
        return []

    raw: list[tuple[str, str, int]] = []
    _collect_constants(
        tree.root_node, language, numeric_types, string_types, raw,
    )

    # Convert 0-based lines to 1-based
    return [(val, ntype, line + 1) for val, ntype, line in raw]


def extract_constants_from_path(
    target: Path,
) -> dict[str, ConstantGroup]:
    """Scan a file or directory tree and return constants grouped by value.

    Args:
        target: A single source file or a directory to scan recursively.

    Returns:
        Mapping of constant value -> ConstantGroup (with count + occurrences).
    """
    loader = LanguageLoader()
    parser_cache: dict[str, Any] = {}
    counter: dict[str, list[ConstantOccurrence]] = {}

    files = _resolve_files(target)

    for fpath in files:
        language = language_for_path(fpath)
        if language is None:
            continue

        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Cannot read %s: %s", fpath, e)
            continue

        constants = extract_constants_from_source(
            source, language, str(fpath), loader, parser_cache,
        )
        for value, ntype, line in constants:
            occ = ConstantOccurrence(
                file_path=str(fpath), line=line, node_type=ntype,
            )
            counter.setdefault(value, []).append(occ)

    groups: dict[str, ConstantGroup] = {}
    for value, occs in counter.items():
        groups[value] = ConstantGroup(
            value=value, count=len(occs), occurrences=occs,
        )
    return groups


def _resolve_files(target: Path) -> list[Path]:
    """Return a sorted list of source files under *target*."""
    if target.is_file():
        return [target]
    if target.is_dir():
        all_exts = set(LANGUAGE_MAP.keys())
        return sorted(
            f for f in target.rglob("*")
            if f.is_file() and f.suffix.lower() in all_exts
        )
    return []
