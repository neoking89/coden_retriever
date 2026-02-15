"""Tree-sitter based string literal extraction from code entities.

Extracts string literals from entity source code, capturing value, line number,
and variable name context for sensitive value detection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import (
    SENSITIVE_VALUE_AST_MAX_DEPTH,
    SENSITIVE_VALUE_MAX_STRING_LENGTH,
    SENSITIVE_VALUE_MIN_STRING_LENGTH,
)

if TYPE_CHECKING:
    from ..language.loader import LanguageLoader
    from ..models.entities import CodeEntity

logger = logging.getLogger(__name__)

# Type alias for file tracking info: (language, covered_lines)
FileInfo = tuple[str, set[int]]

# AST node types for string literals per language family
STRING_LITERAL_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"string"}),
    "javascript": frozenset({"string", "template_string"}),
    "typescript": frozenset({"string", "template_string"}),
    "go": frozenset({"interpreted_string_literal", "raw_string_literal"}),
    "rust": frozenset({"string_literal"}),
    "java": frozenset({"string_literal"}),
    "c": frozenset({"string_literal"}),
    "cpp": frozenset({"string_literal"}),
    "c_sharp": frozenset({"string_literal"}),
    "ruby": frozenset({"string"}),
    "kotlin": frozenset({"string_literal"}),
    "php": frozenset({"string", "encapsed_string"}),
    "swift": frozenset({"string_literal"}),
    "scala": frozenset({"string"}),
}

# AST node types for assignment/variable context
ASSIGNMENT_TYPES: frozenset[str] = frozenset({
    "assignment", "augmented_assignment",
    "variable_declaration", "variable_declarator",
    "lexical_declaration", "const_declaration",
    "short_var_declaration",
    "assignment_expression",
    "expression_statement",
})

# Parent node types that indicate a standalone expression (docstring candidate)
_EXPRESSION_WRAPPER_TYPES: frozenset[str] = frozenset({
    "expression_statement",  # Python, Ruby, JS, TS, PHP
})

# Body/block node types where the first string child is a docstring
_BODY_BLOCK_TYPES: frozenset[str] = frozenset({
    "block",                 # Python
    "body",                  # Ruby
    "statement_block",       # JS, TS
    "compound_statement",    # C, C++
    "declaration_list",      # Rust, Go
    "class_body",            # Java, Kotlin, C#
    "program",               # Top-level module docstring
})


@dataclass
class StringLiteral:
    """A string literal found in source code."""
    value: str
    line: int
    variable_name: str | None


def _get_or_create_parser(
    lang_name: str,
    loader: "LanguageLoader",
    cache: dict[str, Any],
) -> Any | None:
    """Reuse the parser cache pattern from tramp_data/param_extractor.py."""
    if lang_name in cache:
        return cache[lang_name]
    language = loader.load(lang_name)
    if language is None:
        cache[lang_name] = None
        return None
    try:
        from tree_sitter import Parser
    except ImportError:
        cache[lang_name] = None
        return None
    try:
        try:
            parser = Parser(language)
        except TypeError:
            parser = Parser()
            parser.set_language(language)  # type: ignore[attr-defined]
        cache[lang_name] = parser
        return parser
    except Exception as e:
        logger.debug(f"Failed to create parser for {lang_name}: {e}")
        cache[lang_name] = None
        return None


def _strip_quotes(raw: str) -> str:
    """Remove surrounding quotes from a string literal."""
    if len(raw) < 2:
        return raw
    for quote in ('"""', "'''", '`', '"', "'"):
        if raw.startswith(quote) and raw.endswith(quote):
            return raw[len(quote):-len(quote)]
    return raw


def _find_variable_name(node: Any) -> str | None:
    """Walk up the AST to find the variable being assigned to."""
    current = node.parent
    for _ in range(SENSITIVE_VALUE_AST_MAX_DEPTH):
        if current is None:
            return None
        if current.type in ASSIGNMENT_TYPES:
            return _extract_lhs_identifier(current)
        current = current.parent
    return None


def _extract_lhs_identifier(assignment_node: Any) -> str | None:
    """Extract the left-hand-side identifier from an assignment node."""
    for child in assignment_node.children:
        if child.type in ("identifier", "name", "field_identifier"):
            text = child.text
            return text.decode("utf-8") if isinstance(text, bytes) else text
        # Stop at the assignment operator
        if child.type in ("=", ":=", "assignment_operator"):
            break
    return None


def _is_docstring(node: Any) -> bool:
    """Check if a string node is in docstring position (AST-level).

    Docstrings are standalone string expressions appearing as the first
    statement in a function/class/module body. This pattern exists in Python
    and Ruby; other languages use comment syntax for documentation.
    """
    parent = node.parent
    if parent is None:
        return False

    # String must be wrapped in an expression_statement (not an assignment)
    if parent.type not in _EXPRESSION_WRAPPER_TYPES:
        return False

    # The expression_statement must be inside a body/block
    grandparent = parent.parent
    if grandparent is None or grandparent.type not in _BODY_BLOCK_TYPES:
        return False

    # Must be the first non-comment child of the block
    for child in grandparent.children:
        if child.type in ("comment", ":", "{"):
            continue
        return child.id == parent.id

    return False


def _collect_string_nodes(
    node: Any,
    string_types: frozenset[str],
    results: list[tuple[Any, str]],
) -> None:
    """Recursively collect string literal nodes from AST.

    Skips docstrings (AST position check) and multi-line strings
    (secrets are always single-line values).
    """
    if node.type in string_types:
        if _is_docstring(node):
            return
        text = node.text
        raw = text.decode("utf-8") if isinstance(text, bytes) else text
        value = _strip_quotes(raw)
        # Multi-line strings are never secrets
        if "\n" in value:
            return
        if SENSITIVE_VALUE_MIN_STRING_LENGTH <= len(value) <= SENSITIVE_VALUE_MAX_STRING_LENGTH:
            results.append((node, value))
        return
    for child in node.children:
        _collect_string_nodes(child, string_types, results)


def _parse_source_and_collect_nodes(
    source: str,
    language: str,
    string_types: frozenset[str],
    loader: "LanguageLoader",
    parser_cache: dict[str, Any],
    context_name: str,
) -> list[tuple[Any, str]]:
    """Parse source code and collect string literal nodes.

    Args:
        source: Source code to parse
        language: Programming language
        string_types: Set of AST node types for strings in this language
        loader: Language loader for tree-sitter
        parser_cache: Cache of parser instances
        context_name: Name for error logging (entity name or file path)

    Returns:
        List of (node, value) tuples for string literals found.
    """
    parser = _get_or_create_parser(language, loader, parser_cache)
    if parser is None:
        return []

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception as e:
        logger.debug(f"Failed to parse {context_name}: {e}")
        return []

    raw_nodes: list[tuple[Any, str]] = []
    _collect_string_nodes(tree.root_node, string_types, raw_nodes)
    return raw_nodes


def extract_strings_from_entity(
    entity: "CodeEntity",
    loader: "LanguageLoader",
    parser_cache: dict[str, Any],
) -> list[StringLiteral]:
    """Extract string literals from a single entity's source code."""
    string_types = STRING_LITERAL_TYPES.get(entity.language)
    if not string_types:
        return []

    raw_nodes = _parse_source_and_collect_nodes(
        entity.source_code,
        entity.language,
        string_types,
        loader,
        parser_cache,
        entity.name,
    )

    results: list[StringLiteral] = []
    for node, value in raw_nodes:
        # Convert tree-sitter's 0-based line to absolute file line
        absolute_line = entity.line_start + node.start_point[0]
        var_name = _find_variable_name(node)
        results.append(StringLiteral(
            value=value,
            line=absolute_line,
            variable_name=var_name,
        ))

    return results


def _extract_module_level_strings(
    file_path: str,
    language: str,
    covered_lines: set[int],
    loader: "LanguageLoader",
    parser_cache: dict[str, Any],
) -> list[StringLiteral]:
    """Extract string literals from module-level code not covered by entities.

    Secrets are commonly defined at module level (e.g. API_KEY = "sk_live_...").
    These lines aren't part of any function/class entity, so we parse the full
    file and return only strings on uncovered lines.
    """
    string_types = STRING_LITERAL_TYPES.get(language)
    if not string_types:
        return []

    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Failed to read file {file_path}: {e}")
        return []

    raw_nodes = _parse_source_and_collect_nodes(
        source, language, string_types, loader, parser_cache, file_path,
    )

    results: list[StringLiteral] = []
    for node, value in raw_nodes:
        # Convert tree-sitter's 0-based line index to 1-based file line
        line = node.start_point[0] + 1
        if line not in covered_lines:
            var_name = _find_variable_name(node)
            results.append(StringLiteral(
                value=value, line=line, variable_name=var_name,
            ))

    return results


def extract_all_strings(
    entities: dict[str, "CodeEntity"],
) -> dict[str, list[StringLiteral]]:
    """Extract string literals from all entities and module-level code.

    Returns mapping of entity node_id -> list of StringLiteral.
    Module-level strings use a synthetic key: "module::<file_path>".
    """
    from ..language.loader import LanguageLoader

    loader = LanguageLoader()
    parser_cache: dict[str, Any] = {}
    result: dict[str, list[StringLiteral]] = {}

    # Phase 1: Extract from entities
    file_info: dict[str, FileInfo] = {}  # path -> (language, covered_lines)
    for node_id, entity in entities.items():
        strings = extract_strings_from_entity(entity, loader, parser_cache)
        if strings:
            result[node_id] = strings

        # Track covered lines per file
        fp = entity.file_path
        if fp not in file_info:
            file_info[fp] = (entity.language, set())
        for line in range(entity.line_start, entity.line_end + 1):
            file_info[fp][1].add(line)

    # Phase 2: Extract module-level strings from uncovered lines
    for fp, (language, covered_lines) in file_info.items():
        module_strings = _extract_module_level_strings(
            fp, language, covered_lines, loader, parser_cache,
        )
        if module_strings:
            result[f"module::{fp}"] = module_strings

    return result
