"""Tree-sitter based parameter extraction from code entities.

Parses entity source code snippets to extract function parameter names
using AST walking. Handles all languages supported by tree-sitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..language.loader import LanguageLoader
    from ..models.entities import CodeEntity

logger = logging.getLogger(__name__)

# List-like AST nodes that hold multiple parameter children
PARAM_LIST_TYPES: frozenset[str] = frozenset({
    "parameters",
    "formal_parameters",
    "parameter_list",
})

# Individual AST nodes representing a single parameter
PARAM_SINGLE_TYPES: frozenset[str] = frozenset({
    "parameter",
    "typed_parameter",
    "typed_default_parameter",
    "default_parameter",
    "required_parameter",
    "parameter_declaration",
    "formal_parameter",
    "optional_parameter",
    "simple_parameter",
    "rest_parameter",
    "spread_parameter",
    "keyword_parameter",
})

# AST node types that represent identifiers within parameters
PARAM_IDENTIFIER_TYPES: frozenset[str] = frozenset({
    "identifier",
    "field_identifier",
    "simple_identifier",
    "name",
    "word",
})

# Common parameter names that are noise (language boilerplate, not data coupling)
EXCLUDED_PARAM_NAMES: frozenset[str] = frozenset({
    "self", "cls", "args", "kwargs",
    "this", "ctx", "context",
})

# Languages where tree-sitter cannot extract formal parameters
# (bash uses positional $1, $2 only)
LANGUAGES_WITHOUT_PARAMS: frozenset[str] = frozenset({
    "bash",
})


def extract_params_from_entity(
    entity: "CodeEntity",
    loader: "LanguageLoader",
    parser_cache: dict[str, Any],
) -> list[str]:
    """Extract parameter names from a single entity's source code.

    Args:
        entity: Code entity with source_code to parse.
        loader: LanguageLoader for obtaining tree-sitter languages.
        parser_cache: Shared cache mapping lang_name -> parser instance.

    Returns:
        List of parameter name strings (may contain duplicates across
        overloaded params; caller deduplicates as needed).
    """
    if entity.entity_type == "class":
        return []

    if entity.language in LANGUAGES_WITHOUT_PARAMS:
        return []

    parser = _get_or_create_parser(entity.language, loader, parser_cache)
    if parser is None:
        return []

    try:
        source_bytes = entity.source_code.encode("utf-8")
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.debug(f"Failed to parse entity {entity.name}: {e}")
        return []

    params: list[str] = []
    _collect_param_identifiers(tree.root_node, params)
    return [p for p in params if p not in EXCLUDED_PARAM_NAMES]


def _get_or_create_parser(
    lang_name: str,
    loader: "LanguageLoader",
    cache: dict[str, Any],
) -> Any | None:
    """Get a cached parser or create a new one for the language.

    Mirrors RepoParser._get_parser() version-compatible creation.
    """
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


def _collect_param_identifiers(node: Any, params: list[str]) -> None:
    """Recursively walk AST to collect identifier names inside parameter nodes.

    Only descends into parameter-related nodes, never into function bodies.
    """
    if node.type in PARAM_LIST_TYPES:
        _process_param_list(node, params)
        return

    if node.type in PARAM_SINGLE_TYPES:
        _extract_first_identifier(node, params)
        return

    for child in node.children:
        _collect_param_identifiers(child, params)


def _process_param_list(node: Any, params: list[str]) -> None:
    """Process a parameter list node (e.g., `parameters`, `formal_parameters`).

    Iterates children to find individual param nodes or bare identifiers.
    """
    for child in node.children:
        if child.type in PARAM_SINGLE_TYPES:
            _extract_first_identifier(child, params)
        elif child.type in PARAM_IDENTIFIER_TYPES:
            # Bare identifier directly in param list (Python untyped, JS)
            name = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            params.append(name)
        elif child.type in PARAM_LIST_TYPES:
            _process_param_list(child, params)


def _extract_first_identifier(node: Any, params: list[str]) -> None:
    """Extract the first identifier from a single parameter node.

    For typed parameters like `x: int`, we only want `x` (the first identifier).
    """
    for child in node.children:
        if child.type in PARAM_IDENTIFIER_TYPES:
            name = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            params.append(name)
            return
        if child.type in PARAM_SINGLE_TYPES:
            _extract_first_identifier(child, params)
            return


def extract_all_params(
    entities: dict[str, "CodeEntity"],
) -> dict[str, list[str]]:
    """Extract parameters from all entities.

    Creates a single LanguageLoader and parser cache for the batch.

    Args:
        entities: Dict of node_id -> CodeEntity.

    Returns:
        Dict of node_id -> list of parameter names.
    """
    from ..language.loader import LanguageLoader

    loader = LanguageLoader()
    parser_cache: dict[str, Any] = {}
    result: dict[str, list[str]] = {}

    for node_id, entity in entities.items():
        params = extract_params_from_entity(entity, loader, parser_cache)
        if params:
            result[node_id] = params

    return result
