"""Tree-sitter based parameter extraction from code entities.

Parses entity source code snippets to extract function parameter names
using AST walking. Handles all languages supported by tree-sitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..language.parser_utils import (
    get_or_create_parser,
    prepare_entity_source_for_reparse,
)

if TYPE_CHECKING:
    from ..language.loader import LanguageLoader
    from ..models.entities import CodeEntity

logger = logging.getLogger(__name__)

# List-like AST nodes that hold multiple parameter children
PARAM_LIST_TYPES: frozenset[str] = frozenset({
    "parameters",
    "formal_parameters",
    "parameter_list",
    "function_value_parameters",  # Kotlin
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
    # PHP-specific param node types
    "variadic_parameter",
    "property_promotion_parameter",
})

# AST node types that represent identifiers within parameters
PARAM_IDENTIFIER_TYPES: frozenset[str] = frozenset({
    "identifier",
    "field_identifier",
    "simple_identifier",
    "name",
    "word",
})

# Wrapper nodes that hold the parameter identifier as a child (PHP wraps
# every $var in `variable_name > $ + name`); we descend into them rather
# than picking up sibling type names like `union_type > named_type > name`.
PARAM_IDENTIFIER_WRAPPERS: frozenset[str] = frozenset({
    "variable_name",
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

    parser = get_or_create_parser(entity.language, loader, parser_cache)
    if parser is None:
        return []

    try:
        prepared = prepare_entity_source_for_reparse(entity.source_code, entity.language)
        tree = parser.parse(prepared.encode("utf-8"))
    except Exception as e:
        logger.debug(f"Failed to parse entity {entity.name}: {e}")
        return []

    params: list[str] = []
    _collect_param_identifiers(tree.root_node, params)
    return [p for p in params if p not in EXCLUDED_PARAM_NAMES]


def _collect_param_identifiers(node: Any, params: list[str]) -> bool:
    """Recursively walk AST to collect identifier names from the OUTER signature.

    Returns True once the first parameter list (or singleton) is processed so
    the caller stops walking — any later param-list nodes belong to nested
    closures/lambdas/arrow-funcs in the body and would pollute the result.
    Empty outer signatures still halt the walk (no params is the right answer).
    """
    if node.type in PARAM_LIST_TYPES:
        _process_param_list(node, params)
        return True

    if node.type in PARAM_SINGLE_TYPES:
        _extract_first_identifier(node, params)
        return True

    for child in node.children:
        if _collect_param_identifiers(child, params):
            return True
    return False


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
        if child.type in PARAM_IDENTIFIER_WRAPPERS:
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
