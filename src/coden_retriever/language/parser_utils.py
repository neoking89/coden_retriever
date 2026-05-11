"""Shared tree-sitter parser creation and caching.

Provides a single implementation of version-compatible parser creation,
eliminating duplication across string_extractor, param_extractor, and
constants_extractor.
"""
from __future__ import annotations

import logging
from typing import Any

from .loader import LanguageLoader

logger = logging.getLogger(__name__)

# Some tree-sitter grammars cannot parse a bare body fragment because their
# root rule expects a file-level prologue. PHP is the canonical case: without
# `<?php` the parser stays in HTML/text mode and emits zero literals or params.
# Use a space (no newline) so existing node line numbers remain valid.
_LANGUAGE_REPARSE_PROLOGUES: dict[str, str] = {
    "php": "<?php ",
}


def prepare_entity_source_for_reparse(source: str, language: str) -> str:
    """Prepend the grammar prologue needed to parse a body fragment in-mode.

    No-op for languages that don't need one, and for sources that already
    carry the prologue (e.g. full-file scans).
    """
    prologue = _LANGUAGE_REPARSE_PROLOGUES.get(language)
    if prologue and not source.lstrip().startswith("<?"):
        return prologue + source
    return source


def get_or_create_parser(
    lang_name: str,
    loader: LanguageLoader,
    cache: dict[str, Any],
) -> Any | None:
    """Return a cached tree-sitter parser, creating one if needed.

    Handles version differences in the tree-sitter Parser API:
    - 0.21+: Parser(language)
    - 0.20:  Parser() + set_language(language)
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
        logger.debug("Failed to create parser for %s: %s", lang_name, e)
        cache[lang_name] = None
        return None
