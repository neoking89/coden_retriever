"""Tree-sitter based line tokenization for syntactic clone detection.

Provides batch tokenization of function code into per-line token sets.
Uses Tree-sitter for language-aware tokenization with fallback to simple split.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tree_sitter import Parser
    from ..language import LanguageLoader

# Minimum line length to consider (skip trivial lines)
MIN_LINE_LENGTH = 5
# Minimum tokens per line to consider
MIN_TOKENS = 2


class LineTokenizer:
    """Tokenizes function code into per-line token sets using Tree-sitter."""

    def __init__(self) -> None:
        self._parsers: dict[str, Parser | None] = {}
        self._loader: "LanguageLoader | None" = None

    def _get_loader(self) -> "LanguageLoader":
        """Get or create the language loader."""
        if self._loader is None:
            from ..language import LanguageLoader
            self._loader = LanguageLoader()
        return self._loader

    def _get_parser(self, lang: str) -> "Parser | None":
        """Get or create a parser for the given language."""
        if lang in self._parsers:
            return self._parsers[lang]

        loader = self._get_loader()
        language = loader.load(lang)
        if not language:
            self._parsers[lang] = None
            return None

        from tree_sitter import Parser
        try:
            parser = Parser(language)
        except TypeError:
            # Older tree-sitter API
            parser = Parser()
            parser.set_language(language)  # type: ignore[attr-defined]

        self._parsers[lang] = parser
        return parser

    def tokenize_function(
        self,
        code: str,
        lang: str,
        min_line_length: int = MIN_LINE_LENGTH,
        min_tokens: int = MIN_TOKENS,
    ) -> list[tuple[str, frozenset[str]]]:
        """Tokenize entire function at once, returning per-line token sets.

        BATCH optimization: One tree-sitter parse per function instead of per line.

        Args:
            code: Function source code
            lang: Programming language identifier
            min_line_length: Minimum line length to include (skip trivial lines)
            min_tokens: Minimum tokens per line to include

        Returns:
            List of (line_text, token_set) tuples for non-trivial lines
        """
        parser = self._get_parser(lang)
        lines = code.strip().split('\n')
        result: list[tuple[str, frozenset[str]]] = []

        if not parser:
            return self._fallback_tokenize(lines, min_line_length, min_tokens)

        try:
            tree = parser.parse(code.encode("utf-8"))
            line_tokens: dict[int, list[str]] = defaultdict(list)

            # Walk AST to collect tokens per line
            stack = [tree.root_node]
            while stack:
                node = stack.pop()
                if node.child_count == 0:
                    text = node.text.decode("utf-8", errors="replace") if node.text else ""
                    if text.strip():
                        line_tokens[node.start_point[0]].append(text)
                else:
                    stack.extend(reversed(node.children))

            # Build result from lines
            for line_idx, raw_line in enumerate(lines):
                stripped = raw_line.strip()
                if len(stripped) < min_line_length:
                    continue

                tokens = line_tokens.get(line_idx, [])
                if not tokens:
                    tokens = stripped.split()

                token_set = frozenset(tokens)
                if len(token_set) >= min_tokens:
                    result.append((stripped, token_set))

            return result

        except Exception as e:
            logger.debug("Tree-sitter parse failed for %s: %s, using fallback", lang, e)
            return self._fallback_tokenize(lines, min_line_length, min_tokens)

    def _fallback_tokenize(
        self,
        lines: list[str],
        min_line_length: int,
        min_tokens: int,
    ) -> list[tuple[str, frozenset[str]]]:
        """Fallback tokenization using simple split."""
        result: list[tuple[str, frozenset[str]]] = []
        for raw_line in lines:
            stripped = raw_line.strip()
            if len(stripped) >= min_line_length:
                tokens = frozenset(stripped.split())
                if len(tokens) >= min_tokens:
                    result.append((stripped, tokens))
        return result


# Module-level tokenizer instance
_tokenizer: LineTokenizer | None = None


def get_tokenizer() -> LineTokenizer:
    """Get the shared tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = LineTokenizer()
    return _tokenizer


def tokenize_function(
    code: str,
    lang: str,
    min_line_length: int = MIN_LINE_LENGTH,
    min_tokens: int = MIN_TOKENS,
) -> list[tuple[str, frozenset[str]]]:
    """Tokenize function code into per-line token sets.

    Convenience function using the shared tokenizer.
    """
    return get_tokenizer().tokenize_function(code, lang, min_line_length, min_tokens)
