"""Text file value extraction for sensitive value detection.

Extracts string values from non-source text files (.env, .json, .yaml, etc.)
that cannot be parsed by tree-sitter. Values are returned as StringLiteral
instances for compatibility with the existing classifier pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..constants import (
    SENSITIVE_VALUE_MAX_STRING_LENGTH,
    SENSITIVE_VALUE_MIN_STRING_LENGTH,
)
from .string_extractor import StringLiteral

logger = logging.getLogger(__name__)

# Characters that start a comment line in key-value config files
_KV_COMMENT_CHARS = frozenset({"#", ";", "!"})

# Extensions that use key=value or key:value format (includes YAML and TOML
# which are structurally key-value and don't need dedicated parsers)
_KEY_VALUE_EXTENSIONS = frozenset({
    ".env", ".properties", ".ini", ".conf", ".cfg",
    ".yaml", ".yml", ".toml",
})

# JSON needs structured parsing because values can be deeply nested
_JSON_EXTENSIONS = frozenset({".json"})


def _is_valid_length(value: str) -> bool:
    """Check if a string value falls within the analysis length bounds."""
    return SENSITIVE_VALUE_MIN_STRING_LENGTH <= len(value) <= SENSITIVE_VALUE_MAX_STRING_LENGTH


def _strip_surrounding_quotes(value: str) -> str:
    """Remove surrounding single or double quotes from a value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _extract_key_value_lines(text: str) -> list[tuple[int, str]]:
    """Extract values from key=value or key:value formatted text.

    Returns list of (line_number, value) tuples. Skips comments and blank lines.
    """
    results: list[tuple[int, str]] = []
    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped[0] in _KV_COMMENT_CHARS:
            continue
        # Try = first (most common), then : (properties/yaml files)
        for sep in ("=", ":"):
            idx = stripped.find(sep)
            if idx > 0:
                value = stripped[idx + 1:].strip()
                value = _strip_surrounding_quotes(value)
                if _is_valid_length(value) and "\n" not in value:
                    results.append((line_num, value))
                break
    return results


def _build_string_literals_from_pairs(pairs: list[tuple[int, str]]) -> list[StringLiteral]:
    """Convert (line_number, value) pairs to StringLiteral instances."""
    return [
        StringLiteral(value=val, line=line, variable_name=None)
        for line, val in pairs
    ]


def _walk_json_values(
    obj: Any,
    results: list[str],
) -> None:
    """Recursively collect all string values from a parsed JSON structure."""
    if isinstance(obj, str):
        if _is_valid_length(obj) and "\n" not in obj:
            results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_json_values(v, results)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json_values(v, results)


def _extract_json_values(text: str) -> list[str]:
    """Extract all string values from JSON content."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug("Failed to parse JSON: %s", e)
        return []
    results: list[str] = []
    _walk_json_values(data, results)
    return results


def extract_text_file_values(file_path: Path) -> list[StringLiteral]:
    """Extract string values from a text file based on its extension.

    Dispatches to format-specific extractors for .env, .json, .yaml, .toml,
    and other key-value config files.

    Args:
        file_path: Path to the text file to extract values from.

    Returns:
        List of StringLiteral instances with extracted values.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug("Failed to read %s: %s", file_path, e)
        return []

    ext = file_path.suffix.lower()

    if ext in _KEY_VALUE_EXTENSIONS:
        pairs = _extract_key_value_lines(text)
        return _build_string_literals_from_pairs(pairs)

    if ext in _JSON_EXTENSIONS:
        values = _extract_json_values(text)
        return [
            StringLiteral(value=val, line=1, variable_name=None)
            for val in values
        ]

    # Fallback: treat unknown extensions as key-value
    pairs = _extract_key_value_lines(text)
    return _build_string_literals_from_pairs(pairs)
