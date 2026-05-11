"""Detect repeated literal constants (magic numbers/strings) across entities.

Bridges the low-level AST extraction from ``constants_extractor`` with the
entity-based pipeline used by MCP tools, daemon handlers, and ``coden flag``.
"""
from __future__ import annotations

import logging
from typing import Any

from ..constants import (
    MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT,
    MAGIC_CONSTANT_TIER_HIGH,
    MAGIC_CONSTANT_TIER_MODERATE,
    MAGIC_CONSTANT_TRIVIAL_VALUES,
)
from ..constants_extractor import extract_constants_from_source
from ..language.literal_types import NUMERIC_LITERAL_TYPES
from ..language.loader import LanguageLoader

logger = logging.getLogger(__name__)

# Pre-computed union of all numeric node types across languages (used by _node_type_category)
_ALL_NUMERIC_TYPES: frozenset[str] = frozenset().union(*NUMERIC_LITERAL_TYPES.values())

# re-export for convenience (used by formatter)
__all__ = ["detect_magic_constants"]


def detect_magic_constants(
    entities: dict[str, Any],
    min_occurrences: int = MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    min_files: int = MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    exclude_tests: bool = True,
    limit: int | None = MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT,
) -> dict[str, Any]:
    """Detect repeated literal constants across the codebase.

    Returns dict with ``magic_constants`` list and ``summary`` statistics,
    matching the shape of ``detect_tramp_data``.
    """
    grouped = _extract_from_entities(entities, exclude_tests)
    filtered = _filter_trivial_values(grouped)
    results = _apply_thresholds(filtered, min_occurrences, min_files)

    # Sort by occurrence count descending, then by file spread
    results.sort(key=lambda r: (r["count"], r["files"]), reverse=True)

    if limit is not None and limit > 0:
        results = results[:limit]

    return {
        "magic_constants": results,
        "summary": _build_summary(results, len(entities)),
    }


def _extract_from_entities(
    entities: dict[str, Any],
    exclude_tests: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Extract constants from all entities, grouped by value.

    Returns ``{constant_value: [{file, line, entity_name, node_type}]}``.
    """
    loader = LanguageLoader()
    parser_cache: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {}

    for _node_id, entity in entities.items():
        if exclude_tests and getattr(entity, "is_test", False):
            continue
        if getattr(entity, "entity_type", "") == "class":
            continue

        language = getattr(entity, "language", None)
        source = getattr(entity, "source_code", None)
        if not language or not source:
            continue

        file_path = getattr(entity, "file_path", "")
        line_start = getattr(entity, "line_start", 1)

        constants = extract_constants_from_source(
            source, language, file_path, loader, parser_cache,
        )
        for value, ntype, rel_line in constants:
            occ = {
                "file": file_path,
                "line": line_start + rel_line - 1,
                "entity_name": getattr(entity, "name", ""),
                "node_type": ntype,
            }
            groups.setdefault(value, []).append(occ)

    return groups


def _filter_trivial_values(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Remove values that are idiomatic and never considered magic."""
    return {v: occs for v, occs in groups.items() if v not in MAGIC_CONSTANT_TRIVIAL_VALUES}


def _node_type_category(occurrences: list[dict[str, Any]]) -> str:
    """Classify the constant as 'numeric' or 'string' based on node types."""
    for occ in occurrences:
        if occ.get("node_type") in _ALL_NUMERIC_TYPES:
            return "numeric"
    return "string"


def _apply_thresholds(
    groups: dict[str, list[dict[str, Any]]],
    min_occurrences: int,
    min_files: int,
) -> list[dict[str, Any]]:
    """Filter groups by min occurrences and min files, build result dicts."""
    results: list[dict[str, Any]] = []
    for value, occs in groups.items():
        if len(occs) < min_occurrences:
            continue
        distinct_files = len({o["file"] for o in occs})
        if distinct_files < min_files:
            continue
        results.append({
            "value": value,
            "count": len(occs),
            "files": distinct_files,
            "node_type_category": _node_type_category(occs),
            "occurrences": occs,
        })
    return results


def _build_summary(
    results: list[dict[str, Any]],
    total_entities: int,
) -> dict[str, Any]:
    """Build summary statistics for magic constant detection."""
    high = sum(1 for r in results if r["count"] >= MAGIC_CONSTANT_TIER_HIGH)
    moderate = sum(
        1 for r in results
        if MAGIC_CONSTANT_TIER_MODERATE <= r["count"] < MAGIC_CONSTANT_TIER_HIGH
    )
    low = sum(1 for r in results if r["count"] < MAGIC_CONSTANT_TIER_MODERATE)

    return {
        "total_entities_analyzed": total_entities,
        "magic_constants_found": len(results),
        "distribution": {"high": high, "moderate": moderate, "low": low},
    }
