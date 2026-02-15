"""Core sensitive value detection.

Orchestrates: extract strings -> classify -> filter -> return results.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import (
    SENSITIVE_VALUE_DEFAULT_LIMIT,
    SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    SENSITIVE_VALUE_PREVIEW_LENGTH,
    SENSITIVE_VALUE_TIER_HIGH,
    SENSITIVE_VALUE_TIER_MODERATE,
)
from .classifier import classify_batch
from .string_extractor import extract_all_strings

if TYPE_CHECKING:
    from ..models.entities import CodeEntity

logger = logging.getLogger(__name__)


def _truncate_value(value: str, max_len: int = SENSITIVE_VALUE_PREVIEW_LENGTH) -> str:
    """Truncate a value for preview display."""
    if len(value) <= max_len:
        return value
    return value[:max_len - 3] + "..."


def detect_sensitive_values(
    entities: dict[str, "CodeEntity"],
    confidence_threshold: float = SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    exclude_tests: bool = True,
    limit: int | None = SENSITIVE_VALUE_DEFAULT_LIMIT,
    replace_value: str | None = None,
    whitelist: list[str] | None = None,
    root_dir: str | None = None,
) -> dict[str, Any]:
    """Detect sensitive values in string literals across the codebase.

    Args:
        entities: Dict of node_id -> CodeEntity.
        confidence_threshold: Minimum probability to flag as sensitive.
        exclude_tests: Exclude test files from analysis.
        limit: Maximum results to return (None for all).
        replace_value: Replacement string for redaction (None = no replace).
        whitelist: Glob patterns for text files to scan (e.g. ``["*.env"]``).
        root_dir: Project root directory (required when whitelist is set).

    Returns:
        Dict with sensitive_values list and summary statistics.
    """
    filtered_entities = {
        nid: e for nid, e in entities.items()
        if not exclude_tests or not e.is_test
    }

    all_strings = extract_all_strings(filtered_entities)

    # Build flat list for batch classification
    flat_items: list[dict[str, Any]] = []
    flat_texts: list[str] = []

    for node_id, string_literals in all_strings.items():
        # Module-level strings use synthetic "module::<path>" keys to distinguish
        # them from entity-based strings. This allows tracking secrets that appear
        # at file scope outside any function or class definition.
        if node_id.startswith("module::"):
            file_path = node_id[len("module::"):]
            entity_name = Path(file_path).stem
            entity_type = "module"
        else:
            entity = filtered_entities[node_id]
            file_path = entity.file_path
            entity_name = entity.name
            entity_type = entity.entity_type

        for sl in string_literals:
            flat_texts.append(sl.value)
            flat_items.append({
                "value": sl.value,
                "file": file_path,
                "line": sl.line,
                "name": entity_name,
                "variable_name": sl.variable_name,
                "entity_type": entity_type,
            })

    if whitelist and root_dir:
        _collect_whitelist_values(
            root_dir, whitelist, flat_items, flat_texts,
        )

    if not flat_texts:
        return _build_empty_result(len(filtered_entities))

    # Batch classify
    probabilities = classify_batch(flat_texts)

    # Filter by threshold and build results
    results: list[dict[str, Any]] = []
    for item, confidence in zip(flat_items, probabilities):
        if confidence >= confidence_threshold:
            results.append({
                "value_preview": _truncate_value(item["value"]),
                "original_value": item["value"],
                "file": item["file"],
                "line": item["line"],
                "name": item["name"],
                "variable_name": item["variable_name"],
                "confidence": round(confidence, 4),
                "replace_value": replace_value,
            })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    if limit is not None and limit > 0:
        results = results[:limit]

    return {
        "sensitive_values": results,
        "summary": _build_summary(
            results, len(filtered_entities), len(flat_texts),
        ),
    }


def _collect_whitelist_values(
    root_dir: str,
    whitelist: list[str],
    flat_items: list[dict[str, Any]],
    flat_texts: list[str],
) -> None:
    """Scan whitelisted text files and append their values to the flat lists."""
    from .file_scanner import scan_whitelist_files
    from .text_extractor import extract_text_file_values

    matched_files = scan_whitelist_files(Path(root_dir), whitelist)
    for fp in matched_files:
        literals = extract_text_file_values(fp)
        file_str = str(fp)
        file_name = fp.name
        for sl in literals:
            flat_texts.append(sl.value)
            flat_items.append({
                "value": sl.value,
                "file": file_str,
                "line": sl.line,
                "name": file_name,
                "variable_name": sl.variable_name,
                "entity_type": "text_file",
            })


def _build_empty_result(entity_count: int) -> dict[str, Any]:
    """Build result dict when no strings are found."""
    return {
        "sensitive_values": [],
        "summary": {
            "total_entities_analyzed": entity_count,
            "total_strings_scanned": 0,
            "sensitive_values_found": 0,
            "distribution": {"high": 0, "moderate": 0, "low": 0},
        },
    }


def _build_summary(
    results: list[dict[str, Any]],
    entity_count: int,
    string_count: int,
) -> dict[str, Any]:
    """Build summary statistics for sensitive value detection."""
    distribution = {"high": 0, "moderate": 0, "low": 0}
    for r in results:
        conf = r["confidence"]
        if conf >= SENSITIVE_VALUE_TIER_HIGH:
            distribution["high"] += 1
        elif conf >= SENSITIVE_VALUE_TIER_MODERATE:
            distribution["moderate"] += 1
        else:
            distribution["low"] += 1

    return {
        "total_entities_analyzed": entity_count,
        "total_strings_scanned": string_count,
        "sensitive_values_found": len(results),
        "distribution": distribution,
    }
