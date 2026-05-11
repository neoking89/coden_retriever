"""Syntactic (line-by-line) clone detection using Jaccard similarity.

Detects code clones by comparing tokenized lines between functions.
Uses sparse matrices for efficient batch computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..constants import (
    CLONE_EXACT_SYNTACTIC_THRESHOLD,
    CLONE_MIN_BLOCK_SIZE,
    CLONE_NEAR_SYNTACTIC_THRESHOLD,
    CloneCategory,
    DEFAULT_CLONE_RESULT_LIMIT,
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
)
from ..graph_utils import apply_token_budget_filter
from ._constants import TOKEN_OVERHEAD_CLONES, TOKEN_PER_CLONE_PAIR, clone_pair_text
from .sparse_utils import SparseJaccardComputer
from .tokenizer import tokenize_function

if TYPE_CHECKING:
    from ..models import CodeEntity


def _suggest_action(e1: "CodeEntity", e2: "CodeEntity", match_pct: float) -> str:
    """Suggest refactoring action for a clone pair."""
    if e1.name == e2.name and e1.file_path != e2.file_path:
        return f"EXTRACT: Move '{e1.name}' to shared utility module"
    if e1.file_path == e2.file_path:
        return "MERGE: Combine into single parameterized function"
    if match_pct >= 0.90:
        return "CONSOLIDATE: Functions have high line overlap"
    return "REVIEW: Consider if these should be unified"


def _is_nested(e1: "CodeEntity", e2: "CodeEntity") -> bool:
    """Check if one function is nested inside the other."""
    if e1.file_path != e2.file_path:
        return False
    l1_start, l1_end = e1.line_start, e1.line_end
    l2_start, l2_end = e2.line_start, e2.line_end
    return (l1_start <= l2_start and l2_end <= l1_end) or \
           (l2_start <= l1_start and l1_end <= l2_end)


def _get_category(match_pct: float, max_block_size: int) -> str:
    """Determine clone category based on match percentage and block size."""
    if match_pct >= CLONE_EXACT_SYNTACTIC_THRESHOLD:
        return CloneCategory.EXACT
    if match_pct >= CLONE_NEAR_SYNTACTIC_THRESHOLD and max_block_size >= CLONE_MIN_BLOCK_SIZE:
        return CloneCategory.NEAR_CLONE
    if max_block_size >= CLONE_MIN_BLOCK_SIZE:
        return CloneCategory.STRUCTURAL
    return CloneCategory.PARTIAL


def detect_clones_syntactic(
    entities: dict[str, "CodeEntity"],
    line_threshold: float = DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    func_threshold: float = DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    min_shared_lines: int = 2,
    limit: int | None = DEFAULT_CLONE_RESULT_LIMIT,
    exclude_tests: bool = True,
    min_lines: int = 3,
    token_limit: int | None = None,
) -> dict[str, Any]:
    """Detect syntactic (line-by-line) code clones.

    Uses Tree-sitter tokenization and sparse matrix batch Jaccard
    to find functions with similar line structure.

    Args:
        entities: Dict of entity_id -> CodeEntity
        line_threshold: Minimum Jaccard similarity for a line match (0-1)
        func_threshold: Minimum percentage of lines that must match (0-1)
        min_shared_lines: Minimum shared unique lines for candidate consideration
        limit: Maximum number of clone pairs to return (None = no limit)
        exclude_tests: Whether to exclude test functions
        min_lines: Minimum function lines to consider
        token_limit: Soft token limit for output (None = no limit)

    Returns:
        Dict with clones list and summary statistics
    """
    effective_limit: float = float(limit) if limit is not None else float('inf')

    # Filter to functions/methods with enough lines
    func_entities = {
        k: v for k, v in entities.items()
        if v.entity_type in ("function", "method")
        and v.source_code
        and (v.line_end - v.line_start + 1) >= min_lines
        and (not exclude_tests or not v.is_test)
    }

    if len(func_entities) < 2:
        return {
            "clones": [],
            "summary": {
                "mode": "syntactic",
                "total_functions": len(func_entities),
                "clone_pairs_found": 0,
                "exact_duplicates": 0,
                "near_clones": 0,
                "structural": 0,
                "partial": 0,
                "line_threshold_used": line_threshold,
                "func_threshold_used": func_threshold,
            }
        }

    # Tokenize all functions
    tokenized_lines: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for eid, entity in func_entities.items():
        tokenized_lines[eid] = tokenize_function(entity.source_code, entity.language)

    # Filter to functions with enough tokenized lines
    valid_eids = [
        eid for eid in func_entities
        if len(tokenized_lines.get(eid, [])) >= min_lines
    ]

    if len(valid_eids) < 2:
        return {
            "clones": [],
            "summary": {
                "mode": "syntactic",
                "total_functions": len(valid_eids),
                "clone_pairs_found": 0,
                "exact_duplicates": 0,
                "near_clones": 0,
                "structural": 0,
                "partial": 0,
                "line_threshold_used": line_threshold,
                "func_threshold_used": func_threshold,
            }
        }

    # Index functions
    computer = SparseJaccardComputer()
    computer.index_functions(func_entities, tokenized_lines)

    # Find candidates
    candidate_pairs = computer.find_candidates(
        valid_eids,
        min_shared_lines=min_shared_lines,
    )

    # Compare candidates
    clone_pairs: list[dict[str, Any]] = []
    for eid1, eid2 in candidate_pairs:
        e1, e2 = func_entities[eid1], func_entities[eid2]

        # Skip nested functions
        if _is_nested(e1, e2):
            continue

        match = computer.compare_functions(
            eid1, eid2,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
        )

        if match is None:
            continue

        category = _get_category(match.match_percentage, match.max_block_size)

        # Build block info
        blocks_info = []
        for block in match.blocks:
            if block:
                blocks_info.append({
                    "start_line1": block[0].line_idx1 + 1,
                    "start_line2": block[0].line_idx2 + 1,
                    "length": len(block),
                })

        clone_pairs.append({
            "entity1": {
                "name": e1.name,
                "file": e1.file_path,
                "line": e1.line_start,
                "type": e1.entity_type,
                "lines": e1.line_end - e1.line_start + 1,
            },
            "entity2": {
                "name": e2.name,
                "file": e2.file_path,
                "line": e2.line_start,
                "type": e2.entity_type,
                "lines": e2.line_end - e2.line_start + 1,
            },
            "similarity": round(match.match_percentage, 4),
            "syntactic_pct": round(match.match_percentage, 4),
            "matched_lines": match.matched_lines,
            "total_lines": match.total_lines,
            "category": category,
            "blocks": blocks_info,
            "max_block_size": match.max_block_size,
            "suggested_action": _suggest_action(e1, e2, match.match_percentage),
        })

        if len(clone_pairs) >= effective_limit * 10:
            break

    # Sort by match percentage descending
    clone_pairs.sort(key=lambda x: (-x["similarity"], -x["max_block_size"]))

    # Count categories
    exact_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.EXACT)
    near_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.NEAR_CLONE)
    structural_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.STRUCTURAL)
    partial_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.PARTIAL)

    # Apply limit and token budget
    slice_limit = limit if limit is not None else len(clone_pairs)
    filtered_pairs, _, token_budget_exceeded = apply_token_budget_filter(
        clone_pairs[:slice_limit],
        token_limit,
        TOKEN_OVERHEAD_CLONES,
        TOKEN_PER_CLONE_PAIR,
        text_fields=[],
        text_builder=clone_pair_text,
    )

    return {
        "clones": filtered_pairs,
        "summary": {
            "mode": "syntactic",
            "total_functions": len(valid_eids),
            "clone_pairs_found": len(clone_pairs),
            "exact_duplicates": exact_count,
            "near_clones": near_count,
            "structural": structural_count,
            "partial": partial_count,
            "line_threshold_used": line_threshold,
            "func_threshold_used": func_threshold,
            "results_returned": len(filtered_pairs),
            "token_budget_exceeded": token_budget_exceeded,
        },
    }
