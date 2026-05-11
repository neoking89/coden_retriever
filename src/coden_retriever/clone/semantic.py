"""Semantic clone detection using MiniLM ONNX embeddings.

Detects code clones by computing cosine similarity between
function embeddings. Finds functions that "do similar things"
even with different implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..constants import (
    CLONE_EXACT_SEMANTIC_THRESHOLD,
    CLONE_HIGH_SIM_THRESHOLD,
    CLONE_IDENTICAL_NAME_SIM,
    CLONE_LINE_RATIO_THRESHOLD,
    CLONE_NEAR_SEMANTIC_THRESHOLD,
    CLONE_REUSE_MAX_LINES,
    CLONE_SAME_FILE_NAME_SIM,
    CLONE_SHORT_FUNC_MAX_LINES,
    CLONE_UNRELATED_NAME_SIM,
    CLONE_VERY_HIGH_SIM_THRESHOLD,
    CloneCategory,
    DEFAULT_CLONE_RESULT_LIMIT,
    DEFAULT_CLONE_SEMANTIC_THRESHOLD,
)
from ..cache.embedding_cache import encode_with_cache
from ..graph_utils import apply_token_budget_filter
from ._constants import TOKEN_OVERHEAD_CLONES, TOKEN_PER_CLONE_PAIR, clone_pair_text

if TYPE_CHECKING:
    from ..cache.embedding_cache import EmbeddingCache
    from ..models import CodeEntity
    from ..utils.progress import ProgressCallback


def _is_stub_body(entity: "CodeEntity") -> bool:
    """Check if entity is a stub method."""
    return entity.is_stub


def _is_intentional_pair(
    e1: "CodeEntity",
    e2: "CodeEntity",
    body_similarity: float,
    name_similarity: float,
) -> bool:
    """Detect intentional complementary pairs using purely structural patterns.

    Intentional pairs (toggle methods, getter/setter, etc.) have:
    - Very similar body structure (nearly identical code)
    - Same parent class OR same file with similar line counts
    - Different names (but may share common prefix like step_into/step_out)
    - OR identical names across different files (intentional reuse pattern)
    """
    # Skip pairs where both are stub methods (interface definitions)
    if _is_stub_body(e1) and _is_stub_body(e2):
        return True

    # Compute line counts once
    line_count1 = e1.line_end - e1.line_start + 1
    line_count2 = e2.line_end - e2.line_start + 1

    # Very short functions with high body similarity are typically
    # UI handlers, event callbacks, or simple wrappers
    if line_count1 <= CLONE_SHORT_FUNC_MAX_LINES and line_count2 <= CLONE_SHORT_FUNC_MAX_LINES and body_similarity >= CLONE_HIGH_SIM_THRESHOLD:
        return True

    # Same parent class with high similarity = complementary methods
    if e1.parent_class and e1.parent_class == e2.parent_class and body_similarity >= CLONE_HIGH_SIM_THRESHOLD:
        return True

    # Must have very similar body structure for the remaining checks
    if body_similarity <= CLONE_VERY_HIGH_SIM_THRESHOLD:
        return False

    # Identical names in DIFFERENT files = intentional reuse pattern
    if name_similarity >= CLONE_IDENTICAL_NAME_SIM and e1.file_path != e2.file_path:
        if line_count1 <= CLONE_REUSE_MAX_LINES and line_count2 <= CLONE_REUSE_MAX_LINES:
            return True
        line_ratio = min(line_count1, line_count2) / max(line_count1, line_count2, 1)
        if line_ratio > CLONE_LINE_RATIO_THRESHOLD:
            return True

    # Same file handling
    if e1.file_path == e2.file_path:
        # High name similarity in same file = true duplicate (keep it)
        if name_similarity >= CLONE_SAME_FILE_NAME_SIM:
            return False
        # Low name similarity with similar line counts = toggle pair (filter it)
        line_ratio = min(line_count1, line_count2) / max(line_count1, line_count2, 1)
        if line_ratio > CLONE_LINE_RATIO_THRESHOLD:
            return True

    # If no strong structural signal, fall back to name similarity check
    return name_similarity < CLONE_UNRELATED_NAME_SIM


def _suggest_action(e1: "CodeEntity", e2: "CodeEntity", similarity: float) -> str:
    """Suggest refactoring action for a clone pair."""
    if e1.name == e2.name and e1.file_path != e2.file_path:
        return f"EXTRACT: Move '{e1.name}' to shared utility module"
    if e1.file_path == e2.file_path:
        return "MERGE: Combine into single parameterized function"
    if similarity >= CLONE_NEAR_SEMANTIC_THRESHOLD:
        return "CONSOLIDATE: Functions are nearly identical"
    return "REVIEW: Consider if these should be unified"


def detect_clones_semantic(
    entities: dict[str, "CodeEntity"],
    model_path: str,
    threshold: float = DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    limit: int | None = DEFAULT_CLONE_RESULT_LIMIT,
    exclude_tests: bool = True,
    min_lines: int = 3,
    token_limit: int | None = None,
    embedding_cache: "EmbeddingCache | None" = None,
    on_encode_progress: "ProgressCallback | None" = None,
) -> dict[str, Any]:
    """Detect semantic code clones using embeddings.

    Uses MiniLM ONNX embeddings to find functions that do similar things,
    even if they have different variable names or structural differences.

    Args:
        entities: Dict of entity_id -> CodeEntity
        model_path: Unused. Kept for backward compatibility.
        threshold: Minimum similarity threshold (0-1)
        limit: Maximum number of clone pairs to return (None = no limit)
        exclude_tests: Whether to exclude test functions
        min_lines: Minimum function lines to consider
        token_limit: Soft token limit for output (None = no limit)

    Returns:
        Dict with clones list and summary statistics
    """
    effective_limit: float = float(limit) if limit is not None else float('inf')

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
                "mode": "semantic",
                "total_functions": len(func_entities),
                "clone_pairs_found": 0,
                "exact_duplicates": 0,
                "near_clones": 0,
                "semantic_clones": 0,
                "threshold_used": threshold,
            }
        }

    node_ids = list(func_entities.keys())
    texts = [func_entities[nid].source_code for nid in node_ids]
    embeddings = encode_with_cache(texts, embedding_cache, on_batch_done=on_encode_progress)

    # L2-normalized vectors: dot product = cosine similarity
    names = [func_entities[nid].name for nid in node_ids]
    name_embeddings = encode_with_cache(names, embedding_cache, on_batch_done=on_encode_progress)
    name_similarity_matrix = np.dot(name_embeddings, name_embeddings.T)

    similarity_matrix = np.dot(embeddings, embeddings.T)

    n = len(node_ids)
    i_indices, j_indices = np.triu_indices(n, k=1)
    similarities = similarity_matrix[i_indices, j_indices]

    above_threshold = similarities >= threshold
    valid_i = i_indices[above_threshold]
    valid_j = j_indices[above_threshold]
    valid_sims = similarities[above_threshold]

    sort_order = np.argsort(-valid_sims)
    if effective_limit == float('inf'):
        max_candidates = len(sort_order)
    else:
        max_candidates = min(len(sort_order), int(effective_limit * 10))
    sort_order = sort_order[:max_candidates]

    clone_pairs: list[dict[str, Any]] = []
    for idx in sort_order:
        i, j = int(valid_i[idx]), int(valid_j[idx])
        sim = float(valid_sims[idx])
        e1 = func_entities[node_ids[i]]
        e2 = func_entities[node_ids[j]]

        if e1.file_path == e2.file_path:
            l1_start, l1_end = e1.line_start, e1.line_end
            l2_start, l2_end = e2.line_start, e2.line_end
            if (l1_start <= l2_start and l2_end <= l1_end) or \
               (l2_start <= l1_start and l1_end <= l2_end):
                continue

        name_sim = float(name_similarity_matrix[i, j])
        if _is_intentional_pair(e1, e2, sim, name_sim):
            continue

        if sim >= CLONE_EXACT_SEMANTIC_THRESHOLD:
            category = CloneCategory.EXACT
        elif sim >= CLONE_NEAR_SEMANTIC_THRESHOLD:
            category = CloneCategory.NEAR_CLONE
        else:
            category = CloneCategory.SEMANTIC

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
            "similarity": round(sim, 4),
            "semantic_sim": round(sim, 4),
            "category": category,
            "suggested_action": _suggest_action(e1, e2, sim),
        })

        if len(clone_pairs) >= effective_limit:
            break

    clone_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    exact_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.EXACT)
    near_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.NEAR_CLONE)
    semantic_count = sum(1 for c in clone_pairs if c["category"] == CloneCategory.SEMANTIC)

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
            "mode": "semantic",
            "total_functions": len(func_entities),
            "clone_pairs_found": len(clone_pairs),
            "exact_duplicates": exact_count,
            "near_clones": near_count,
            "semantic_clones": semantic_count,
            "threshold_used": threshold,
            "results_returned": len(filtered_pairs),
            "token_budget_exceeded": token_budget_exceeded,
        },
    }
