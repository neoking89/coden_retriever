"""Combined semantic + syntactic clone detection.

Provides comprehensive clone detection by combining:
1. Semantic similarity (MiniLM ONNX embeddings)
2. Syntactic similarity (line-by-line Jaccard)

Uses weighted harmonic mean for score aggregation with block bonus.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..constants import (
    CLONE_CONSOLIDATE_SCORE,
    CLONE_DIFFERENT_IMPL_SYN,
    CLONE_EXACT_SEMANTIC_THRESHOLD,
    CLONE_EXACT_SYNTACTIC_THRESHOLD,
    CLONE_HIGH_SIM_THRESHOLD,
    CLONE_IDENTICAL_NAME_SIM,
    CLONE_LINE_OVERLAP_SYN,
    CLONE_LINE_RATIO_THRESHOLD,
    CLONE_MIN_BLOCK_SIZE,
    CLONE_NEAR_SEMANTIC_THRESHOLD,
    CLONE_NEAR_SYNTACTIC_THRESHOLD,
    CLONE_REUSE_MAX_LINES,
    CLONE_SAME_FILE_NAME_SIM,
    CLONE_SEM_STRUCT_SYNTACTIC_THRESHOLD,
    CLONE_SHORT_FUNC_MAX_LINES,
    CLONE_STRUCTURAL_SYNTACTIC_THRESHOLD,
    CLONE_UNRELATED_NAME_SIM,
    CLONE_VERY_HIGH_SIM_THRESHOLD,
    CloneCategory,
    DEFAULT_CLONE_RESULT_LIMIT,
    DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    DEFAULT_CLONE_SEMANTIC_WEIGHT,
    DEFAULT_CLONE_SYNTACTIC_WEIGHT,
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
)
from ..cache.embedding_cache import encode_with_cache
from ..graph_utils import apply_token_budget_filter
from ._constants import (
    TOKEN_OVERHEAD_CLONES,
    TOKEN_PER_COMBINED_CLONE_PAIR,
    clone_pair_text,
)
from .sparse_utils import SparseJaccardComputer
from .tokenizer import tokenize_function

if TYPE_CHECKING:
    from ..cache.embedding_cache import EmbeddingCache
    from ..models import CodeEntity
    from ..utils.progress import ProgressCallback

# Block bonus constants for consecutive matching lines
BLOCK_BONUS_THRESHOLD = 5   # Minimum block size to trigger bonus
BLOCK_BONUS_VALUE = 0.03    # Bonus added for large consecutive blocks


def compute_combined_score(
    semantic_sim: float | None,
    syntactic_pct: float | None,
    max_block_size: int = 0,
    semantic_weight: float = DEFAULT_CLONE_SEMANTIC_WEIGHT,
    syntactic_weight: float = DEFAULT_CLONE_SYNTACTIC_WEIGHT,
) -> float:
    """Compute combined clone score using weighted harmonic mean.

    When both semantic and syntactic scores are available and > 0, uses
    weighted harmonic mean: (w_sem + w_syn) / (w_sem/sem + w_syn/syn)

    When only one score is available, returns that score directly.
    Missing values (None) are NOT treated as zeros for the harmonic mean.

    Args:
        semantic_sim: Semantic similarity (0-1), None if not computed
        syntactic_pct: Syntactic match percentage (0-1), None if not computed
        max_block_size: Largest consecutive matching block size
        semantic_weight: Weight for semantic similarity in harmonic mean (default 0.65)
        syntactic_weight: Weight for syntactic similarity in harmonic mean (default 0.35)

    Returns:
        Combined similarity score (0-1), capped at 1.0

    Raises:
        ValueError: If weights are both zero (would cause division by zero)
    """
    # Handle missing values - None means "not computed", not "zero"
    if semantic_sim is None and syntactic_pct is not None:
        return min(1.0, syntactic_pct)
    if syntactic_pct is None and semantic_sim is not None:
        return min(1.0, semantic_sim)
    if semantic_sim is None and syntactic_pct is None:
        return 0.0

    # Validate weights to prevent division by zero
    if semantic_weight == 0 and syntactic_weight == 0:
        raise ValueError("semantic_weight and syntactic_weight cannot both be zero")

    # At this point, both values are not None (due to early returns above)
    sem = cast(float, semantic_sim)
    syn = cast(float, syntactic_pct)

    # Both available: weighted harmonic mean (only when both > 0)
    if sem > 0 and syn > 0:
        # Handle edge case where one weight is zero
        if semantic_weight == 0:
            combined = syn
        elif syntactic_weight == 0:
            combined = sem
        else:
            combined = (semantic_weight + syntactic_weight) / (
                semantic_weight / sem + syntactic_weight / syn
            )
    elif sem == 0 and syn == 0:
        combined = 0.0
    else:
        # One is zero, one is positive: use weighted average
        # This ensures 0% syntactic actually lowers the combined score
        # e.g., 95% semantic + 0% syntactic with 0.6/0.4 weights = 57%
        combined = (semantic_weight * sem + syntactic_weight * syn) / (semantic_weight + syntactic_weight)

    # Block bonus: consecutive matches indicate true duplication
    if max_block_size >= BLOCK_BONUS_THRESHOLD:
        combined = combined + BLOCK_BONUS_VALUE

    # Cap at 1.0 to ensure valid similarity score
    return min(1.0, combined)


def _get_combined_category(
    semantic_sim: float | None,
    syntactic_pct: float | None,
    max_block_size: int,
) -> str:
    """Determine clone category for combined detection."""
    sem = semantic_sim or 0
    syn = syntactic_pct or 0

    if sem >= CLONE_EXACT_SEMANTIC_THRESHOLD and syn >= CLONE_EXACT_SYNTACTIC_THRESHOLD:
        return CloneCategory.EXACT
    if sem >= CLONE_NEAR_SEMANTIC_THRESHOLD and syn >= CLONE_NEAR_SYNTACTIC_THRESHOLD:
        return CloneCategory.NEAR_CLONE
    if sem >= DEFAULT_CLONE_SEMANTIC_THRESHOLD and syn >= CLONE_SEM_STRUCT_SYNTACTIC_THRESHOLD:
        return CloneCategory.SEMANTIC_STRUCTURAL
    if syn >= CLONE_STRUCTURAL_SYNTACTIC_THRESHOLD and max_block_size >= CLONE_MIN_BLOCK_SIZE:
        return CloneCategory.STRUCTURAL
    if sem >= DEFAULT_CLONE_SEMANTIC_THRESHOLD:
        return CloneCategory.SEMANTIC
    return CloneCategory.PARTIAL


def _suggest_action(
    e1: "CodeEntity",
    e2: "CodeEntity",
    combined_score: float,
    semantic_sim: float | None,
    syntactic_pct: float | None,
) -> str:
    """Suggest refactoring action for a clone pair."""
    if e1.name == e2.name and e1.file_path != e2.file_path:
        return f"EXTRACT: Move '{e1.name}' to shared utility module"
    if e1.file_path == e2.file_path:
        return "MERGE: Combine into single parameterized function"
    if combined_score >= CLONE_CONSOLIDATE_SCORE:
        return "CONSOLIDATE: High semantic and structural overlap"
    if (semantic_sim or 0) >= CLONE_HIGH_SIM_THRESHOLD and (syntactic_pct or 0) < CLONE_DIFFERENT_IMPL_SYN:
        return "REVIEW: Similar behavior, different implementation"
    if (syntactic_pct or 0) >= CLONE_LINE_OVERLAP_SYN:
        return "CONSOLIDATE: High line-by-line overlap"
    return "REVIEW: Consider if these should be unified"


def _is_nested(e1: "CodeEntity", e2: "CodeEntity") -> bool:
    """Check if one function is nested inside the other."""
    if e1.file_path != e2.file_path:
        return False
    l1_start, l1_end = e1.line_start, e1.line_end
    l2_start, l2_end = e2.line_start, e2.line_end
    return (l1_start <= l2_start and l2_end <= l1_end) or \
           (l2_start <= l1_start and l1_end <= l2_end)


def _is_intentional_pair(
    e1: "CodeEntity",
    e2: "CodeEntity",
    semantic_sim: float,
    name_similarity: float,
) -> bool:
    """Detect intentional complementary pairs."""
    # Skip pairs where both are stub methods
    if e1.is_stub and e2.is_stub:
        return True

    line_count1 = e1.line_end - e1.line_start + 1
    line_count2 = e2.line_end - e2.line_start + 1

    # Very short functions with high similarity
    if line_count1 <= CLONE_SHORT_FUNC_MAX_LINES and line_count2 <= CLONE_SHORT_FUNC_MAX_LINES and semantic_sim >= CLONE_HIGH_SIM_THRESHOLD:
        return True

    # Same parent class with high similarity
    if e1.parent_class and e1.parent_class == e2.parent_class and semantic_sim >= CLONE_HIGH_SIM_THRESHOLD:
        return True

    if semantic_sim <= CLONE_VERY_HIGH_SIM_THRESHOLD:
        return False

    # Identical names in different files = intentional reuse
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

    return name_similarity < CLONE_UNRELATED_NAME_SIM


def _build_summary(
    clone_pairs: list[dict[str, Any]],
    func_entities: dict[str, "CodeEntity"],
    filtered_pairs: list[dict[str, Any]],
    token_budget_exceeded: bool,
    *,
    semantic_threshold: float,
    line_threshold: float,
    func_threshold: float,
    semantic_weight: float,
    syntactic_weight: float,
) -> dict[str, Any]:
    """Build the combined-mode summary dict.

    Single source of truth for the summary schema — both the early-return
    (no work done) and the populated path go through here.
    """
    counts = Counter(c["category"] for c in clone_pairs)
    return {
        "mode": "combined",
        "total_functions": len(func_entities),
        "clone_pairs_found": len(clone_pairs),
        "exact_duplicates": counts[CloneCategory.EXACT],
        "near_clones": counts[CloneCategory.NEAR_CLONE],
        "semantic_structural": counts[CloneCategory.SEMANTIC_STRUCTURAL],
        "structural": counts[CloneCategory.STRUCTURAL],
        "semantic": counts[CloneCategory.SEMANTIC],
        "partial": counts[CloneCategory.PARTIAL],
        "semantic_threshold_used": semantic_threshold,
        "line_threshold_used": line_threshold,
        "func_threshold_used": func_threshold,
        "semantic_weight": semantic_weight,
        "syntactic_weight": syntactic_weight,
        "results_returned": len(filtered_pairs),
        "token_budget_exceeded": token_budget_exceeded,
    }


def detect_clones_combined(
    entities: dict[str, "CodeEntity"],
    model_path: str,
    semantic_threshold: float = DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    line_threshold: float = DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    func_threshold: float = DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    min_shared_lines: int = 2,
    limit: int | None = DEFAULT_CLONE_RESULT_LIMIT,
    exclude_tests: bool = True,
    min_lines: int = 3,
    token_limit: int | None = None,
    semantic_weight: float = DEFAULT_CLONE_SEMANTIC_WEIGHT,
    syntactic_weight: float = DEFAULT_CLONE_SYNTACTIC_WEIGHT,
    embedding_cache: "EmbeddingCache | None" = None,
    on_encode_progress: "ProgressCallback | None" = None,
) -> dict[str, Any]:
    """Detect code clones using combined semantic + syntactic analysis.

    Runs both detection methods and merges results with weighted scoring.

    Args:
        entities: Dict of entity_id -> CodeEntity
        model_path: Path to the MiniLM ONNX model
        semantic_threshold: Minimum semantic similarity threshold (0-1)
        line_threshold: Minimum Jaccard similarity for a line match (0-1)
        func_threshold: Minimum percentage of lines that must match (0-1)
        min_shared_lines: Minimum shared unique lines for syntactic candidates
        limit: Maximum number of clone pairs to return (None = no limit)
        exclude_tests: Whether to exclude test functions
        min_lines: Minimum function lines to consider
        token_limit: Soft token limit for output (None = no limit)
        semantic_weight: Weight for semantic similarity in combined score (default 0.65)
        syntactic_weight: Weight for syntactic similarity in combined score (default 0.35)

    Returns:
        Dict with clones list and summary statistics
    """
    # Filter to functions/methods
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
            "summary": _build_summary(
                clone_pairs=[],
                func_entities=func_entities,
                filtered_pairs=[],
                token_budget_exceeded=False,
                semantic_threshold=semantic_threshold,
                line_threshold=line_threshold,
                func_threshold=func_threshold,
                semantic_weight=semantic_weight,
                syntactic_weight=syntactic_weight,
            ),
        }

    node_ids = list(func_entities.keys())
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    n = len(node_ids)

    texts = [func_entities[nid].source_code for nid in node_ids]
    embeddings = encode_with_cache(texts, embedding_cache, on_batch_done=on_encode_progress)

    # L2-normalized vectors: dot product = cosine similarity
    names = [func_entities[nid].name for nid in node_ids]
    name_embeddings = encode_with_cache(names, embedding_cache, on_batch_done=on_encode_progress)
    name_similarity_matrix = np.dot(name_embeddings, name_embeddings.T)

    similarity_matrix = np.dot(embeddings, embeddings.T)

    tokenized_lines: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for eid, entity in func_entities.items():
        tokenized_lines[eid] = tokenize_function(entity.source_code, entity.language)

    valid_eids = [
        eid for eid in node_ids
        if len(tokenized_lines.get(eid, [])) >= min_lines
    ]

    computer = SparseJaccardComputer()
    computer.index_functions(func_entities, tokenized_lines)
    syntactic_candidates = sorted(computer.find_candidates(valid_eids, min_shared_lines))

    i_indices, j_indices = np.triu_indices(n, k=1)
    semantic_sims = similarity_matrix[i_indices, j_indices]

    clone_pairs: list[dict[str, Any]] = []
    pair_data: dict[tuple[str, str], dict[str, Any]] = {}

    above_semantic = semantic_sims >= semantic_threshold
    for idx in np.where(above_semantic)[0]:
        i, j = int(i_indices[idx]), int(j_indices[idx])
        eid1, eid2 = node_ids[i], node_ids[j]
        pair_key = (eid1, eid2) if eid1 < eid2 else (eid2, eid1)

        if pair_key not in pair_data:
            pair_data[pair_key] = {
                "i": i, "j": j,
                "semantic_sim": float(semantic_sims[idx]),
                "syntactic_pct": None,
                "syntactic_match": None,
            }
        else:
            pair_data[pair_key]["semantic_sim"] = float(semantic_sims[idx])

    for eid1, eid2 in syntactic_candidates:
        pair_key = (eid1, eid2) if eid1 < eid2 else (eid2, eid1)

        # Get indices
        idx1: int | None = node_id_to_idx.get(eid1)
        idx2: int | None = node_id_to_idx.get(eid2)
        if idx1 is None or idx2 is None:
            continue

        match = computer.compare_functions(
            eid1, eid2,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
        )

        if match is not None:
            if pair_key not in pair_data:
                pair_data[pair_key] = {
                    "i": idx1, "j": idx2,
                    "semantic_sim": float(similarity_matrix[idx1, idx2]),
                    "syntactic_pct": match.match_percentage,
                    "syntactic_match": match,
                }
            else:
                pair_data[pair_key]["syntactic_pct"] = match.match_percentage
                pair_data[pair_key]["syntactic_match"] = match

    for pair_key, data in pair_data.items():
        eid1, eid2 = pair_key
        i, j = data["i"], data["j"]
        e1, e2 = func_entities[eid1], func_entities[eid2]

        if _is_nested(e1, e2):
            continue

        semantic_sim = data["semantic_sim"]
        syntactic_pct = data.get("syntactic_pct")
        syntactic_match = data.get("syntactic_match")

        name_sim = float(name_similarity_matrix[i, j])
        if _is_intentional_pair(e1, e2, semantic_sim, name_sim):
            continue

        # Treat missing syntactic as 0.0 so the weighted average properly
        # penalizes pairs that have no syntactic evidence.
        max_block_size = syntactic_match.max_block_size if syntactic_match else 0
        effective_syntactic = syntactic_pct if syntactic_pct is not None else 0.0
        combined = compute_combined_score(
            semantic_sim, effective_syntactic, max_block_size,
            semantic_weight=semantic_weight, syntactic_weight=syntactic_weight
        )
        category = _get_combined_category(semantic_sim, syntactic_pct, max_block_size)

        blocks_info = []
        if syntactic_match:
            for block in syntactic_match.blocks:
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
            "similarity": round(combined, 4),
            "semantic_sim": round(semantic_sim, 4) if semantic_sim else None,
            "syntactic_pct": round(syntactic_pct, 4) if syntactic_pct else None,
            "matched_lines": syntactic_match.matched_lines if syntactic_match else None,
            "total_lines": syntactic_match.total_lines if syntactic_match else None,
            "category": category,
            "blocks": blocks_info,
            "max_block_size": max_block_size,
            "suggested_action": _suggest_action(e1, e2, combined, semantic_sim, syntactic_pct),
        })

    clone_pairs.sort(key=lambda x: (-x["similarity"], -x.get("max_block_size", 0)))

    slice_limit = limit if limit is not None else len(clone_pairs)
    filtered_pairs, _, token_budget_exceeded = apply_token_budget_filter(
        clone_pairs[:slice_limit],
        token_limit,
        TOKEN_OVERHEAD_CLONES,
        TOKEN_PER_COMBINED_CLONE_PAIR,
        text_fields=[],
        text_builder=clone_pair_text,
    )

    return {
        "clones": filtered_pairs,
        "summary": _build_summary(
            clone_pairs=clone_pairs,
            func_entities=func_entities,
            filtered_pairs=filtered_pairs,
            token_budget_exceeded=token_budget_exceeded,
            semantic_threshold=semantic_threshold,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            semantic_weight=semantic_weight,
            syntactic_weight=syntactic_weight,
        ),
    }
