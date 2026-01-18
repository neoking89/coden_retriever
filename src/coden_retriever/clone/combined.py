"""Combined semantic + syntactic clone detection.

Provides comprehensive clone detection by combining:
1. Semantic similarity (Model2Vec embeddings)
2. Syntactic similarity (line-by-line Jaccard)

Uses weighted harmonic mean for score aggregation with block bonus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..search.semantic import get_cached_model
from ..token_estimator import count_tokens
from .sparse_utils import SparseJaccardComputer
from .tokenizer import tokenize_function

if TYPE_CHECKING:
    from ..models import CodeEntity

_TOKEN_OVERHEAD_CLONES = 200
_TOKEN_PER_CLONE_PAIR = 100  # Slightly higher for combined output

# Block bonus constants for consecutive matching lines
BLOCK_BONUS_THRESHOLD = 5   # Minimum block size to trigger bonus
BLOCK_BONUS_VALUE = 0.03    # Bonus added for large consecutive blocks


def compute_combined_score(
    semantic_sim: float | None,
    syntactic_pct: float | None,
    max_block_size: int = 0,
    semantic_weight: float = 0.65,
    syntactic_weight: float = 0.35,
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

    if sem >= 0.9999 and syn >= 0.95:
        return "EXACT"
    if sem >= 0.98 and syn >= 0.80:
        return "NEAR-CLONE"
    if sem >= 0.95 and syn >= 0.50:
        return "SEMANTIC-STRUCTURAL"
    if syn >= 0.70 and max_block_size >= 5:
        return "STRUCTURAL"
    if sem >= 0.95:
        return "SEMANTIC"
    return "PARTIAL"


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
    if combined_score >= 0.95:
        return "CONSOLIDATE: High semantic and structural overlap"
    if (semantic_sim or 0) >= 0.95 and (syntactic_pct or 0) < 0.50:
        return "REVIEW: Similar behavior, different implementation"
    if (syntactic_pct or 0) >= 0.70:
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
    if line_count1 <= 5 and line_count2 <= 5 and semantic_sim >= 0.95:
        return True

    # Same parent class with high similarity
    if e1.parent_class and e1.parent_class == e2.parent_class and semantic_sim >= 0.95:
        return True

    if semantic_sim <= 0.97:
        return False

    # Identical names in different files = intentional reuse
    if name_similarity >= 0.99 and e1.file_path != e2.file_path:
        if line_count1 <= 10 and line_count2 <= 10:
            return True
        line_ratio = min(line_count1, line_count2) / max(line_count1, line_count2, 1)
        if line_ratio > 0.7:
            return True

    # Same file handling
    if e1.file_path == e2.file_path:
        # High name similarity in same file = true duplicate (keep it)
        if name_similarity >= 0.80:
            return False
        # Low name similarity with similar line counts = toggle pair (filter it)
        line_ratio = min(line_count1, line_count2) / max(line_count1, line_count2, 1)
        if line_ratio > 0.7:
            return True

    return name_similarity < 0.85


def detect_clones_combined(
    entities: dict[str, "CodeEntity"],
    model_path: str,
    semantic_threshold: float = 0.95,
    line_threshold: float = 0.70,
    func_threshold: float = 0.50,
    min_shared_lines: int = 2,
    limit: int | None = 50,
    exclude_tests: bool = True,
    min_lines: int = 3,
    token_limit: int | None = None,
    semantic_weight: float = 0.65,
    syntactic_weight: float = 0.35,
) -> dict[str, Any]:
    """Detect code clones using combined semantic + syntactic analysis.

    Runs both detection methods and merges results with weighted scoring.

    Args:
        entities: Dict of entity_id -> CodeEntity
        model_path: Path to the Model2Vec model
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
    effective_limit: float = float(limit) if limit is not None else float('inf')

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
            "summary": {
                "mode": "combined",
                "total_functions": len(func_entities),
                "clone_pairs_found": 0,
                "exact_duplicates": 0,
                "near_clones": 0,
                "semantic_structural": 0,
                "structural": 0,
                "semantic": 0,
                "partial": 0,
                "semantic_threshold_used": semantic_threshold,
                "line_threshold_used": line_threshold,
                "func_threshold_used": func_threshold,
                "semantic_weight": semantic_weight,
                "syntactic_weight": syntactic_weight,
            }
        }

    node_ids = list(func_entities.keys())
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    n = len(node_ids)

    # === SEMANTIC ANALYSIS ===
    model = get_cached_model(model_path)
    texts = [func_entities[nid].source_code for nid in node_ids]
    embeddings = model.encode(texts)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    # Name embeddings for intentional pair detection
    names = [func_entities[nid].name for nid in node_ids]
    name_embeddings = model.encode(names)
    name_norms = np.linalg.norm(name_embeddings, axis=1, keepdims=True)
    name_norms = np.where(name_norms == 0, 1, name_norms)
    name_embeddings = name_embeddings / name_norms
    name_similarity_matrix = np.dot(name_embeddings, name_embeddings.T)

    similarity_matrix = np.dot(embeddings, embeddings.T)

    # === SYNTACTIC ANALYSIS ===
    tokenized_lines: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for eid, entity in func_entities.items():
        tokenized_lines[eid] = tokenize_function(entity.source_code, entity.language)

    valid_eids = [
        eid for eid in node_ids
        if len(tokenized_lines.get(eid, [])) >= min_lines
    ]

    computer = SparseJaccardComputer()
    computer.index_functions(func_entities, tokenized_lines)
    syntactic_candidates = set(computer.find_candidates(valid_eids, min_shared_lines))

    # === FIND CLONE PAIRS ===
    # Consider pairs that meet EITHER threshold
    i_indices, j_indices = np.triu_indices(n, k=1)
    semantic_sims = similarity_matrix[i_indices, j_indices]

    clone_pairs: list[dict[str, Any]] = []
    pair_data: dict[tuple[str, str], dict[str, Any]] = {}

    # Process all pairs that meet semantic threshold
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

    # Process syntactic candidates
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

    # Build clone pairs from merged data
    for pair_key, data in pair_data.items():
        eid1, eid2 = pair_key
        i, j = data["i"], data["j"]
        e1, e2 = func_entities[eid1], func_entities[eid2]

        # Skip nested functions
        if _is_nested(e1, e2):
            continue

        semantic_sim = data["semantic_sim"]
        syntactic_pct = data.get("syntactic_pct")
        syntactic_match = data.get("syntactic_match")

        # Skip intentional pairs
        name_sim = float(name_similarity_matrix[i, j])
        if _is_intentional_pair(e1, e2, semantic_sim, name_sim):
            continue

        # Compute combined score
        # In combined mode, treat None syntactic as 0.0 (no match found)
        # so the weighted average properly penalizes missing syntactic matches
        max_block_size = syntactic_match.max_block_size if syntactic_match else 0
        effective_syntactic = syntactic_pct if syntactic_pct is not None else 0.0
        combined = compute_combined_score(
            semantic_sim, effective_syntactic, max_block_size,
            semantic_weight=semantic_weight, syntactic_weight=syntactic_weight
        )
        category = _get_combined_category(semantic_sim, syntactic_pct, max_block_size)

        # Build block info
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

        if len(clone_pairs) >= effective_limit * 10:
            break

    # Sort by combined score
    clone_pairs.sort(key=lambda x: (-x["similarity"], -x.get("max_block_size", 0)))

    # Count categories
    exact_count = sum(1 for c in clone_pairs if c["category"] == "EXACT")
    near_count = sum(1 for c in clone_pairs if c["category"] == "NEAR-CLONE")
    sem_struct_count = sum(1 for c in clone_pairs if c["category"] == "SEMANTIC-STRUCTURAL")
    structural_count = sum(1 for c in clone_pairs if c["category"] == "STRUCTURAL")
    semantic_count = sum(1 for c in clone_pairs if c["category"] == "SEMANTIC")
    partial_count = sum(1 for c in clone_pairs if c["category"] == "PARTIAL")

    # Apply limit and token budget
    slice_limit = limit if limit is not None else len(clone_pairs)
    if token_limit is None:
        filtered_pairs = clone_pairs[:slice_limit]
        token_budget_exceeded = False
    else:
        used_tokens = _TOKEN_OVERHEAD_CLONES
        token_budget_exceeded = False
        filtered_pairs = []

        for pair in clone_pairs[:slice_limit]:
            pair_text = f"{pair['entity1']['name']} {pair['entity1']['file']} {pair['entity2']['name']} {pair['entity2']['file']} {pair['suggested_action']}"
            pair_tokens = count_tokens(pair_text, is_code=False) + _TOKEN_PER_CLONE_PAIR
            if used_tokens + pair_tokens > token_limit:
                token_budget_exceeded = True
                break
            used_tokens += pair_tokens
            filtered_pairs.append(pair)

    return {
        "clones": filtered_pairs,
        "summary": {
            "mode": "combined",
            "total_functions": len(func_entities),
            "clone_pairs_found": len(clone_pairs),
            "exact_duplicates": exact_count,
            "near_clones": near_count,
            "semantic_structural": sem_struct_count,
            "structural": structural_count,
            "semantic": semantic_count,
            "partial": partial_count,
            "threshold_used": semantic_threshold,  # Backward compat
            "semantic_threshold_used": semantic_threshold,
            "line_threshold_used": line_threshold,
            "func_threshold_used": func_threshold,
            "semantic_weight": semantic_weight,
            "syntactic_weight": syntactic_weight,
            "results_returned": len(filtered_pairs),
            "token_budget_exceeded": token_budget_exceeded,
        },
    }
