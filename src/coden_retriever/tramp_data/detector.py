"""Core tramp data detection using grouped parameter mining.

Identifies parameter GROUPS that travel together across many functions.
The real anti-pattern: (host, port, timeout) passed to 15 functions
instead of a ConnectionConfig object.

Algorithm: frequent pair mining → greedy group expansion.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

from ..constants import (
    TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    TRAMP_DATA_MIN_GROUP_SIZE,
    TRAMP_DATA_TIER_HIGH,
    TRAMP_DATA_TIER_LOW,
    TRAMP_DATA_TIER_MODERATE,
)
from .param_extractor import extract_all_params

if TYPE_CHECKING:
    from ..models.entities import CodeEntity


def detect_tramp_data(
    entities: dict[str, "CodeEntity"],
    min_occurrences: int = TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    exclude_tests: bool = True,
    limit: int | None = 50,
    min_group_size: int = TRAMP_DATA_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Detect tramp data parameter groups across the codebase.

    Args:
        entities: Dict of node_id -> CodeEntity.
        min_occurrences: Minimum functions a group must co-occur in.
        exclude_tests: Exclude test functions from analysis.
        limit: Maximum results to return (None for all).
        min_group_size: Minimum parameters in a group.

    Returns:
        Dict with tramp_data list (grouped) and summary statistics.
    """
    func_params = _build_function_params(entities, exclude_tests)
    pair_counts = _count_param_pairs(func_params, min_occurrences)
    groups = _expand_groups(pair_counts, func_params, min_occurrences, min_group_size)

    # Populate functions list for each group
    for group_item in groups:
        frozen = frozenset(group_item["group"])
        group_item["functions"] = _get_group_functions(frozen, func_params, entities)

    # Score and sort: group_size * function_count (descending)
    groups.sort(key=lambda g: _score_group(g), reverse=True)

    total_params = len({p for params in func_params.values() for p in params})
    total_functions = len(func_params)

    if limit is not None and limit > 0:
        groups = groups[:limit]

    return {
        "tramp_data": groups,
        "summary": _build_summary(groups, total_functions, total_params),
    }


def _build_function_params(
    entities: dict[str, "CodeEntity"],
    exclude_tests: bool,
) -> dict[str, set[str]]:
    """Build mapping of function_key -> set of param names."""
    filtered = {
        nid: e for nid, e in entities.items()
        if e.entity_type != "class" and (not exclude_tests or not e.is_test)
    }

    all_params = extract_all_params(filtered)

    func_params: dict[str, set[str]] = {}
    for node_id, params in all_params.items():
        if params:
            func_params[node_id] = set(params)

    return func_params


def _count_param_pairs(
    func_params: dict[str, set[str]],
    min_occurrences: int,
) -> dict[frozenset[str], int]:
    """Count how many functions each 2-param pair co-occurs in.

    Args:
        func_params: Mapping of function_id -> set of param names.
        min_occurrences: Threshold to keep a pair.

    Returns:
        Dict of frozenset({param_a, param_b}) -> function count.
        Only pairs meeting min_occurrences are included.
    """
    pair_counts: dict[frozenset[str], int] = {}

    for params in func_params.values():
        if len(params) < 2:
            continue
        for pair in combinations(sorted(params), 2):
            key = frozenset(pair)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    return {
        pair: count for pair, count in pair_counts.items()
        if count >= min_occurrences
    }


def _count_group_functions(
    group: frozenset[str],
    func_params: dict[str, set[str]],
) -> int:
    """Count functions containing ALL parameters in the group."""
    return sum(1 for params in func_params.values() if group <= params)


def _get_group_functions(
    group: frozenset[str],
    func_params: dict[str, set[str]],
    entities: dict[str, Any],
) -> list[dict[str, Any]]:
    """Get function info dicts for functions containing all group params."""
    results: list[dict[str, Any]] = []
    for node_id, params in func_params.items():
        if group <= params:
            entity = entities.get(node_id)
            if entity:
                results.append({
                    "name": entity.name,
                    "file": entity.file_path,
                    "line": entity.line_start,
                })
    return results


def _expand_groups(
    pair_counts: dict[frozenset[str], int],
    func_params: dict[str, set[str]],
    min_occurrences: int,
    min_group_size: int,
) -> list[dict[str, Any]]:
    """Expand frequent pairs into larger groups via greedy expansion.

    Algorithm:
    1. Start with highest-frequency pair
    2. Try adding params that co-occur with ALL group members
    3. Record expanded group AND any higher-frequency sub-groups
    4. Remove consumed pairs, repeat

    Reports both expanded groups and their sub-groups when the sub-group
    has a higher function count (avoids losing information about frequent
    subsets, e.g. (host, port) in 20 funcs vs (host, port, timeout) in 5).
    """
    all_pair_params = _collect_all_pair_params(pair_counts)
    remaining_pairs = dict(pair_counts)
    groups: list[dict[str, Any]] = []
    used_groups: set[frozenset[str]] = set()

    while remaining_pairs:
        best_pair = max(remaining_pairs, key=lambda p: remaining_pairs[p])
        seed_count = remaining_pairs[best_pair]
        current_group = set(best_pair)

        _try_expand_group(current_group, all_pair_params, func_params, min_occurrences)

        frozen_group = frozenset(current_group)
        expanded_count = _count_group_functions(frozen_group, func_params)

        # Report the seed pair if it has higher frequency than the expanded group
        _maybe_report_seed(
            best_pair, seed_count, frozen_group, expanded_count,
            min_group_size, used_groups, groups,
        )

        # Report the expanded group
        _maybe_report_group(
            frozen_group, expanded_count, min_occurrences,
            min_group_size, used_groups, groups,
        )

        # Remove all pairs consumed by this expanded group
        consumed = [p for p in remaining_pairs if p <= frozen_group]
        for p in consumed:
            del remaining_pairs[p]

    return groups


def _maybe_report_seed(
    seed_pair: frozenset[str],
    seed_count: int,
    expanded_group: frozenset[str],
    expanded_count: int,
    min_group_size: int,
    used_groups: set[frozenset[str]],
    groups: list[dict[str, Any]],
) -> None:
    """Report the seed pair separately if it has higher frequency than expanded."""
    if seed_pair == expanded_group:
        return
    if seed_count <= expanded_count:
        return
    if len(seed_pair) < min_group_size:
        return
    if seed_pair in used_groups:
        return

    used_groups.add(seed_pair)
    groups.append({
        "group": sorted(seed_pair),
        "count": seed_count,
        "functions": [],
    })


def _maybe_report_group(
    frozen_group: frozenset[str],
    count: int,
    min_occurrences: int,
    min_group_size: int,
    used_groups: set[frozenset[str]],
    groups: list[dict[str, Any]],
) -> None:
    """Report a group if it meets all thresholds and hasn't been seen."""
    if frozen_group in used_groups:
        return
    if len(frozen_group) < min_group_size:
        return
    if count < min_occurrences:
        return

    used_groups.add(frozen_group)
    groups.append({
        "group": sorted(frozen_group),
        "count": count,
        "functions": [],
    })


def _collect_all_pair_params(
    pair_counts: dict[frozenset[str], int],
) -> set[str]:
    """Collect all parameter names that appear in at least one frequent pair."""
    return {p for pair in pair_counts for p in pair}


def _try_expand_group(
    current_group: set[str],
    candidate_params: set[str],
    func_params: dict[str, set[str]],
    min_occurrences: int,
) -> None:
    """Try to expand group by adding params that co-occur with all members.

    Modifies current_group in place. Keeps expanding as long as adding a
    candidate still meets min_occurrences threshold.
    """
    changed = True
    while changed:
        changed = False
        candidates = candidate_params - current_group
        for param in sorted(candidates):
            trial = frozenset(current_group | {param})
            if _count_group_functions(trial, func_params) >= min_occurrences:
                current_group.add(param)
                changed = True


def _score_group(group: dict[str, Any]) -> tuple[int, int]:
    """Score a group for sorting: (group_size * count, group_size)."""
    size = len(group.get("group", []))
    count = group.get("count", 0)
    return (size * count, size)


def _build_summary(
    results: list[dict[str, Any]],
    total_functions: int,
    total_params: int,
) -> dict[str, Any]:
    """Build summary statistics for tramp data detection."""
    distribution = {"high": 0, "moderate": 0, "low": 0}
    for r in results:
        count = r["count"]
        if count >= TRAMP_DATA_TIER_HIGH:
            distribution["high"] += 1
        elif count >= TRAMP_DATA_TIER_MODERATE:
            distribution["moderate"] += 1
        elif count >= TRAMP_DATA_TIER_LOW:
            distribution["low"] += 1

    return {
        "total_functions_analyzed": total_functions,
        "total_unique_params": total_params,
        "tramp_groups_found": len(results),
        "distribution": distribution,
    }
