"""Sparse matrix utilities for batch Jaccard computation.

Uses scipy sparse matrices for efficient pairwise similarity computation
between tokenized function lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from ..models import CodeEntity


@dataclass
class LineMatch:
    """A matching line pair between two functions."""

    line_idx1: int
    line_idx2: int
    similarity: float


@dataclass
class FunctionMatch:
    """Match result between two functions."""

    match_percentage: float
    matched_lines: int
    total_lines: int
    line_matches: list[LineMatch]
    blocks: list[list[LineMatch]]
    max_block_size: int


class SparseJaccardComputer:
    """Computes batch Jaccard similarity using sparse matrices."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._vocab_size: int = 0
        self._func_lines: dict[str, list[tuple[str, tuple[int, ...], int]]] = {}
        self._func_sparse: dict[str, sparse.csr_matrix] = {}
        self._func_lens: dict[str, np.ndarray] = {}
        self._line_hash_to_funcs: dict[int, set[str]] = {}

    def _get_token_id(self, token: str) -> int:
        """Get or create token index."""
        if token not in self._vocab:
            self._vocab[token] = self._vocab_size
            self._vocab_size += 1
        return self._vocab[token]

    def index_functions(
        self,
        entities: dict[str, "CodeEntity"],
        tokenized_lines: dict[str, list[tuple[str, frozenset[str]]]],
    ) -> None:
        """Index functions for efficient candidate finding and comparison.

        Args:
            entities: Entity dict (eid -> CodeEntity)
            tokenized_lines: Tokenized lines per entity (eid -> [(text, tokens), ...])
        """
        self._vocab.clear()
        self._vocab_size = 0
        self._func_lines.clear()
        self._line_hash_to_funcs.clear()

        # Build vocabulary and line hashes
        for eid, lines in tokenized_lines.items():
            func_data: list[tuple[str, tuple[int, ...], int]] = []
            for stripped, tokens in lines:
                token_ids = tuple(self._get_token_id(t) for t in tokens)
                line_hash = hash(tuple(sorted(tokens)))
                func_data.append((stripped, token_ids, line_hash))

                if eid not in self._line_hash_to_funcs.get(line_hash, set()):
                    if line_hash not in self._line_hash_to_funcs:
                        self._line_hash_to_funcs[line_hash] = set()
                    self._line_hash_to_funcs[line_hash].add(eid)

            self._func_lines[eid] = func_data

        # Build global sparse matrix
        self._build_sparse_matrices()

    def _build_sparse_matrices(self) -> None:
        """Build sparse matrices for all indexed functions."""
        if not self._func_lines or self._vocab_size == 0:
            return

        all_rows: list[int] = []
        all_cols: list[int] = []
        func_row_ranges: dict[str, tuple[int, int]] = {}
        current_row = 0

        for eid, lines in self._func_lines.items():
            if not lines:
                continue
            n = len(lines)
            start_row = current_row

            for i, (_, token_ids, _) in enumerate(lines):
                for tid in token_ids:
                    all_rows.append(current_row + i)
                    all_cols.append(tid)

            func_row_ranges[eid] = (start_row, start_row + n)
            current_row += n

        if current_row == 0:
            return

        total_rows = current_row
        all_rows_np = np.array(all_rows, dtype=np.int32)
        all_cols_np = np.array(all_cols, dtype=np.int32)
        all_data = np.ones(len(all_rows), dtype=np.float32)

        global_sparse = sparse.csr_matrix(
            (all_data, (all_rows_np, all_cols_np)),
            shape=(total_rows, self._vocab_size),
            dtype=np.float32
        )

        all_lens = np.array(global_sparse.sum(axis=1)).flatten()

        for eid, (start, end) in func_row_ranges.items():
            self._func_sparse[eid] = global_sparse[start:end]
            self._func_lens[eid] = all_lens[start:end]

    def find_candidates(
        self,
        valid_eids: list[str],
        min_shared_lines: int = 2,
        max_line_freq: int = 20,
    ) -> list[tuple[str, str]]:
        """Find candidate function pairs based on shared exact lines.

        Args:
            valid_eids: List of valid entity IDs to consider
            min_shared_lines: Minimum shared unique lines required
            max_line_freq: Maximum line frequency (skip very common lines)

        Returns:
            List of (eid1, eid2) candidate pairs
        """
        valid_set = set(valid_eids)
        pair_shared: dict[tuple[str, str], int] = {}

        for line_hash, funcs in self._line_hash_to_funcs.items():
            valid_funcs = [f for f in funcs if f in valid_set]
            if len(valid_funcs) < 2 or len(valid_funcs) > max_line_freq:
                continue

            funcs_list = list(valid_funcs)
            for i in range(len(funcs_list)):
                for j in range(i + 1, len(funcs_list)):
                    pair = (funcs_list[i], funcs_list[j]) if funcs_list[i] < funcs_list[j] else (funcs_list[j], funcs_list[i])
                    pair_shared[pair] = pair_shared.get(pair, 0) + 1

        return [pair for pair, count in pair_shared.items() if count >= min_shared_lines]

    def compare_functions(
        self,
        eid1: str,
        eid2: str,
        line_threshold: float = 0.70,
        func_threshold: float = 0.50,
    ) -> FunctionMatch | None:
        """Compare two functions using batch Jaccard.

        Args:
            eid1: First entity ID
            eid2: Second entity ID
            line_threshold: Minimum Jaccard similarity for a line match
            func_threshold: Minimum percentage of lines that must match

        Returns:
            FunctionMatch if above threshold, None otherwise
        """
        if eid1 not in self._func_sparse or eid2 not in self._func_sparse:
            return None

        mat1 = self._func_sparse[eid1]
        mat2 = self._func_sparse[eid2]
        n1, n2 = mat1.shape[0], mat2.shape[0]

        if n1 == 0 or n2 == 0:
            return None

        min_match = int(n1 * func_threshold)
        if min_match == 0:
            min_match = 1

        # Hash lookups for exact matches
        L1 = self._func_lines[eid1]
        L2 = self._func_lines[eid2]
        h1_list = [h for (_, _, h) in L1]
        h2_map = {h: i for i, (_, _, h) in enumerate(L2)}

        # Batch Jaccard computation
        intersection = (mat1 @ mat2.T).toarray()
        lens1 = self._func_lens[eid1]
        lens2 = self._func_lens[eid2]
        union = lens1.reshape(-1, 1) + lens2.reshape(1, -1) - intersection

        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard_matrix = np.where(union > 0, intersection / union, 0.0)

        # Fast path: exact hash matches set to 1.0
        for i1, h1 in enumerate(h1_list):
            if h1 in h2_map:
                jaccard_matrix[i1, h2_map[h1]] = 1.0

        # Find best matches per line
        best_indices = np.argmax(jaccard_matrix, axis=1)
        best_sims = jaccard_matrix[np.arange(n1), best_indices]

        above_threshold = best_sims >= line_threshold
        matched = int(np.sum(above_threshold))

        if matched < min_match:
            return None

        # Extract line matches
        line_matches: list[LineMatch] = []
        for i1 in range(n1):
            if above_threshold[i1]:
                line_matches.append(LineMatch(
                    line_idx1=i1,
                    line_idx2=int(best_indices[i1]),
                    similarity=float(best_sims[i1]),
                ))

        # Detect consecutive blocks
        blocks = self._find_blocks(line_matches, threshold=0.95)
        max_block_size = max((len(b) for b in blocks), default=0)

        return FunctionMatch(
            match_percentage=matched / n1,
            matched_lines=matched,
            total_lines=n1,
            line_matches=line_matches,
            blocks=blocks,
            max_block_size=max_block_size,
        )

    def _find_blocks(
        self,
        matches: list[LineMatch],
        threshold: float = 0.95,
    ) -> list[list[LineMatch]]:
        """Find consecutive matching line blocks."""
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda x: x.line_idx1)
        blocks: list[list[LineMatch]] = []
        current_block: list[LineMatch] = [sorted_matches[0]]

        for i in range(1, len(sorted_matches)):
            prev = sorted_matches[i - 1]
            curr = sorted_matches[i]

            if (curr.line_idx1 == prev.line_idx1 + 1 and
                curr.line_idx2 == prev.line_idx2 + 1 and
                curr.similarity >= threshold):
                current_block.append(curr)
            else:
                if len(current_block) >= 2:
                    blocks.append(current_block)
                current_block = [curr]

        if len(current_block) >= 2:
            blocks.append(current_block)

        return blocks

    def get_function_line_count(self, eid: str) -> int:
        """Get the number of tokenized lines for a function."""
        return len(self._func_lines.get(eid, []))
