"""
BM25 search module.

Implements Okapi BM25 algorithm for lexical code search with inverted index
optimization for O(df) query time instead of O(N).
"""
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import Config
from ..utils.optional_deps import get_numpy
from .base import SearchIndex

if TYPE_CHECKING:
    from numpy import ndarray


# Pre-compiled regex patterns for tokenization (module-level for performance)
_CAMEL_CASE_PATTERN = re.compile(r"([a-z])([A-Z])")
_NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9]")


@lru_cache(maxsize=2048)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    """Cached tokenization for repeated queries.

    Returns a tuple (immutable) for caching compatibility.
    Uses pre-compiled regex patterns for better performance.
    """
    text = _CAMEL_CASE_PATTERN.sub(r"\1 \2", text)
    text = _NON_ALPHANUM_PATTERN.sub(" ", text.lower())
    return tuple(t for t in text.split() if len(t) >= 2)


class BM25Index(SearchIndex):
    """
    Okapi BM25 implementation with inverted index for fast lexical code search.

    BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks
    documents based on the query terms appearing in each document.

    Key features:
    - Term frequency saturation (via k1 parameter)
    - Document length normalization (via b parameter)
    - Inverse document frequency (IDF) weighting

    Performance optimizations:
    - Inverted index for O(df) query time instead of O(N)
    - NumPy vectorized operations for batch scoring
    - Pre-computed document length penalties
    - LRU-cached tokenization
    """

    def __init__(self, k1: float = Config.BM25_K1, b: float = Config.BM25_B):
        self.k1 = k1
        self.b = b
        self._corpus_size: int = 0
        self._avg_doc_length: float = 0.0
        self._doc_ids: list[str] = []
        self._idf: dict[str, float] = {}
        # Inverted index: term -> (doc_indices array, term_frequencies array)
        self._inverted_index: dict[str, tuple["ndarray", "ndarray"]] = {}
        # Pre-computed: k1 * (1 - b + b * (doc_len / avg_len)) per document
        np = get_numpy()
        self._doc_denom_part: "ndarray" = np.array([], dtype=np.float32)

    @staticmethod
    def tokenize(text: str) -> tuple[str, ...]:
        """Tokenize text for indexing/searching.

        Returns a tuple for consistency with cached version.
        """
        return _tokenize_cached(text)

    def index(self, documents: dict[str, str]) -> None:
        """Build the BM25 index from documents using inverted index structure."""
        self._corpus_size = len(documents)
        if self._corpus_size == 0:
            return

        np = get_numpy()
        self._doc_ids = list(documents.keys())
        doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}

        # Temporary storage for building inverted index
        temp_inverted: dict[str, list[int]] = defaultdict(list)
        temp_freqs: dict[str, list[int]] = defaultdict(list)
        doc_lengths = np.zeros(self._corpus_size, dtype=np.float32)

        for doc_id, text in documents.items():
            idx = doc_id_to_idx[doc_id]
            tokens = self.tokenize(text)
            doc_lengths[idx] = len(tokens)

            counts = Counter(tokens)
            for term, freq in counts.items():
                temp_inverted[term].append(idx)
                temp_freqs[term].append(freq)

        self._avg_doc_length = float(np.mean(doc_lengths)) if self._corpus_size else 1.0

        # Pre-compute document-specific denominator part
        # BM25 denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
        self._doc_denom_part = self.k1 * (
            1 - self.b + self.b * (doc_lengths / self._avg_doc_length)
        )

        # Finalize inverted index and calculate IDF
        for term in temp_inverted:
            indices = np.array(temp_inverted[term], dtype=np.int32)
            freqs = np.array(temp_freqs[term], dtype=np.float32)
            self._inverted_index[term] = (indices, freqs)

            # IDF calculation (BM25+ variant - always positive)
            df = len(indices)
            numerator = self._corpus_size - df + 0.5
            denominator = df + 0.5
            self._idf[term] = math.log((numerator / denominator) + 1)

    def score(self, query: str, doc_id: str) -> float:
        """Score a document against a query."""
        if doc_id not in self._doc_ids:
            return 0.0

        try:
            doc_idx = self._doc_ids.index(doc_id)
        except ValueError:
            return 0.0

        np = get_numpy()
        score = 0.0
        for term in self.tokenize(query):
            if term not in self._inverted_index:
                continue

            indices, freqs = self._inverted_index[term]
            # Find if this document contains the term
            positions = np.where(indices == doc_idx)[0]
            if len(positions) == 0:
                continue

            tf = freqs[positions[0]]
            idf = self._idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self._doc_denom_part[doc_idx]
            score += idf * (numerator / denominator)

        return score

    def score_all(self, query: str) -> dict[str, float]:
        """Score all documents against a query using inverted index.

        Only iterates over documents that contain query terms (O(df) vs O(N)).
        Uses NumPy vectorized operations for batch scoring.
        """
        query_terms = self.tokenize(query)
        if not query_terms or self._corpus_size == 0:
            return {}

        np = get_numpy()
        # Initialize scores for all documents
        scores = np.zeros(self._corpus_size, dtype=np.float32)

        # Only process unique terms that exist in our index
        for term in set(query_terms):
            if term not in self._inverted_index:
                continue

            indices, freqs = self._inverted_index[term]
            idf = self._idf[term]

            # BM25 formula: idf * (f * (k1 + 1)) / (f + doc_denom_part)
            # Vectorized: update only documents that contain the term
            numerator = freqs * (self.k1 + 1)
            denominator = freqs + self._doc_denom_part[indices]
            scores[indices] += idf * (numerator / denominator)

        # Convert to dict, filtering zero scores
        relevant_indices = np.nonzero(scores)[0]
        return {self._doc_ids[idx]: float(scores[idx]) for idx in relevant_indices}
