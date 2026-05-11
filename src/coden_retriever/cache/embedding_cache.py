"""Content-addressable embedding cache.

Maps text content hashes to pre-computed ONNX embeddings, avoiding
redundant inference when code hasn't changed.

Persistence: text_embeddings.npy (N x EMBEDDING_DIM) + text_embedding_index.json (MD5 -> row).
"""
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils.onnx_encoder import EMBEDDING_DIM, onnx_encode

if TYPE_CHECKING:
    from ..utils.progress import ProgressCallback

logger = logging.getLogger(__name__)

CACHE_EMBEDDINGS_FILE = "text_embeddings.npy"
CACHE_INDEX_FILE = "text_embedding_index.json"

# Pre-allocate 512 rows: typical codebase has ~100-400 unique function/comment
# strings for echo+clone analysis, so 512 avoids any resize on the first run
# while keeping the cold-start footprint under 1 MB (512 * 384 * 4 bytes).
_INITIAL_CAPACITY = 512


def _cache_key(text: str, model_dir: "str | Path | None") -> str:
    """Hash text + model identity. Different models must not share embeddings."""
    model_id = str(Path(model_dir).resolve()) if model_dir else "default"
    return hashlib.md5(f"{model_id}\0{text}".encode("utf-8")).hexdigest()


def encode_with_cache(
    texts: list[str],
    cache: "EmbeddingCache | None",
    model_dir: "str | Path | None" = None,
    on_batch_done: "ProgressCallback | None" = None,
) -> np.ndarray:
    """Centralizes cache-or-encode so every call site is one line."""
    if cache is not None:
        return cache.get_or_encode(texts, model_dir=model_dir, on_batch_done=on_batch_done)
    return onnx_encode(texts, model_dir=model_dir, on_batch_done=on_batch_done)


class EmbeddingCache:
    """Content-addressable cache mapping text hashes to ONNX embeddings.

    Thread-safe for concurrent get_or_encode calls (daemon use case).
    """

    def __init__(self) -> None:
        self._hash_to_row: dict[str, int] = {}
        self._embeddings: np.ndarray = np.empty(
            (_INITIAL_CAPACITY, EMBEDDING_DIM), dtype=np.float32,
        )
        self._size = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get_or_encode(
        self,
        texts: list[str],
        model_dir: str | Path | None = None,
        on_batch_done: "ProgressCallback | None" = None,
    ) -> np.ndarray:
        """Return embeddings for texts, batch-encoding only cache misses."""
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        hashes = [_cache_key(t, model_dir) for t in texts]
        result = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)

        miss_indices: list[int] = []
        miss_texts: list[str] = []

        with self._lock:
            for i, h in enumerate(hashes):
                row = self._hash_to_row.get(h)
                if row is not None:
                    result[i] = self._embeddings[row]
                    self._hits += 1
                else:
                    miss_indices.append(i)
                    miss_texts.append(texts[i])
                    self._misses += 1

        if not miss_texts:
            # All cache hits — report full progress immediately
            if on_batch_done:
                on_batch_done(len(texts))
            return result

        miss_embeddings = onnx_encode(miss_texts, model_dir=model_dir, on_batch_done=on_batch_done)

        with self._lock:
            self._ensure_capacity(len(miss_texts))
            for j, idx in enumerate(miss_indices):
                h = hashes[idx]
                if h not in self._hash_to_row:
                    row = self._size
                    self._hash_to_row[h] = row
                    self._embeddings[row] = miss_embeddings[j]
                    self._size += 1
                else:
                    row = self._hash_to_row[h]
                result[idx] = self._embeddings[row]

        return result

    def _ensure_capacity(self, additional: int) -> None:
        """Must hold self._lock."""
        needed = self._size + additional
        if needed <= self._embeddings.shape[0]:
            return
        new_cap = max(self._embeddings.shape[0] * 2, needed)
        new_arr = np.empty((new_cap, EMBEDDING_DIM), dtype=np.float32)
        new_arr[: self._size] = self._embeddings[: self._size]
        self._embeddings = new_arr

    def save(self, cache_dir: Path) -> None:
        """Persist cache to disk as .npy + .json files."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            trimmed = self._embeddings[: self._size].copy()
            index_copy = dict(self._hash_to_row)

        np.save(cache_dir / CACHE_EMBEDDINGS_FILE, trimmed)
        with open(cache_dir / CACHE_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_copy, f)

        logger.info(
            "Embedding cache saved: %d entries, hits=%d misses=%d",
            len(index_copy), self._hits, self._misses,
        )

    @classmethod
    def load(cls, cache_dir: Path) -> "EmbeddingCache | None":
        """Load cache from disk. Returns None if files are missing or corrupt."""
        emb_path = cache_dir / CACHE_EMBEDDINGS_FILE
        idx_path = cache_dir / CACHE_INDEX_FILE

        if not emb_path.exists() or not idx_path.exists():
            return None

        try:
            embeddings = np.load(emb_path)
            with open(idx_path, "r", encoding="utf-8") as f:
                hash_to_row = json.load(f)
        except (ValueError, json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load embedding cache: %s", e)
            return None

        # Stale cache from a different ONNX model dimension must be discarded
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            logger.warning(
                "Embedding cache has wrong shape %s, expected (N, %d) — discarding",
                embeddings.shape,
                EMBEDDING_DIM,
            )
            return None

        n_rows = embeddings.shape[0]
        if any(row < 0 or row >= n_rows for row in hash_to_row.values()):
            logger.warning(
                "Embedding cache index contains out-of-bounds row references — discarding"
            )
            return None

        cache = cls()
        cache._hash_to_row = hash_to_row
        cache._embeddings = embeddings
        cache._size = embeddings.shape[0]

        logger.info("Embedding cache loaded: %d entries", cache._size)
        return cache

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_cached": self._size,
            }
