"""
Semantic search module.

Implements vector-based semantic code search using MiniLM ONNX embeddings.
"""
import logging
from typing import Dict, TYPE_CHECKING

import numpy as np

from ..config import Config
from ..models import CodeEntity
from ..utils.onnx_encoder import onnx_encode
from .base import EntitySearchIndex

if TYPE_CHECKING:
    from ..utils.progress import ProgressCallback

logger = logging.getLogger(__name__)


class SemanticIndex(EntitySearchIndex):
    """
    Semantic search index using MiniLM ONNX for vector embeddings.

    Uses in-memory numpy arrays for embeddings (No vector DB needed).
    Embedding computation is handled by the shared onnx_encoder module
    which caches the ONNX session internally.

    Implements the EntitySearchIndex interface for compatibility with SearchEngine.
    Unlike BM25Index which works with raw text, this class works with CodeEntity
    objects to leverage entity metadata for richer semantic understanding.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the semantic index.

        Args:
            model_path: Directory containing the ONNX embedding model. None means
                use the bundled MiniLM model (resolved lazily by onnx_encode).
        """
        self._embeddings: np.ndarray | None = None
        self._node_ids: list[str] = []
        self._model_path = model_path

    def index(
        self,
        entities: Dict[str, CodeEntity],
        on_batch_done: "ProgressCallback | None" = None,
    ) -> None:
        """
        Generate embeddings for all entities.

        Args:
            entities: Dictionary mapping node_id to CodeEntity.
            on_batch_done: Optional callback receiving the count of texts
                encoded in each batch, used to advance a progress bar.
        """
        if not entities:
            logger.warning("No entities provided for semantic indexing")
            return

        # Extract node IDs and texts
        self._node_ids = list(entities.keys())
        texts = [entities[nid].semantic_searchable_text for nid in self._node_ids]

        logger.info(f"Generating semantic embeddings for {len(texts)} entities...")

        # onnx_encode returns L2-normalized embeddings
        self._embeddings = onnx_encode(texts, model_dir=self._model_path, on_batch_done=on_batch_done)

        logger.info("Semantic indexing complete")

    def score_all(self, query: str) -> Dict[str, float]:
        """
        Return cosine similarity scores for all nodes.

        Args:
            query: Natural language query string.

        Returns:
            Dictionary mapping node_id to similarity score (0-1 range).
        """
        if self._embeddings is None:
            logger.warning("Semantic index not initialized. Call index() first.")
            return {}

        if not query.strip():
            return {}

        # onnx_encode returns L2-normalized vectors
        query_vec = onnx_encode([query], model_dir=self._model_path)[0]

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            logger.warning("Query embedding is zero vector")
            return {}

        # Dot product of normalized vectors = cosine similarity
        scores = np.dot(self._embeddings, query_vec)

        # Filter out low scores (cutoff threshold to reduce noise)
        return {
            nid: float(score)
            for nid, score in zip(self._node_ids, scores)
            if score > Config.SEMANTIC_SCORE_THRESHOLD
        }
