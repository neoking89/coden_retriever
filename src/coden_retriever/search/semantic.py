"""
Semantic search module.

Implements vector-based semantic code search using Model2Vec.
"""
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from model2vec import StaticModel

from ..models import CodeEntity
from .base import EntitySearchIndex

logger = logging.getLogger(__name__)

# Module-level cache for loaded models (singleton pattern)
# StaticModel is thread-safe and read-only after loading, so safe to share
_model_cache: dict[str, "StaticModel"] = {}

# Directory for mmap cache files (OS filesystem cache keeps these in RAM)
_MMAP_CACHE_DIR = Path(tempfile.gettempdir()) / "coden_retriever_model_cache"


def _get_mmap_cache_path(model_path: str) -> Path:
    """Get the mmap cache file path for a model."""
    # Create a unique cache filename based on model path
    model_path_resolved = Path(model_path).resolve()
    # Use the model directory name as cache key
    cache_name = f"{model_path_resolved.name}_embeddings.npy"
    return _MMAP_CACHE_DIR / cache_name


def _ensure_mmap_cache(model: "StaticModel", model_path: str) -> None:
    """
    Ensure the mmap cache file exists for the model's embeddings.

    Creates the cache file if it doesn't exist. The OS filesystem cache
    will keep frequently accessed mmap'd files in RAM automatically.
    """
    cache_path = _get_mmap_cache_path(model_path)

    if not cache_path.exists():
        logger.info(f"Creating mmap cache at {cache_path}")
        _MMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Save embeddings as .npy file for fast mmap loading
        np.save(cache_path, model.embedding)
        logger.info(f"Mmap cache created ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")


def _load_model_with_mmap(model_path: str) -> "StaticModel":
    """
    Load a Model2Vec model with mmap'd embeddings for cross-process caching.

    The first call creates a .npy cache file. Subsequent calls (even from
    different processes) load the embeddings via mmap, which the OS keeps
    in its filesystem cache (RAM). This means:

    1. First CLI call: loads model normally, creates cache (~2-3s)
    2. Subsequent CLI calls: mmap hits OS cache (~50-100ms)

    The OS automatically manages the cache - frequently accessed files
    stay in RAM until memory pressure forces eviction.
    """
    cache_path = _get_mmap_cache_path(model_path)

    if cache_path.exists():
        # Fast path: load model structure, then mmap the embeddings
        logger.info("Loading model with mmap'd embeddings from cache")
        model = StaticModel.from_pretrained(model_path)

        # Replace embeddings with mmap'd version
        # mmap_mode='r' = read-only, OS caches in RAM
        mmap_embeddings = np.load(cache_path, mmap_mode='r')
        model.embedding = mmap_embeddings

        logger.info("Model loaded with mmap'd embeddings (OS-cached)")
        return model
    else:
        # Slow path: first load, create cache for next time
        logger.info(f"Loading semantic model from {model_path}")
        model = StaticModel.from_pretrained(model_path)

        # Create mmap cache for subsequent loads
        _ensure_mmap_cache(model, model_path)

        logger.info("Semantic model loaded and mmap cache created")
        return model


def get_cached_model(model_path: str) -> "StaticModel":
    """
    Get a cached model instance with mmap-based cross-process caching.

    This function provides two levels of caching:
    1. In-process cache: shares model across SemanticIndex instances in same process
    2. OS filesystem cache: mmap'd embeddings stay in RAM across CLI invocations

    The mmap approach works across separate CLI calls because:
    - np.load(..., mmap_mode='r') maps the file into virtual memory
    - First access loads pages from disk into OS filesystem cache (RAM)
    - Subsequent processes mmap the same file -> hits filesystem cache
    - No disk I/O on subsequent loads (until OS evicts under memory pressure)

    Args:
        model_path: Path to the Model2Vec static model.

    Returns:
        StaticModel: The cached model instance.
    """
    global _model_cache

    # Normalize path for consistent cache keys
    cache_key = str(Path(model_path).resolve())

    if cache_key not in _model_cache:
        if StaticModel is None:
            raise ImportError("model2vec is required. Install with: pip install model2vec")

        # Load with mmap for cross-process caching
        _model_cache[cache_key] = _load_model_with_mmap(model_path)
    else:
        logger.debug(f"Using in-process cached model for {model_path}")

    return _model_cache[cache_key]


def distill_to_static_model(
    source_model: str,
    output_path: Union[str, Path],
    pca_dims: Optional[int] = 256,
    trust_remote_code: bool = False,
) -> "StaticModel":
    """
    Convert a Hugging Face transformer model to a static Model2Vec embedding model.

    This function distills a full transformer model into a lightweight static embedding
    model that can be used for fast semantic search. The distillation process creates
    token-level embeddings that can be averaged for sentence embeddings.

    Args:
        source_model: Hugging Face model identifier (e.g., 'jinaai/jina-embeddings-v2-base-code')
                      or path to a local model directory.
        output_path: Directory path where the distilled model will be saved.
        pca_dims: Number of dimensions for PCA reduction. Set to None to keep original
                  dimensions. Default 256 provides good balance of speed and quality.
        trust_remote_code: Whether to trust remote code when loading models that require
                          custom code execution. Default False.

    Returns:
        StaticModel: The distilled static embedding model ready for use.

    Raises:
        ImportError: If required packages (transformers, model2vec) are not installed.
        ValueError: If the source model cannot be loaded or distillation fails.

    Example:
        >>> model = distill_to_static_model(
        ...     source_model="jinaai/jina-embeddings-v2-base-code",
        ...     output_path="./my_static_model",
        ...     pca_dims=256
        ... )
        >>> embeddings = model.encode(["def hello(): pass"])
    """
    # Lazy imports - these are heavy dependencies only needed for distillation
    try:
        from model2vec.distill import distill_from_model
    except ImportError:
        logger.error("model2vec not installed. Install with: pip install model2vec")
        raise ImportError(
            "model2vec is required for model distillation. "
            "Install with: pip install model2vec"
        )

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers")
        raise ImportError(
            "transformers is required for model distillation. "
            "Install with: pip install transformers"
        )

    output_path = Path(output_path)

    logger.info(f"Loading source model: {source_model}")
    try:
        model = AutoModel.from_pretrained(source_model, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    except Exception as e:
        logger.error(f"Failed to load model '{source_model}': {e}")
        raise ValueError(f"Could not load model '{source_model}': {e}") from e

    logger.info(
        f"Starting distillation (pca_dims={pca_dims}, sif_coefficient=1e-4)..."
    )
    try:
        distilled_model = distill_from_model(
            model=model,
            tokenizer=tokenizer,
            pca_dims=pca_dims,
        )
    except Exception as e:
        logger.error(f"Distillation failed: {e}")
        raise ValueError(f"Model distillation failed: {e}") from e

    logger.info(f"Saving distilled model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    distilled_model.save_pretrained(str(output_path))

    logger.info("Model distillation complete")
    return distilled_model


def load_or_distill_model(
    output_path: Union[str, Path],
    source_model: Optional[str] = None,
    pca_dims: Optional[int] = 256,
    trust_remote_code: bool = False,
    force_distill: bool = False,
) -> "StaticModel":
    """
    Load an existing static model or distill from source if not found.

    This is a convenience function that checks if a distilled model exists at the
    output path and loads it, otherwise distills from the source model.

    Args:
        output_path: Directory path where the static model is/will be saved.
        source_model: Hugging Face model identifier for distillation if model doesn't exist.
                      Required if the model needs to be distilled.
        pca_dims: Number of dimensions for PCA reduction (only used if distilling).
        trust_remote_code: Whether to trust remote code (only used if distilling).
        force_distill: If True, always distill even if model exists. Default False.

    Returns:
        StaticModel: The loaded or newly distilled static embedding model.

    Raises:
        ValueError: If model doesn't exist and source_model is not provided.
        ImportError: If required packages are not installed.
    """
    if StaticModel is None:
        raise ImportError("model2vec is required. Install with: pip install model2vec")

    output_path = Path(output_path)

    if not force_distill and output_path.exists() and (output_path / "model.safetensors").exists():
        logger.info(f"Loading existing static model from: {output_path}")
        return StaticModel.from_pretrained(str(output_path))

    if source_model is None:
        raise ValueError(
            f"No existing model found at '{output_path}' and no source_model provided. "
            "Please provide a source_model to distill from."
        )

    return distill_to_static_model(
        source_model=source_model,
        output_path=output_path,
        pca_dims=pca_dims,
        trust_remote_code=trust_remote_code,
    )


class SemanticIndex(EntitySearchIndex):
    """
    Semantic search index using Model2Vec for vector embeddings.

    Uses in-memory numpy arrays for embeddings (No vector DB needed).
    Uses cached model loading to share models across instances (saves memory and load time).

    Implements the EntitySearchIndex interface for compatibility with SearchEngine.
    Unlike BM25Index which works with raw text, this class works with CodeEntity
    objects to leverage entity metadata for richer semantic understanding.
    """

    def __init__(self, model_path: str):
        """
        Initialize the semantic index.

        Args:
            model_path: Path to the Model2Vec static model.
        """
        self.model_path = model_path
        self._model = None
        self._embeddings = None
        self._node_ids = []

    @classmethod
    def from_huggingface(
        cls,
        source_model: str,
        output_path: Union[str, Path],
        pca_dims: Optional[int] = 256,
        trust_remote_code: bool = False,
        force_distill: bool = False,
    ) -> "SemanticIndex":
        """
        Create a SemanticIndex by distilling a Hugging Face model to static embeddings.

        This factory method handles the entire workflow of checking for an existing
        distilled model, distilling if necessary, and creating the SemanticIndex.

        Args:
            source_model: Hugging Face model identifier (e.g., 'jinaai/jina-embeddings-v2-base-code').
            output_path: Directory where the distilled model will be saved/loaded from.
            pca_dims: Number of dimensions for PCA reduction. Default 256.
            trust_remote_code: Whether to trust remote code for models like Jina. Default False.
            force_distill: If True, re-distill even if model exists. Default False.

        Returns:
            SemanticIndex: Configured semantic index ready for use.

        Example:
            >>> index = SemanticIndex.from_huggingface(
            ...     source_model="jinaai/jina-embeddings-v2-base-code",
            ...     output_path="./models/code_embeddings",
            ...     pca_dims=256,
            ...     trust_remote_code=True
            ... )
            >>> index.index(entities)
            >>> scores = index.score_all("find authentication function")
        """
        output_path = Path(output_path)

        # Distill or load the model
        load_or_distill_model(
            output_path=output_path,
            source_model=source_model,
            pca_dims=pca_dims,
            trust_remote_code=trust_remote_code,
            force_distill=force_distill,
        )

        return cls(model_path=str(output_path))

    def _load_model(self):
        """Lazy load the Model2Vec model using the global cache."""
        if self._model is None:
            # Use cached model loading - shares model across instances
            self._model = get_cached_model(self.model_path)

    def index(self, entities: Dict[str, CodeEntity]) -> None:
        """
        Generate embeddings for all entities.

        Args:
            entities: Dictionary mapping node_id to CodeEntity.
        """
        if not entities:
            logger.warning("No entities provided for semantic indexing")
            return

        # Lazy load model
        self._load_model()

        # Extract node IDs and texts
        self._node_ids = list(entities.keys())
        texts = [entities[nid].semantic_searchable_text for nid in self._node_ids]

        logger.info(f"Generating semantic embeddings for {len(texts)} entities...")

        # Create embeddings
        assert self._model is not None  # _load_model() ensures this
        self._embeddings = self._model.encode(texts)

        # Normalize embeddings for cosine similarity (||v|| = 1)
        # This allows us to use dot product instead of computing cosine each time
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = self._embeddings / norms

        logger.info("Semantic indexing complete")

    def score_all(self, query: str) -> Dict[str, float]:
        """
        Return cosine similarity scores for all nodes.

        Args:
            query: Natural language query string.

        Returns:
            Dictionary mapping node_id to similarity score (0-1 range).
        """
        if self._embeddings is None or self._model is None:
            logger.warning("Semantic index not initialized. Call index() first.")
            return {}

        if not query.strip():
            return {}

        # Encode query
        assert self._model is not None  # Checked above
        query_vec = self._model.encode([query])[0]

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            logger.warning("Query embedding is zero vector")
            return {}
        query_vec = query_vec / query_norm

        # Compute cosine similarity (dot product of normalized vectors)
        scores = np.dot(self._embeddings, query_vec)

        # Filter out low scores (cutoff threshold to reduce noise)
        # Map back to node_ids
        return {
            nid: float(score)
            for nid, score in zip(self._node_ids, scores)
            if score > 0.1  # Only return scores above threshold
        }
