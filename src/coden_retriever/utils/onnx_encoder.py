"""Shared ONNX embedding encoder using all-MiniLM-L6-v2 (INT8 quantized).

Provides the single source of truth for text-to-embedding conversion across
the codebase: semantic search, clone detection, echo comments, tool filtering.

Returns L2-normalized 384-dim embeddings suitable for cosine similarity via
dot product.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from .progress import ProgressCallback

logger = logging.getLogger(__name__)

# ONNX model directory (all-MiniLM-L6-v2 INT8 quantized, ~23MB)
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "embeddings" / "minilm_onnx"
_ONNX_MODEL_FILE = "model_quantized.onnx"
_TOKENIZER_FILE = "tokenizer.json"
# MiniLM-L6-v2 max sequence length
_MAX_TOKEN_LENGTH = 256

# MiniLM-L6-v2 output embedding dimension (384-d vectors, L2-normalized)
EMBEDDING_DIM = 384

# Small epsilon to avoid division by zero in mean pooling and L2 normalization.
# 1e-9 is well below float32 precision floor so it never distorts real vectors.
_NORM_EPSILON = 1e-9

# Mini-batch size for progress-aware encoding. 64 texts per session.run()
# adds ~2ms overhead per batch call — negligible vs the ~400ms inference time.
_ENCODE_BATCH_SIZE = 64

# Module-level caches for ONNX session and tokenizer (loaded once, reused)
_onnx_session_cache: dict[str, Any] = {}
_onnx_tokenizer_cache: dict[str, Any] = {}


def _get_onnx_resources(model_dir: str | Path) -> tuple[ort.InferenceSession, Tokenizer]:
    """Lazy-load and cache ONNX session + tokenizer for a model directory."""
    cache_key = str(Path(model_dir).resolve())

    if cache_key not in _onnx_session_cache:
        model_path = Path(model_dir) / _ONNX_MODEL_FILE
        tokenizer_path = Path(model_dir) / _TOKENIZER_FILE

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
        )

        tok = Tokenizer.from_file(str(tokenizer_path))
        tok.enable_truncation(max_length=_MAX_TOKEN_LENGTH)
        tok.enable_padding(length=_MAX_TOKEN_LENGTH)

        _onnx_session_cache[cache_key] = session
        _onnx_tokenizer_cache[cache_key] = tok
        logger.info(f"Loaded ONNX model from {model_path}")

    return _onnx_session_cache[cache_key], _onnx_tokenizer_cache[cache_key]


def _run_inference(texts: list[str], session: ort.InferenceSession, tok: Tokenizer) -> np.ndarray:
    """Run ONNX inference on a single batch. Returns L2-normalized embeddings."""
    encodings = tok.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    input_names = [inp.name for inp in session.get_inputs()]
    feeds: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

    last_hidden = session.run(None, feeds)[0]

    # Mean pooling with attention mask
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    pooled = np.sum(last_hidden * mask_expanded, axis=1)
    pooled = pooled / np.clip(mask_expanded.sum(axis=1), a_min=_NORM_EPSILON, a_max=None)

    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, a_min=_NORM_EPSILON, a_max=None)


def onnx_encode(
    texts: list[str],
    model_dir: str | Path | None = None,
    on_batch_done: "ProgressCallback | None" = None,
) -> np.ndarray:
    """Encode texts into L2-normalized embeddings using MiniLM ONNX.

    Texts are always processed in mini-batches of _ENCODE_BATCH_SIZE so peak memory
    is bounded by the batch size, not the corpus size. A single session.run() over the
    whole corpus allocates activation/attention tensors sized to N: measured ~10.9 MB of
    commit per text at the 256-token padding, so a cold whole-repo run (~7k texts) wanted
    ~77 GB and thrashed the box. Batching caps every session.run() at _ENCODE_BATCH_SIZE
    regardless of N. *on_batch_done*, when given, fires once per mini-batch for progress.
    """
    model_dir = model_dir or _DEFAULT_MODEL_DIR
    session, tok = _get_onnx_resources(model_dir)

    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    parts: list[np.ndarray] = []
    for i in range(0, len(texts), _ENCODE_BATCH_SIZE):
        batch = texts[i : i + _ENCODE_BATCH_SIZE]
        parts.append(_run_inference(batch, session, tok))
        if on_batch_done:
            on_batch_done(len(batch))
    return np.vstack(parts)
