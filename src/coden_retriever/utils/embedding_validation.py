"""Strict pre-flight validator for a custom semantic-model directory.

Used by `/config set semantic_model_path <dir>` to hard-reject directories
that wouldn't be loadable, so the failure surfaces at config time rather
than later as a silent fallback to BM25.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .onnx_encoder import EMBEDDING_DIM, _ONNX_MODEL_FILE, _TOKENIZER_FILE


def validate_embedding_model_dir(path: Path) -> tuple[bool, str]:
    """Validate that *path* is a usable embedding model directory.

    Returns (ok, error_message). error_message is empty on success.
    """
    for required in (_ONNX_MODEL_FILE, _TOKENIZER_FILE):
        if not (path / required).is_file():
            return False, f"missing {required} in {path}"

    try:
        session = ort.InferenceSession(
            str(path / _ONNX_MODEL_FILE), providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        return False, f"failed to load {_ONNX_MODEL_FILE}: {exc}"

    try:
        tok = Tokenizer.from_file(str(path / _TOKENIZER_FILE))
    except Exception as exc:
        return False, f"failed to load {_TOKENIZER_FILE}: {exc}"

    try:
        last_hidden = _run_one_token_inference(session, tok)
    except Exception as exc:
        return False, f"1-token inference failed: {exc}"

    actual_dim = last_hidden.shape[-1]
    if actual_dim != EMBEDDING_DIM:
        return False, (
            f"embedding dimension mismatch: model produces {actual_dim}, "
            f"the index requires {EMBEDDING_DIM}"
        )

    return True, ""


def _run_one_token_inference(session: ort.InferenceSession, tokenizer: Tokenizer) -> np.ndarray:
    """Run one tiny inference to confirm the model produces a tensor."""
    encoded = tokenizer.encode("x")
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    input_names = [inp.name for inp in session.get_inputs()]
    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
    return session.run(None, feeds)[0]
