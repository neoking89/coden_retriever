"""SVM-RBF classifier for sensitive value detection.

Trained lazily on first call using the golden data set.
Training takes ~5ms. SVM-RBF captures non-linear feature interactions
that linear models miss, improving validation F1 from 0.86 to 0.92.

Source modules (features.py, golden_data.py) are hot-reloaded when
modified on disk so the daemon picks up changes without restart.
"""
from __future__ import annotations

import importlib
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..constants import SENSITIVE_VALUE_CLASSIFIER_REGULARIZATION
# Module-level imports for hot-reload via importlib.reload()
from . import features as _features_mod
from . import golden_data as _golden_data_mod

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Module-level singleton state for lazy initialization.
# sklearn is imported inside _ensure_trained() (not at module top) so MCP
# server startup doesn't pay ~3s + drag onnxruntime/numpy dependencies.
# `mcp/sensitive_values.py::detect_sensitive_values_tool` calls the public
# `ensure_trained()` wrapper on the main event loop BEFORE dispatching to
# asyncio.to_thread, preserving the Windows DLL-load-must-be-main-thread
# guarantee that the previous module-top import enforced.
_model: SVC | None = None
_scaler: StandardScaler | None = None

# Track source file mtimes so the daemon can hot-reload on disk changes
_SOURCE_DIR = Path(__file__).parent
_last_features_mtime: float = 0.0
_last_golden_mtime: float = 0.0


def _source_files_changed() -> bool:
    """Check if classifier source files have been modified on disk.

    Compares mtime of features.py and golden_data.py against last-seen
    values.  Updates stored mtimes on every call so subsequent calls
    detect the *next* change.
    """
    global _last_features_mtime, _last_golden_mtime

    try:
        feat_mtime = (_SOURCE_DIR / "features.py").stat().st_mtime
        gold_mtime = (_SOURCE_DIR / "golden_data.py").stat().st_mtime
    except OSError:
        return False

    changed = (
        feat_mtime != _last_features_mtime
        or gold_mtime != _last_golden_mtime
    )
    _last_features_mtime = feat_mtime
    _last_golden_mtime = gold_mtime
    return changed


def _reload_source_modules() -> None:
    """Reload feature extraction and golden data modules from disk."""
    importlib.reload(_features_mod)
    importlib.reload(_golden_data_mod)
    logger.info("Reloaded sensitive value classifier source modules")


def _ensure_trained() -> bool:
    """Train the classifier on first call or retrain when sources change."""
    global _model, _scaler

    sources_changed = _source_files_changed()

    if _model is not None and not sources_changed:
        return True

    if sources_changed and _model is not None:
        _reload_source_modules()

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    all_values = (
        list(_golden_data_mod.SENSITIVE_VALUES)
        + list(_golden_data_mod.SAFE_VALUES)
    )
    labels = np.array(
        [1] * len(_golden_data_mod.SENSITIVE_VALUES)
        + [0] * len(_golden_data_mod.SAFE_VALUES)
    )
    features = np.array([
        _features_mod.extract_features(v) for v in all_values
    ])

    _scaler = StandardScaler()
    features_scaled = _scaler.fit_transform(features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = SVC(
            C=SENSITIVE_VALUE_CLASSIFIER_REGULARIZATION,
            kernel="rbf",
            probability=True,
            random_state=42,
        )
        _model.fit(features_scaled, labels)

    return True


def ensure_trained() -> bool:
    """Public prewarm wrapper. Call from the main event loop before any
    thread/worker uses classify_value/classify_batch — on Windows, first
    sklearn DLL load inside asyncio.to_thread deadlocks the MCP subprocess.
    """
    return _ensure_trained()


def classify_value(text: str) -> float:
    """Classify a single string value. Returns probability [0.0, 1.0]."""
    if not _ensure_trained():
        return 0.0

    assert _scaler is not None and _model is not None
    features = np.array([_features_mod.extract_features(text)])
    features_scaled = _scaler.transform(features)
    return float(_model.predict_proba(features_scaled)[0, 1])


def classify_batch(texts: list[str]) -> list[float]:
    """Classify multiple strings in batch. Returns list of probabilities."""
    if not texts:
        return []
    if not _ensure_trained():
        return [0.0] * len(texts)

    assert _scaler is not None and _model is not None
    features = np.array([
        _features_mod.extract_features(t) for t in texts
    ])
    features_scaled = _scaler.transform(features)
    return [float(p) for p in _model.predict_proba(features_scaled)[:, 1]]
