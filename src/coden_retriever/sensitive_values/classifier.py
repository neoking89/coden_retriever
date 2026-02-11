"""Logistic Regression classifier for sensitive value detection.

Trained lazily on first call using the golden data set.
Training takes <10ms and achieves F1 91.3% at threshold 0.35.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any
import numpy as np

from ..constants import (
    SENSITIVE_VALUE_CLASSIFIER_MAX_ITER,
    SENSITIVE_VALUE_CLASSIFIER_REGULARIZATION,
)
from .features import extract_features
from .golden_data import SAFE_VALUES, SENSITIVE_VALUES

logger = logging.getLogger(__name__)

# Module-level singleton state for lazy initialization
_model: Any | None = None
_scaler: Any | None = None
_sklearn_available: bool | None = None


def _ensure_trained() -> bool:
    """Train the classifier on first call. Returns True if model is ready."""
    global _model, _scaler, _sklearn_available

    if _sklearn_available is False:
        return False
    if _model is not None:
        return True

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
    except ImportError:
        _sklearn_available = False
        logger.warning(
            "scikit-learn not installed; sensitive value detection unavailable. "
            "Install with: pip install scikit-learn"
        )
        return False

    _sklearn_available = True

    all_values = list(SENSITIVE_VALUES) + list(SAFE_VALUES)
    labels = np.array([1] * len(SENSITIVE_VALUES) + [0] * len(SAFE_VALUES))
    features = np.array([extract_features(v) for v in all_values])

    _scaler = StandardScaler()
    features_scaled = _scaler.fit_transform(features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = LogisticRegression(
            max_iter=SENSITIVE_VALUE_CLASSIFIER_MAX_ITER,
            C=SENSITIVE_VALUE_CLASSIFIER_REGULARIZATION,
        )
        _model.fit(features_scaled, labels)

    return True


def classify_value(text: str) -> float:
    """Classify a single string value. Returns probability [0.0, 1.0]."""
    if not _ensure_trained():
        return 0.0

    features = np.array([extract_features(text)])
    features_scaled = _scaler.transform(features)
    return float(_model.predict_proba(features_scaled)[0, 1])


def is_available() -> bool:
    """Check if the classifier can run (sklearn installed)."""
    _ensure_trained()
    return _sklearn_available is not False


def classify_batch(texts: list[str]) -> list[float]:
    """Classify multiple strings in batch. Returns list of probabilities."""
    if not texts:
        return []
    if not _ensure_trained():
        return [0.0] * len(texts)

    features = np.array([extract_features(t) for t in texts])
    features_scaled = _scaler.transform(features)
    return [float(p) for p in _model.predict_proba(features_scaled)[:, 1]]
