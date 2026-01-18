"""Utility modules for coden-retriever."""

from .optional_deps import (
    MissingDependencyError,
    get_numpy,
    is_feature_available,
    require_feature,
)

__all__ = [
    "MissingDependencyError",
    "get_numpy",
    "is_feature_available",
    "require_feature",
]
