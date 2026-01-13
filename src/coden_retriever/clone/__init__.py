"""Clone detection module.

Provides syntactic (line-by-line) and semantic clone detection.
"""

from .combined import detect_clones_combined
from .semantic import detect_clones_semantic
from .syntactic import detect_clones_syntactic

__all__ = [
    "detect_clones_combined",
    "detect_clones_semantic",
    "detect_clones_syntactic",
]
