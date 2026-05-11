"""Language support package for coden-retriever."""

from .definitions import LANGUAGE_MAP, LANGUAGE_QUERIES, language_for_path
from .loader import LanguageLoader

__all__ = [
    "LANGUAGE_MAP",
    "LANGUAGE_QUERIES",
    "LanguageLoader",
    "language_for_path",
]
