"""
Cache module for code retriever.

Provides persistent caching of search indices for fast startup times.
"""
from .embedding_cache import EmbeddingCache, encode_with_cache
from .layout import LITE_LAYOUT, STATIC_LAYOUT, CacheLayout
from .manager import CacheManager
from .models import CachedIndices, ChangeSet, LiteCachedIndices

__all__ = [
    "LITE_LAYOUT",
    "STATIC_LAYOUT",
    "CacheLayout",
    "CacheManager",
    "CachedIndices",
    "ChangeSet",
    "EmbeddingCache",
    "LiteCachedIndices",
    "encode_with_cache",
]
