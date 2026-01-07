"""
Cache module for code retriever.

Provides persistent caching of search indices for fast startup times.
"""
from .manager import CacheManager
from .models import CachedIndices, ChangeSet

__all__ = ["CacheManager", "CachedIndices", "ChangeSet"]
