"""
Code Retriever - Enhanced code search and context engine.
"""

__version__ = "11.4.0"

# Core configuration
from .config import Config, EntityType, OutputFormat

# Models
from .models import CodeEntity, DependencyContext, IndexStats, SearchResult

# Search engine
from .search import SearchEngine

# Cache
from .cache import CacheManager, CachedIndices, ChangeSet

# Formatters
from .formatters import get_formatter

# Utilities
from .token_estimator import count_tokens

__all__ = [
    # Configuration
    "Config",
    "EntityType",
    "OutputFormat",
    # Models
    "CodeEntity",
    "DependencyContext",
    "IndexStats",
    "SearchResult",
    # Search
    "SearchEngine",
    # Cache
    "CacheManager",
    "CachedIndices",
    "ChangeSet",
    # Formatters
    "get_formatter",
    # Utilities
    "count_tokens",
]
