"""
Base classes for search indices.

Defines abstract interfaces for search algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict

from ..models import CodeEntity


class SearchIndex(ABC):
    """
    Abstract base class for search indices.

    All search index implementations (BM25, Semantic, future TF-IDF, etc.)
    should implement this interface to be usable with SearchEngine.

    This follows the Strategy Pattern - different search algorithms can be
    swapped without changing the SearchEngine's search logic.
    """

    @abstractmethod
    def index(self, documents: Dict[str, str]) -> None:
        """
        Build the search index from documents.

        Args:
            documents: Dictionary mapping document_id (node_id) to document text.
                       The text should be the searchable representation of the entity.

        Note:
            After calling index(), the implementation should be ready to
            accept score() and score_all() calls.
        """
        pass

    @abstractmethod
    def score(self, query: str, doc_id: str) -> float:
        """
        Score a single document against a query.

        Args:
            query: The search query string.
            doc_id: The document identifier to score.

        Returns:
            Relevance score (higher = more relevant). Score scale varies
            by implementation but should be non-negative.
        """
        pass

    @abstractmethod
    def score_all(self, query: str) -> Dict[str, float]:
        """
        Score all indexed documents against a query.

        Args:
            query: The search query string.

        Returns:
            Dictionary mapping document_id to relevance score.
            Only documents with score > 0 should be included.
        """
        pass


class EntitySearchIndex(ABC):
    """
    Abstract base class for search indices that work with CodeEntity objects.

    Unlike SearchIndex which works with raw text documents, EntitySearchIndex
    has access to the full CodeEntity structure, enabling richer indexing
    (e.g., considering entity type, parent class, etc.).

    Used by SemanticIndex which needs entity metadata for context.
    """

    @abstractmethod
    def index(self, entities: Dict[str, CodeEntity]) -> None:
        """
        Build the search index from code entities.

        Args:
            entities: Dictionary mapping node_id to CodeEntity objects.
                      Implementations can extract relevant text/features
                      from entities for indexing.

        Note:
            After calling index(), the implementation should be ready to
            accept score_all() calls.
        """
        pass

    @abstractmethod
    def score_all(self, query: str) -> Dict[str, float]:
        """
        Score all indexed entities against a query.

        Args:
            query: The search query string (natural language or keywords).

        Returns:
            Dictionary mapping node_id to relevance score.
            Only entities with score > threshold should be included.
        """
        pass
