"""
Base formatter module.

Defines the abstract interface for output formatters.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from ..models import SearchResult
from ..token_estimator import count_tokens


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @staticmethod
    def get_relative_path(file_path: str, root: Path) -> str:
        """Get relative path from root, with fallback to filename only."""
        try:
            return str(Path(file_path).relative_to(root))
        except ValueError:
            return Path(file_path).name

    @staticmethod
    def filter_by_token_budget(
        results: list[SearchResult],
        token_budget: int | None,
        show_deps: bool = False,
        base_overhead: int = 100,
        per_item_overhead: int = 30
    ) -> list[SearchResult]:
        """Filter results to fit within token budget.

        Args:
            results: List of search results to filter
            token_budget: Maximum token budget, or None for unlimited
            show_deps: Whether dependencies are shown (adds to token count)
            base_overhead: Base token overhead for the output format
            per_item_overhead: Per-item token overhead (formatting, metadata)

        Returns:
            Filtered list of results that fit within budget
        """
        if token_budget is None:
            return results

        included = []
        used_tokens = base_overhead

        for result in results:
            code = result.entity.get_context_snippet()
            item_tokens = count_tokens(code) + per_item_overhead

            if show_deps and result.dependency_context and not result.dependency_context.is_empty():
                item_tokens += count_tokens(result.dependency_context.format_compact())

            if used_tokens + item_tokens > token_budget:
                continue

            used_tokens += item_tokens
            included.append(result)

        return included

    @abstractmethod
    def format_results(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        """Format search results for output."""
        pass

    def format_map(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        """Format repository map for output. Default delegates to format_results."""
        return self.format_results(results, root, token_budget, show_deps)
