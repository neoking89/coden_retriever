"""
Search result models.

Contains data structures for search results and indexing statistics.
"""
from dataclasses import dataclass, field

from .entities import CodeEntity, DependencyContext


@dataclass
class SearchResult:
    """A single search result with scoring breakdown."""

    rank: int
    entity: CodeEntity
    score: float
    components: dict[str, float] = field(default_factory=dict)
    dependency_context: DependencyContext | None = None

    def __repr__(self) -> str:
        return f"<Rank {self.rank}: {self.entity.name} (Score: {self.score:.4f})>"

    @property
    def explanation(self) -> str:
        """Human-readable explanation of scoring."""
        parts = []
        for key, val in sorted(self.components.items()):
            if val > 0:
                parts.append(f"{key}={val:.3f}")
        return ", ".join(parts) if parts else "baseline"


@dataclass
class IndexStats:
    """Statistics about the indexed repository."""

    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    total_entities: int = 0
    total_edges: int = 0
    entities_by_type: dict[str, int] = field(default_factory=dict)
    entities_by_language: dict[str, int] = field(default_factory=dict)
    index_time_ms: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"Files: {self.parsed_files}/{self.total_files} parsed ({self.failed_files} failed)",
            f"Entities: {self.total_entities}",
            f"Edges: {self.total_edges}",
            f"By type: {dict(self.entities_by_type)}",
            f"By language: {dict(self.entities_by_language)}",
        ]
        return "\n".join(lines)
