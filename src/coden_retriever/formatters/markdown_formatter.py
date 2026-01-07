"""
Markdown formatter module.

Formats search results as Markdown.
"""
from pathlib import Path

from ..models import SearchResult
from .base import OutputFormatter


class MarkdownFormatter(OutputFormatter):
    """Markdown output format - readable and LLM-friendly."""

    def format_results(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        filtered = self.filter_by_token_budget(results, token_budget, show_deps, base_overhead=50)
        output = ["# Search Results\n"]

        for result in filtered:
            rel_path = self.get_relative_path(result.entity.file_path, root)
            code = result.entity.get_context_snippet()

            output.append(f"## {result.entity.name}")
            output.append(f"**File:** `{rel_path}` (lines {result.entity.line_start}-{result.entity.line_end})")
            output.append(f"**Type:** {result.entity.entity_type} | **Score:** {result.score:.3f}")

            if show_deps and result.dependency_context and not result.dependency_context.is_empty():
                output.append(f"**Dependencies:** {result.dependency_context.format_compact()}")

            output.append(f"```{result.entity.language}")
            output.append(code)
            output.append("```\n")

        return "\n".join(output)
