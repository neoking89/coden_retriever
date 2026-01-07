"""
Tree formatter module.

Formats search results as a hierarchical tree view with Rich colors and clickable links.
"""
from pathlib import Path

from ..models import SearchResult
from .base import OutputFormatter
from .terminal_style import get_terminal_style


class TreeFormatter(OutputFormatter):
    """Tree output format with Rich colors and clickable hyperlinks."""

    def format_results(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        return self.format_map(results, root, token_budget, show_deps)

    def format_map(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        filtered = self.filter_by_token_budget(
            results, token_budget, show_deps, base_overhead=100, per_item_overhead=15
        )

        if not filtered:
            return "No results found."

        # Get Rich-based terminal styling
        style = get_terminal_style()

        # Calculate max score for color normalization
        global_max_score = max(r.score for r in filtered)

        # Group results by file
        files_to_results: dict[str, list[SearchResult]] = {}
        file_order: list[str] = []
        rel_to_abs: dict[str, str] = {}

        for result in filtered:
            rel_path = self.get_relative_path(result.entity.file_path, root)
            if rel_path not in files_to_results:
                files_to_results[rel_path] = []
                file_order.append(rel_path)
                rel_to_abs[rel_path] = result.entity.file_path
            files_to_results[rel_path].append(result)

        output = [f"Repository Context Map ({len(filtered)} definitions)\n{'=' * 60}"]

        # Sort files by max score
        if len(filtered) >= 2:
            is_descending = filtered[0].score >= filtered[-1].score
            file_order.sort(
                key=lambda path: max(r.score for r in files_to_results[path]),
                reverse=is_descending
            )

        for rel_path in file_order:
            file_results = sorted(files_to_results[rel_path], key=lambda r: r.entity.line_start)
            file_max_score = max(r.score for r in file_results)
            abs_path = rel_to_abs[rel_path]

            # Clickable, colored file header with rank - all in one append
            file_header = style.format_file_header(
                rel_path=rel_path,
                abs_path=abs_path,
                score=file_max_score,
                max_score=global_max_score,
            )
            colored_rank = style.format_rank(file_max_score, global_max_score)
            output.append(f"\n{file_header}\n   (Rank: {colored_rank})\n")

            for r in file_results:
                # Clickable, colored entity
                entity_line = style.format_entity(
                    name=r.entity.name,
                    entity_type=r.entity.entity_type,
                    file_path=r.entity.file_path,
                    line=r.entity.line_start,
                    score=r.score,
                    max_score=global_max_score,
                )

                dep_line = ""
                if show_deps and r.dependency_context and not r.dependency_context.is_empty():
                    dep_line = f"\n   {r.dependency_context.format_compact()}"

                snippet = r.entity.get_context_snippet()
                lines = snippet.splitlines()
                start_line = r.entity.line_start

                # Build entity block with clickable line numbers in one go
                numbered_lines = [
                    f"  {style.format_line_number(start_line + i, r.entity.file_path)}: {line}"
                    for i, line in enumerate(lines)
                ]
                output.append(f"{entity_line}{dep_line}\n" + "\n".join(numbered_lines) + "\n")

        return "\n".join(output)
