"""
JSON formatter module.

Formats search results as JSON.
"""
import json
from pathlib import Path

from ..models import SearchResult
from .base import OutputFormatter


class JSONFormatter(OutputFormatter):
    """JSON output format - for programmatic consumption."""

    def format_results(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        filtered = self.filter_by_token_budget(results, token_budget, show_deps)
        output = []

        for result in filtered:
            rel_path = self.get_relative_path(result.entity.file_path, root)
            code = result.entity.get_context_snippet()

            item = {
                "name": result.entity.name,
                "type": result.entity.entity_type,
                "file": rel_path,
                "lines": [result.entity.line_start, result.entity.line_end],
                "score": round(result.score, 4),
                "components": {k: round(v, 4) for k, v in result.components.items()},
                "code": code,
            }

            if show_deps and result.dependency_context:
                item["dependencies"] = {
                    "callers": [{"name": name, "type": etype} for _, name, etype, _ in result.dependency_context.callers],
                    "callees": [{"name": name, "type": etype} for _, name, etype, _ in result.dependency_context.callees],
                }

            output.append(item)

        return json.dumps(output, indent=2)
