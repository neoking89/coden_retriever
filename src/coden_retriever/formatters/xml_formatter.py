"""
XML formatter module.

Formats search results as XML.
"""
from pathlib import Path

from ..models import SearchResult
from .base import OutputFormatter


class XMLFormatter(OutputFormatter):
    """XML output format - structured and easy to parse."""

    def format_results(
        self,
        results: list[SearchResult],
        root: Path,
        token_budget: int | None,
        show_deps: bool = False
    ) -> str:
        filtered = self.filter_by_token_budget(results, token_budget, show_deps, base_overhead=50)
        output = ['<search_results>']

        for result in filtered:
            rel_path = self.get_relative_path(result.entity.file_path, root)
            code = result.entity.get_context_snippet()
            code_escaped = (code
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

            deps_xml = ""
            if show_deps and result.dependency_context and not result.dependency_context.is_empty():
                deps_xml = f"\n    <deps>{result.dependency_context.format_compact()}</deps>"

            output.append(f'''  <item file="{rel_path}" lang="{result.entity.language}" type="{result.entity.entity_type}">
    <name>{result.entity.name}</name>
    <lines>{result.entity.line_start}-{result.entity.line_end}</lines>
    <score value="{result.score:.4f}">{result.explanation}</score>{deps_xml}
    <code>{code_escaped}</code>
  </item>''')

        output.append("</search_results>")
        return "\n".join(output)
