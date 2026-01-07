"""Formatters package for coden-retriever."""

from ..config import OutputFormat
from .base import OutputFormatter
from .directory_tree_formatter import generate_shallow_tree
from .json_formatter import JSONFormatter
from .markdown_formatter import MarkdownFormatter
from .tree_formatter import TreeFormatter
from .xml_formatter import XMLFormatter


def get_formatter(format_type: OutputFormat) -> OutputFormatter:
    """Factory function to get the appropriate formatter."""
    formatters = {
        OutputFormat.XML: XMLFormatter,
        OutputFormat.MARKDOWN: MarkdownFormatter,
        OutputFormat.TREE: TreeFormatter,
        OutputFormat.JSON: JSONFormatter,
    }
    return formatters[format_type]()


__all__ = [
    "OutputFormatter",
    "XMLFormatter",
    "MarkdownFormatter",
    "TreeFormatter",
    "JSONFormatter",
    "get_formatter",
    "generate_shallow_tree",
]
