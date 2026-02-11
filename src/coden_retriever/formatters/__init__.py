"""Formatters package for coden-retriever."""

from ..config import OutputFormat
from .base import OutputFormatter
from .cli_metrics import (
    BaseCLIMetricFormatter,
    CLIMetricFormatter,
    SeverityTier,
)
from .clone_formatter import CloneFormatter
from .dead_code_formatter import DeadCodeFormatter
from .directory_tree_formatter import generate_shallow_tree
from .propagation_formatter import PropagationFormatter
from .sensitive_value_formatter import SensitiveValueFormatter
from .tramp_data_formatter import TrampDataFormatter
from .json_formatter import JSONFormatter
from .markdown_formatter import MarkdownFormatter
from .tree_formatter import TreeFormatter
from .xml_formatter import XMLFormatter


def get_formatter(format_type: OutputFormat) -> OutputFormatter:
    """Factory function to get the appropriate formatter."""
    formatters: dict[OutputFormat, type[OutputFormatter]] = {
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
    "CLIMetricFormatter",
    "BaseCLIMetricFormatter",
    "SeverityTier",
    "CloneFormatter",
    "DeadCodeFormatter",
    "PropagationFormatter",
    "SensitiveValueFormatter",
    "TrampDataFormatter",
]
