"""Argparse builder for `coden architecture`."""
from __future__ import annotations

import argparse

from ..architecture.core.constants import TOP_FINDINGS_DEFAULT
from .utils import DefaultValueHelpFormatter


def create_architecture_parser() -> argparse.ArgumentParser:
    """Parser for `coden architecture <path> [--top N] [--exclude] [--json] [--lang]`."""
    parser = argparse.ArgumentParser(
        prog="coden architecture",
        description=(
            "Audit a codebase for architectural drift: cycles, kitchen-sink "
            "packages, oversized files, shallow packages, and imports moved "
            "inside functions. Read-only. Exits 1 if cycles are found."
        ),
        formatter_class=DefaultValueHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Directory to audit (e.g. src/coden_retriever).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_FINDINGS_DEFAULT,
        help="Cap each section at N rows.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated directory names to skip on top of repo-default excludes.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of the human view.",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Force a language adapter (e.g. python, stub). Auto-detected if omitted.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging to stderr.",
    )
    return parser
