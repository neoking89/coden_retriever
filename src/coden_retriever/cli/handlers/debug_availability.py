"""CLI handler for debug adapter availability checks."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rich.console import Console
from rich.markup import escape

from ...mcp.adapters import DebugAvailability, check_debug_availability
from ..utils import DefaultValueHelpFormatter

_ALL_LANGUAGES = "all"
_FORMAT_TEXT = "text"
_FORMAT_JSON = "json"
_STATUS_AVAILABLE = "available"
_STATUS_UNAVAILABLE = "unavailable"
_ICON_AVAILABLE = "✓"
_ICON_UNAVAILABLE = "✗"
_STYLE_AVAILABLE = "green"
_STYLE_UNAVAILABLE = "red"
_STYLE_HINT = "cyan"
_STYLE_LABEL = "bold"

# Exit codes: shell callers (CI matrix, status checks) rely on a non-zero
# exit to signal "at least one adapter is unavailable" without having to
# parse stdout. 0 = every requested adapter usable, 1 = at least one gap.
_EXIT_OK = 0
_EXIT_UNAVAILABLE = 1


def handle_debug_availability_command(
    args: list[str], console: Console | None = None,
) -> int:
    """Handle `coden debug-availability [language]`.

    ``console`` is injected so tests can capture output without monkey-patching
    a module-global — the CLI default falls back to a fresh ``Console()``.
    """

    parser = _create_debug_availability_parser()
    parsed = parser.parse_args(args)
    result = check_debug_availability(parsed.language)
    reports = _coerce_reports(result)

    if parsed.output_format == _FORMAT_JSON:
        _print_json_result(result)
    else:
        _print_text_result(reports, console or Console())

    return _EXIT_OK if all(report.can_debug for report in reports) else _EXIT_UNAVAILABLE


def _create_debug_availability_parser() -> argparse.ArgumentParser:
    """Build the parser for the debug-availability subcommand."""

    parser = argparse.ArgumentParser(
        prog="coden debug-availability",
        description="Check whether debugging is available for one language or all adapters.",
        formatter_class=DefaultValueHelpFormatter,
    )
    parser.add_argument(
        "language",
        nargs="?",
        default=_ALL_LANGUAGES,
        help="Language or adapter name (default: all)",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=[_FORMAT_TEXT, _FORMAT_JSON],
        default=_FORMAT_TEXT,
        help="Output format",
    )
    return parser


def _coerce_reports(
    result: DebugAvailability | tuple[DebugAvailability, ...],
) -> tuple[DebugAvailability, ...]:
    """Normalize the registry result into a tuple for text rendering."""

    if isinstance(result, tuple):
        return result
    return (result,)


def _print_json_result(
    result: DebugAvailability | tuple[DebugAvailability, ...],
) -> None:
    """Render the command result as JSON."""

    if isinstance(result, tuple):
        payload: object = [asdict(report) for report in result]
    else:
        payload = asdict(result)
    print(json.dumps(payload, indent=2))


def _print_text_result(
    reports: tuple[DebugAvailability, ...], console: Console,
) -> None:
    """Render the command result in a human-readable text format."""

    for index, report in enumerate(reports):
        if index:
            console.print()
        console.print(_format_report_header(report), markup=True)
        for dependency in report.dependencies:
            console.print(
                _format_dependency_line(
                    dependency.kind,
                    dependency.name,
                    dependency.installed,
                ),
                markup=True,
            )
            if dependency.detail and dependency.detail != report.reason:
                console.print(f"    detail: {escape(dependency.detail)}", markup=True)
            if dependency.install_hint and not dependency.installed:
                console.print(
                    f"    [{_STYLE_HINT}]hint:[/] {escape(dependency.install_hint)}",
                    markup=True,
                )


def _format_report_header(report: DebugAvailability) -> str:
    """Build the first line for one availability report."""

    icon, style, status = _status_parts(report.can_debug)
    base = (
        f"[{style}]{icon}[/] "
        f"[{_STYLE_LABEL}]{escape(report.language)}[/]: "
        f"[{style}]{status}[/]"
    )
    if report.can_debug:
        return base
    return f"{base} - {escape(report.reason)}"


def _format_dependency_line(kind: str, name: str, installed: bool) -> str:
    """Build one dependency-status line."""

    icon, style, _status = _status_parts(installed)
    return f"  [{style}]{icon}[/] {escape(kind)}: {escape(name)}"


def _status_parts(is_available: bool) -> tuple[str, str, str]:
    """Return icon, color, and status label for availability values."""

    if is_available:
        return (_ICON_AVAILABLE, _STYLE_AVAILABLE, _STATUS_AVAILABLE)
    return (_ICON_UNAVAILABLE, _STYLE_UNAVAILABLE, _STATUS_UNAVAILABLE)
