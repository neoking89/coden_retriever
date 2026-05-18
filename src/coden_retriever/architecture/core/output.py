"""Render an architecture `Report` as the spec's five-section text or JSON.

The text format is pure ASCII (greppable, pipe-safe). Severity tags
`[FAIL]` / `[WARN]` / `[INFO]` / `[OK]` are at column 0 of each section
header so a `grep '^\\['` reads every section line.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .constants import LOC_DISPLAY_DIVISOR

if TYPE_CHECKING:
    from .files import OversizedFile
    from .graph import Cycle, KitchenSinkFacts
    from .metrics import InFunctionStats, PackageMetric
    from .runner import Report


_INDENT = " " * 8
_LOC_ABBREV_FLOOR = 1000
# Why: below 1k LOC we show the raw integer for accuracy ("524 LOC body");
# at/above 1k we abbreviate ("2.6k LOC body") so the report stays compact.


def render_text(report: "Report") -> str:
    """Render the report in the human-readable five-section format."""
    parts: list[str] = []
    parts.append(_stat_line(report))
    if report.layout_warning is not None:
        parts.append("")
        parts.append(f"[WARN]  {report.layout_warning}")
    parts.append("")
    parts.append(_render_cycles(report.cycles))
    parts.append("")
    parts.append(_render_kitchen_sinks(report.kitchen_sinks))
    parts.append("")
    parts.append(_render_oversized(report.oversized_files))
    parts.append("")
    parts.append(_render_shallow(report.shallow_packages))
    parts.append("")
    parts.append(_render_in_function(report.in_function_stats))
    parts.append("")
    parts.extend(_footer(report))
    return "\n".join(parts) + "\n"


def render_json(report: "Report") -> str:
    """Render the report as a structured JSON document."""
    payload = {
        "stats": {
            "language": report.language,
            "modules": report.n_modules,
            "packages": report.n_packages,
            "files": report.n_files,
            "loc": report.total_loc,
        },
        "layout_warning": report.layout_warning,
        "cycles": [
            {
                "members": list(c.members),
                "workaround_imports": c.workaround_count,
            }
            for c in report.cycles
        ],
        "kitchen_sinks": [
            {
                "name": k.name,
                "body_loc": k.body_loc,
                "files": k.files,
                "fan_out": k.fan_out,
            }
            for k in report.kitchen_sinks
        ],
        "oversized_files": [
            {
                "path": str(o.path).replace("\\", "/"),
                "loc": o.loc,
                "top_imports": o.top_imports,
            }
            for o in report.oversized_files
        ],
        "shallow_packages": [
            {
                "name": s.name,
                "files": s.files,
                "public_symbols": s.public_symbols,
                "public_params": s.public_params,
                "body_loc": s.body_loc,
                "depth_ratio": s.depth_ratio,
            }
            for s in report.shallow_packages
        ],
        "in_function_imports": {
            "total": report.in_function_stats.total,
            "by_package": [
                {"package": p, "count": n}
                for p, n in report.in_function_stats.by_package
            ],
            "elsewhere": report.in_function_stats.elsewhere,
        },
        "exit_code": 1 if report.cycles else 0,
    }
    return json.dumps(payload, indent=2)


def _stat_line(report: "Report") -> str:
    """`  Python · N packages · M files · K.Lk LOC`

    Workspace audits (``n_modules >= 1``) inject `· N module(s) ·` between
    the language name and package count. Single-module audits (``n_modules
    == 0``) omit the segment entirely so output stays byte-identical to
    the pre-workspace formatter.
    """
    language = report.language.capitalize()
    loc_text = _format_loc(report.total_loc, force_abbrev=True)
    modules_segment = ""
    if report.n_modules > 0:
        noun = _plural(report.n_modules, "module", "modules")
        modules_segment = f"{report.n_modules} {noun} · "
    return (
        f"  {language} · {modules_segment}{report.n_packages} packages · "
        f"{report.n_files} files · {loc_text} LOC"
    )


def _render_cycles(cycles: tuple["Cycle", ...]) -> str:
    if not cycles:
        return "[OK]    no cycles"
    header = f"[FAIL]  {len(cycles)} {_plural(len(cycles), 'cycle', 'cycles')}"
    body_lines = [_format_cycle_row(c) for c in cycles]
    width = max(len(line) for line in body_lines) if body_lines else 0
    annotated = [
        _annotate_cycle(line, c, width)
        for line, c in zip(body_lines, cycles)
    ]
    return "\n".join([header, *annotated])


def _format_cycle_row(cycle: "Cycle") -> str:
    if len(cycle.members) == 2:
        return f"{_INDENT}{cycle.members[0]} ↔ {cycle.members[1]}"
    inner = ", ".join(cycle.members)
    return f"{_INDENT}{{{inner}}}"


def _annotate_cycle(line: str, cycle: "Cycle", width: int) -> str:
    pad = " " * (width - len(line) + 3)
    n = cycle.workaround_count
    if n == 0:
        return line
    noun = "import" if n == 1 else "imports"
    function = "a function" if n == 1 else "functions"
    return f"{line}{pad}(workaround: {n} {noun} moved inside {function})"


def _render_kitchen_sinks(rows: tuple["KitchenSinkFacts", ...]) -> str:
    if not rows:
        return "[OK]    no kitchen-sink packages"
    header = (
        f"[WARN]  {len(rows)} kitchen-sink "
        f"{_plural(len(rows), 'package', 'packages')}"
    )
    name_w = max(len(r.name) + 1 for r in rows)
    loc_w = max(len(_format_loc(r.body_loc, force_abbrev=True)) for r in rows)
    files_w = max(len(str(r.files)) for r in rows)
    body_lines: list[str] = []
    for r in rows:
        name = f"{r.name}/"
        loc_txt = _format_loc(r.body_loc, force_abbrev=True)
        body_lines.append(
            f"{_INDENT}{name:<{name_w}}  "
            f"{loc_txt:>{loc_w}} LOC · "
            f"{r.files:>{files_w}} files · "
            f"depends on {r.fan_out} other packages"
        )
    return "\n".join([header, *body_lines])


def _render_oversized(rows: tuple["OversizedFile", ...]) -> str:
    if not rows:
        return "[OK]    no oversized files"
    header = (
        f"[WARN]  {len(rows)} oversized "
        f"{_plural(len(rows), 'file', 'files')}"
    )
    path_strs = [str(r.path).replace("\\", "/") for r in rows]
    path_w = max(len(p) for p in path_strs)
    loc_w = max(len(str(r.loc)) for r in rows)
    body_lines = [
        f"{_INDENT}{p:<{path_w}}  {r.loc:>{loc_w}} LOC · "
        f"{r.top_imports} imports at top"
        for p, r in zip(path_strs, rows)
    ]
    return "\n".join([header, *body_lines])


def _render_shallow(rows: tuple["PackageMetric", ...]) -> str:
    if not rows:
        return "[OK]    no shallow packages"
    header = (
        f"[INFO]  {len(rows)} shallow "
        f"{_plural(len(rows), 'package', 'packages')}"
    )
    name_w = max(len(r.name) + 1 for r in rows)
    sym_w = max(len(str(r.public_symbols)) for r in rows)
    body_lines = [
        f"{_INDENT}{r.name + '/':<{name_w}}  "
        f"{r.public_symbols:>{sym_w}} exports / "
        f"{_format_loc(r.body_loc, force_abbrev=False)} LOC body"
        for r in rows
    ]
    return "\n".join([header, *body_lines])


def _render_in_function(stats: "InFunctionStats") -> str:
    if stats.total == 0:
        return "[OK]    no imports inside functions"
    header = (
        f"[INFO]  {stats.total} imports inside functions  "
        "(usually cycle workarounds)"
    )
    pieces: list[str] = []
    for pkg, n in stats.by_package:
        pieces.append(f"{n} {pkg}/")
    if stats.elsewhere > 0:
        pieces.append(f"{stats.elsewhere} elsewhere")
    if not pieces:
        return header
    breakdown = " · ".join(pieces)
    return "\n".join([header, f"{_INDENT}{breakdown}"])


def _footer(report: "Report") -> list[str]:
    if report.cycles:
        return ["  → exit 1 (cycles present)"]
    return ["  → exit 0"]


def _format_loc(loc: int, force_abbrev: bool) -> str:
    """`524` or `2.6k`. `force_abbrev=True` always uses abbreviation."""
    if loc < _LOC_ABBREV_FLOOR and not force_abbrev:
        return str(loc)
    scaled = loc / LOC_DISPLAY_DIVISOR
    return f"{scaled:.1f}k"


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural
