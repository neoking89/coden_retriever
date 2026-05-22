"""Architecture audit — MCP wrapper around `coden architecture <path>`."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from ..architecture.core.constants import TOP_FINDINGS_DEFAULT
from ..architecture.core.messages import (
    is_unsupported_language_message,
    parse_unsupported_language,
    supported_languages,
)
from .tool_timeout import worker_safe
from .validation import validate_root_directory

logger = logging.getLogger(__name__)


@worker_safe
async def architecture(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    top: Annotated[
        int,
        Field(
            description="Per-section findings cap (cycles, kitchen-sinks, oversized files, ...)",
            ge=0,
        ),
    ] = TOP_FINDINGS_DEFAULT,
    excludes: Annotated[
        str,
        Field(description="Comma-separated directory names to skip (e.g. 'tests,vendor')"),
    ] = "",
    lang: Annotated[
        str | None,
        Field(
            description=(
                "Force a specific language adapter (e.g. 'python'). Omit to auto-detect "
                "from the file mix under `root_directory`."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """Five-section architecture audit: cycles, kitchen-sinks, oversized files, shallow packages, in-function imports.

    Runs the same analysis as `coden architecture <path>` but returns a structured
    dict so an agent can act on findings directly (rather than parsing CLI text).

    WHEN TO USE:
    - Before a refactor — establish baseline architectural debt.
    - After a refactor — confirm cycles/kitchen-sinks were actually reduced.
    - When the user asks "what's wrong with this codebase architecturally?".
    - As a pre-commit gate — `exit_code == 1` means cycles exist.

    WHEN NOT TO USE:
    - To find specific coupled function pairs (use `coupling_hotspots`).
    - To measure overall coupling intensity (use `propagation_cost`).
    - To find dead code or duplication (use `detect_dead_code` / `detect_clones`).

    OUTPUT:
    - On success: dict with `stats`, `cycles`, `kitchen_sinks`, `oversized_files`,
      `shallow_packages`, `in_function_imports`, `exit_code`, and a pre-rendered
      `text_report` string mirroring the CLI's human-readable output.
    - On unsupported language: dict with `warning`, `supported_languages`, and
      `detected_language` keys (NOT `error` — the path is valid, just no adapter yet).
    - On hard error (missing path, no source files): dict with `error` key.
    """
    validation_error = validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    from ..architecture import run_audit

    excludes_tuple = tuple(part.strip() for part in excludes.split(",") if part.strip())
    root = Path(root_directory).resolve()

    report, err = await asyncio.to_thread(
        run_audit, root, lang, top, excludes_tuple
    )

    if err is not None:
        if is_unsupported_language_message(err):
            # Prefer the caller-supplied lang (could contain arbitrary characters);
            # fall back to extracting from the message only on auto-detect, where
            # the runner derived the name from `LANGUAGE_MAP` (quote-free by construction).
            detected = lang if lang is not None else parse_unsupported_language(err)
            return {
                "warning": err,
                "supported_languages": list(supported_languages()),
                "detected_language": detected,
            }
        return {"error": err}

    assert report is not None
    return _report_to_dict(report)


def _report_to_dict(report: Any) -> dict[str, Any]:
    """Mirror `render_json`'s field shape, plus a `text_report` for human display."""
    from ..architecture import render_text
    return {
        "stats": {
            "language": report.language,
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
        "text_report": render_text(report),
    }


def register_architecture_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register the architecture audit tool with the MCP server."""
    disabled = disabled_tools or set()
    tools = [("architecture", architecture)]
    for tool_name, tool_func in tools:
        if tool_name not in disabled:
            mcp.tool()(tool_func)
