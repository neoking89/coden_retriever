"""Dead code detection MCP tool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


logger = logging.getLogger(__name__)

# Limits for MCP context windows
DEFAULT_RESULT_LIMIT = 50   # Prevent overwhelming results
DEFAULT_CONFIDENCE = 0.5    # Balance between precision and recall
DEFAULT_MIN_LINES = 3       # Skip trivial one-liners

# Validation bounds for MCP parameters
MAX_RESULTS = 500           # Upper bound prevents memory issues


def _load_and_detect(
    root_directory: str,
    confidence_threshold: float,
    exclude_tests: bool,
    include_private: bool,
    min_lines: int,
    limit: int,
) -> dict[str, Any]:
    """Load cache and run dead code detection (sync helper)."""
    from ..cache import CacheManager
    from ..dead_code.detector import detect_unused_functions

    cache = CacheManager(Path(root_directory), enable_semantic=False)
    indices = cache.load_or_rebuild()

    return detect_unused_functions(
        entities=indices.entities,
        graph=indices.graph,
        confidence_threshold=confidence_threshold,
        exclude_tests=exclude_tests,
        include_private=include_private,
        min_lines=min_lines,
        limit=limit,
    )


async def detect_dead_code(
    root_directory: Annotated[str, Field(description="Project root directory")],
    confidence_threshold: Annotated[
        float, Field(description="Minimum confidence (0.0-1.0)", ge=0.0, le=1.0)
    ] = DEFAULT_CONFIDENCE,
    limit: Annotated[
        int, Field(description="Max results", ge=1, le=MAX_RESULTS)
    ] = DEFAULT_RESULT_LIMIT,
    exclude_tests: Annotated[
        bool, Field(description="Exclude test functions")
    ] = True,
    include_private: Annotated[
        bool, Field(description="Include private functions")
    ] = False,
    min_lines: Annotated[
        int, Field(description="Min function lines", ge=1)
    ] = DEFAULT_MIN_LINES,
) -> dict[str, Any]:
    """Detect potentially dead (unused) code in the codebase."""
    if not root_directory:
        return {"error": "root_directory is required"}

    return await asyncio.to_thread(
        _load_and_detect,
        root_directory,
        confidence_threshold,
        exclude_tests,
        include_private,
        min_lines,
        limit,
    )


def register_dead_code_tools(
    mcp: "FastMCP",
    disabled_tools: set[str] | None = None,
) -> None:
    """Register dead code detection tools with the MCP server."""
    disabled = disabled_tools or set()
    if "detect_dead_code" not in disabled:
        mcp.tool()(detect_dead_code)
