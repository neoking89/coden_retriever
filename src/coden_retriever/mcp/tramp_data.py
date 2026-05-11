"""Tramp data detection MCP tool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


from ..constants import (
    TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    TRAMP_DATA_DEFAULT_RESULT_LIMIT,
    TRAMP_DATA_MAX_RESULTS,
    TRAMP_DATA_MIN_GROUP_SIZE,
)
from .validation import validate_root_directory

logger = logging.getLogger(__name__)


def _load_and_detect(
    root_directory: str,
    min_occurrences: int,
    exclude_tests: bool,
    limit: int,
    min_group_size: int,
) -> dict[str, Any]:
    """Load cache and run tramp data detection (sync helper)."""
    from ..cache import CacheManager
    from ..tramp_data.detector import detect_tramp_data

    cache = CacheManager(Path(root_directory))
    indices = cache.load_or_rebuild()

    return detect_tramp_data(
        entities=indices.entities,
        min_occurrences=min_occurrences,
        exclude_tests=exclude_tests,
        limit=limit,
        min_group_size=min_group_size,
    )


async def detect_tramp_data_tool(
    root_directory: Annotated[str, Field(description="Project root directory")],
    min_occurrences: Annotated[
        int, Field(description="Min functions a param group must appear in", ge=1)
    ] = TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    limit: Annotated[
        int, Field(description="Max results", ge=1, le=TRAMP_DATA_MAX_RESULTS)
    ] = TRAMP_DATA_DEFAULT_RESULT_LIMIT,
    exclude_tests: Annotated[
        bool, Field(description="Exclude test functions")
    ] = True,
    min_group_size: Annotated[
        int, Field(description="Min parameters in a group", ge=2)
    ] = TRAMP_DATA_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Detect tramp data - parameter groups traveling together across functions."""
    if err := validate_root_directory(root_directory):
        return err

    return await asyncio.to_thread(
        _load_and_detect,
        root_directory,
        min_occurrences,
        exclude_tests,
        limit,
        min_group_size,
    )


def register_tramp_data_tools(
    mcp: "FastMCP",
    disabled_tools: set[str] | None = None,
) -> None:
    """Register tramp data detection tools with the MCP server."""
    disabled = disabled_tools or set()
    if "detect_tramp_data" not in disabled:
        mcp.tool(name="detect_tramp_data")(detect_tramp_data_tool)
