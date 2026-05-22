"""Magic constant detection MCP tool."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP

from ..constants import (
    MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT,
    MAGIC_CONSTANT_MAX_RESULTS,
)
from .tool_timeout import worker_safe
from .validation import validate_root_directory

logger = logging.getLogger(__name__)


def _load_and_detect(
    root_directory: str,
    min_occurrences: int,
    min_files: int,
    exclude_tests: bool,
    limit: int,
) -> dict[str, Any]:
    """Load cache and run magic constant detection (sync helper)."""
    from ..cache import CacheManager
    from ..magic_constants.detector import detect_magic_constants

    cache = CacheManager(Path(root_directory))
    indices = cache.load_or_rebuild()

    return detect_magic_constants(
        entities=indices.entities,
        min_occurrences=min_occurrences,
        min_files=min_files,
        exclude_tests=exclude_tests,
        limit=limit,
    )


@worker_safe
async def detect_magic_constants_tool(
    root_directory: Annotated[str, Field(description="Project root directory")],
    min_occurrences: Annotated[
        int, Field(description="Min occurrences to flag a value", ge=2)
    ] = MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    min_files: Annotated[
        int, Field(description="Min distinct files a value must appear in", ge=1)
    ] = MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    limit: Annotated[
        int, Field(description="Max results", ge=1, le=MAGIC_CONSTANT_MAX_RESULTS)
    ] = MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT,
    exclude_tests: Annotated[
        bool, Field(description="Exclude test files")
    ] = True,
) -> dict[str, Any]:
    """Detect magic constants - repeated literal values across the codebase.

    Finds numeric and string literals that appear multiple times in different
    files, suggesting they should be extracted to named constants.
    """
    if err := validate_root_directory(root_directory):
        return err

    return await asyncio.to_thread(
        _load_and_detect,
        root_directory,
        min_occurrences,
        min_files,
        exclude_tests,
        limit,
    )


def register_magic_constant_tools(
    mcp: "FastMCP",
    disabled_tools: set[str] | None = None,
) -> None:
    """Register magic constant detection tools with the MCP server."""
    disabled = disabled_tools or set()
    if "detect_magic_constants" not in disabled:
        mcp.tool(name="detect_magic_constants")(detect_magic_constants_tool)
