"""Sensitive value detection MCP tool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP

from ..constants import (
    SENSITIVE_VALUE_DEFAULT_LIMIT,
    SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    SENSITIVE_VALUE_MAX_RESULTS,
)
from .validation import validate_root_directory

# Pre-import classifier (and its sklearn dependency) at module level.
# On Windows, first importing sklearn DLLs inside asyncio.to_thread
# causes a deadlock in the MCP subprocess.
from ..sensitive_values import classifier  # noqa: F401

logger = logging.getLogger(__name__)


def _load_and_detect(
    root_directory: str,
    confidence_threshold: float,
    exclude_tests: bool,
    limit: int,
    replace_value: str | None,
    whitelist: list[str] | None = None,
) -> dict[str, Any]:
    """Load cache and run sensitive value detection (sync helper)."""
    from ..cache import CacheManager
    from ..sensitive_values.detector import detect_sensitive_values

    cache = CacheManager(Path(root_directory))
    indices = cache.load_or_rebuild()

    return detect_sensitive_values(
        entities=indices.entities,
        confidence_threshold=confidence_threshold,
        exclude_tests=exclude_tests,
        limit=limit,
        replace_value=replace_value,
        whitelist=whitelist,
        root_dir=root_directory,
    )


async def detect_sensitive_values_tool(
    root_directory: Annotated[str, Field(description="Project root directory")],
    confidence_threshold: Annotated[
        float, Field(description="Min confidence to flag (0.0-1.0)", ge=0.0, le=1.0)
    ] = SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    limit: Annotated[
        int, Field(description="Max results", ge=1, le=SENSITIVE_VALUE_MAX_RESULTS)
    ] = SENSITIVE_VALUE_DEFAULT_LIMIT,
    exclude_tests: Annotated[
        bool, Field(description="Exclude test files")
    ] = True,
    replace_value: Annotated[
        str | None, Field(description="Replacement string for redaction (None = detect only)")
    ] = None,
    whitelist: Annotated[
        list[str] | None, Field(description="Glob patterns for text files to scan (e.g. ['*.env', '*.json'])")
    ] = None,
) -> dict[str, Any]:
    """Detect hardcoded sensitive values (secrets, credentials, API keys) in source code."""
    if err := validate_root_directory(root_directory):
        return err

    return await asyncio.to_thread(
        _load_and_detect,
        root_directory,
        confidence_threshold,
        exclude_tests,
        limit,
        replace_value,
        whitelist,
    )


def register_sensitive_value_tools(
    mcp: "FastMCP",
    disabled_tools: set[str] | None = None,
) -> None:
    """Register sensitive value detection tools with the MCP server."""
    disabled = disabled_tools or set()
    if "detect_sensitive_values" not in disabled:
        mcp.tool(name="detect_sensitive_values")(detect_sensitive_values_tool)
