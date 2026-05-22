"""Sensitive value detection MCP tool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP

from ..config_loader import daemon_enabled
from ..constants import (
    SENSITIVE_VALUE_DEFAULT_LIMIT,
    SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    SENSITIVE_VALUE_MAX_RESULTS,
)
from ..daemon.client import try_daemon_sensitive_values
from ..daemon.protocol import SensitiveValueParams
from .validation import validate_root_directory

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

    # Try the daemon first: it keeps the index + classifier warm across calls,
    # avoiding the cold cache rebuild + first-call classifier training that
    # otherwise pushes large repos past the client read-timeout.
    if daemon_enabled():
        daemon_result = try_daemon_sensitive_values(
            SensitiveValueParams(
                source_dir=str(Path(root_directory).resolve()),
                confidence_threshold=confidence_threshold,
                limit=limit,
                exclude_tests=exclude_tests,
                replace_value=replace_value,
                whitelist=whitelist,
            ),
            auto_start=False,
        )
        if daemon_result is not None:
            return daemon_result

    # In-process fallback. Prewarm the classifier on the main event loop. On
    # Windows, first sklearn DLL load inside asyncio.to_thread deadlocks the
    # MCP subprocess. Subsequent calls short-circuit at _ensure_trained's
    # `_model is not None` check, so the only cost is on the first invocation.
    from ..sensitive_values import classifier
    classifier.ensure_trained()

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
