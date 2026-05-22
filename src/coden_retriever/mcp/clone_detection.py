"""Clone detection using semantic and syntactic analysis.

This module provides MCP tools for detecting code clones.
Supports three modes:
- combined (default): Both semantic + syntactic analysis
- semantic: MiniLM ONNX embeddings for similar behavior detection
- syntactic: Line-by-line Jaccard for copy-paste detection
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from ..cache import CacheManager
from ..semantic_config import SemanticConfig
from ..constants import (
    DEFAULT_CLONE_RESULT_LIMIT,
    DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    DEFAULT_CLONE_SEMANTIC_WEIGHT,
    DEFAULT_CLONE_SYNTACTIC_WEIGHT,
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    MAX_TOKEN_LIMIT,
    MIN_TOKEN_LIMIT,
)
from ._defaults import RESOLVED_DEFAULT_TOKEN_BUDGET
from ..clone import (
    detect_clones_combined,
    detect_clones_semantic,
    detect_clones_syntactic,
)
from ..config_loader import get_semantic_model_path, daemon_enabled
from ..daemon.client import try_daemon_clones as _daemon_clones
from ..daemon.protocol import CloneDetectionParams
from .tool_timeout import worker_safe
from .validation import validate_root_directory

if TYPE_CHECKING:
    from ..cache.models import CachedIndices

logger = logging.getLogger(__name__)


async def _load_cached_indices(
    root_directory: str,
    semantic: SemanticConfig,
) -> "CachedIndices":
    def _load_sync() -> "CachedIndices":
        cache = CacheManager(
            Path(root_directory),
            semantic=semantic,
        )
        return cache.load_or_rebuild()
    return await asyncio.to_thread(_load_sync)


@worker_safe
async def detect_clones(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    mode: Annotated[
        Literal["combined", "semantic", "syntactic"],
        Field(
            description=(
                "'combined' (default): Both semantic + syntactic analysis; "
                "'semantic': MiniLM ONNX embeddings for similar behavior; "
                "'syntactic': Line-by-line Jaccard for copy-paste detection"
            )
        ),
    ] = "combined",
    similarity_threshold: Annotated[
        float,
        Field(description="Semantic similarity threshold (0.0-1.0)", ge=0.8, le=1.0),
    ] = DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    line_threshold: Annotated[
        float,
        Field(description="Line-by-line Jaccard threshold for syntactic mode (0.0-1.0)", ge=0.5, le=1.0),
    ] = DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    func_threshold: Annotated[
        float,
        Field(description="Percentage of lines that must match for syntactic mode (0.0-1.0)", ge=0.3, le=1.0),
    ] = DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    limit: Annotated[
        int,
        Field(description="Maximum number of clone pairs to return", ge=1, le=500),
    ] = DEFAULT_CLONE_RESULT_LIMIT,
    exclude_tests: Annotated[
        bool,
        Field(description="Exclude test functions from analysis"),
    ] = True,
    min_lines: Annotated[
        int,
        Field(description="Minimum function lines to consider", ge=1),
    ] = 3,
    token_limit: Annotated[
        int | None,
        Field(description="Soft limit on return size in tokens (None=no limit)", ge=MIN_TOKEN_LIMIT, le=MAX_TOKEN_LIMIT),
    ] = RESOLVED_DEFAULT_TOKEN_BUDGET,
    semantic_weight: Annotated[
        float,
        Field(description="Weight for semantic similarity in combined score (0.0-1.0)", ge=0.0, le=1.0),
    ] = DEFAULT_CLONE_SEMANTIC_WEIGHT,
    syntactic_weight: Annotated[
        float,
        Field(description="Weight for syntactic similarity in combined score (0.0-1.0)", ge=0.0, le=1.0),
    ] = DEFAULT_CLONE_SYNTACTIC_WEIGHT,
) -> dict[str, Any]:
    """Detect code clones using semantic, syntactic, or combined analysis.

    MODES:
    - combined (default): Both semantic + syntactic analysis, most comprehensive
    - semantic: MiniLM ONNX embeddings for functions that "do similar things"
    - syntactic: Line-by-line Jaccard for copy-paste detection with blocks

    WHEN TO USE EACH MODE:
    - combined: General refactoring analysis (recommended)
    - semantic: Find functions with similar behavior but different implementations
    - syntactic: Find copy-paste code with specific line matches

    OUTPUT:
    - clones: List of function pairs with similarity scores
    - Each clone includes: file, line, name, similarity, suggested_action
    - For syntactic/combined: includes blocks (consecutive matching lines)
    - summary: Statistics about clone distribution
    """
    validation_error = validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    model_path = get_semantic_model_path()

    # Try daemon first
    daemon_params = CloneDetectionParams(
        source_dir=str(Path(root_directory).resolve()),
        mode=mode,
        similarity_threshold=similarity_threshold,
        line_threshold=line_threshold,
        func_threshold=func_threshold,
        limit=limit,
        exclude_tests=exclude_tests,
        min_lines=min_lines,
        token_limit=token_limit,
        semantic_weight=semantic_weight,
        syntactic_weight=syntactic_weight,
    )
    daemon_result = None
    if daemon_enabled():
        daemon_result = _daemon_clones(daemon_params, auto_start=False)
    if daemon_result is not None:
        return daemon_result

    # Fallback to direct computation
    try:
        enable_semantic = mode in ("combined", "semantic")
        indices = await _load_cached_indices(
            root_directory,
            SemanticConfig(enabled=enable_semantic, model_path=model_path),
        )
    except Exception as e:
        logger.exception("Failed to load cache for clone detection")
        return {"error": f"Failed to load cache: {e}"}

    if mode == "syntactic":
        return detect_clones_syntactic(
            entities=indices.entities,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=limit,
            exclude_tests=exclude_tests,
            min_lines=min_lines,
            token_limit=token_limit,
        )
    elif mode == "semantic":
        return detect_clones_semantic(
            entities=indices.entities,
            model_path=model_path,
            threshold=similarity_threshold,
            limit=limit,
            exclude_tests=exclude_tests,
            min_lines=min_lines,
            token_limit=token_limit,
            embedding_cache=indices.embedding_cache,
        )
    else:  # combined (default)
        return detect_clones_combined(
            entities=indices.entities,
            model_path=model_path,
            semantic_threshold=similarity_threshold,
            line_threshold=line_threshold,
            func_threshold=func_threshold,
            limit=limit,
            exclude_tests=exclude_tests,
            min_lines=min_lines,
            token_limit=token_limit,
            semantic_weight=semantic_weight,
            syntactic_weight=syntactic_weight,
            embedding_cache=indices.embedding_cache,
        )


def register_clone_detection_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register clone detection tools with the MCP server."""
    disabled = disabled_tools or set()
    tools = [("detect_clones", detect_clones)]
    for tool_name, tool_func in tools:
        if tool_name not in disabled:
            mcp.tool()(tool_func)
