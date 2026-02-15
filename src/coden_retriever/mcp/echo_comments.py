"""Echo comment detection using semantic similarity.

This module provides MCP tools for detecting "echo comments" - redundant comments
that merely restate what the identifier already conveys without adding value.
Uses Model2Vec embeddings for language-agnostic semantic similarity analysis.

Requires the 'semantic' extra:
    pip install 'coden-retriever[semantic]'
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from ..cache import CacheManager
from ..config_loader import get_semantic_model_path
from ..language import LanguageLoader
from ..search.semantic import get_cached_model
from ..token_estimator import count_tokens
from ..utils.optional_deps import get_numpy
from .flag_insertion import CODEN_MARKER

if TYPE_CHECKING:
    from ..models import CodeEntity

logger = logging.getLogger(__name__)

# Module-level constants
DEFAULT_ECHO_THRESHOLD = 0.85
MIN_COMMENT_LENGTH = 3  # Filter out very short comments

# Token budget constants for MCP
_TOKEN_OVERHEAD_ECHO = 150
_TOKEN_PER_ECHO_COMMENT = 60


def _extract_comment_text(comment_node, source_bytes: bytes) -> str | None:
    """Extract clean comment text from a tree-sitter comment node.

    Args:
        comment_node: Tree-sitter comment node
        source_bytes: Source code as bytes

    Returns:
        Clean comment text without prefix/markers, or None if invalid
    """
    # Get raw comment text from source
    text = source_bytes[comment_node.start_byte:comment_node.end_byte].decode("utf-8", errors="ignore")
    text = text.strip()

    # Skip CODEN markers
    if CODEN_MARKER in text:
        return None

    # Remove common comment prefixes
    for prefix in ["//", "#", "--", "/*", "*/"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
        if text.endswith(prefix):
            text = text[:-len(prefix)].strip()

    # Handle block comment markers
    if text.startswith('"""') or text.startswith("'''"):
        text = text[3:].strip()
    if text.endswith('"""') or text.endswith("'''"):
        text = text[:-3].strip()

    # Remove leading * from multi-line comments
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("*"):
            line = line[1:].strip()
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines).strip()

    # Filter out very short comments
    if len(text) < MIN_COMMENT_LENGTH:
        return None

    return text


def _extract_identifier_from_node(node, source_bytes: bytes) -> str | None:
    """Extract the primary identifier from a code node.

    For function definitions: returns function name
    For class definitions: returns class name
    For variable assignments: returns variable name
    For other nodes: returns the text of the node

    Args:
        node: Tree-sitter node
        source_bytes: Source code as bytes

    Returns:
        Identifier string or None
    """
    # Function/method definitions
    if node.type in ("function_definition", "function_declaration", "function_item",
                     "method_definition", "method_declaration"):
        # Look for name node (child with type "identifier" or "name")
        for child in node.children:
            if child.type in ("identifier", "name"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

    # Class definitions
    if node.type in ("class_definition", "class_declaration", "class_item"):
        for child in node.children:
            if child.type in ("identifier", "name", "type_identifier"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

    # Variable declarations/assignments
    if node.type in ("variable_declaration", "variable_declarator", "assignment",
                     "lexical_declaration", "variable_definition"):
        # Look for identifier on the left side
        for child in node.children:
            if child.type in ("identifier", "name"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

    # For other nodes, try to find any identifier child
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

    # Fallback: return the text of the node itself (limited to 100 chars)
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
    text = " ".join(text.split())  # Normalize whitespace
    if len(text) > 100:
        text = text[:100]
    return text if text else None


def _find_associated_code_node(comment_node, tree_root):
    """Find the code node that a comment is describing.

    Uses tree-sitter AST structure to find the semantically related code node.
    Looks for the next sibling node or parent's next sibling.

    Args:
        comment_node: Tree-sitter comment node
        tree_root: Root of the syntax tree

    Returns:
        Associated code node or None
    """
    # Strategy 1: Look at next sibling (most common case)
    # Comments and their associated code are often siblings
    parent = comment_node.parent
    if parent:
        found_comment = False
        for sibling in parent.children:
            if sibling == comment_node:
                found_comment = True
                continue

            if found_comment and sibling.type != "comment":
                # Found the next non-comment sibling
                return sibling

    # Strategy 2: Look at parent's next sibling
    # For nested structures
    if parent and parent.parent:
        found_parent = False
        for uncle in parent.parent.children:
            if uncle == parent:
                found_parent = True
                continue

            if found_parent and uncle.type != "comment":
                return uncle

    # Strategy 3: Find the next node in document order
    # Walk forward from comment position
    comment_end_line = comment_node.end_point[0]

    def find_next_meaningful_node(node, min_line):
        """Recursively find next meaningful node after a given line."""
        if node.type == "comment":
            return None

        # If this node starts after the comment, it might be our target
        if node.start_point[0] > min_line:
            # Look for meaningful node types
            if node.type in ("function_definition", "function_declaration", "function_item",
                           "method_definition", "method_declaration",
                           "class_definition", "class_declaration", "class_item",
                           "variable_declaration", "variable_declarator", "assignment",
                           "lexical_declaration", "variable_definition"):
                return node

            # If node has identifier children, it might be meaningful
            for child in node.children:
                if child.type in ("identifier", "name", "type_identifier"):
                    return node

        # Recurse into children
        for child in node.children:
            result = find_next_meaningful_node(child, min_line)
            if result:
                return result

        return None

    return find_next_meaningful_node(tree_root, comment_end_line)


def _extract_all_comments_from_file(file_path: str, language: str) -> list[tuple[int, str, str]]:
    """Extract ALL comments from a file using tree-sitter.

    Args:
        file_path: Path to source file
        language: Programming language

    Returns:
        List of (line_number, comment_text, context_identifier) tuples
        where context_identifier is the semantically associated code identifier
    """
    if not os.path.exists(file_path):
        return []

    # Read source code
    try:
        with open(file_path, "rb") as f:
            source_bytes = f.read()
    except IOError:
        return []

    # Parse with tree-sitter
    try:
        from tree_sitter import Parser
    except ImportError:
        logger.warning("tree-sitter not installed")
        return []

    try:
        loader = LanguageLoader()
        ts_language = loader.load(language)
        if not ts_language:
            logger.warning(f"Failed to load language for {language}")
            return []

        try:
            parser = Parser(ts_language)
        except TypeError:
            parser = Parser()
            parser.set_language(ts_language)  # type: ignore[attr-defined]

        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
        return []

    comments: list[tuple[int, str, str]] = []

    # Walk the tree and find ALL comment nodes
    def walk_tree(node):
        if node.type == "comment":
            # Extract comment text
            comment_text = _extract_comment_text(node, source_bytes)
            if comment_text:
                # Find associated code node using AST structure
                code_node = _find_associated_code_node(node, tree.root_node)
                if code_node:
                    # Extract identifier from code node
                    identifier = _extract_identifier_from_node(code_node, source_bytes)
                    if identifier:
                        # Line numbers are 1-indexed for user display
                        line_number = node.start_point[0] + 1
                        comments.append((line_number, comment_text, identifier))

        # Recurse into children
        for child in node.children:
            walk_tree(child)

    # Start tree walk from root
    walk_tree(tree.root_node)

    return comments


def compute_echo_comments(
    entities: dict[str, CodeEntity],
    echo_threshold: float = DEFAULT_ECHO_THRESHOLD,
    token_limit: int | None = None,
    include_tests: bool = False,
    include_private: bool = False,
) -> dict[str, Any]:
    """Compute semantic similarity between ALL comments and their associated identifiers.

    Uses model2vec embeddings and cosine similarity to detect "echo comments"
    that merely restate the identifier without adding architectural value.

    Performance: Uses batch encoding for all comments per file to minimize
    model inference overhead on large codebases.

    Args:
        entities: Dict of entity_id -> CodeEntity (used to get file paths and language)
        echo_threshold: Similarity threshold for echo detection (0.0-1.0)
        token_limit: Max tokens for MCP context (None = unlimited)
        include_tests: Whether to analyze test files
        include_private: Whether to analyze private entities

    Returns:
        {
            "echo_comments": [
                {
                    "file_path": str,
                    "line": int,
                    "comment_text": str,
                    "context_identifier": str,
                    "similarity_score": float,
                    "severity": str
                }
            ],
            "summary": {
                "total_files_analyzed": int,
                "total_comments_found": int,
                "echo_comments_count": int,
                "echo_ratio": float,
                "files_affected": int,
                "avg_similarity": float,
                "distribution": dict
            },
            "token_budget_exceeded": bool
        }
    """
    # Load model (uses cache internally)
    model_path = get_semantic_model_path()
    model = get_cached_model(model_path)

    echo_comments: list[dict] = []
    total_comments = 0
    files_affected: set[str] = set()

    # Get unique files to analyze
    files_to_analyze: dict[str, str] = {}  # file_path -> language
    for entity_id, entity in entities.items():
        if not include_tests and entity.is_test:
            continue
        if not include_private and entity.is_private:
            continue
        if entity.file_path:
            files_to_analyze[entity.file_path] = entity.language

    total_files = len(files_to_analyze)

    # Process each file and extract ALL comments
    for file_path, language in files_to_analyze.items():
        comments = _extract_all_comments_from_file(file_path, language)
        if not comments:
            continue

        total_comments += len(comments)

        # === BATCH ENCODING FOR PERFORMANCE ===
        # Encode all comments and identifiers in batch rather than one-by-one
        # This significantly improves performance on large codebases
        comment_texts = [c[1] for c in comments]  # Extract comment text
        identifier_texts = [c[2] for c in comments]  # Extract identifier

        # Batch encode all texts at once
        all_texts = comment_texts + identifier_texts
        all_embeddings = model.encode(all_texts)

        # Split embeddings back into comments and identifiers
        n_comments = len(comments)
        comment_embeddings = all_embeddings[:n_comments]
        identifier_embeddings = all_embeddings[n_comments:]

        # Lazy load numpy for norm computation
        np = get_numpy()
        norm = np.linalg.norm

        # Compute norms for all embeddings
        comment_norms = norm(comment_embeddings, axis=1)
        identifier_norms = norm(identifier_embeddings, axis=1)

        # Process each comment with its pre-computed embedding
        for idx, (line_num, comment_text, identifier) in enumerate(comments):
            c_norm = comment_norms[idx]
            i_norm = identifier_norms[idx]

            # Skip if either embedding has zero norm (empty/invalid text)
            if c_norm == 0 or i_norm == 0:
                continue

            # Cosine similarity using pre-computed embeddings
            similarity = float(
                np.dot(comment_embeddings[idx], identifier_embeddings[idx])
                / (c_norm * i_norm)
            )

            if similarity >= echo_threshold:
                # Determine severity
                if similarity >= 0.95:
                    severity = "CRITICAL"
                elif similarity >= 0.90:
                    severity = "HIGH"
                elif similarity >= 0.85:
                    severity = "ELEVATED"
                else:
                    severity = "MODERATE"

                echo_comments.append(
                    {
                        "file_path": file_path,
                        "line": line_num,
                        "comment_text": comment_text,
                        "context_identifier": identifier,
                        "similarity_score": similarity,
                        "severity": severity,
                    }
                )

                files_affected.add(file_path)

    # Sort by similarity (descending)
    echo_comments.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Apply token limit for MCP mode
    token_budget_exceeded = False
    if token_limit is not None:
        used_tokens = _TOKEN_OVERHEAD_ECHO
        filtered_comments = []

        for comment in echo_comments:
            comment_str = f"{comment['file_path']}:{comment['line']} {comment['comment_text']} {comment['context_identifier']}"
            comment_tokens = count_tokens(comment_str, is_code=False) + _TOKEN_PER_ECHO_COMMENT
            if used_tokens + comment_tokens > token_limit:
                token_budget_exceeded = True
                break
            used_tokens += comment_tokens
            filtered_comments.append(comment)

        echo_comments = filtered_comments

    # Compute summary
    echo_count = len(echo_comments)
    echo_ratio = echo_count / total_comments if total_comments > 0 else 0.0
    avg_similarity = (
        sum(c["similarity_score"] for c in echo_comments) / echo_count
        if echo_count > 0
        else 0.0
    )

    distribution = {
        "critical": sum(1 for c in echo_comments if c["similarity_score"] >= 0.95),
        "high": sum(
            1 for c in echo_comments if 0.90 <= c["similarity_score"] < 0.95
        ),
        "elevated": sum(
            1 for c in echo_comments if 0.85 <= c["similarity_score"] < 0.90
        ),
        "moderate": sum(
            1 for c in echo_comments if 0.75 <= c["similarity_score"] < 0.85
        ),
    }

    return {
        "echo_comments": echo_comments,
        "summary": {
            "total_files_analyzed": total_files,
            "total_comments_found": total_comments,
            "echo_comments_count": echo_count,
            "echo_ratio": echo_ratio,
            "files_affected": len(files_affected),
            "avg_similarity": avg_similarity,
            "distribution": distribution,
        },
        "token_budget_exceeded": token_budget_exceeded,
    }


def _validate_root_directory(root_directory: str) -> dict[str, Any] | None:
    """Validate that root_directory exists."""
    if not root_directory:
        return {"error": "root_directory is required"}
    if not os.path.isdir(root_directory):
        return {"error": f"Root directory not found: {root_directory}"}
    return None


async def _load_cached_indices(root_directory: str, model_path: str | None = None):
    """Load cached indices asynchronously."""
    from ..cache.models import CachedIndices

    def _load_sync() -> CachedIndices:
        cache = CacheManager(Path(root_directory), enable_semantic=True, model_path=model_path)
        return cache.load_or_rebuild()

    return await asyncio.to_thread(_load_sync)


async def detect_echo_comments(
    root_directory: Annotated[
        str,
        Field(description="Absolute path to the project root directory"),
    ],
    echo_threshold: Annotated[
        float,
        Field(description="Minimum similarity to flag as echo (0.0-1.0)", ge=0.5, le=1.0),
    ] = 0.85,
    include_tests: Annotated[
        bool,
        Field(description="Include test files in analysis"),
    ] = False,
    include_private: Annotated[
        bool,
        Field(description="Include private/internal entities"),
    ] = False,
    token_limit: Annotated[
        int | None,
        Field(description="Soft limit on return size in tokens (None=no limit)", ge=100, le=100000),
    ] = 4000,
) -> dict[str, Any]:
    """Detect echo comments - redundant comments that merely restate the code.

    Uses semantic embeddings to find comments that are semantically similar
    to their associated identifiers, indicating they add no value.

    WHEN TO USE:
    - During code review to find low-value comments
    - Before documentation cleanup to prioritize what to improve
    - To enforce documentation standards (comments should explain WHY, not WHAT)

    WHEN NOT TO USE:
    - To find all comments (use grep for that)
    - To analyze comment quality beyond redundancy

    OUTPUT:
    - echo_comments: List of redundant comments with similarity scores
    - Each comment includes: file, line, text, identifier, severity
    - summary: Statistics about echo comment distribution
    """
    validation_error = _validate_root_directory(root_directory)
    if validation_error:
        return validation_error

    model_path = get_semantic_model_path()

    try:
        indices = await _load_cached_indices(root_directory, model_path)
    except Exception as e:
        logger.exception("Failed to load cache for echo comment detection")
        return {"error": f"Failed to load cache: {e}"}

    return compute_echo_comments(
        entities=indices.entities,
        echo_threshold=echo_threshold,
        token_limit=token_limit,
        include_tests=include_tests,
        include_private=include_private,
    )


def register_echo_comment_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register echo comment detection tools with the MCP server."""
    disabled = disabled_tools or set()
    tools = [("detect_echo_comments", detect_echo_comments)]
    for tool_name, tool_func in tools:
        if tool_name not in disabled:
            mcp.tool()(tool_func)
