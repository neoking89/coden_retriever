"""
Stacktrace parsing and debug_stacktrace MCP tool.

Provides universal stacktrace parsing for multiple languages and maps
frames to local codebase indexed by SearchEngine.

Performance: Uses daemon when available for fast in-memory access.
Falls back to disk-based cache when daemon is unavailable.
"""
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from ..cache import CacheManager
from ..config_loader import daemon_enabled
from ..daemon.client import try_daemon_stacktrace as _daemon_stacktrace
from ..daemon.protocol import StacktraceParams
from ..search import SearchEngine

logger = logging.getLogger(__name__)


@dataclass
class StackFrame:
    """Represents a single frame extracted from a stacktrace."""
    file_path: str        # Raw path as extracted from stacktrace
    line_number: int      # Line number from the stacktrace
    function_name: str | None = None  # Optional function name if captured
    resolved_path: str | None = None  # Resolved to local codebase path
    is_external: bool = False  # True if not found in local index


class StacktraceParser:
    """
    Universal stacktrace parser supporting multiple programming languages.

    Uses a multi-pattern heuristic strategy with regex patterns matched
    in priority order to extract stack frames from error logs.
    """

    # Regex patterns in priority order
    # Each pattern captures: file, line, and optionally func
    PATTERNS = [
        # Python: File "src/app.py", line 42, in main
        (
            "python",
            re.compile(r'File "(?P<file>.*?)", line (?P<line>\d+)(?:, in (?P<func>\w+))?')
        ),
        # Ruby: /app/lib/utils.rb:25:in `process'
        (
            "ruby",
            re.compile(r'(?P<file>[^\s]+?):(?P<line>\d+):in')
        ),
        # Java/C#: at com.example.App.main(App.java:15)
        (
            "java",
            re.compile(r'\((?P<file>[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+):(?P<line>\d+)\)')
        ),
        # Unix/Go/Rust/Node: /app/main.go:42 or src/utils.ts:10
        # More permissive pattern - must come after more specific ones
        (
            "generic",
            re.compile(r'(?P<file>[a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+):(?P<line>\d+)')
        ),
    ]

    # Error type patterns for extraction
    ERROR_PATTERNS = [
        # Python: ZeroDivisionError: division by zero
        re.compile(r'^(?P<type>[A-Z][a-zA-Z]*(?:Error|Exception|Fault)):\s*(?P<msg>.+)$', re.MULTILINE),
        # Java: java.lang.NullPointerException or NullPointerException
        re.compile(r'(?:java\.\w+\.)?(?P<type>[A-Z][a-zA-Z]*(?:Exception|Error))(?:\s|$)', re.MULTILINE),
        # Go panic: panic: runtime error: <message>
        re.compile(r'panic:\s*(?:runtime error:\s*)?(?P<msg>.+?)(?:\n|$)', re.MULTILINE),
        # Node.js: Error: <message>
        re.compile(r'^Error:\s*(?P<msg>.+)$', re.MULTILINE),
        # Rust: panicked at '<message>'
        re.compile(r"panicked at '(?P<msg>[^']+)'", re.MULTILINE),
    ]

    def parse(self, stacktrace: str) -> list[StackFrame]:
        """
        Parse a stacktrace from any supported language.

        Args:
            stacktrace: Raw stacktrace or error log text

        Returns:
            List of StackFrame objects extracted from the trace
        """
        frames: list[StackFrame] = []
        seen_frames: set[tuple[str, int]] = set()  # Deduplicate

        # Normalize line separators - some copy-pasted stacktraces use |
        # or other characters instead of newlines
        normalized_trace = stacktrace
        # Common patterns: " | " or " |   " used as line separators
        if '|' in normalized_trace and normalized_trace.count('|') > normalized_trace.count('\n'):
            # Replace pipe-based separators with newlines
            normalized_trace = re.sub(r'\s*\|\s*', '\n', normalized_trace)

        for line in normalized_trace.splitlines():
            line = line.strip()
            if not line:
                continue

            # Try each pattern in priority order
            for lang, pattern in self.PATTERNS:
                match = pattern.search(line)
                if match:
                    groups = match.groupdict()
                    file_path = groups.get("file", "")

                    try:
                        line_num = int(groups.get("line", 0))
                    except (ValueError, TypeError):
                        continue

                    # Skip if we've seen this exact frame
                    frame_key = (file_path, line_num)
                    if frame_key in seen_frames:
                        break
                    seen_frames.add(frame_key)

                    func_name = groups.get("func")

                    frames.append(StackFrame(
                        file_path=file_path,
                        line_number=line_num,
                        function_name=func_name,
                    ))
                    break  # Found a match, don't try other patterns

        return frames

    def resolve_path(self, raw_path: str, local_files: set[str]) -> str | None:
        """
        Maps raw stacktrace paths to local indexed files.

        Resolution Strategy:
        1. Normalization: Convert all backslashes to forward slashes
        2. Exact Match: Direct key lookup in file set
        3. Suffix Match: If any local_path.endswith(raw_path), it's a match
        4. Basename Match (Java/C#): If raw path has no directory, find any local file with that basename

        Args:
            raw_path: Path extracted from stacktrace (e.g., "/app/src/main.py")
            local_files: Set of indexed file paths from SearchEngine

        Returns:
            Resolved local path or None if not found
        """
        # Step 1: Normalize path separators
        normalized = raw_path.replace("\\", "/")

        # Step 2: Exact match
        if normalized in local_files:
            return normalized

        # Step 3: Suffix match (handles Docker/CI prefixes like /app/src/...)
        for local_path in local_files:
            local_normalized = local_path.replace("\\", "/")
            # Check if local path ends with our raw path
            if local_normalized.endswith(normalized):
                return local_path
            # Also check if our path ends with the local path
            if normalized.endswith(local_normalized):
                return local_path

        # Step 4: Basename match (for Java/C# where only filename is provided)
        basename = Path(normalized).name
        if "/" not in normalized and "\\" not in raw_path:
            # Only basename provided, find matching file
            for local_path in local_files:
                if Path(local_path).name == basename:
                    return local_path

        return None

    def extract_error_info(self, stacktrace: str) -> dict[str, str]:
        """
        Extract error type and message from a stacktrace.

        Args:
            stacktrace: Raw stacktrace text

        Returns:
            Dict with 'error_type' and 'error_message' keys
        """
        error_type = ""
        error_message = ""

        for pattern in self.ERROR_PATTERNS:
            match = pattern.search(stacktrace)
            if match:
                groups = match.groupdict()
                if "type" in groups and groups["type"]:
                    error_type = groups["type"]
                if "msg" in groups and groups["msg"]:
                    error_message = groups["msg"]
                if error_type or error_message:
                    break

        return {
            "error_type": error_type,
            "error_message": error_message,
        }


async def _create_engine(root_directory: str) -> SearchEngine:
    """Create a SearchEngine using cached indices when possible."""
    def _create_sync() -> SearchEngine:
        cache = CacheManager(Path(root_directory))
        cached_indices = cache.load_or_rebuild()
        return SearchEngine.from_cached_indices(cached_indices)

    return await asyncio.to_thread(_create_sync)


async def debug_stacktrace(
    root_directory: Annotated[
        str,
        Field(description="MUST be the absolute path to the project root directory")
    ],
    stacktrace: Annotated[
        str,
        Field(description="Raw stacktrace or error log to parse and analyze")
    ],
    include_context: Annotated[
        bool,
        Field(description="Include source code snippets for matched frames")
    ] = True,
    include_dependencies: Annotated[
        bool,
        Field(description="Include caller/callee relationships for matched functions")
    ] = False,
    max_context_lines: Annotated[
        int,
        Field(description="Maximum lines of source code context per frame", ge=5, le=50)
    ] = 15,
) -> dict[str, Any]:
    """Parse a stacktrace and map frames to local codebase with rich context.

    WHEN TO USE:
    - When debugging an error and you have a stacktrace from any language
    - To understand which parts of YOUR code (not library code) are involved in an error
    - To get source context and dependency information for error locations

    WHEN NOT TO USE:
    - For general code search (use code_search instead)
    - When you only have an error message without a stacktrace
    """
    if not os.path.isdir(root_directory):
        return {"error": f"Root directory not found: {root_directory}"}

    # Try daemon first for fast in-memory access
    daemon_params = StacktraceParams(
        source_dir=str(Path(root_directory).resolve()),
        stacktrace=stacktrace,
        context_lines=max_context_lines,
        show_dependencies=include_dependencies,
    )
    daemon_result = None
    if daemon_enabled():
        daemon_result = _daemon_stacktrace(daemon_params, auto_start=False)
    if daemon_result is not None:
        return daemon_result

    # Fallback to disk-based cache with full processing
    try:
        parser = StacktraceParser()
        frames = parser.parse(stacktrace)

        if not frames:
            return {
                "error": "No stack frames found in the provided stacktrace",
                "hint": "Make sure the stacktrace includes file paths and line numbers"
            }

        # Extract error info
        error_info = parser.extract_error_info(stacktrace)

        # Create engine for file resolution and context
        engine = await _create_engine(root_directory)

        # Get the set of local files from the engine
        local_files = set(engine._file_to_entities.keys())

        # Also add file paths from _file_scopes for complete coverage
        local_files.update(engine._file_scopes.keys())

        # Normalize local file paths for comparison
        normalized_local_files: set[str] = set()
        path_mapping: dict[str, str] = {}  # normalized -> original
        for f in local_files:
            normalized = f.replace("\\", "/")
            normalized_local_files.add(normalized)
            path_mapping[normalized] = f

        # Resolve frames and categorize
        user_frames = []
        external_frames = []

        for frame in frames:
            # Try to resolve the path
            resolved = parser.resolve_path(frame.file_path, normalized_local_files)

            if resolved:
                # Map back to original path if needed
                original_path = path_mapping.get(resolved, resolved)
                frame.resolved_path = original_path
                frame.is_external = False

                # Build frame info dict
                frame_info: dict[str, Any] = {
                    "file": _make_relative_path(original_path, root_directory),
                    "line": frame.line_number,
                    "function": frame.function_name,
                }

                # Find entity at this location for context
                entity = None
                entity_node_id = None

                # Look for entity containing this line
                scopes = engine._file_scopes.get(original_path, [])
                for start, end, node_id in scopes:
                    if start <= frame.line_number <= end:
                        entity = engine._entities.get(node_id)
                        entity_node_id = node_id
                        break

                if entity:
                    frame_info["entity_type"] = entity.entity_type
                    frame_info["entity_name"] = entity.name

                    if include_context:
                        frame_info["context"] = entity.get_context_snippet(max_lines=max_context_lines)

                    if include_dependencies and entity_node_id:
                        dep_context = engine.get_dependency_context(entity_node_id)
                        if not dep_context.is_empty():
                            frame_info["dependencies"] = {
                                "callers": [name for _, name, _, _ in dep_context.callers],
                                "callees": [name for _, name, _, _ in dep_context.callees],
                            }

                user_frames.append(frame_info)
            else:
                # External frame
                frame.is_external = True
                external_frames.append({
                    "file": frame.file_path,
                    "line": frame.line_number,
                    "function": frame.function_name,
                    "note": "External library - not in project index"
                })

        # Build call chain visualization
        call_chain = _build_call_chain(user_frames, error_info)

        # Build summary
        summary = f"Found {len(user_frames)} user-code frames, {len(external_frames)} external frames filtered"

        result: dict[str, Any] = {
            "summary": summary,
            "user_frames": user_frames,
            "external_frames": external_frames,
            "call_chain": call_chain,
        }

        if error_info["error_type"]:
            result["error_type"] = error_info["error_type"]
        if error_info["error_message"]:
            result["error_message"] = error_info["error_message"]

        return result

    except Exception as e:
        logger.exception(f"Error parsing stacktrace: {e}")
        return {"error": str(e)}


def _make_relative_path(path: str, root: str) -> str:
    """Convert an absolute path to relative if within root."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _build_call_chain(user_frames: list[dict], error_info: dict) -> str:
    """Build a visual call chain from user frames."""
    if not user_frames:
        return ""

    chain_parts = []
    for frame in user_frames:
        name = frame.get("entity_name") or frame.get("function") or Path(frame["file"]).stem
        chain_parts.append(name)

    chain = " -> ".join(chain_parts)

    if error_info.get("error_type"):
        chain += f" -> [{error_info['error_type']}]"

    return chain
