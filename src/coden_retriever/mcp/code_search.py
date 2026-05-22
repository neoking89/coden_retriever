"""
Code Search MCP Tools module.

Provides code search tool implementations for the Model Context Protocol server.
Uses daemon for fast searches when available, with fallback to direct search.
"""
import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from ..cache import CacheManager
from ..constants import MAX_TOKEN_LIMIT, MIN_TOKEN_LIMIT
from ._defaults import RESOLVED_DEFAULT_TOKEN_BUDGET
from .tool_timeout import worker_safe
from .validation import validate_root_directory
from ..config import MapMode, OutputFormat
from ..config_loader import daemon_enabled
from ..daemon.client import try_daemon_search as _daemon_search, try_daemon_trace_dependency as _daemon_trace_dependency
from ..daemon.protocol import SearchParams, TraceDependencyParams
from ..formatters import get_formatter
from ..formatters.base import OutputFormatter
from ..search import SearchEngine
from ..search.engine import strategy_for
from ..semantic_config import SemanticConfig
from .git_analysis import code_evolution, find_hotspots
from .inspection import git_history_context, read_source_range, read_source_ranges
from .stacktrace import debug_stacktrace

logger = logging.getLogger(__name__)

# Strips ANSI SGR sequences and OSC hyperlinks from formatter output.
# The tree/context-map formatters embed terminal colors and clickable
# vscode:// links that are useful in the CLI but pure noise for an LLM.
_ANSI_RE = re.compile(r"\x1b(?:\]8;[^\\]*\\|\[[0-9;]*[A-Za-z])")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text before it enters the LLM context."""
    return _ANSI_RE.sub("", text)


async def _create_engine(
    root_directory: str,
    semantic: SemanticConfig = SemanticConfig(),
) -> SearchEngine:
    """Create a SearchEngine using cached indices when possible.

    Uses CacheManager for fast startup when codebase hasn't changed.

    Args:
        root_directory: Absolute path to the root directory.
        semantic: Semantic search configuration. None means disabled.

    Returns:
        SearchEngine instance for the project.
    """
    def _create_sync() -> SearchEngine:
        cache = CacheManager(Path(root_directory), semantic=semantic)
        cached_indices = cache.load_or_rebuild()
        return SearchEngine.from_cached_indices(cached_indices)

    return await asyncio.to_thread(_create_sync)


def _build_engine_for_mode(root_directory: str, map_mode: MapMode) -> SearchEngine:
    """Build a `SearchEngine` for `map_mode` via the strategy registry.

    Picks the cache layout (LITE for SIMPLE, STATIC otherwise) and the
    strategy's `semantic_override` so the direct MCP fallback gets the same
    per-mode cache shape and warm-cache speedup as the CLI. The MCP entry
    points do not currently propagate the caller's `enable_semantic`, so
    semantic scoring is off here by default.
    """
    strategy = strategy_for(map_mode)
    semantic = strategy.semantic_override or SemanticConfig()
    cache = CacheManager(
        Path(root_directory),
        semantic=semantic,
        layout=strategy.layout,
    )
    return strategy.build_engine(cache, verbose=False)


async def _code_search_impl(
    root_directory: str,
    query: str,
    token_limit: int,
    output_format: str,
    show_dependencies: bool,
    show_tree: bool,
    enable_semantic: bool,
) -> dict[str, Any]:
    """Internal implementation for code search - shared by keyword and semantic search tools."""
    if err := validate_root_directory(root_directory):
        return err

    search_mode = "semantic" if enable_semantic else "keyword"

    try:
        # Try daemon first for fast sub-200ms response
        params = SearchParams(
            source_dir=str(Path(root_directory).resolve()),
            query=query,
            enable_semantic=enable_semantic,
            tokens=token_limit,
            show_deps=show_dependencies,
            output_format=output_format,
            dir_tree=show_tree,
            stats=True,
        )
        daemon_result = None
        if daemon_enabled():
            daemon_result = await asyncio.to_thread(_daemon_search, params, auto_start=False)

        if daemon_result is not None:
            # Daemon returned results - reformat for MCP response
            logger.debug(f"Using daemon for code_search(mode='{search_mode}') (fast path)")
            result_dict = {}
            if show_tree and "directory_tree" in daemon_result:
                result_dict["directory_tree"] = _strip_ansi(daemon_result["directory_tree"])
            result_dict.update({
                "results": _strip_ansi(daemon_result.get("output", "")),
                "count": daemon_result.get("result_count", 0),
                "total_found": daemon_result.get("total_matched", 0),
                "stats": daemon_result.get("stats", ""),
                "search_time_ms": daemon_result.get("search_time_ms", 0),
                "source": "daemon",
            })
            # Pass through warning from daemon if present
            if "warning" in daemon_result:
                result_dict["warning"] = daemon_result["warning"]
            return result_dict

        # Fallback: Create a fresh engine for each request
        logger.debug(f"Daemon not available, using direct search for code_search(mode='{search_mode}') (fallback)")
        engine = await _create_engine(root_directory, semantic=SemanticConfig(enabled=enable_semantic))

        # Run blocking search operations in thread pool for true async
        results = await asyncio.to_thread(
            engine.search,
            query=query,
            include_deps=show_dependencies,
        )

        # Apply token budget
        included_results = OutputFormatter.filter_by_token_budget(results, token_limit, show_dependencies)

        fmt = OutputFormat(output_format.lower())
        formatter = get_formatter(fmt)
        output = formatter.format_results(
            included_results, Path(root_directory), token_limit, show_dependencies
        )

        result_dict = {}

        # Add directory tree first for LLM attention
        if show_tree:
            tree_output = await asyncio.to_thread(
                engine.generate_directory_tree, included_results
            )
            result_dict["directory_tree"] = _strip_ansi(tree_output)

        result_dict.update({
            "results": _strip_ansi(output),
            "count": len(included_results),
            "total_found": len(results),
            "stats": str(engine.get_stats()),
            "source": "direct",
        })

        # Add warning when token budget filtered out all results
        if len(results) > 0 and len(included_results) == 0:
            result_dict["warning"] = (
                f"All {len(results)} matching results were filtered out due to token_limit={token_limit}. "
                f"Increase token_limit (e.g., 4000 or higher) to get meaningful results."
            )

        return result_dict

    except Exception as e:
        logger.exception(f"Search error: {e}")
        return {"error": str(e)}


@worker_safe
async def code_search(
    root_directory: Annotated[
        str,
        Field(
            description="MUST be the absolute path to the project root directory"
        )
    ],
    query: Annotated[
        str,
        Field(
            description="Search query - use code terminology for 'keyword' mode, natural language for 'semantic' mode"
        )
    ],
    mode: Annotated[
        Literal["keyword", "semantic"],
        Field(
            description=(
                "'keyword': BM25 lexical matching for exact code terminology; "
                "'semantic': MiniLM ONNX embeddings for natural language questions"
            )
        )
    ] = "keyword",
    token_limit: Annotated[
        int,
        Field(
            description="Soft limit on the return size in tokens",
            ge=MIN_TOKEN_LIMIT,
            le=MAX_TOKEN_LIMIT
        )
    ] = RESOLVED_DEFAULT_TOKEN_BUDGET,
    output_format: Annotated[
        Literal["tree", "xml", "markdown", "json"],
        Field(
            description="Format of the output results"
        )
    ] = "tree",
    show_dependencies: Annotated[
        bool,
        Field(
            description="Set to True to include caller/callee relationship information"
        )
    ] = False,
    show_tree: Annotated[
        bool,
        Field(
            description="Set to True to show recursive directory tree of results"
        )
    ] = False,
) -> dict[str, Any]:
    """Search code using keyword or semantic matching.

    MODES:
    - 'keyword': BM25 lexical matching - finds exact word matches, fast and precise
    - 'semantic': MiniLM ONNX embeddings - understands meaning, finds conceptually similar code

    WHEN TO USE:
    - mode='keyword': when you know exact terminology (e.g., 'password reset', 'UserFactory')
    - mode='semantic': for natural language questions (e.g., 'how is auth handled?')

    WHEN NOT TO USE:
    - For precise symbol lookups if you know the name - use find_identifier instead

    EXAMPLES:
    - Keyword: 'database connection pool', 'UserAuthentication login'
    - Semantic: 'how is authentication handled?', 'where are API routes defined?'
    """
    return await _code_search_impl(
        root_directory=root_directory,
        query=query,
        token_limit=token_limit,
        output_format=output_format,
        show_dependencies=show_dependencies,
        show_tree=show_tree,
        enable_semantic=(mode == "semantic"),
    )


@worker_safe
async def code_map(
    root_directory: Annotated[
        str,
        Field(
            description="MUST be the absolute path to the project root directory"
        )
    ],
    token_limit: Annotated[
        int,
        Field(
            description="Maximum tokens for the map output",
            ge=MIN_TOKEN_LIMIT,
            le=MAX_TOKEN_LIMIT
        )
    ] = RESOLVED_DEFAULT_TOKEN_BUDGET,
    output_format: Annotated[
        Literal["tree", "xml", "markdown", "json"],
        Field(
            description="Format of the output results"
        )
    ] = "tree",
    show_dependencies: Annotated[
        bool,
        Field(
            description="Include high-level dependency flows between modules"
        )
    ] = False,
    show_tree: Annotated[
        bool,
        Field(
            description="Include the recursive directory tree structure"
        )
    ] = True,
    map_mode: Annotated[
        MapMode,
        Field(
            description=(
                "Ranking strategy. 'static' (default): combined-signal score "
                "(PageRank + Betweenness + BM25 + ...). 'simple': rank by "
                "per-file git commit count with line-count fallback when no git "
                "history is present. 'simple' empirically beats 'static' by "
                "+0.260 mean P@10 on a 5-repo click-GT fixture. Output is "
                "structurally identical between modes."
            )
        )
    ] = MapMode.STATIC,
) -> dict[str, Any]:
    """Generates a high-level architectural overview of the entire repository. Use this FIRST when exploring a new codebase.

    WHEN TO USE:
    - Use this FIRST when entering a new repository to understand its structure
    - Use when you need to understand the relationships between top-level modules
    - Use to generate a 'mental map' of where key components are located

    WHEN NOT TO USE:
    - Do NOT use to find specific functions or lines of code (use code_search or find_identifier)
    - Do NOT use if you only need to list files in a directory
    """
    if err := validate_root_directory(root_directory):
        return err

    try:
        # Try daemon first for fast sub-200ms response
        params = SearchParams(
            source_dir=str(Path(root_directory).resolve()),
            query="",
            tokens=token_limit,
            show_deps=show_dependencies,
            output_format=output_format,
            show_map=True,
            dir_tree=show_tree,
            stats=True,
            map_mode=map_mode.value,
        )
        daemon_result = None
        if daemon_enabled():
            daemon_result = await asyncio.to_thread(_daemon_search, params, auto_start=False)

        if daemon_result is not None:
            # Daemon returned results - reformat for MCP response
            logger.debug("Using daemon for code_map (fast path)")
            result_dict = {}
            if show_tree and "directory_tree" in daemon_result:
                result_dict["directory_tree"] = _strip_ansi(daemon_result["directory_tree"])
            result_dict.update({
                "map": _strip_ansi(daemon_result.get("output", "")),
                "stats": daemon_result.get("stats", ""),
                "search_time_ms": daemon_result.get("search_time_ms", 0),
                "source": "daemon",
            })
            # Pass through warning from daemon if present
            if "warning" in daemon_result:
                result_dict["warning"] = daemon_result["warning"]
            return result_dict

        # Fallback: Create a fresh engine for each request via the strategy
        # matching `map_mode`. SIMPLE resolves the lite cache (or full lite
        # rebuild) via the same path the CLI uses, skipping the heavy
        # centrality build; STATIC builds the full graph.
        logger.debug("Daemon not available, using direct map (fallback)")
        engine = await asyncio.to_thread(
            _build_engine_for_mode, root_directory, map_mode
        )

        # Run blocking search operations in thread pool for true async
        results = await asyncio.to_thread(
            engine.search,
            query="",
            use_architecture=True,
            include_deps=show_dependencies,
            limit=500,
            map_mode=map_mode,
        )

        # Apply token budget
        included_results = OutputFormatter.filter_by_token_budget(results, token_limit, show_dependencies)

        fmt = OutputFormat(output_format.lower())
        formatter = get_formatter(fmt)
        map_output = formatter.format_map(
            included_results, Path(root_directory), token_limit, show_dependencies
        )

        result_dict = {}

        # Add directory tree first for LLM attention
        if show_tree:
            tree_output = await asyncio.to_thread(
                engine.generate_directory_tree, included_results
            )
            result_dict["directory_tree"] = _strip_ansi(tree_output)

        result_dict.update({
            "map": _strip_ansi(map_output),
            "stats": str(engine.get_stats()),
            "source": "direct",
        })

        # Add warning when token budget filtered out all results
        if len(results) > 0 and len(included_results) == 0:
            result_dict["warning"] = (
                f"All {len(results)} matching results were filtered out due to token_limit={token_limit}. "
                f"Increase token_limit (e.g., 4000 or higher) to get meaningful results."
            )

        return result_dict

    except Exception as e:
        logger.exception(f"Map generation error: {e}")
        return {"error": str(e)}


@worker_safe
async def find_identifier(
    root_directory: Annotated[
        str,
        Field(
            description="MUST be the absolute path to the project root directory"
        )
    ],
    identifier: Annotated[
        str,
        Field(
            description="The specific name of the symbol to find (e.g., 'UserFactory', 'process_payment'). Can be exact or partial."
        )
    ],
    max_results: Annotated[
        int,
        Field(
            description="Maximum number of matching results to return",
            ge=1,
            le=500
        )
    ] = 20,
    show_dependencies: Annotated[
        bool,
        Field(
            description="Include context on where this identifier is used/called"
        )
    ] = False,
    show_tree: Annotated[
        bool,
        Field(
            description="Set to False to hide recursive directory tree of results"
        )
    ] = True,
) -> dict[str, Any]:
    """Precise lookup for specific symbols (functions, classes, variables) by name. Use when you know the exact or partial name.

    WHEN TO USE:
    - When you know the exact or partial name of a symbol (e.g., 'UserFactory', 'process_payment')
    - When you need to find where a specific class or function is defined
    - This is more precise than code_search for symbol lookups

    WHEN NOT TO USE:
    - Do NOT use for general concepts (e.g., 'how does auth work?'). Use code_search for that
    """
    if err := validate_root_directory(root_directory):
        return err

    try:
        # Try daemon first for fast sub-200ms response
        params = SearchParams(
            source_dir=str(Path(root_directory).resolve()),
            find_identifier=identifier,
            limit=max_results,
            show_deps=show_dependencies,
            dir_tree=show_tree,
            stats=True,
        )
        daemon_result = None
        if daemon_enabled():
            daemon_result = await asyncio.to_thread(_daemon_search, params, auto_start=False)

        if daemon_result is not None:
            # Daemon returned results - return formatted output
            logger.debug("Using daemon for find_identifier (fast path)")
            result_dict = {}
            if show_tree and "directory_tree" in daemon_result:
                result_dict["directory_tree"] = _strip_ansi(daemon_result["directory_tree"])
            result_dict.update({
                "results": _strip_ansi(daemon_result.get("output", "")),
                "count": daemon_result.get("result_count", 0),
                "search_time_ms": daemon_result.get("search_time_ms", 0),
                "source": "daemon",
            })
            return result_dict

        # Fallback: Create a fresh engine for each request
        logger.debug("Daemon not available, using direct find (fallback)")
        engine = await _create_engine(root_directory)

        # Run blocking find operation in thread pool for true async
        results = await asyncio.to_thread(
            engine.find_identifiers,
            identifier,
            limit=max_results,
            include_deps=show_dependencies
        )

        output = []
        for r in results:
            try:
                rel_path = str(Path(r.entity.file_path).relative_to(root_directory))
            except ValueError:
                rel_path = r.entity.file_path

            item = {
                "name": r.entity.name,
                "type": r.entity.entity_type,
                "file": rel_path,
                "line": r.entity.line_start,
                "score": r.score,
                "snippet": r.entity.get_context_snippet(max_lines=10),
            }

            if show_dependencies and r.dependency_context:
                item["dependencies"] = r.dependency_context.format_compact()

            output.append(item)

        result_dict = {}

        # Add directory tree first for LLM attention
        if show_tree:
            tree_output = await asyncio.to_thread(
                engine.generate_directory_tree, results
            )
            result_dict["directory_tree"] = _strip_ansi(tree_output)

        result_dict["results"] = output
        result_dict["source"] = "direct"

        return result_dict

    except Exception as e:
        logger.exception(f"Find identifier error: {e}")
        return {"error": str(e)}


async def check_python_virtual_env(
    root_directory: Annotated[
        str,
        Field(
            description="Absolute path to the project root directory to search for virtual environments"
        )
    ]
) -> dict[str, Any]:
    """Detects if a Python virtual environment exists in the project. Use this before running Python scripts or debugging import errors.

    WHEN TO USE:
    - Use this before attempting to run python scripts or install packages
    - Use this to verify the environment state before debugging import errors
    - Use this as a prerequisite step before calling get_python_package_path
    """
    # Run blocking file I/O in thread pool for true async
    def _check_venv_sync(root_path: Path) -> dict[str, Any]:
        # Search for directories containing pyvenv.cfg (limit depth to avoid deep recursion)
        for item in root_path.iterdir():
            if item.is_dir():
                pyvenv_cfg = item / "pyvenv.cfg"
                if pyvenv_cfg.exists():
                    return {"exists": True, "path": str(item)}
        return {"exists": False, "message": "No virtual environment found."}

    return await asyncio.to_thread(_check_venv_sync, Path(root_directory))


async def get_python_package_path(
    root_directory: Annotated[
        str,
        Field(
            description="Absolute path to the project root directory"
        )
    ],
    package_name: Annotated[
        str,
        Field(
            description="The import name of the package (e.g., 'numpy', 'yaml' not 'PyYAML')"
        )
    ]
) -> dict[str, Any]:
    """Locates the installation path of a third-party Python package so you can inspect its source code.

    WHEN TO USE:
    - When you need to read the source code of an external library (e.g., 'how does requests implement sessions?')
    - Use this to get the absolute path, then use code_search or code_map on that path

    USAGE STRATEGY:
    1. Call check_python_virtual_env to ensure an environment exists
    2. Call get_python_package_path with the import name (e.g., 'numpy')
    3. Use the returned path with code_search to inspect the library code
    """
    venv_info = await check_python_virtual_env(root_directory)
    if not venv_info["exists"]:
        return {"error": "No virtual environment found."}

    venv_path = Path(venv_info["path"])

    # Find Python executable (run in thread pool to avoid blocking)
    def _find_python_exe() -> Path | None:
        if (venv_path / "Scripts" / "python.exe").exists():
            return venv_path / "Scripts" / "python.exe"
        elif (venv_path / "bin" / "python").exists():
            return venv_path / "bin" / "python"
        return None

    python_exe = await asyncio.to_thread(_find_python_exe)
    if python_exe is None:
        return {"error": "Python executable not found."}

    # Try importlib approach first (best) - use async subprocess
    try:
        proc = await asyncio.create_subprocess_exec(
            str(python_exe), "-c",
            f"import {package_name}; print({package_name}.__file__)",
            stdin=asyncio.subprocess.DEVNULL,  # Prevent stdin conflicts with MCP
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

        if proc.returncode == 0:
            module_file = Path(stdout.decode().strip())
            if module_file.name == "__init__.py":
                return {"path": str(module_file.parent)}
            return {"path": str(module_file)}

    except (asyncio.TimeoutError, Exception):
        # Fallback: search site-packages directly
        try:
            proc = await asyncio.create_subprocess_exec(
                str(python_exe), "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                stdin=asyncio.subprocess.DEVNULL,  # Prevent stdin conflicts with MCP
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode == 0:
                site_packages = Path(stdout.decode().strip())

                # Check paths in thread pool (file I/O)
                def _check_package_paths():
                    pkg_path = site_packages / package_name
                    if pkg_path.exists() and pkg_path.is_dir():
                        return {"path": str(pkg_path)}

                    pkg_file = site_packages / f"{package_name}.py"
                    if pkg_file.exists():
                        return {"path": str(pkg_file)}
                    return None

                result = await asyncio.to_thread(_check_package_paths)
                if result:
                    return result

        except Exception:
            pass

    return {"error": f"Package '{package_name}' not found."}


@worker_safe
async def trace_dependency_path(
    root_directory: Annotated[
        str,
        Field(
            description="MUST be the absolute path to the project root directory"
        )
    ],
    start_identifier: Annotated[
        str,
        Field(
            description="The name of the function/class to start tracing from (e.g., 'process_payment')"
        )
    ],
    end_identifier: Annotated[
        str | None,
        Field(
            description="Optional target to check for connectivity (e.g., 'PaymentDatabase'). Leave empty to find all reachable nodes."
        )
    ] = None,
    direction: Annotated[
        Literal["upstream", "downstream", "both"],
        Field(
            description="upstream=who calls me, downstream=what do I call, both=bidirectional trace"
        )
    ] = "downstream",
    max_depth: Annotated[
        int,
        Field(
            description="How many layers deep to trace through the call graph",
            ge=1,
            le=20
        )
    ] = 5,
    limit_paths: Annotated[
        int,
        Field(
            description="Maximum number of paths to return",
            ge=1,
            le=100
        )
    ] = 10
) -> dict[str, Any]:
    """Trace execution or dependency paths between symbols to understand data flow and impact analysis.

    WHEN TO USE:
    - When you need to know 'Who calls this function?' or 'What does this function eventually trigger?'
    - To map the path between a Controller (API endpoint) and a Database Model to understand the logic chain
    - To analyze impact: 'If I change this utility, what high-level features break?'

    WHEN NOT TO USE:
    - Do NOT use for simple 'Find Usages' (use find_identifier with show_dependencies=True)
    - Do NOT use to list all files in a folder
    """
    # Try daemon first for fast in-memory access
    daemon_params = TraceDependencyParams(
        source_dir=str(Path(root_directory).resolve()),
        start_identifier=start_identifier,
        end_identifier=end_identifier,
        direction=direction,
        max_depth=max_depth,
        limit_paths=limit_paths,
    )
    daemon_result = None
    if daemon_enabled():
        daemon_result = _daemon_trace_dependency(daemon_params, auto_start=False)
    if daemon_result is not None:
        return daemon_result

    # Fallback to disk-based cache
    try:
        engine = await _create_engine(root_directory)

        result = await asyncio.to_thread(
            engine.trace_call_path,
            start_identifier=start_identifier,
            end_identifier=end_identifier,
            direction=direction,
            max_depth=max_depth,
            limit_paths=limit_paths
        )

        # Return enhanced output format optimized for LLM consumption
        return result.format_enhanced_output(root_directory=root_directory)

    except Exception as e:
        logger.error(f"Error tracing dependency path: {e}", exc_info=True)
        return {"error": str(e)}


def register_code_search_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register code search MCP tools on the given FastMCP instance.

    Args:
        mcp: FastMCP instance to register tools on.
        disabled_tools: Optional set of tool names to skip registration.
    """
    disabled = disabled_tools or set()

    all_tools = [
        code_search,
        code_map,
        find_identifier,
        trace_dependency_path,
        check_python_virtual_env,
        get_python_package_path,
        debug_stacktrace,
        read_source_range,
        read_source_ranges,
        git_history_context,
        find_hotspots,
        code_evolution,
    ]

    for func in all_tools:
        if func.__name__ not in disabled:
            mcp.tool()(func)
