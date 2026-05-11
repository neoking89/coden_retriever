"""
Configuration module for coden-retriever.

Contains enums, tuning parameters, and configuration settings.
"""
import hashlib
from enum import Enum
from pathlib import Path

CENTRAL_CACHE_DIR = ".coden-retriever"


def get_central_cache_root() -> Path:
    """Get the central cache root directory (~/.coden-retriever/).

    All project caches are stored under this directory, organized by
    a hash of the project path for easy management.

    Returns:
        Path to ~/.coden-retriever/
    """
    return Path.home() / CENTRAL_CACHE_DIR


def get_project_cache_key(source_dir: Path) -> str:
    """Generate a unique cache key for a project directory.

    Uses a combination of the directory name and a hash of the full path
    for human-readability while ensuring uniqueness.

    Args:
        source_dir: The project source directory (should be resolved/absolute)

    Returns:
        A string like "my_project_a1b2c3d4" that can be used as a directory name
    """
    source_dir = Path(source_dir).resolve()
    # Use the directory name + short hash of full path for readability + uniqueness
    dir_name = source_dir.name or "root"
    # Clean directory name to be filesystem-safe
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dir_name)
    # Create hash of full path for uniqueness (16 chars = 64 bits for better collision resistance)
    path_hash = hashlib.sha256(str(source_dir).encode()).hexdigest()[:16]
    return f"{safe_name}_{path_hash}"


def get_project_cache_dir(source_dir: Path) -> Path:
    """Get the cache directory for a specific project.

    Args:
        source_dir: The project source directory

    Returns:
        Path to ~/.coden-retriever/{project_key}/
    """
    cache_key = get_project_cache_key(source_dir)
    return get_central_cache_root() / cache_key


class EntityType(Enum):
    """Types of code entities we extract."""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    STRUCT = "struct"
    INTERFACE = "interface"
    CONSTANT = "constant"
    VARIABLE = "variable"


class OutputFormat(Enum):
    """Output format options."""
    XML = "xml"
    MARKDOWN = "markdown"
    TREE = "tree"
    JSON = "json"


class MapMode(str, Enum):
    """Ranking strategy for code-map output.

    `str` mixin makes values JSON-serialisable for the daemon protocol and
    interchangeable with bare strings (`mode == "simple"` works).
    """
    STATIC = "static"   # Combined-signal score (PageRank + Betweenness + BM25 + ...)
    SIMPLE = "simple"   # Per-file git change_count + per-object blame


class Config:
    """Centralized configuration for the search engine."""

    # Edge weights for different relationship types
    EDGE_WEIGHTS: dict[str, float] = {
        "inherit": 3.0,   # Class inheritance (strongest coupling)
        "call": 1.5,      # Function/method calls
        "import": 0.5,    # Module imports (loose coupling)
        "usage": 1.0,     # Variable/type usage
        "type": 1.0,      # Type annotations — between import (0.5) and call (1.5); "X parameterizes Y" is structural but weaker than "X calls Y"
    }

    # Functions that are utility sinks (high in-degree, low informational value)
    UTILITY_NAMES: set[str] = {
        "print", "println", "printf", "eprint", "eprintln",
        "log", "debug", "info", "warn", "error", "trace", "fatal",
        "console", "assert", "panic", "exit", "len", "str", "int",
        "float", "bool", "list", "dict", "set", "tuple", "range",
        "open", "close", "read", "write", "append", "extend",
        "get", "set", "has", "delete", "remove", "pop", "push",
        "toString", "valueOf", "hasOwnProperty", "getElementById",
        "querySelector", "addEventListener", "setTimeout", "setInterval",
    }

    # Directories to skip during indexing
    SKIP_DIRS: set[str] = {
        "venv", "env", ".venv", ".env",
        "node_modules", "bower_components",
        ".git", ".svn", ".hg",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "dist", "build", "target", "out", "bin", "obj",
        "vendor", "third_party", "external", "deps",
        ".idea", ".vscode", ".vs",
        "coverage", ".coverage", "htmlcov",
        ".tox", ".nox",
    }

    # Files to skip during indexing
    SKIP_FILES: set[str] = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "Pipfile.lock", "Cargo.lock",
        "go.sum", "composer.lock", "Gemfile.lock",
    }

    # BM25 hyperparameters
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    BM25_NAME_BOOST: int = 7  # Repeat entity name N times to boost name matches

    # RRF fusion constant
    RRF_K: int = 30  # Lower values favor top-ranked items more strongly

    # Ranking weights for search mode (BM25/lexical)
    WEIGHT_BM25: float = 4.0              # Lexical match priority (leading signal in query mode)
    WEIGHT_PAGERANK_BM25: float = 0.5     # Structural importance (BM25 mode)
    WEIGHT_BETWEENNESS_BM25: float = 0.5  # Architectural bridges
    WEIGHT_TYPE_REF_BM25: float = 0.3     # Type-annotation centrality (lower in query mode — query already constrains)

    # Ranking weights for semantic search mode (separate to allow semantic to dominate)
    WEIGHT_SEMANTIC: float = 6.0      # Semantic similarity
    WEIGHT_PAGERANK_SEMANTIC: float = 0.3   # Lower PR weight when semantic active
    WEIGHT_BETWEENNESS_SEMANTIC: float = 0.5  # Lower betweenness when semantic active
    WEIGHT_TYPE_REF_SEMANTIC: float = 0.3

    # Ranking weights for map mode (no query)
    MAP_WEIGHT_PAGERANK: float = 0.5
    MAP_WEIGHT_BETWEENNESS: float = 0.5
    MAP_WEIGHT_DISPATCHER: float = 0.5      # Hotspots-style risk score; surfaces high-CC, high-fan_out dispatchers
    MAP_WEIGHT_ENTRY: float = 1.5           # Reverse-PageRank; surfaces entry-point roots (e.g. main, serverCron) that PR/BT/Disp systematically bury — needs more weight than Disp on class-heavy codebases (Python) where class aggregation otherwise dominates BT/PR-light functions
    MAP_WEIGHT_BM25: float = 0.3            # Static IDF distinctiveness in map mode; lower than structural signals because lexical alone over-rewards verbose docstrings
    MAP_WEIGHT_TYPE_REF: float = 1.0        # Type-annotation centrality; surfaces dataclasses/configs/DTOs. Stronger than baseline structural signals because passive data classes get ~0 contribution from method aggregation
    MAP_AGGREGATION_DAMPENING: float = 0.3  # How much method scores boost parent classes (0=none, 1=full)
    MAP_PENALTY_METHOD: int = 5             # Mild penalty for methods/functions in map mode (lower = more balanced)


    # Penalty factors
    PENALTY_PRIVATE: int = 20         # Rank positions to penalize private methods
    PENALTY_SINKHOLE: float = 0.4     # Multiplier for utility sinks
    PENALTY_TINY_FUNC: float = 1   # Multiplier for functions < 5 lines
    PENALTY_UTILITY: float = 0.1      # Multiplier for known utility functions
    PENALTY_TEST: float = 0.3         # Multiplier for test files/functions
    PENALTY_METHOD: int = 300         # Rank penalty for methods/functions in map mode (favor class-level entities)
    QUERY_PENALTY_METHOD: int = 0     # No penalty in query mode (user searched for something specific)

    # Thresholds
    SINKHOLE_IN_THRESHOLD: float = 5.0
    SINKHOLE_OUT_THRESHOLD: float = 1.0
    TINY_FUNCTION_LINES: int = 5

    # Context extraction settings
    CONTEXT_LINES_BEFORE: int = 2     # Lines before definition
    CONTEXT_LINES_AFTER: int = 3      # Lines after definition starts
    MAX_BODY_LINES: int = 15          # Max lines to show from function body
    COLLAPSE_THRESHOLD: int = 20      # Collapse bodies longer than this

    # Dependency context settings
    DEPENDENCY_MAX_CALLERS: int = 3   # Max callers to show
    DEPENDENCY_MAX_CALLEES: int = 3   # Max callees to show
    DEPENDENCY_MIN_WEIGHT: float = 0.1  # Min edge weight to include

    # Output settings
    MAX_CODE_LINES: int = 50
    TREE_INDENT: str = "  "

    # Important files to always consider
    IMPORTANT_FILES: set[str] = {
        "README.md", "README.txt", "README.rst", "README",
        "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "Cargo.toml", "go.mod", "pom.xml",
        "Makefile", "CMakeLists.txt", "Dockerfile",
        "main.py", "app.py", "index.js", "index.ts", "main.go",
        "main.rs", "Main.java", "Program.cs",
    }

    # Semantic search settings
    SEMANTIC_SCORE_THRESHOLD: float = 0.1  # Minimum similarity score to include
    SEMANTIC_IRRELEVANT_PENALTY: float = 0.4  # Score multiplier for entities with no semantic match
