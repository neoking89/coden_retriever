"""Optional dependency management for coden-retriever.

Provides utilities to check and require optional features like semantic search,
MCP server, and agent mode. Each feature requires additional dependencies that
can be installed via pip extras.

Features:
    - semantic: Model2Vec, numpy, scipy for semantic code search
    - mcp: FastMCP for MCP server mode
    - agent: Pydantic-AI for interactive agent mode

Example:
    >>> from coden_retriever.utils.optional_deps import require_feature
    >>> require_feature("semantic")  # Raises MissingDependencyError if not installed
"""

from types import ModuleType
from typing import Literal

# Feature names as literals for type safety
FeatureName = Literal["semantic", "mcp", "agent"]

# Maps features to their pip install commands
_FEATURE_INSTALL_COMMANDS: dict[FeatureName, str] = {
    "semantic": "pip install 'coden-retriever[semantic]'",
    "mcp": "pip install 'coden-retriever[mcp]'",
    "agent": "pip install 'coden-retriever[agent]'",
}

# Maps features to human-readable descriptions
_FEATURE_DESCRIPTIONS: dict[FeatureName, str] = {
    "semantic": "Semantic search",
    "mcp": "MCP server",
    "agent": "Interactive agent",
}

# Maps features to their required packages (for availability check)
_FEATURE_PACKAGES: dict[FeatureName, list[str]] = {
    "semantic": ["model2vec", "numpy", "scipy"],
    "mcp": ["fastmcp"],
    "agent": ["pydantic_ai"],
}


class MissingDependencyError(ImportError):
    """Raised when a required optional dependency is missing.

    Provides a helpful error message with installation instructions.
    """

    def __init__(self, feature: FeatureName):
        self.feature = feature
        description = _FEATURE_DESCRIPTIONS[feature]
        install_cmd = _FEATURE_INSTALL_COMMANDS[feature]
        message = (
            f"{description} requires additional dependencies.\n"
            f"Install with: {install_cmd}"
        )
        super().__init__(message)


def is_feature_available(feature: FeatureName) -> bool:
    """Check if all packages for a feature are importable.

    Args:
        feature: The feature to check (semantic, mcp, or agent).

    Returns:
        True if all required packages are available, False otherwise.
    """
    packages = _FEATURE_PACKAGES.get(feature, [])
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            return False
    return True


def require_feature(feature: FeatureName) -> None:
    """Require a feature to be available, raising an error if not.

    Args:
        feature: The feature to require (semantic, mcp, or agent).

    Raises:
        MissingDependencyError: If any required package is not installed.
    """
    if not is_feature_available(feature):
        raise MissingDependencyError(feature)


def get_numpy() -> ModuleType:
    """Lazy import numpy with helpful error message.

    Returns the numpy module, raising MissingDependencyError if not installed.
    This centralizes numpy imports across all semantic-related modules.

    Returns:
        The numpy module.

    Raises:
        MissingDependencyError: If numpy is not installed.
    """
    try:
        import numpy as np
        return np
    except ImportError:
        raise MissingDependencyError("semantic")
