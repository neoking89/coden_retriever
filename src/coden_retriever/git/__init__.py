"""Git utilities for code analysis.

This module provides cross-platform git command execution and output parsing
for the git-aware code analysis tools.
"""

from coden_retriever.git.commands import (
    run_git_command,
    is_git_repository,
    get_git_root,
)

__all__ = [
    "run_git_command",
    "is_git_repository",
    "get_git_root",
]
