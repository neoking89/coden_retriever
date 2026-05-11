"""Shared MCP tool validation helpers.

Centralizes common validation patterns used across all MCP tools
to avoid duplicating error messages and validation logic.
"""

import os
from typing import Any

# Shared error messages for consistent user-facing errors
ERR_ROOT_REQUIRED = "root_directory is required"
ERR_ROOT_NOT_FOUND_FMT = "Root directory not found: {}"


def validate_root_directory(root_directory: str) -> dict[str, Any] | None:
    """Validate that root_directory is provided and exists.

    Returns:
        Error dict if validation fails, None if valid.
    """
    if not root_directory:
        return {"error": ERR_ROOT_REQUIRED}
    if not os.path.isdir(root_directory):
        return {"error": ERR_ROOT_NOT_FOUND_FMT.format(root_directory)}
    return None
