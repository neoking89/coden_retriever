"""Cross-platform git command execution utilities.

This module provides async git command execution with:
- Windows compatibility (CREATE_NO_WINDOW to prevent console popup)
- Timeout handling with proper process cleanup
- Consistent error handling and return format
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path


async def run_git_command(
    args: list[str],
    cwd: str,
    timeout: float = 30.0,
) -> tuple[int, str, str]:
    """Execute a git command with cross-platform compatibility.

    Args:
        args: Git command arguments (without 'git' prefix).
        cwd: Working directory for the command.
        timeout: Maximum time to wait in seconds (default: 30).

    Returns:
        Tuple of (returncode, stdout, stderr).
        returncode is -1 for timeout or other errors.
    """
    try:
        # Set environment to disable interactive prompts
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"  # Disable credential prompts

        # Windows-specific: CREATE_NO_WINDOW prevents console popup
        creationflags = 0
        if sys.platform == "win32":
            creationflags = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )

        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            creationflags=creationflags,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        # Kill the process if it times out
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return -1, "", f"Command timed out after {timeout}s (cwd: {cwd})"
    except FileNotFoundError:
        return -1, "", "git command not found"
    except Exception as e:
        return -1, "", str(e)


async def is_git_repository(path: str) -> bool:
    """Check if the given path is inside a git repository.

    Args:
        path: Path to check (file or directory).

    Returns:
        True if path is inside a git repository.
    """
    # Use the directory if path is a file
    check_path = path if Path(path).is_dir() else str(Path(path).parent)
    returncode, _, _ = await run_git_command(
        ["rev-parse", "--git-dir"],
        cwd=check_path,
        timeout=5.0,
    )
    return returncode == 0


async def get_git_root(path: str) -> str | None:
    """Get the root directory of the git repository containing the path.

    Args:
        path: Path inside the repository (file or directory).

    Returns:
        Absolute path to repository root, or None if not a git repo.
    """
    # Use the directory if path is a file
    check_path = path if Path(path).is_dir() else str(Path(path).parent)
    returncode, stdout, _ = await run_git_command(
        ["rev-parse", "--show-toplevel"],
        cwd=check_path,
        timeout=5.0,
    )
    if returncode == 0:
        return stdout.strip()
    return None
