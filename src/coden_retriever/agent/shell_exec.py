"""Execute shell commands typed with the `!` prefix inside coden -a.

Inherits the parent shell (bash / PowerShell / cmd). Output is captured,
truncated at a size ceiling, and returned as a ShellResult for rendering
and optional hand-off to the LLM.
"""

import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from ..constants import SHELL_EXIT_COMMAND_NOT_FOUND, SHELL_OUTPUT_MAX_SIZE


class ShellKind(str, Enum):
    """Which shell family the user launched coden from."""
    BASH_FAMILY = "bash"
    POWERSHELL = "pwsh"
    CMD = "cmd"


@dataclass
class ShellResult:
    """Captured result of a single shell command invocation."""
    cmd: str
    stdout: str
    stderr: str
    returncode: int
    elapsed_s: float
    timed_out: bool
    truncated: bool
    shell_kind: ShellKind


_BASH_FAMILY_NAMES = {"bash", "zsh", "sh", "dash", "fish", "ash"}


def detect_shell() -> tuple[ShellKind, str]:
    """Detect the parent shell and return (kind, executable_path).

    Resolution order:
      1. Explicit override via ``$CODEN_SHELL``.
      2. POSIX ``$SHELL`` pointing at a bash-family shell.
      3. Windows with ``$PSModulePath`` set -> PowerShell (pwsh > powershell).
      4. Windows fallback -> cmd.exe.
      5. POSIX fallback -> /bin/sh.
    """
    override = os.environ.get("CODEN_SHELL", "").strip().lower()
    if override:
        return _resolve_override(override)

    shell_env = os.environ.get("SHELL", "")
    if shell_env:
        name = Path(shell_env).stem.lower()
        if name in _BASH_FAMILY_NAMES:
            return ShellKind.BASH_FAMILY, shell_env

    if sys.platform == "win32":
        if os.environ.get("PSModulePath"):
            return ShellKind.POWERSHELL, _find_powershell()
        return ShellKind.CMD, os.environ.get("COMSPEC") or "cmd.exe"

    return ShellKind.BASH_FAMILY, "/bin/sh"


def _resolve_override(value: str) -> tuple[ShellKind, str]:
    """Resolve the CODEN_SHELL override to a concrete (kind, path)."""
    if value in {"pwsh", "powershell"}:
        return ShellKind.POWERSHELL, _find_powershell(prefer=value)
    if value == "cmd":
        return ShellKind.CMD, os.environ.get("COMSPEC") or "cmd.exe"
    # Anything else is treated as a bash-family executable name or path.
    path = shutil.which(value) or value
    return ShellKind.BASH_FAMILY, path


def _find_powershell(prefer: str = "pwsh") -> str:
    """Pick the best available PowerShell binary, preferring PS 7 (pwsh)."""
    order = ("pwsh", "powershell") if prefer == "pwsh" else ("powershell", "pwsh")
    for name in order:
        found = shutil.which(name)
        if found:
            return found
    return "powershell"  # Windows always ships this name even if not on PATH


def _build_argv(cmd: str, shell: ShellKind, shell_path: str) -> list[str]:
    """Map (shell kind, user command) -> argv for create_subprocess_exec."""
    if shell is ShellKind.POWERSHELL:
        return [shell_path, "-NoProfile", "-NonInteractive", "-Command", cmd]
    if shell is ShellKind.CMD:
        return [shell_path, "/C", cmd]
    # WSL from the Windows side: wsl.exe doesn't accept `-c` directly —
    # it expects `wsl.exe -- <program> <args...>`. Detect by executable stem.
    if Path(shell_path).stem.lower() == "wsl":
        return [shell_path, "--", "bash", "-c", cmd]
    return [shell_path, "-c", cmd]


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Clip text to *limit* chars; flag whether truncation occurred."""
    if len(text) <= limit:
        return text, False
    suffix = f"\n... [output truncated at {limit:,} chars]"
    return text[: limit - len(suffix)] + suffix, True


async def _spawn_process(
    argv: list[str], cwd: str,
) -> "asyncio.subprocess.Process | None":
    """Launch the subprocess; return None (with error printed) on missing binary."""
    try:
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None


async def _collect_output(
    proc: "asyncio.subprocess.Process", timeout: float,
) -> tuple[bytes, bytes, bool]:
    """Wait for process output with a timeout; kill on expiry or cancellation.

    Returns (stdout_bytes, stderr_bytes, timed_out).
    """
    stdout_b: bytes = b""
    stderr_b: bytes = b""
    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout_b, stderr_b = await proc.communicate()
    except (KeyboardInterrupt, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    return stdout_b, stderr_b, timed_out


async def execute_shell(
    cmd: str,
    cwd: str,
    timeout: float,
    shell: ShellKind,
    shell_path: str,
) -> ShellResult:
    """Run *cmd* in the given *shell* and return a :class:`ShellResult`."""
    argv = _build_argv(cmd, shell, shell_path)
    start = time.monotonic()
    proc = await _spawn_process(argv, cwd)
    if proc is None:
        # Shell binary itself missing — surface clearly instead of crashing.
        return ShellResult(
            cmd=cmd, stdout="",
            stderr=f"shell not found: {shell_path!r}",
            returncode=SHELL_EXIT_COMMAND_NOT_FOUND, elapsed_s=0.0,
            timed_out=False, truncated=False, shell_kind=shell,
        )

    stdout_b, stderr_b, timed_out = await _collect_output(proc, timeout)
    elapsed = time.monotonic() - start
    stdout, trunc_out = _truncate(_decode(stdout_b), SHELL_OUTPUT_MAX_SIZE)
    stderr, trunc_err = _truncate(_decode(stderr_b), SHELL_OUTPUT_MAX_SIZE)
    return ShellResult(
        cmd=cmd,
        stdout=stdout,
        stderr=stderr,
        returncode=proc.returncode if proc.returncode is not None else -1,
        elapsed_s=elapsed,
        timed_out=timed_out,
        truncated=trunc_out or trunc_err,
        shell_kind=shell,
    )


def _decode(raw: Optional[bytes]) -> str:
    """Decode subprocess bytes with a safe fallback for OEM/ANSI output."""
    return raw.decode("utf-8", errors="replace") if raw else ""


def format_shell_message(result: ShellResult, query: str) -> str:
    """Format a :class:`ShellResult` + follow-up query for the LLM.

    Shape mirrors the `@file` reference format from file_reference.py so the
    model sees a familiar structure: command echo, fenced stdout, optional
    stderr, exit-status footer, then the user's question.
    """
    lines = [f"[Output of `{result.cmd}` ({result.shell_kind.value})]"]
    lines.append("```")
    lines.append(result.stdout.rstrip("\n"))
    lines.append("```")
    if result.stderr.strip():
        lines.append("[stderr]")
        lines.append("```")
        lines.append(result.stderr.rstrip("\n"))
        lines.append("```")
    footer_bits = [f"exit {result.returncode}"]
    if result.timed_out:
        footer_bits.append("timed out")
    if result.truncated:
        footer_bits.append("truncated")
    lines.append(f"({', '.join(footer_bits)})")
    if query:
        lines.append("")
        lines.append(query)
    return "\n".join(lines)
