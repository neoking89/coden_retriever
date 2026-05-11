"""Bash adapter via `vscode-bash-debug` (Node bridge → bashdb).

The upstream Node bridge (`bash-debug` npm package) wraps `bashdb`
and speaks DAP on stdio. Linux/macOS only — `bashdb` is not packaged
for Windows, and the bridge's FIFO machinery is POSIX-specific.
Windows callers get a hard-false `detect_installed()` with a
"use WSL" hint.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from ._node_bridge import NODE_BINARY, resolve_bridge_script
from .availability import (
    DebugDependencyStatus,
    binary_dependency_status,
    missing_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY


def _which_or(binary: str, fallback: str) -> str:
    """Return `shutil.which(binary)` or `fallback` when absent — the bridge
    calls `child_process.spawnSync` on every path field, and `undefined`
    crashes it with ERR_INVALID_ARG_TYPE before emitting `initialized`.
    """
    return shutil.which(binary) or fallback

_BRIDGE_ENV_VAR = "VSCODE_BASH_DEBUG"
_BRIDGE_RELATIVE_PATHS = (
    "node_modules/bash-debug/out/bashDebug.js",
    "lib/node_modules/bash-debug/out/bashDebug.js",
)
_BASH_BINARY = "bash"
_BASHDB_BINARY = "bashdb"
_INSTALL_HINT = (
    "Install bashdb (Linux: apt install bashdb; macOS: brew install bashdb) "
    "AND: npm install -g bash-debug"
)
_BASH_RUNTIME_HINT = "Install Bash and ensure `bash` is on PATH"
_WINDOWS_HINT = (
    "vscode-bash-debug is not supported on native Windows — use WSL instead"
)
# `debugConsole` is the upstream default that does NOT emit a
# runInTerminal reverse-request; Phase 5 keeps it as the default here too
# so the reverse-request hook (Phase 5 C1) can respond with a well-formed
# refusal for users who explicitly override to "integrated"/"external".
_DEFAULT_TERMINAL_KIND = "debugConsole"


class BashAdapter(DebugAdapter):
    """bash-debug (bashdb wrapper) Node-bridge DAP adapter for Bash."""

    name = "bash"
    file_extensions = (".sh", ".bash")
    transport_type = "stdio"
    adapter_id = "bashdb"

    def detect_installed(self) -> tuple[bool, str]:
        if sys.platform == "win32":
            return (False, _WINDOWS_HINT)
        if shutil.which(NODE_BINARY) is None:
            return (False, _INSTALL_HINT)
        if shutil.which(_BASHDB_BINARY) is None:
            return (False, _INSTALL_HINT)
        if resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        if sys.platform == "win32":
            return (
                missing_dependency_status(
                    kind="platform",
                    name="POSIX shell environment",
                    install_hint=_WINDOWS_HINT,
                    detail=_WINDOWS_HINT,
                ),
            )
        return (
            binary_dependency_status(
                binary=_BASH_BINARY,
                kind="runtime",
                name="Bash runtime",
                install_hint=_BASH_RUNTIME_HINT,
            ),
            binary_dependency_status(
                binary=NODE_BINARY,
                kind="debugger",
                name="Node.js",
                install_hint=_INSTALL_HINT,
            ),
            binary_dependency_status(
                binary=_BASHDB_BINARY,
                kind="debugger",
                name="bashdb",
                install_hint=_INSTALL_HINT,
            ),
            resolver_dependency_status(
                kind="debugger",
                name="bash-debug",
                install_hint=_INSTALL_HINT,
                resolver=lambda: resolve_bridge_script(
                    _BRIDGE_ENV_VAR,
                    _BRIDGE_RELATIVE_PATHS,
                ),
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        bridge = resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS)
        if bridge is None:
            raise RuntimeError(_INSTALL_HINT)
        return [NODE_BINARY, bridge]

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        # The Node bridge's `validatePath` feeds every `path*` field into
        # child_process.spawnSync. Any undefined field throws
        # ERR_INVALID_ARG_TYPE and the bridge exits before it can emit the
        # DAP `initialized` event, so we fill in PATH-resolved fallbacks
        # rather than relying on the extension's JSON-schema defaults.
        return {
            "program": cfg.program,
            "args": list(cfg.args),
            "argsString": cfg.extras.get("args_string", ""),
            "cwd": cfg.cwd or self._default_cwd(cfg),
            "env": cfg.extras.get("env", {}),
            "stopOnEntry": cfg.stop_on_entry,
            "pathBash": cfg.extras.get("bash_path", _which_or("bash", "/bin/bash")),
            "pathBashdb": cfg.extras.get("bashdb_path", _BASHDB_BINARY),
            "pathBashdbLib": cfg.extras.get("bashdb_lib", "/usr/share/bashdb"),
            "pathCat": cfg.extras.get("cat_path", _which_or("cat", "/bin/cat")),
            "pathMkfifo": cfg.extras.get("mkfifo_path", _which_or("mkfifo", "/usr/bin/mkfifo")),
            "pathPkill": cfg.extras.get("pkill_path", _which_or("pkill", "/usr/bin/pkill")),
            "terminalKind": cfg.extras.get("terminal_kind", _DEFAULT_TERMINAL_KIND),
            "showDebugOutput": cfg.extras.get("show_debug_output", False),
            "trace": cfg.extras.get("trace", False),
        }


    def _default_cwd(self, cfg: LaunchConfig) -> str:
        """Derive a non-empty cwd: bridge's `cd "${cwd}"` silently breaks on empty."""
        program = Path(cfg.program)
        if program.exists():
            return str(program.parent)
        return os.getcwd()


REGISTRY.register(BashAdapter())
