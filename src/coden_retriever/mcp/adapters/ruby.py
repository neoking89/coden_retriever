"""Ruby / `rdbg` adapter.

Socket-transport DAP adapter powered by the `debug` gem (stdlib on Ruby
≥ 3.1). rdbg has **no stdio DAP mode** — `--stdio` is not a recognized
flag on any released version (OptionParser rejects it). The only DAP-
capable transport is TCP via `--open=vscode --port=<N> --host=<H>`,
where the `vscode` frontend tag selects DAP framing over the socket.

`--open=vscode` has an unfortunate side effect: `lib/debug/open.rb`
invokes `system("code", "--open-url", ...)` to pop a VS Code window.
We don't want that during MCP-driven debug sessions, so
`build_launch_env` scrubs `code` (and its `.cmd`/`.exe` aliases) out
of the subprocess PATH. rdbg's `system(...)` call then fails silently
and DAP-over-TCP continues uninterrupted.

`ruby -S rdbg` is used instead of bare `rdbg` because RubyInstaller
ships `rdbg.bat` on Windows and `subprocess.Popen(shell=False)` cannot
execute a `.bat` wrapper. `ruby -S` searches PATH for the target script
and re-execs the interpreter on it, which works uniformly across OSes.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .availability import DebugDependencyStatus, binary_dependency_status
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_RUBY_BINARY = "ruby"
_RDBG_SCRIPT = "rdbg"
_INSTALL_HINT = "gem install debug"
_RUBY_RUNTIME_HINT = "Install Ruby and ensure `ruby` is on PATH"
_LISTEN_HOST = "127.0.0.1"
# rdbg's `-O/--open=FRONTEND` accepts {rdbg, vscode, chrome}. `vscode`
# selects DAP framing over the TCP socket; the other two speak rdbg's
# own console or Chrome DevTools protocols and are unusable here.
_DAP_FRONTEND = "vscode"
# rdbg's VSCode.open_vscode invokes `code` via PATH lookup. Dropping any
# PATH directory whose resolved `code` is one of these names prevents the
# auto-launch without breaking the DAP socket. Both .cmd and .exe covered
# because Windows VS Code installs both wrappers in the same bin/ dir.
_VSCODE_LAUNCHER_NAMES = ("code", "code.cmd", "code.exe")


class RubyAdapter(DebugAdapter):
    """rdbg-based DAP adapter for Ruby."""

    name = "ruby"
    file_extensions = (".rb",)
    transport_type = "socket"
    adapter_id = "ruby"

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(_RUBY_BINARY) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_RUBY_BINARY,
                kind="runtime",
                name="Ruby runtime",
                install_hint=_RUBY_RUNTIME_HINT,
            ),
            binary_dependency_status(
                binary=_RDBG_SCRIPT,
                kind="debugger",
                name="rdbg",
                install_hint=_INSTALL_HINT,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        if port is None:
            raise ValueError("RubyAdapter requires a port (socket transport)")
        argv = [
            _RUBY_BINARY,
            "-S",
            _RDBG_SCRIPT,
            f"--open={_DAP_FRONTEND}",
            f"--port={port}",
            f"--host={_LISTEN_HOST}",
            "--",
            cfg.program,
        ]
        argv.extend(cfg.args)
        return argv

    def build_launch_env(self, cfg: LaunchConfig) -> dict[str, str] | None:
        # Filter PATH entries whose directory contains a `code` launcher so
        # rdbg's VSCode.open_vscode can't find it. Compare by resolved path
        # rather than string-matching PATH entries — case differences and
        # trailing separators are common on Windows.
        path = os.environ.get("PATH", "")
        kept: list[str] = []
        for entry in path.split(os.pathsep):
            entry_clean = entry.strip('"').rstrip(os.sep)
            if not entry_clean:
                continue
            try:
                directory = Path(entry_clean)
            except (OSError, ValueError):
                kept.append(entry)
                continue
            has_code = any(
                (directory / name).is_file() for name in _VSCODE_LAUNCHER_NAMES
            )
            if not has_code:
                kept.append(entry)
        return {"PATH": os.pathsep.join(kept)}

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        return {
            "program": cfg.program,
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "stopOnEntry": cfg.stop_on_entry,
        }


REGISTRY.register(RubyAdapter())
