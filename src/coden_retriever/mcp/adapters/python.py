"""Python / debugpy adapter.

Encapsulates the debugpy-specific logic currently hardcoded in `dap_client.py`:
- launch argv construction (lines 290-295)
- `adapterID: "debugpy"` in the initialize request (lines 201, 354)
- the `justMyCode` / `redirectOutput` body of the attach-phase request (lines 378-380)

Phase 0 builds this class but does NOT wire it — `DAPClient` still uses the
hardcoded bits. Phase 1 deletes those hardcodes and routes through here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from .availability import (
    DebugDependencyStatus,
    module_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_ADAPTER_ID = "debugpy"
_INSTALL_HINT = "pip install debugpy"
_RUNTIME_INSTALL_HINT = "Install Python 3 and run coden with that interpreter"


class PythonAdapter(DebugAdapter):
    """debugpy-based DAP adapter for Python."""

    name = "python"
    file_extensions = (".py",)
    transport_type = "socket"
    adapter_id = _ADAPTER_ID
    # debugpy oddity: connect over TCP after listener is up, then send DAP
    # `attach` (with justMyCode/redirectOutput) — DAP spec `launch` would
    # spawn a second process. See dap_client.py launch flow.
    launch_request_command = "attach"

    def detect_installed(self) -> tuple[bool, str]:
        if importlib.util.find_spec("debugpy") is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            resolver_dependency_status(
                kind="runtime",
                name="Python runtime",
                install_hint=_RUNTIME_INSTALL_HINT,
                resolver=lambda: Path(sys.executable).is_file(),
            ),
            module_dependency_status(
                module_name="debugpy",
                kind="debugger",
                name="debugpy",
                install_hint=_INSTALL_HINT,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        if port is None:
            raise ValueError("PythonAdapter requires a port (socket transport)")
        python = sys.executable
        argv = [
            python, "-m", "debugpy",
            "--listen", f"127.0.0.1:{port}",
            "--wait-for-client",
            cfg.program,
        ]
        argv.extend(cfg.args)
        return argv

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        return {"justMyCode": False, "redirectOutput": True}


REGISTRY.register(PythonAdapter())
