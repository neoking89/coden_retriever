"""Go / `dlv dap` adapter.

Socket-transport DAP adapter powered by Delve. `dlv dap` has been headless-
TCP-only since at least v1.20 — its own `--help` text reads *"Starts a
headless TCP server"* and there is no stdio mode. The adapter binds dlv to
an ephemeral loopback port and `DAPClient.SocketTransport` connects to it.

Unlike `PythonAdapter` this sticks to the DAP-spec default
`launch_request_command = "launch"` — the `"attach"` override is a debugpy
quirk, not a shared convention.

Mode selection follows `cfg.extras.get("mode", "debug")` so MCP callers
who eventually need `"exec"` / `"test"` / `"replay"` / `"core"` can pass
them through without an adapter-specific API. MVP only ships `"debug"`
— it compiles + runs from source and is the only mode exercised in
Phase 3 tests.
"""
from __future__ import annotations

import shutil
from typing import Any

from .availability import DebugDependencyStatus, binary_dependency_status
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_DLV_BINARY = "dlv"
_GO_BINARY = "go"
_INSTALL_HINT = "go install github.com/go-delve/delve/cmd/dlv@latest"
_GO_RUNTIME_HINT = "Install Go from https://go.dev/dl/"
# "debug" auto-compiles with -gcflags="all=-N -l" (inlining/optimizations
# off — required for line-accurate breakpoints). "exec" on a pre-built
# binary requires the user to have passed those flags manually; see
# archive/debugger_mcp_docs/debug-adapters.md.
_DEFAULT_MODE = "debug"
_LISTEN_HOST = "127.0.0.1"


class GoAdapter(DebugAdapter):
    """Delve-based DAP adapter for Go."""

    name = "go"
    file_extensions = (".go",)
    transport_type = "socket"
    adapter_id = "go"

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(_DLV_BINARY) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_GO_BINARY,
                kind="runtime",
                name="Go toolchain",
                install_hint=_GO_RUNTIME_HINT,
            ),
            binary_dependency_status(
                binary=_DLV_BINARY,
                kind="debugger",
                name="Delve debugger",
                install_hint=_INSTALL_HINT,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        if port is None:
            raise ValueError("GoAdapter requires a port (socket transport)")
        return [_DLV_BINARY, "dap", "--listen", f"{_LISTEN_HOST}:{port}"]

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        return {
            "mode": cfg.extras.get("mode", _DEFAULT_MODE),
            "program": cfg.program,
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "stopOnEntry": cfg.stop_on_entry,
        }


REGISTRY.register(GoAdapter())
