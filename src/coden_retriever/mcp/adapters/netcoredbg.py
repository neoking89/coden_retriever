"""Samsung `netcoredbg` adapter for C# / .NET.

stdio-transport DAP adapter powered by the Samsung `netcoredbg` binary
(permissively licensed alternative to Microsoft's `vsdbg`, which cannot
be redistributed outside Visual Studio products).

Compiled-language semantics match LLDBAdapter: `cfg.program` is the
string the user typed (often a `.cs` source); `cfg.extras["executable"]`
overrides with the built `.dll` / native binary path that netcoredbg
actually runs.

Docker / container symbol lookup often needs `symbolSearchPaths` — we
forward `cfg.extras["symbol_search_paths"]` (a list of directories) to
the DAP `launch` body when provided. The key is omitted entirely when
the user doesn't supply it, so the default DAP body stays minimal.
"""
from __future__ import annotations

import shutil
from typing import Any

from .availability import DebugDependencyStatus, binary_dependency_status
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_NETCOREDBG_BINARY = "netcoredbg"
_DOTNET_BINARY = "dotnet"
_INSTALL_HINT = (
    "Download from github.com/Samsung/netcoredbg/releases "
    "(avoid vsdbg — MS license)"
)
_DOTNET_RUNTIME_HINT = (
    "Install the .NET SDK or runtime from https://dotnet.microsoft.com/download"
)


class NetcoredbgAdapter(DebugAdapter):
    """netcoredbg-based DAP adapter for .NET."""

    name = "dotnet"
    language_aliases = ("csharp", "cs")
    file_extensions = (".cs",)
    transport_type = "stdio"
    # adapterID is the DAP-protocol handshake token; netcoredbg checks for
    # "coreclr" specifically. The user-facing `adapter.name = "dotnet"` is
    # separate.
    adapter_id = "coreclr"

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(_NETCOREDBG_BINARY) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_DOTNET_BINARY,
                kind="runtime",
                name="dotnet runtime",
                install_hint=_DOTNET_RUNTIME_HINT,
            ),
            binary_dependency_status(
                binary=_NETCOREDBG_BINARY,
                kind="debugger",
                name="netcoredbg",
                install_hint=_INSTALL_HINT,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        # `--interpreter=vscode` puts netcoredbg into DAP-on-stdio mode.
        # Without it, netcoredbg defaults to an MI-style interpreter.
        return [_NETCOREDBG_BINARY, "--interpreter=vscode"]

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        body: dict[str, Any] = {
            "program": cfg.extras.get("executable", cfg.program),
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "stopOnEntry": cfg.stop_on_entry,
        }
        symbol_paths = cfg.extras.get("symbol_search_paths")
        if symbol_paths:
            # Normalize to a list even if caller passed a tuple or other iterable.
            body["symbolSearchPaths"] = list(symbol_paths)
        return body


REGISTRY.register(NetcoredbgAdapter())
