"""Lua adapter via `local-lua-debugger-vscode` (Node bridge).

Unlike the Tier-1 native-binary adapters, Lua speaks DAP through a
Node.js bridge script bundled as an npm package. The bridge in turn
spawns the configured Lua interpreter. First use of the shared
`_node_bridge.resolve_bridge_script` helper — Bash and PHP reuse it.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ._node_bridge import NODE_BINARY, resolve_bridge_script
from .availability import (
    DebugDependencyStatus,
    binary_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_LUA_BINARY = "lua"
_BRIDGE_ENV_VAR = "LOCAL_LUA_DEBUGGER"
# Standard npm global-install layouts. Both POSIX (`lib/node_modules/`) and
# Windows (direct `node_modules/` under `%APPDATA%\npm`) prefixes are tried.
_BRIDGE_RELATIVE_PATHS = (
    "node_modules/local-lua-debugger-vscode/extension/debugAdapter.js",
    "lib/node_modules/local-lua-debugger-vscode/extension/debugAdapter.js",
)
_INSTALL_HINT = (
    "Install Node.js 18+ AND: npm install -g local-lua-debugger-vscode"
)
_LUA_RUNTIME_HINT = "Install a Lua interpreter and ensure `lua` is on PATH"


class LuaAdapter(DebugAdapter):
    """local-lua-debugger-vscode Node-bridge DAP adapter for Lua."""

    name = "lua"
    file_extensions = (".lua",)
    transport_type = "stdio"
    adapter_id = "lua-local"

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(NODE_BINARY) is None:
            return (False, _INSTALL_HINT)
        if resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_LUA_BINARY,
                kind="runtime",
                name="Lua runtime",
                install_hint=_LUA_RUNTIME_HINT,
            ),
            binary_dependency_status(
                binary=NODE_BINARY,
                kind="debugger",
                name="Node.js",
                install_hint=_INSTALL_HINT,
            ),
            resolver_dependency_status(
                kind="debugger",
                name="local-lua-debugger-vscode",
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
            # Race guard: DAPClient.launch gates on detect_installed, so this
            # only fires if the bridge disappeared between the gate and the
            # argv build. A RuntimeError surfaces at the launch layer cleanly.
            raise RuntimeError(_INSTALL_HINT)
        return [NODE_BINARY, bridge]

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        # local-lua-debugger-vscode accepts `program` as either a path string
        # OR a {lua, file} dict specifying the interpreter + script. We use
        # the dict form so `cfg.extras["lua_interpreter"]` picks the Lua
        # flavor (lua / lua5.3 / luajit / ...). Default "lua" matches the
        # upstream extension default.
        #
        # cwd and workspacePath both default to the script's parent directory.
        # luaDebugSession.js:152 calls `path.resolve(workspacePath, cwd)` when
        # cwd isn't absolute — if either is undefined, Node's path.resolve
        # throws `ERR_INVALID_ARG_TYPE` and the adapter crashes before
        # spawning lua.
        program_dir = str(Path(cfg.program).resolve().parent)
        cwd = cfg.cwd or program_dir
        workspace_path = cfg.extras.get("workspace_path") or program_dir
        body: dict[str, Any] = {
            "program": {
                "lua": cfg.extras.get("lua_interpreter", "lua"),
                "file": cfg.program,
            },
            "args": list(cfg.args),
            "cwd": cwd,
            "workspacePath": workspace_path,
            "stopOnEntry": cfg.stop_on_entry,
            # Bridge's debugAdapter.js lives at <ext>/extension/debugAdapter.js;
            # the parent's parent is the "extension" root containing debugger/
            # lldebugger.lua — which the adapter interpolates into LUA_PATH at
            # luaDebugSession.js:170,658. Without this, LUA_PATH picks up
            # "undefined/debugger/?.lua", require('lldebugger') fails, and the
            # lua process exits before any breakpoint can attach.
            "extensionPath": self._resolve_extension_path(),
        }
        script_roots = cfg.extras.get("script_roots")
        if script_roots:
            body["scriptRoots"] = list(script_roots)
        return body

    def _resolve_extension_path(self) -> str:
        bridge = resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS)
        if bridge is None:
            raise RuntimeError(_INSTALL_HINT)
        return str(Path(bridge).resolve().parent.parent)


REGISTRY.register(LuaAdapter())
