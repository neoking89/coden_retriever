"""PHP adapter via `vscode-php-debug` (Node bridge → Xdebug).

The `vscode-php-debug` npm package ships a Node DAP bridge that speaks
DBGp to Xdebug on a listening port (default 9003). In launch mode the
bridge spawns PHP itself with Xdebug configured to start on request —
callers pass a `.php` script and the Xdebug plumbing stays invisible.

Unlike Lua / Bash this adapter also surfaces `pathMappings` and Xdebug
hostname/port via `cfg.extras` for container / remote setups.
"""
from __future__ import annotations

import shutil
from typing import Any

from ._node_bridge import NODE_BINARY, resolve_bridge_script
from .availability import (
    DebugDependencyStatus,
    binary_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_BRIDGE_ENV_VAR = "VSCODE_PHP_DEBUG"
_BRIDGE_RELATIVE_PATHS = (
    "node_modules/php-debug/out/phpDebug.js",
    "lib/node_modules/php-debug/out/phpDebug.js",
    "node_modules/vscode-php-debug/out/phpDebug.js",
    "lib/node_modules/vscode-php-debug/out/phpDebug.js",
)
_PHP_BINARY = "php"
_INSTALL_HINT = (
    "Install Node.js 18+ AND: npm install -g php-debug. PHP itself must "
    "have Xdebug enabled (see https://xdebug.org/wizard) with php.ini: "
    "zend_extension=xdebug, xdebug.mode=debug, xdebug.start_with_request=yes"
)
_PHP_RUNTIME_HINT = "Install PHP and ensure `php` is on PATH"
# Xdebug 3.x default; Xdebug 2.x used 9000 but is EOL as of 2021.
_XDEBUG_DEFAULT_PORT = 9003
# Explicit IPv4 loopback (not "localhost") so the vscode-php-debug bridge
# binds where Xdebug actually dials. On Windows 11 "localhost" resolves to
# IPv6 `::1` first; the bridge then binds only to `::1`, while Xdebug's
# default `client_host=127.0.0.1` talks IPv4 — producing a 200 ms timeout
# and a PHP run that never halts.
_XDEBUG_DEFAULT_HOST = "127.0.0.1"

# PHP CLI `-d` flags injected into every launch so breakpoints halt even
# when the host php.ini leaves xdebug.mode unset (observed on Ubuntu 24.04
# apt php-xdebug, where the default mode is empty and Xdebug 3.x silently
# skips halting without an explicit debug mode + start_with_request).
_XDEBUG_RUNTIME_FLAGS: tuple[str, ...] = (
    "-d", "xdebug.mode=debug",
    "-d", "xdebug.start_with_request=yes",
)


class PHPAdapter(DebugAdapter):
    """vscode-php-debug Node-bridge DAP adapter for PHP (Xdebug-backed)."""

    name = "php"
    file_extensions = (".php",)
    transport_type = "stdio"
    adapter_id = "php"

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(NODE_BINARY) is None:
            return (False, _INSTALL_HINT)
        if shutil.which(_PHP_BINARY) is None:
            return (False, _INSTALL_HINT)
        if resolve_bridge_script(_BRIDGE_ENV_VAR, _BRIDGE_RELATIVE_PATHS) is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_PHP_BINARY,
                kind="runtime",
                name="PHP runtime",
                install_hint=_PHP_RUNTIME_HINT,
            ),
            binary_dependency_status(
                binary=NODE_BINARY,
                kind="debugger",
                name="Node.js",
                install_hint=_INSTALL_HINT,
            ),
            resolver_dependency_status(
                kind="debugger",
                name="php-debug",
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
        body: dict[str, Any] = {
            "program": cfg.program,
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "stopOnEntry": cfg.stop_on_entry,
            "runtimeExecutable": cfg.extras.get("php_executable", _PHP_BINARY),
            "runtimeArgs": list(_XDEBUG_RUNTIME_FLAGS)
                + list(cfg.extras.get("runtime_args", ())),
            "port": cfg.extras.get("xdebug_port", _XDEBUG_DEFAULT_PORT),
            "hostname": cfg.extras.get("xdebug_host", _XDEBUG_DEFAULT_HOST),
        }
        path_mappings = cfg.extras.get("path_mappings")
        if path_mappings:
            body["pathMappings"] = dict(path_mappings)
        return body


REGISTRY.register(PHPAdapter())
