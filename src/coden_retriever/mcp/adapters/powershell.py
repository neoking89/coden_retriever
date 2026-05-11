"""PowerShell Editor Services (PSES) adapter.

PSES is not a standalone DAP binary — it's a PowerShell module hosted by
`pwsh` that starts a DAP listener on stdin/stdout. We spawn `pwsh` with
a one-liner that imports the module and calls `Start-EditorServices`.

`Start-EditorServices -DebugServiceOnly` flips PSES into pure DAP-over-stdio
mode; without it the process serves LSP and the DAP reader never sees an
`initialize` response. The legacy `powerShell/getVersion` handshake was
LSP-side and is not needed (and not served) in DebugServiceOnly mode.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .availability import (
    DebugDependencyStatus,
    binary_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_PWSH_BINARY = "pwsh"
_INSTALL_HINT = "Install PowerShell 7+ (https://aka.ms/powershell)"
_PSES_INSTALL_HINT = (
    "Install the PowerShellEditorServices module with `Install-Module "
    "PowerShellEditorServices -Scope CurrentUser`, install the VS Code "
    "PowerShell extension, or set PSES_BUNDLE_PATH to the "
    "PowerShellEditorServices module directory"
)
_PSES_ENV_VAR = "PSES_BUNDLE_PATH"
_PSES_MANIFEST = "PowerShellEditorServices.psd1"
_PSES_MODULE_NAME = "PowerShellEditorServices"
# PowerShell startup is materially slower than simple PATH probes, but this
# still needs to fail fast during adapter detection. Two seconds is enough for
# a healthy local `pwsh` to enumerate installed modules without hanging UI.
_PSES_DISCOVERY_TIMEOUT_SECONDS = 2.0
_PSES_MODULE_PROBE = (
    f"if (Get-Module -ListAvailable -Name '{_PSES_MODULE_NAME}') {{ exit 0 }} "
    "else { exit 1 }"
)


def _normalize_pses_bundle_path(candidate: str | None) -> str | None:
    """Return the bundle directory when `candidate` points at a valid PSES install."""
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    candidate_dirs: list[Path] = []
    if path.is_file():
        if path.name.lower() == _PSES_MANIFEST.lower():
            candidate_dirs.append(path.parent)
    else:
        candidate_dirs.append(path)
        candidate_dirs.append(path / "modules" / "PowerShellEditorServices")
    for directory in candidate_dirs:
        if (directory / _PSES_MANIFEST).is_file():
            return str(directory)
    return None


def resolve_pses_bundle_path() -> str | None:
    """Best-effort local resolution of the VS Code-bundled PSES module directory."""
    env_bundle = _normalize_pses_bundle_path(os.environ.get(_PSES_ENV_VAR))
    if env_bundle is not None:
        return env_bundle

    for root in (
        Path.home() / ".vscode" / "extensions",
        Path.home() / ".vscode-insiders" / "extensions",
    ):
        if not root.is_dir():
            continue
        for extension_dir in sorted(root.glob("ms-vscode.powershell-*"), reverse=True):
            discovered = _normalize_pses_bundle_path(str(extension_dir))
            if discovered is not None:
                return discovered
    return None


def _has_globally_available_pses_module(pwsh_binary: str) -> bool:
    """True when `pwsh` can discover PowerShellEditorServices on PSModulePath."""
    try:
        result = subprocess.run(
            [pwsh_binary, "-NoProfile", "-NoLogo", "-Command", _PSES_MODULE_PROBE],
            capture_output=True,
            text=True,
            timeout=_PSES_DISCOVERY_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def _resolve_pses_runtime(pwsh_binary: str | None) -> str | None:
    """Return a truthy marker when PSES is available to the launch bootstrap."""
    if pwsh_binary is None:
        return None
    bundle = resolve_pses_bundle_path()
    if bundle is not None:
        return bundle
    if _has_globally_available_pses_module(pwsh_binary):
        return _PSES_MODULE_NAME
    return None

# PSES bootstrap: import the module and start Editor Services in stdio mode.
# Log + session files land in the OS temp dir (`$env:TEMP` is unset on Linux
# so we use `[System.IO.Path]::GetTempPath()` which returns the right thing
# cross-platform). PSES needs both paths to exist.
# FeatureFlags is intentionally empty (PSES 3.x requires it, even as @()).
# BundledModulesPath is optional but has [ValidateNotNullOrEmpty] — passing
# `$null` throws. We omit it entirely; if the user (or adapter auto-discovery)
# supplied $env:PSES_BUNDLE_PATH we forward it, otherwise PSES resolves its
# own module directory.
_PSES_BOOTSTRAP = (
    # PSES is rarely on the default PSModulePath — it ships as a sidecar
    # inside the vscode-powershell extension bundle. Import by manifest path
    # via $env:PSES_BUNDLE_PATH so a bare `Import-Module PowerShellEditorServices`
    # can't silently fail to resolve, leaving Start-EditorServices undefined
    # and the DAP reader hung on initialize.
    f"$bundled = $env:{_PSES_ENV_VAR}; "
    "if ($bundled) { "
    "  Import-Module (Join-Path $bundled 'PowerShellEditorServices.psd1') "
    "} else { "
    "  Import-Module PowerShellEditorServices "
    "}; "
    "$tmp = [System.IO.Path]::GetTempPath(); "
    "$bp_arg = if ($bundled) { @{ BundledModulesPath = $bundled } } else { @{} }; "
    "Start-EditorServices @bp_arg "
    "-HostName 'mcp-debugger' -HostProfileId 'mcp' -HostVersion '1.0.0' "
    "-LogPath (Join-Path $tmp 'pses.log') -LogLevel Normal "
    "-SessionDetailsPath (Join-Path $tmp 'pses-session.json') "
    "-FeatureFlags @() "
    "-Stdio "
    # DAP-only mode: PSES defaults to LSP-on-stdio; without this flag the
    # DAP reader never sees an `initialize` response because the process
    # is talking LSP, not DAP. -DebugServiceOnly flips it to DAP-on-stdio.
    "-DebugServiceOnly"
)

class PSESAdapter(DebugAdapter):
    """PowerShell Editor Services adapter (stdio via `pwsh`)."""

    name = "powershell"
    file_extensions = (".ps1",)
    transport_type = "stdio"
    # adapterID for PSES is capital 'P' PowerShell — PSES checks
    # this exact handshake token.
    adapter_id = "PowerShell"
    # PSES cold start JIT-loads ~20 .NET assemblies (10-30s on Windows),
    # so the default 30s launch budget — shared across connect + initialize
    # + launch + configurationDone — routinely times out at initialize.
    launch_timeout_seconds = 60.0

    def detect_installed(self) -> tuple[bool, str]:
        pwsh_binary = shutil.which(_PWSH_BINARY)
        if pwsh_binary is None:
            return (False, _INSTALL_HINT)
        if _resolve_pses_runtime(pwsh_binary) is None:
            return (False, _PSES_INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_PWSH_BINARY,
                kind="runtime",
                name="PowerShell runtime",
                install_hint=_INSTALL_HINT,
            ),
            resolver_dependency_status(
                kind="debugger",
                name="PowerShellEditorServices",
                install_hint=_PSES_INSTALL_HINT,
                resolver=lambda: _resolve_pses_runtime(shutil.which(_PWSH_BINARY)),
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        # `-NoProfile`: don't source the user's profile.ps1 (faster, and
        # prevents surprise interference with PSES's module path).
        # `-NoLogo`: suppress the PS startup banner.
        return [_PWSH_BINARY, "-NoProfile", "-NoLogo", "-Command", _PSES_BOOTSTRAP]

    def build_launch_env(self, cfg: LaunchConfig) -> dict[str, str] | None:
        bundle = resolve_pses_bundle_path()
        if bundle is None:
            return None
        return {_PSES_ENV_VAR: bundle}

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        return {
            # PSES uses `script`, not `program` — that's its contract.
            "script": cfg.program,
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "createTemporaryIntegratedConsole": False,
            "stopOnEntry": cfg.stop_on_entry,
        }

REGISTRY.register(PSESAdapter())
