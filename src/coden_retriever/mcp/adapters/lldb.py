"""CodeLLDB adapter for C / C++ (and Rust via subclass).

Stdio-transport DAP adapter powered by the `codelldb` binary shipped
inside the `vadimcn/codelldb` VSCode extension. CodeLLDB bundles its own
LLDB (21.1 as of v1.12.2) rather than dynamically linking against a
system LLVM install, which structurally pre-empts two upstream
`lldb-dap` defects this project hit on LLVM ≤ 22.1:

  (1) Windows stdio silence (llvm#121722) — lldb-dap emits zero bytes on
      stdin-driven stdio. Fix merged to LLVM main but unverified in 22.1.x.
  (2) Linux `setBreakpoints` returns `verified=false` on Ubuntu 24.04's
      apt `lldb-dap 18` (llvm#112629, closed by PR #129589 → LLDB 20+).

Empirical smoketest of CodeLLDB 1.12.2 recorded 4/4 green cells
(cpp + rust × Windows + Linux) — see `scripts/smoketest_codelldb.py`.

Install: download `codelldb-x86_64-windows.vsix` /
`codelldb-x86_64-linux.vsix` from https://github.com/vadimcn/codelldb/releases,
unzip, and point the `CODELLDB` environment variable at the extracted
`extension/adapter/codelldb` (POSIX) or `extension/adapter/codelldb.exe`
(Windows) binary. If that bare binary name is already on PATH, the
adapter finds it without the env var.

Compiled-language note (decision D3): `LaunchConfig.program` remains a
string the user typed — which may be a source file path even though the
adapter debugs a pre-built executable. Adapters interpret that gap via
`cfg.extras["executable"]`, which overrides the path sent in the DAP
`launch` body's `program` field. No change to `LaunchConfig` itself.

Rust is registered via the `RustAdapter(LLDBAdapter)` subclass at the
bottom of this file: subclassing lets `adapter.name` introspection
surface "rust" in logs and error envelopes even though the underlying
debug binary is the same.

DAP flow note: CodeLLDB defers the `initialized` event until after
`launch` has been received. `DAPClient._send_launch_request` already
tolerates a deferred launch response (see its docstring) and
`_wait_for_initialized` is the real gate, so no client-side changes are
needed to adopt CodeLLDB.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from .availability import DebugDependencyStatus, resolver_dependency_status
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_BINARY_POSIX = "codelldb"
_BINARY_WINDOWS = "codelldb.exe"
_ENV_OVERRIDE = "CODELLDB"
_INSTALL_HINT = (
    "Install: download codelldb VSIX from "
    "https://github.com/vadimcn/codelldb/releases, unzip, and set "
    f"{_ENV_OVERRIDE} to the extracted extension/adapter/codelldb[.exe] path "
    "(or place the binary on PATH)."
)


def _resolve_codelldb() -> str | None:
    override = os.environ.get(_ENV_OVERRIDE)
    if override and os.path.isfile(override):
        return override
    candidate = _BINARY_WINDOWS if sys.platform == "win32" else _BINARY_POSIX
    return shutil.which(candidate)


class LLDBAdapter(DebugAdapter):
    """CodeLLDB-based DAP adapter for C / C++."""

    name = "lldb"
    file_extensions = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")
    language_aliases = ("cpp", "c++", "c")
    transport_type = "stdio"
    adapter_id = "lldb"

    def detect_installed(self) -> tuple[bool, str]:
        if _resolve_codelldb() is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            resolver_dependency_status(
                kind="debugger",
                name="CodeLLDB",
                install_hint=_INSTALL_HINT,
                resolver=_resolve_codelldb,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        # codelldb speaks DAP framing on its own stdin/stdout when launched
        # with no flags. The target binary is sent via `launch` body.
        binary = _resolve_codelldb()
        if binary is None:
            raise RuntimeError(_INSTALL_HINT)
        return [binary]

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        # Compiled-language split: `cfg.program` may point at the source
        # file the user typed, but the adapter needs the built binary path.
        # `cfg.extras["executable"]` is the explicit override; when absent,
        # we pass `cfg.program` through (common for pre-built binaries).
        executable = cfg.extras.get("executable", cfg.program)
        return {
            "program": executable,
            "args": list(cfg.args),
            "cwd": cfg.cwd or "",
            "stopOnEntry": cfg.stop_on_entry,
        }

    def transform_eval_expression(self, expression: str) -> str:
        # CodeLLDB's DAP `evaluate` with context=="repl" routes the
        # expression through LLDB's command interpreter, so a bare `1` is
        # parsed as a command (and rejected). `?<expr>` forces LLDB's
        # simple-expression evaluator regardless of context. `/<fmt>` is
        # the formatted-print variant — leave it untouched if the caller
        # is already using either.
        stripped = expression.lstrip()
        if stripped.startswith(("?", "/")):
            return expression
        return f"?{expression}"


class RustAdapter(LLDBAdapter):
    """CodeLLDB adapter specialized for Rust.

    Subclass only for clean `adapter.name == "rust"` introspection and
    install-error messaging. All launch logic inherits from LLDBAdapter,
    including `cfg.extras["executable"]` for pointing at `target/debug/<bin>`.

    `detect_installed` inherits the codelldb probe — `rustc` is a *build*
    dependency, not a *debug* dependency; integration tests that compile
    Rust fixtures must gate on `rustc` themselves.
    """

    name = "rust"
    file_extensions = (".rs",)
    language_aliases = ()


REGISTRY.register(LLDBAdapter())
REGISTRY.register(RustAdapter())
