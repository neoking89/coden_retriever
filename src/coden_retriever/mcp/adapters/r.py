"""R / `vscDebugger` adapter.

DAP adapter powered by the `vscDebugger` R package
(ManuelHentschel/VSCode-R-Debugger). R's debugger is not a standalone
binary — it lives inside an interactive R session that imports the
package and enters its DAP server loop via
`.vsc.listenForDAP(port=N, server=TRUE)`. The function only supports
TCP listening (no stdio mode), so we use the socket transport and let
DAPClient dial the R-hosted listener.

A vanilla Rscript (non-interactive) process can't be driven via
vscDebugger: when a breakpoint fires, vscDebugger's `.vsc.preBreakpoint`
triggers `browser()`, which in non-interactive R returns immediately
without pausing, so the DAP listener never gets a chance to answer
subsequent requests before the program terminates. We therefore launch
`R --interactive` with a live stdin pipe and funnel vscDebugger's
custom `writeToStdin` events back into R's stdin — the same flow-control
pattern VS Code's R extension uses.

Three non-obvious knobs bolted onto this adapter:

1. `sendWriteToStdinForFlowControl` is monkey-patched in the bootstrap
   to always emit the follow-up `vscDebugger::.vsc.listenForDAP()` write
   event. Upstream only emits it when `session$useDapSocket` is `TRUE`,
   but initializeRequest unconditionally resets that flag to `FALSE`
   (reading the `useDapSocket` key from the initialize body), and
   passing `useDapSocket=true` there causes vscDebugger to open an
   outbound client socket — incompatible with our server-mode listener.

2. The launch body uses vscDebugger's native names (`file`,
   `workingDirectory`, `debugMode="file"`) plus `supportsWriteToStdinEvent=true`
   to opt the debugger into writeToStdin flow-control. A plain
   `stopOnEntry`+`program` body leaves vscDebugger in workspace mode,
   which never runs the user file.

3. `handle_adapter_event` services the `custom` events carrying
   writeToStdin payloads — without that handler, R sits forever at
   `browser()` waiting for input that would never arrive.

The package is GitHub-only (not on CRAN) — install hint matches that.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .availability import DebugDependencyStatus, binary_dependency_status
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

if TYPE_CHECKING:
    from ..dap_client import DAPClient

_R_BINARY = "R"
_RSCRIPT_BINARY = "Rscript"
_INSTALL_HINT = (
    "Install R, then: Rscript -e 'remotes::install_github("
    "\"ManuelHentschel/VSCode-R-Debugger\")'"
)
_R_RUNTIME_HINT = "Install R and ensure `R` is on PATH"
_VSCDEBUGGER_HINT = (
    "R is on PATH but vscDebugger is not installed. "
    "Run: Rscript -e 'install.packages(\"vscDebugger\", repos=\"https://manuelhentschel.r-universe.dev\")'"
)
# Cache the vscDebugger probe — Rscript spawn is ~500ms on a cold cache;
# detect_installed runs once per pytest collection, but tests/MCP tooling
# may call it many times in a session. None = not yet probed; True / False
# = result.
_VSCDEBUGGER_PRESENT_CACHE: bool | None = None


def _vscdebugger_installed() -> bool:
    """Probe whether vscDebugger is loadable in this R installation.

    Spawns Rscript with a one-shot script that exits non-zero if the
    package isn't in installed.packages(). The result is cached so
    repeat calls in the same process don't pay the spawn cost.
    """
    global _VSCDEBUGGER_PRESENT_CACHE
    if _VSCDEBUGGER_PRESENT_CACHE is not None:
        return _VSCDEBUGGER_PRESENT_CACHE
    rscript = shutil.which(_RSCRIPT_BINARY)
    if rscript is None:
        _VSCDEBUGGER_PRESENT_CACHE = False
        return False
    try:
        result = subprocess.run(
            [rscript, "-e",
             'if (!"vscDebugger" %in% rownames(installed.packages())) quit(status=1)'],
            capture_output=True, timeout=10,
        )
        _VSCDEBUGGER_PRESENT_CACHE = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        _VSCDEBUGGER_PRESENT_CACHE = False
    return _VSCDEBUGGER_PRESENT_CACHE
# Patch + kickoff script fed through R's stdin at startup. Two pieces:
#   1) Override `sendWriteToStdinForFlowControl` so it ALWAYS emits the
#      follow-up `.vsc.listenForDAP()` writeToStdin event regardless of
#      `session$useDapSocket` (which initializeRequest resets to FALSE).
#   2) Enter the DAP listener on the requested port.
# The interactive R session reads this from stdin, then sits in the DAP
# loop. Later writeToStdin events reuse the same stdin pipe to steer
# `browser()` (the "n" step) and re-enter the DAP loop
# (`.vsc.listenForDAP()`). `.vsc.onError` is NOT patched here — upstream
# sends a `browserPrompt` writeToStdin event with empty text (requires
# `supportsStdoutReading` to work), so we synthesize the listen-call
# ourselves in `handle_adapter_event` below when we see a
# `stopped/reason=exception` event.
_R_BOOTSTRAP_TEMPLATE = (
    "assignInNamespace('sendWriteToStdinForFlowControl', function(text) {{"
    "vscDebugger:::sendWriteToStdinEvent(text, when='now');"
    "vscDebugger:::sendWriteToStdinEvent("
    "'vscDebugger::.vsc.listenForDAP()', when='now')"
    "}}, ns='vscDebugger')\n"
    "invisible(vscDebugger::.vsc.listenForDAP(port={port}, server=TRUE))\n"
)
# Listen-call echoed into R's stdin when the DAP session pauses on an
# exception. Upstream `.vsc.onError` emits a `browserPrompt` writeToStdin
# event with empty text (relying on `supportsStdoutReading` to time the
# send), which we can't satisfy without a real TTY — so we emit the
# listen-call unconditionally as soon as a stopped-exception arrives.
_R_LISTEN_REENTER = b"vscDebugger::.vsc.listenForDAP()\n"


class RAdapter(DebugAdapter):
    """vscDebugger-based DAP adapter for R (socket + stdin-pipe transport)."""

    name = "r"
    file_extensions = (".r", ".R")
    transport_type = "socket"
    adapter_id = "R-Debugger"
    # R needs a live stdin so vscDebugger can funnel breakpoint flow-control
    # (the `n` step + `.vsc.listenForDAP()` re-enter call) back into the
    # R session while paused at `browser()`.
    wants_stdin_pipe = True
    # vscDebugger's trace-based breakpoints can't be replaced while paused
    # at one — the subsequent setBreakpoints would untrace the currently
    # executing trace and tear down the session. The matrix's stop_on_entry
    # launches rely on `.vsc.onError` pausing on the fixture's `stop()` call
    # instead; no bp swap happens because there's never an active trace bp.
    supports_entry_line_autopause = False

    def detect_installed(self) -> tuple[bool, str]:
        if shutil.which(_R_BINARY) is None:
            return (False, _INSTALL_HINT)
        # Rscript-only is not enough — the adapter loads vscDebugger
        # inside the R session, so without that package the launch fails
        # with `r adapter exited immediately with code 1`. Probe for the
        # package so callers (and the matrix tests) skip cleanly.
        if not _vscdebugger_installed():
            return (False, _VSCDEBUGGER_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_R_BINARY,
                kind="runtime",
                name="R runtime",
                install_hint=_R_RUNTIME_HINT,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        if port is None:
            raise ValueError("RAdapter requires a socket port — socket transport only")
        # `--interactive` is load-bearing: in non-interactive R (including
        # Rscript), `browser()` auto-returns and no DAP request can be
        # serviced while the program is "paused" — the program simply runs
        # to completion. `--quiet --no-save --no-restore` suppress startup
        # banner and workspace load/save noise.
        return [_R_BINARY, "--interactive", "--quiet", "--no-save", "--no-restore"]

    async def bootstrap_stdin(self, cfg: LaunchConfig, port: int | None) -> bytes | None:
        if port is None:
            raise ValueError("RAdapter requires a socket port for bootstrap_stdin")
        return _R_BOOTSTRAP_TEMPLATE.format(port=port).encode("utf-8")

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        # vscDebugger's launchRequest reads `file`, `workingDirectory` and
        # `debugMode` — not `program`, `cwd`, `stopOnEntry`. `debugMode="file"`
        # runs the user program via `.vsc.debugSource`; leaving it unset
        # defaults to "workspace" which never runs the file and terminates
        # the listener on configurationDone (stopListeningOnPort=TRUE).
        # `supportsWriteToStdinEvent=TRUE` opts into the flow-control events
        # that keep `browser()` responsive while the DAP client is driving.
        # stopOnEntry is intentionally omitted — vscDebugger only implements
        # it for non-"file"/"function" debugMode values, which don't run
        # the program. For the matrix tests, real pauses come from either
        # `pre_launch_breakpoints` (set before configurationDone) or from
        # `.vsc.onError` pausing on the fixture's `stop()` call — both
        # fire while the DAP listener is reachable via the stdin re-entry
        # patch in `_R_BOOTSTRAP_TEMPLATE`.
        return {
            "file": cfg.program,
            "workingDirectory": cfg.cwd or self._default_cwd(cfg),
            "debugMode": "file",
            "allowGlobalDebugging": False,
            "supportsWriteToStdinEvent": True,
        }

    def _default_cwd(self, cfg: LaunchConfig) -> str:
        program = Path(cfg.program)
        if program.exists():
            return str(program.parent)
        return os.getcwd()

    async def handle_adapter_event(
        self, client: "DAPClient", event_name: str, body: Mapping[str, Any],
    ) -> None:
        # vscDebugger wraps flow-control writes as `custom` events with a
        # writeToStdin-shaped body. Funnel the text back into R's stdin so
        # `browser()` can advance and `.vsc.listenForDAP()` can re-enter.
        if event_name != "custom":
            return
        if body.get("reason") != "writeToStdin" and "text" not in body:
            return
        text = str(body.get("text", ""))
        if not text:
            # Upstream `.vsc.onError` sends an empty-text `browserPrompt`
            # writeToStdin (a no-op when `supportsStdoutReading` is false);
            # `on_stopped` below injects the real listen-call on our side.
            return
        add_newline = body.get("addNewLine", True)
        payload = (text + ("\n" if add_newline else "")).encode("utf-8")
        self._write_stdin(client, payload)

    def on_stopped(self, client: "DAPClient", body: Mapping[str, Any]) -> None:
        # When vscDebugger pauses on an exception via `.vsc.onError`, it
        # doesn't emit the flow-control writeToStdin that would keep the
        # DAP channel reachable. Inject the listen-call ourselves so the
        # client can still drive `setBreakpoints`, `stackTrace`, etc.
        if body.get("reason") != "exception":
            return
        self._write_stdin(client, _R_LISTEN_REENTER)

    def _write_stdin(self, client: "DAPClient", payload: bytes) -> None:
        proc = client.transport.process
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # R likely exited (normal termination path) — swallow so the
            # DAP session can finalize via `terminated`/`exited` events.
            return


REGISTRY.register(RAdapter())
