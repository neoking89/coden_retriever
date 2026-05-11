"""Base types for debug adapters.

Defines the contract every language-specific DAP adapter must implement:
- `LaunchConfig`: immutable launch parameters passed from MCP tool to adapter.
- `PathMapper`: structural interface for translating paths between the
  adapter's filesystem view and the MCP caller's.
- `DebugAdapter`: ABC each language subclasses.

Phase 0 only *defines* these types. `DAPClient` does not consume them yet —
the wiring happens in Phase 1 alongside the transport split.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, runtime_checkable

from .availability import (
    DebugAvailability,
    DebugDependencyStatus,
    availability_from_dependencies,
    missing_dependency_status,
)

if TYPE_CHECKING:
    from ..dap_client import DAPClient


@dataclass(frozen=True)
class LaunchConfig:
    """Immutable launch parameters for a debug session.

    Frozen because instances cross the MCP → DAPClient → adapter boundary and
    mutation along the way would be a correctness bug.
    """

    program: str = ""
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    stop_on_entry: bool = False
    console: Literal["internal", "external"] = "internal"
    extras: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PathMapper(Protocol):
    """Translate paths between the adapter's view and the MCP caller's view.

    Adapters running in WSL, containers, or remote machines see a different
    filesystem than the MCP client. Each adapter supplies a `PathMapper` to
    normalize paths on both the outbound (breakpoint) and inbound
    (stack-frame) sides.
    """

    def to_client(self, path: str | None) -> str | None:
        """Adapter-side path → MCP-caller-side path. None passes through."""
        ...

    def to_adapter(self, path: str | None) -> str | None:
        """MCP-caller-side path → adapter-side path. None passes through."""
        ...


class IdentityPathMapper:
    """Default pass-through mapper for same-host adapters."""

    def to_client(self, path: str | None) -> str | None:
        return path

    def to_adapter(self, path: str | None) -> str | None:
        return path


_DEFAULT_CLIENT_ID = "mcp-debugger"
_DEFAULT_CLIENT_NAME = "MCP Debug Client"


class DebugAdapter(abc.ABC):
    """Abstract base for one language's DAP adapter.

    Subclasses encapsulate everything debugpy-specific that lives hardcoded
    in `dap_client.py` today: the launch argv, the `adapterID` sent in
    `initialize`, and the launch/attach request body.
    """

    name: str
    file_extensions: tuple[str, ...]
    transport_type: Literal["socket", "stdio"]
    # DAP handshake token sent as `adapterID` on the initialize request.
    # Subclasses must set this (the base `initialize_args` asserts non-empty).
    # jsdebug/powershell/netcoredbg use different tokens than their `name` —
    # the DAP spec value is what the upstream server checks, not a display
    # label. Empty string means the subclass overrides `initialize_args`
    # entirely (e.g. jsdebug needs to remember the args for child sessions).
    adapter_id: str = ""
    # Alternative language names that resolve to this adapter. `name` itself
    # is the canonical identity surfaced in logs and error envelopes; aliases
    # exist purely so callers can type `language="cpp"` instead of the less
    # obvious `language="lldb"`. Conflicts across aliases/names are rejected
    # at `AdapterRegistry.register()` time, same as extension collisions.
    language_aliases: tuple[str, ...] = ()
    # False when this adapter is declared unsupported in production despite
    # being registered (e.g. cpp/rust via lldb-dap — see
    # `archive/debugger_mcp_docs/polyglot_debugger/work/prod-ready-exit.md`). The MCP resolver
    # short-circuits with `adapter_unsupported_in_production` instead of
    # proceeding into a known-broken launch path.
    production_supported: bool = True
    # Short, caller-facing reason for `production_supported=False`. Embedded
    # in the `adapter_unsupported_in_production` error's `message`.
    production_unsupported_reason: str = ""
    # "launch" is DAP-spec-correct; debugpy is the odd one that wants "attach"
    # on an already-spawned process (see PythonAdapter).
    launch_request_command: Literal["launch", "attach"] = "launch"
    # True when `cfg.program` is a language-level symbol (e.g., a JVM
    # fully-qualified main class) rather than a filesystem path. Tool-layer
    # and DAPClient path-existence guards skip their checks when this is True.
    program_is_class_name: bool = False
    # True when the adapter swallows `configurationDone` without sending a
    # response — its launch response is the combined success gate (e.g.,
    # fwcd/kotlin-debug-adapter). DAPClient switches to fire-and-forget
    # instead of waiting on `_send_request("configurationDone", ...)`.
    configuration_done_fire_and_forget: bool = False
    # True when the adapter process needs a live stdin pipe (vscDebugger's
    # `.vsc.preBreakpoint` funnels flow-control back through R's stdin). The
    # socket-transport spawn flow uses `stdin=subprocess.PIPE` instead of
    # `DEVNULL`, and `bootstrap_stdin()` is awaited right after spawn to
    # push any startup script through the pipe.
    wants_stdin_pipe: bool = False
    # Per-adapter override for the DAPClient.launch() overall timeout (covers
    # connect + initialize + launch + configurationDone). Default 30s suits
    # fast starters (debugpy, dlv, netcoredbg); PSES on cold start routinely
    # spends 10-30s JIT-loading ~20 .NET assemblies before initialize even
    # returns, overflowing the 30s combined budget.
    launch_timeout_seconds: float = 30.0
    # True when the adapter pauses at a bootstrap/entry frame before the
    # user script is parsed, and the caller's pre-launch breakpoints are
    # expected to halt on a *subsequent* stop. js-debug is the canonical
    # case: CDP breakpoints need the script to parse before they bind, so
    # the first `reason=entry` stop must be auto-continued so launch()
    # returns on the real user-frame pause instead of the empty bootstrap.
    skip_entry_stop_when_pre_launch_bp: bool = False
    # True when the adapter CANNOT honour the debugpy-style "stop at first
    # executable line" fallback that DAPClient implements for `stop_on_entry`.
    # vscDebugger can technically install that bp, but once the program
    # pauses at it, any subsequent setBreakpoints that would replace the
    # active bp crashes the session (the trace-based bp is currently
    # executing the browser() inside its own callback — untracing it while
    # paused at it tears down the stack). Adapters that opt out of this
    # fallback get no auto-entry bp; `stop_on_entry=True` without explicit
    # `pre_launch_breakpoints` silently degrades to run-to-completion
    # (or run-until-error — vscDebugger's `.vsc.onError` pauses on the
    # fixture's `stop()` call, which is the matrix's deterministic pause
    # point anyway).
    supports_entry_line_autopause: bool = True

    async def bootstrap_stdin(self, cfg: LaunchConfig, port: int | None) -> bytes | None:
        """Bytes to feed the adapter's stdin right after spawn (pre-DAP).

        Only called when `wants_stdin_pipe=True`. Return `None` to skip.
        `port` is the socket-transport listen port (or None for stdio),
        so adapters that run an interactive interpreter loop (e.g. R's
        `.vsc.listenForDAP(port=N)`) can embed the port in the kickoff
        script. Used by the R adapter to push the `.vsc.listenForDAP(...)`
        kickoff into an interactive R session that was otherwise started
        without any `-e` argv.
        """
        return None

    async def handle_adapter_event(
        self, client: "DAPClient", event_name: str, body: Mapping[str, Any],
    ) -> None:
        """Fires for DAP events the client's built-in handlers don't cover.

        Default no-op — most adapters don't need custom events. The R
        adapter overrides this to service vscDebugger's `custom/writeToStdin`
        by funneling the payload back into R's stdin pipe.
        """
        return None

    def on_stopped(self, client: "DAPClient", body: Mapping[str, Any]) -> None:
        """Side-effect hook fired alongside the built-in stopped handler.

        Synchronous because it runs inside `_on_stopped`, which is invoked
        from the message-dispatch event loop. Default no-op. The R adapter
        overrides to push a `.vsc.listenForDAP()` kickoff into R's stdin
        when paused on exception — vscDebugger's own onError doesn't emit
        the flow-control writeToStdin, so without this the DAP channel
        goes dark the instant the program throws.
        """
        return None

    @abc.abstractmethod
    def detect_installed(self) -> tuple[bool, str]:
        """Check that the adapter is runnable on this machine.

        Returns:
            (True, "") if the adapter binary / package is present.
            (False, install_hint) otherwise. `install_hint` is a shell
            command the user can copy-paste.
        """

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        """Best-effort preflight prerequisites for this adapter.

        Default fallback preserves the old `detect_installed()` contract for
        adapters that have not split runtime-vs-debugger checks yet.
        """

        ok, install_hint = self.detect_installed()
        if ok:
            return ()
        adapter_name = f"{self.name} debug adapter"
        return (
            missing_dependency_status(
                kind="debugger",
                name=adapter_name,
                install_hint=install_hint,
                detail=install_hint or f"{adapter_name} is not installed",
            ),
        )

    def debug_availability(self) -> DebugAvailability:
        """Return a structured preflight answer for this adapter."""

        return availability_from_dependencies(self.name, self.dependency_statuses())

    @abc.abstractmethod
    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        """Construct the argv for spawning the adapter.

        `port` is provided only for socket-transport adapters; stdio adapters
        ignore it. Phase 1 will move port into `LaunchConfig` once the
        transport split is in place.
        """

    def build_launch_env(self, cfg: LaunchConfig) -> dict[str, str] | None:
        """Adapter-specific env overrides merged into the subprocess env.

        Default `None` means the subprocess inherits the current process env
        (plus any `cfg.env` the caller supplied). Override to return a dict
        of env changes; `cfg.env` still wins if it overrides the same keys.
        Used by the Ruby adapter to scrub `code` from PATH so rdbg's
        `--open=vscode` side-effect (auto-launching VS Code) is suppressed
        without breaking the DAP socket.
        """
        return None

    def initialize_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        """Body of the DAP `initialize` request, including `adapterID`.

        Default implementation returns the DAP-spec-standard initialize body
        using the subclass's `adapter_id` class attribute. Adapters that need
        side-effects or extra fields (e.g. jsdebug echoing args to a child
        session) override.
        """
        if not self.adapter_id:
            raise NotImplementedError(
                f"{type(self).__name__} must set `adapter_id` or override `initialize_args`"
            )
        return {
            "clientID": _DEFAULT_CLIENT_ID,
            "clientName": _DEFAULT_CLIENT_NAME,
            "adapterID": self.adapter_id,
            "pathFormat": "path",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "supportsVariableType": True,
        }

    @abc.abstractmethod
    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        """Body of the DAP `launch` (or `attach`) request after `initialize`."""

    async def post_initialize(self, client: "DAPClient") -> None:
        """Hook for adapter-specific handshake quirks (e.g. PSES getVersion).

        Default no-op. Awaited once after `initialize` succeeds, before the
        launch/attach request. Raise to abort the launch with a structured
        `adapter_handshake_failed` error at the tool layer.
        """

    def transform_eval_expression(self, expression: str) -> str:
        """Rewrite an eval expression before it hits DAP `evaluate`.

        Default identity. Adapters whose evaluator rejects DAP-caller-native
        syntax override. CodeLLDB is the motivating case: its `repl` context
        routes bare tokens through LLDB's command interpreter, so literals
        like `1` or variable names need a prefix to force expression mode.
        """
        return expression

    async def prepare_launch(self, cfg: LaunchConfig) -> int | None:
        """Run adapter-specific pre-launch work; return a socket port or None.

        Fires in `DAPClient.launch()` BEFORE transport spawn. Default returns
        `None` (no pre-launch work; normal spawn flow runs). Adapters that need
        a side-channel — e.g. Java's LSP `workspace/executeCommand` call to
        obtain java-debug's random listen port — override and return an int.
        The dispatcher honours a non-None return only when
        `transport_type == "socket"` AND `build_launch_argv(cfg)` returns `[]`;
        both guards must hold, otherwise the normal spawn flow runs. This
        hook is protocol-level plumbing, not capability-gated.
        """
        return None

    async def handle_reverse_request(
        self,
        command: str,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Handle a server→client DAP `request` message.

        Returns `(success, body)`; DAPProtocol constructs the response envelope.
        Default refuses with a well-formed error so the DAP spec still sees a
        response. Adapters that need real handling (bash's runInTerminal,
        js-debug's startDebugging) override and return `(True, body)` after
        doing the work. Not capability-gated — this is protocol-level plumbing,
        not an adapter capability advertised via `initialize`.
        """
        return (False, {"error": f"reverse request {command!r} unsupported"})

    @property
    def path_mapper(self) -> PathMapper:
        """Path translation strategy. Override for WSL / container / remote."""
        return IdentityPathMapper()
