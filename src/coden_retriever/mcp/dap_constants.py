"""Shared timing constants for the DAP transport/protocol layer.

Centralises timeouts and poll intervals used by multiple DAP files
(dap_client.py, dap_protocol.py, dap_transport.py) so a single edit
updates every site and each value carries a `# Why:` rationale.
"""
from __future__ import annotations

# Why: debugpy historically replies to `initialize` / `attach` within a few
# hundred ms on loopback; 30s is the DAP community default and matches what
# VS Code uses. Large enough to survive a cold Python interpreter start,
# small enough that a genuinely wedged adapter fails the handshake instead
# of hanging the user.
DAP_DEFAULT_REQUEST_TIMEOUT: float = 30.0

# Why: TCP connect to an already-listening debugpy usually succeeds on the
# first attempt; 10s covers adapter-spawn races on slow CI hosts where the
# listen() call lags the child process start. Any larger and the user feels
# the UI hang; any smaller and cold-start Windows debug sessions flake.
DAP_CONNECT_TIMEOUT: float = 10.0

# Why: `wait_for(message_processor_task, ...)` after cancelling. The task
# yields immediately on its `CancelledError` except-branch, so 1s is an order
# of magnitude above the expected completion time — if we exceed it the task
# is genuinely stuck and the TimeoutError path is the right escape.
DAP_TASK_CANCEL_TIMEOUT: float = 1.0

# Why: CDP-backed adapters (js-debug) only verify source-file breakpoints
# after script parse, which lands within a few event-loop ticks of the
# setBreakpoints response. 100ms is the smallest interval that both (a)
# avoids busy-looping the adapter and (b) keeps total rebind latency well
# under the 2s timeout the caller passes in.
DAP_REBIND_POLL_INTERVAL: float = 0.1

# Why: `_process_messages` polls the transport queue when it's empty. 10ms
# is the standard asyncio "yield without starving" cadence — short enough
# that a landed message surfaces within a single frame of UI latency, long
# enough that an idle session doesn't pin a CPU.
DAP_MSG_POLL_INTERVAL: float = 0.01

# Why: Post-stop frame-context queries (stackTrace, threads) run under a
# fixed short timeout so a slow adapter can't stall the `_wait_for_stop`
# caller indefinitely. 5s is generous for what are purely introspection
# requests against a paused program — no user code runs during them.
DAP_STACK_PAUSE_TIMEOUT: float = 5.0

# Why: Lines drained from the adapter's stderr are tagged with this prefix
# before joining the normal program-output stream. Centralised so writer
# (transports) and reader (dap_client.py when it surfaces the exit tail)
# agree on a single literal — avoids drift when someone tweaks the format.
DAP_STDERR_LINE_PREFIX: str = "[stderr] "
