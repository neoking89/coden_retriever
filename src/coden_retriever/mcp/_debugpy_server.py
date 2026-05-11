"""Out-of-process debugpy listener.

Spawned by debug_server(action='start') so debugpy never loads in the MCP
process. Prints LISTENING <port> to stdout once the port is bound, then
sleeps forever until the parent terminates it. If wait_for_client is set,
also prints CONNECTED <port> after the first client attaches.
"""
import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _debugpy_server.py <port> <wait_for_client: 0|1>", file=sys.stderr)
        return 2
    port = int(sys.argv[1])
    wait_for_client = sys.argv[2] == "1"

    try:
        import debugpy  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR debugpy-not-installed", flush=True)
        return 3

    try:
        debugpy.listen(("127.0.0.1", port))
    except Exception as exc:  # pragma: no cover — surfaced to parent via stdout
        print(f"ERROR listen-failed: {exc}", flush=True)
        return 4

    print(f"LISTENING {port}", flush=True)

    if wait_for_client:
        debugpy.wait_for_client()
        print(f"CONNECTED {port}", flush=True)

    # Idle until parent terminates us. One-hour ceiling so a runaway helper
    # can't outlive a dead parent on platforms where terminate() is missed.
    idle_seconds_cap = 3600  # safety guard against orphaned helper lingering
    slept = 0
    poll_interval = 1
    while slept < idle_seconds_cap:
        time.sleep(poll_interval)
        slept += poll_interval
    return 0


if __name__ == "__main__":
    sys.exit(main())
