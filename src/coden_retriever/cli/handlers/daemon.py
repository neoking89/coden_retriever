"""Handler for daemon subcommands."""
import json
import subprocess
import sys
import time

from ...config_loader import get_config
from ...constants import (
    DAEMON_CACHE_TIMEOUT_SECONDS,
    DAEMON_MAX_POLL_ATTEMPTS,
    DAEMON_POLL_INTERVAL_SECONDS,
    DAEMON_RESTART_DELAY_SECONDS,
)
from ...daemon.client import DaemonClient, get_daemon_status, stop_daemon
from ...daemon.protocol import WINDOWS_CREATE_NEW_PROCESS_GROUP, WINDOWS_DETACHED_PROCESS
from ...daemon.server import get_log_file, is_daemon_running, run_daemon
from ..utils import parse_duration


def _daemon_start(
    host: str, port: int, max_projects: int,
    idle_timeout: str | None, verbose: bool, no_watch: bool = False,
) -> int:
    """Start daemon in background."""
    running, pid = is_daemon_running()
    if running:
        print(f"Daemon is already running (PID: {pid})")
        return 0

    if sys.platform == "win32":
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")
    else:
        python_exe = sys.executable

    cmd = [
        python_exe, "-m", "coden_retriever",
        "daemon", "run",
        "--daemon-host", host,
        "--daemon-port", str(port),
    ]

    if max_projects:
        cmd.extend(["--max-projects", str(max_projects)])
    if idle_timeout:
        cmd.extend(["--idle-timeout", str(idle_timeout)])
    if verbose:
        cmd.append("--verbose")
    if no_watch:
        cmd.append("--no-watch")

    if sys.platform == "win32":
        subprocess.Popen(
            cmd,
            creationflags=WINDOWS_DETACHED_PROCESS | WINDOWS_CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    for _ in range(DAEMON_MAX_POLL_ATTEMPTS):
        time.sleep(DAEMON_POLL_INTERVAL_SECONDS)
        status = get_daemon_status(host, port)
        if status:
            _, pid = is_daemon_running()
            print(f"Daemon started (PID: {pid})")
            print(f"  Address: {host}:{port}")
            print(f"  Log: {get_log_file()}")
            return 0

    print("Daemon failed to start. Check log:", file=sys.stderr)
    print(f"  {get_log_file()}", file=sys.stderr)
    return 1


def _daemon_stop(host: str, port: int) -> int:
    """Stop the daemon."""
    running, pid = is_daemon_running()
    if not running:
        print("Daemon is not running")
        return 0

    if stop_daemon(host, port):
        print(f"Daemon stopped (was PID: {pid})")
        print(f"  Address: {host}:{port}")
        return 0
    else:
        print(f"Failed to stop daemon (PID: {pid})", file=sys.stderr)
        return 1


def _daemon_status(host: str, port: int) -> int:
    """Show daemon status."""
    status = get_daemon_status(host, port)
    if status:
        print("Daemon is running")
        print(json.dumps(status, indent=2))
        return 0

    running, pid = is_daemon_running()
    if running:
        print(f"Daemon process exists (PID: {pid}) but not responding")
        return 1
    else:
        print("Daemon is not running")
        return 1


def _daemon_restart(
    host: str, port: int, max_projects: int,
    idle_timeout: str | None, verbose: bool, no_watch: bool = False,
) -> int:
    """Restart the daemon."""
    stop_daemon(host, port)
    time.sleep(DAEMON_RESTART_DELAY_SECONDS)
    return _daemon_start(host, port, max_projects, idle_timeout, verbose, no_watch)


def _daemon_run(
    host: str, port: int, max_projects: int,
    idle_timeout: str | None, verbose: bool, no_watch: bool = False,
) -> int:
    """Run daemon in foreground."""
    config = get_config()
    max_projects = max_projects or config.daemon.max_projects
    timeout_seconds = parse_duration(idle_timeout) if idle_timeout else None

    return run_daemon(
        host=host,
        port=port,
        max_projects=max_projects,
        idle_timeout=timeout_seconds,
        verbose=verbose,
        foreground=True,
        enable_watch=not no_watch,
    )


def _daemon_clear_cache(host: str, port: int, clear_path: str | None, clear_all: bool) -> int:
    """Clear daemon cache."""
    client = DaemonClient(host=host, port=port, timeout=DAEMON_CACHE_TIMEOUT_SECONDS)
    try:
        result = client.invalidate(source_dir=clear_path, all=clear_all)
        print(f"Cache cleared: {result.get('invalidated', 'none')}")
        return 0
    except Exception as e:
        print(f"Failed to clear cache: {e}", file=sys.stderr)
        return 1


def handle_daemon_command(args) -> int:
    """Handle daemon subcommands by dispatching to specific handlers."""
    config = get_config()
    host = getattr(args, 'daemon_host', config.daemon.host)
    port = getattr(args, 'daemon_port', config.daemon.port)
    verbose = getattr(args, 'verbose', False)
    no_watch = getattr(args, 'no_watch', False)

    action = args.daemon_action

    if action == "start":
        return _daemon_start(
            host, port,
            getattr(args, 'max_projects', config.daemon.max_projects),
            getattr(args, 'idle_timeout', None),
            verbose, no_watch,
        )
    elif action == "stop":
        return _daemon_stop(host, port)
    elif action == "status":
        return _daemon_status(host, port)
    elif action == "restart":
        return _daemon_restart(
            host, port,
            getattr(args, 'max_projects', config.daemon.max_projects),
            getattr(args, 'idle_timeout', None),
            verbose, no_watch,
        )
    elif action == "run":
        return _daemon_run(
            host, port,
            getattr(args, 'max_projects', config.daemon.max_projects),
            getattr(args, 'idle_timeout', None),
            verbose, no_watch,
        )
    elif action == "clear-cache":
        return _daemon_clear_cache(
            host, port,
            getattr(args, 'clear_path', None),
            getattr(args, 'clear_all', False),
        )

    return 0
