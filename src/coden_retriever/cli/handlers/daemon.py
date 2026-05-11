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
from ...daemon.address import DaemonAddress
from ...daemon.client import DaemonClient, get_daemon_status, stop_daemon
from ...daemon.protocol import WINDOWS_CREATE_NEW_PROCESS_GROUP, WINDOWS_DETACHED_PROCESS
from ...daemon.server import get_log_file, is_daemon_running, run_daemon
from ..utils import parse_duration


def _daemon_start(
    address: DaemonAddress, max_projects: int,
    idle_timeout: str | None, verbose: bool, no_watch: bool = False,
    daemon_timeout: float | None = None,
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
        "--daemon-host", address.host,
        "--daemon-port", str(address.port),
    ]

    if max_projects:
        cmd.extend(["--max-projects", str(max_projects)])
    if idle_timeout:
        cmd.extend(["--idle-timeout", str(idle_timeout)])
    if verbose:
        cmd.append("--verbose")
    if no_watch:
        cmd.append("--no-watch")
    if daemon_timeout is not None:
        cmd.extend(["--daemon-timeout", str(daemon_timeout)])

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
        status = get_daemon_status(address)
        if status:
            _, pid = is_daemon_running()
            print(f"Daemon started (PID: {pid})")
            print(f"  Address: {address.host}:{address.port}")
            print(f"  Log: {get_log_file()}")
            return 0

    print("Daemon failed to start. Check log:", file=sys.stderr)
    print(f"  {get_log_file()}", file=sys.stderr)
    return 1


def _daemon_stop(address: DaemonAddress) -> int:
    """Stop the daemon."""
    running, pid = is_daemon_running()
    if not running:
        print("Daemon is not running")
        return 0

    if stop_daemon(address):
        print(f"Daemon stopped (was PID: {pid})")
        print(f"  Address: {address.host}:{address.port}")
        return 0
    else:
        print(f"Failed to stop daemon (PID: {pid})", file=sys.stderr)
        return 1


def _daemon_status(address: DaemonAddress) -> int:
    """Show daemon status."""
    status = get_daemon_status(address)
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
    address: DaemonAddress, max_projects: int,
    idle_timeout: str | None, verbose: bool, no_watch: bool = False,
    daemon_timeout: float | None = None,
) -> int:
    """Restart the daemon."""
    stop_daemon(address)
    time.sleep(DAEMON_RESTART_DELAY_SECONDS)
    return _daemon_start(address, max_projects, idle_timeout, verbose, no_watch, daemon_timeout)


def _daemon_run(
    address: DaemonAddress, max_projects: int,
    idle_timeout: str | None, verbose: bool, no_watch: bool = False,
    daemon_timeout: float | None = None,
) -> int:
    """Run daemon in foreground."""
    config = get_config()
    max_projects = max_projects or config.daemon.max_projects
    timeout_seconds = parse_duration(idle_timeout) if idle_timeout else None
    # CLI arg takes precedence over config; fall back to config value
    effective_daemon_timeout = daemon_timeout if daemon_timeout is not None else config.daemon.daemon_timeout

    return run_daemon(
        address=address,
        max_projects=max_projects,
        idle_timeout=timeout_seconds,
        verbose=verbose,
        foreground=True,
        enable_watch=not no_watch,
        daemon_timeout=effective_daemon_timeout,
    )


def _daemon_clear_cache(address: DaemonAddress, clear_path: str | None, clear_all: bool) -> int:
    """Clear daemon cache."""
    client = DaemonClient(address=address, timeout=DAEMON_CACHE_TIMEOUT_SECONDS)
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
    address = DaemonAddress(
        host=getattr(args, 'daemon_host', config.daemon.host),
        port=getattr(args, 'daemon_port', config.daemon.port),
    )
    verbose = getattr(args, 'verbose', False)
    no_watch = getattr(args, 'no_watch', False)
    daemon_timeout = getattr(args, 'daemon_timeout', None)

    action = args.daemon_action

    if action == "stop":
        return _daemon_stop(address)
    if action == "status":
        return _daemon_status(address)
    if action == "clear-cache":
        return _daemon_clear_cache(
            address,
            getattr(args, 'clear_path', None),
            getattr(args, 'clear_all', False),
        )

    # start, restart, run all share the same settings parameters
    max_projects = getattr(args, 'max_projects', config.daemon.max_projects)
    idle_timeout = getattr(args, 'idle_timeout', None)
    settings_args = (address, max_projects, idle_timeout, verbose, no_watch, daemon_timeout)

    if action == "start":
        return _daemon_start(*settings_args)
    if action == "restart":
        return _daemon_restart(*settings_args)
    if action == "run":
        return _daemon_run(*settings_args)

    return 0
