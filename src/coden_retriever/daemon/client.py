"""
Daemon client for coden-retriever.

Lightweight client to communicate with the daemon server.
"""
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from ..constants import (
    DEFAULT_CLIENT_TIMEOUT,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DEFAULT_HEAVY_ANALYSIS_TIMEOUT,
    DEFAULT_DAEMON_TIMEOUT,
)
from .protocol import (
    MESSAGE_DELIMITER,
    CloneDetectionParams,
    DeadCodeParams,
    FlagClearParams,
    FlagParams,
    PropagationCostParams,
    GraphAnalysisParams,
    Request,
    Response,
    SearchParams,
    StacktraceParams,
    TraceDependencyParams,
)
from .server import get_log_file, is_daemon_running, get_pid_file


logger = logging.getLogger(__name__)


class DaemonConnectionError(Exception):
    """Raised when unable to connect to daemon."""
    pass


class DaemonRequestError(Exception):
    """Raised when daemon returns an error response."""
    def __init__(self, code: int, message: str, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")


class DaemonClient:
    """
    Client for communicating with the daemon server.

    Provides methods for search, status, and control operations.
    """

    def __init__(
        self,
        host: str = DEFAULT_DAEMON_HOST,
        port: int = DEFAULT_DAEMON_PORT,
        timeout: float = DEFAULT_DAEMON_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    def _send_request(self, method: str, params: dict | None = None) -> Response:
        """Send a request and get the response."""
        request = Request(
            id=self._next_id(),
            method=method,
            params=params or {},
        )

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            try:
                # Send request
                sock.sendall(request.to_bytes())

                # Receive response
                buffer = b""
                while MESSAGE_DELIMITER not in buffer:
                    data = sock.recv(4096)
                    if not data:
                        raise DaemonConnectionError("Connection closed by daemon")
                    buffer += data

                message = buffer.split(MESSAGE_DELIMITER, 1)[0]
                response = Response.from_json(message.decode("utf-8"))

                if response.error:
                    raise DaemonRequestError(
                        response.error.get("code", -1),
                        response.error.get("message", "Unknown error"),
                        response.error.get("data"),
                    )

                return response

            finally:
                sock.close()

        except socket.timeout:
            raise DaemonConnectionError(f"Connection timed out after {self.timeout}s")
        except socket.error as e:
            raise DaemonConnectionError(f"Connection failed: {e}")

    def is_running(self) -> bool:
        """Check if daemon is running and responding."""
        try:
            self.ping()
            return True
        except DaemonConnectionError:
            return False

    def ping(self) -> dict:
        """Ping the daemon."""
        response = self._send_request("ping")
        return response.result

    def status(self) -> dict:
        """Get daemon status."""
        response = self._send_request("status")
        return response.result

    def invalidate(self, source_dir: str | None = None, all: bool = False) -> dict:
        """
        Invalidate daemon cache.

        Args:
            source_dir: Path to invalidate, or None
            all: If True, invalidate all cached projects
        """
        params: dict[str, str | bool] = {}
        if source_dir:
            params["source_dir"] = str(Path(source_dir).resolve())
        if all:
            params["all"] = True

        response = self._send_request("invalidate", params)
        return response.result

    def shutdown(self) -> dict:
        """Request daemon shutdown."""
        response = self._send_request("shutdown")
        return response.result

    def _execute_search(self, method: str, params: SearchParams) -> dict:
        """Common search execution logic."""
        response = self._send_request(method, params.to_dict())
        return response.result

    def search(self, params: SearchParams) -> dict:
        """
        Perform a search via the daemon.

        Args:
            params: SearchParams dataclass with all search options

        Returns:
            dict with keys: output, result_count, total_matched, search_time_ms, tokens_used
        """
        return self._execute_search("search", params)

    def find(self, params: SearchParams) -> dict:
        """
        Find an identifier via the daemon.

        Args:
            params: SearchParams dataclass with find_identifier set

        Returns:
            dict with keys: output, result_count, total_matched, search_time_ms, tokens_used
        """
        return self._execute_search("find", params)

    def _validate_result(self, result: dict) -> dict:
        """Validate result and raise DaemonRequestError if it contains an error."""
        if isinstance(result, dict) and "error" in result:
            raise DaemonRequestError(-1, result["error"], result)
        return result

    def architectural_bottlenecks(self, params: GraphAnalysisParams) -> dict:
        """Find architectural bottlenecks using in-memory indices."""
        response = self._send_request("architectural_bottlenecks", params.to_dict())
        return self._validate_result(response.result)

    def coupling_hotspots(self, params: GraphAnalysisParams) -> dict:
        """Find coupling hotspots using in-memory indices."""
        response = self._send_request("coupling_hotspots", params.to_dict())
        return self._validate_result(response.result)

    def change_impact_radius(self, params: GraphAnalysisParams) -> dict:
        """Analyze change impact radius using in-memory indices."""
        response = self._send_request("change_impact_radius", params.to_dict())
        return self._validate_result(response.result)

    def trace_dependency(self, params: TraceDependencyParams) -> dict:
        """Trace dependency paths using in-memory indices."""
        response = self._send_request("trace_dependency", params.to_dict())
        return self._validate_result(response.result)

    def debug_stacktrace(self, params: StacktraceParams) -> dict:
        """Debug stacktrace using in-memory indices."""
        response = self._send_request("debug_stacktrace", params.to_dict())
        return self._validate_result(response.result)

    def detect_clones(self, params: CloneDetectionParams) -> dict:
        """Detect code clones using in-memory indices."""
        response = self._send_request("detect_clones", params.to_dict())
        return self._validate_result(response.result)

    def propagation_cost(self, params: PropagationCostParams) -> dict:
        """Compute propagation cost using in-memory indices."""
        response = self._send_request("propagation_cost", params.to_dict())
        return self._validate_result(response.result)

    def flag_code(self, params: FlagParams) -> dict:
        """Flag code objects with [CODEN] comments based on analysis."""
        response = self._send_request("flag_code", params.to_dict())
        return self._validate_result(response.result)

    def flag_clear(self, params: FlagClearParams) -> dict:
        """Remove all [CODEN] comments from source files."""
        response = self._send_request("flag_clear", params.to_dict())
        return self._validate_result(response.result)

    def detect_dead_code(self, params: DeadCodeParams) -> dict:
        """Detect potentially dead code using in-memory indices."""
        response = self._send_request("detect_dead_code", params.to_dict())
        return self._validate_result(response.result)


def _try_daemon_request(
    client_method: str,
    params,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
    timeout: float = DEFAULT_CLIENT_TIMEOUT,
) -> dict | None:
    """
    Generic helper to execute a daemon request with auto-start support.

    This consolidates the common pattern of:
    1. Ensuring daemon is running (auto-starting if needed)
    2. Creating a client with appropriate timeout
    3. Calling the specified method
    4. Handling connection/request errors

    Args:
        client_method: Name of the DaemonClient method to call
        params: Parameters to pass to the method
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)
        timeout: Request timeout in seconds (default: DEFAULT_CLIENT_TIMEOUT)

    Returns:
        Result dict, or None if daemon unavailable and couldn't start
    """
    if auto_start:
        if not ensure_daemon_running(host=host, port=port, auto_start=True):
            return None
    else:
        client = DaemonClient(host=host, port=port, timeout=1.0)
        try:
            client.ping()
        except DaemonConnectionError:
            return None

    client = DaemonClient(host=host, port=port, timeout=timeout)

    try:
        method = getattr(client, client_method)
        return method(params)
    except (DaemonConnectionError, DaemonRequestError):
        return None


def try_daemon_search(
    params: SearchParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to perform a search via daemon, auto-starting if needed.

    Args:
        params: Search parameters (use SearchParams dataclass)
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Search result dict, or None if daemon unavailable and couldn't start
    """
    method = "find" if params.find_identifier else "search"
    return _try_daemon_request(method, params, host, port, auto_start)


def try_daemon_hotspots(
    params: GraphAnalysisParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to get hotspots via daemon, auto-starting if needed.

    Args:
        params: GraphAnalysisParams with source_dir and options
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Hotspots result dict, or None if daemon unavailable and couldn't start
    """
    return _try_daemon_request("coupling_hotspots", params, host, port, auto_start)


def try_daemon_clones(
    params: CloneDetectionParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
    timeout: float | None = None,
) -> dict | None:
    """
    Try to detect clones via daemon, auto-starting if needed.

    Args:
        params: CloneDetectionParams with source_dir and options
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)
        timeout: Request timeout (default: uses config daemon_timeout or 60s)

    Returns:
        Clone detection result dict, or None if daemon unavailable
    """
    # Clone detection is compute-intensive, use longer timeout
    # Default to config daemon_timeout, fallback to heavy analysis timeout
    effective_timeout = timeout if timeout is not None else DEFAULT_HEAVY_ANALYSIS_TIMEOUT
    return _try_daemon_request(
        "detect_clones", params, host, port, auto_start,
        timeout=effective_timeout
    )


def try_daemon_propagation_cost(
    params: PropagationCostParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to compute propagation cost via daemon, auto-starting if needed.

    Args:
        params: PropagationCostParams with source_dir and options
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Propagation cost result dict, or None if daemon unavailable
    """
    # Propagation cost analysis can be heavy on large codebases
    return _try_daemon_request(
        "propagation_cost", params, host, port, auto_start,
        timeout=DEFAULT_HEAVY_ANALYSIS_TIMEOUT
    )


def try_daemon_graph_analysis(
    method: str,
    params: GraphAnalysisParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to perform graph analysis via daemon, auto-starting if needed.

    Args:
        method: One of "change_impact_radius", "coupling_hotspots", "architectural_bottlenecks"
        params: GraphAnalysisParams with source_dir and options
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Graph analysis result dict, or None if daemon unavailable
    """
    return _try_daemon_request(method, params, host, port, auto_start)


def try_daemon_stacktrace(
    params: StacktraceParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to debug stacktrace via daemon, auto-starting if needed.

    Args:
        params: StacktraceParams with stacktrace and source_dir
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Stacktrace analysis result dict, or None if daemon unavailable
    """
    return _try_daemon_request("debug_stacktrace", params, host, port, auto_start)


def try_daemon_trace_dependency(
    params: TraceDependencyParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to trace dependencies via daemon, auto-starting if needed.

    Args:
        params: TraceDependencyParams with identifier and source_dir
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Dependency trace result dict, or None if daemon unavailable
    """
    return _try_daemon_request("trace_dependency", params, host, port, auto_start)


def try_daemon_flag(
    params: FlagParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to flag code via daemon, auto-starting if needed.

    Args:
        params: FlagParams with source_dir and flag options
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Flag result dict, or None if daemon unavailable
    """
    return _try_daemon_request("flag_code", params, host, port, auto_start)


def try_daemon_flag_clear(
    params: FlagClearParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to clear flags via daemon, auto-starting if needed.

    Args:
        params: FlagClearParams with source_dir
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Flag clear result dict, or None if daemon unavailable
    """
    return _try_daemon_request("flag_clear", params, host, port, auto_start)


def try_daemon_dead_code(
    params: DeadCodeParams,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> dict | None:
    """
    Try to detect dead code via daemon, auto-starting if needed.

    Args:
        params: DeadCodeParams with source_dir and options
        host: Daemon host
        port: Daemon port
        auto_start: Auto-start daemon if not running (default: True)

    Returns:
        Dead code detection result dict, or None if daemon unavailable
    """
    return _try_daemon_request("detect_dead_code", params, host, port, auto_start)


def stop_daemon(host: str = DEFAULT_DAEMON_HOST, port: int = DEFAULT_DAEMON_PORT) -> bool:
    """
    Stop the daemon server.

    Returns:
        True if successfully stopped, False otherwise.
    """
    client = DaemonClient(host=host, port=port, timeout=DEFAULT_CLIENT_TIMEOUT)

    try:
        client.shutdown()
        time.sleep(0.5)
        return True
    except DaemonConnectionError:
        # Try killing by PID
        running, pid = is_daemon_running()
        if running and pid:
            return _kill_daemon_process(pid)
        return False
    except DaemonRequestError:
        return False


def _kill_daemon_process(pid: int, force: bool = False) -> bool:
    """
    Kill daemon process by PID (platform-specific).

    Args:
        pid: Process ID to kill
        force: If True, use SIGKILL (Unix) for immediate termination.
               On Windows, taskkill /F is always used regardless of this flag.

    Returns:
        True if kill signal was sent successfully, False otherwise
    """
    try:
        if sys.platform == "win32":
            # Windows: taskkill /F is always a force kill
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, text=True
            )
            success = result.returncode == 0
        else:
            # Unix: SIGKILL for force, SIGTERM for graceful
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            success = True

        # Only clean up PID file if kill succeeded
        if success:
            pid_file = get_pid_file()
            if pid_file.exists():
                pid_file.unlink()

        return success

    except (ProcessLookupError, PermissionError, OSError):
        return False


def get_daemon_status(host: str = DEFAULT_DAEMON_HOST, port: int = DEFAULT_DAEMON_PORT) -> dict | None:
    """
    Get daemon status.

    Returns:
        Status dict if daemon is running, None otherwise.
    """
    client = DaemonClient(host=host, port=port, timeout=2.0)

    try:
        return client.status()
    except (DaemonConnectionError, DaemonRequestError):
        return None


DAEMON_START_TIMEOUT = 10.0
DAEMON_START_POLL_INTERVAL = 0.1
PORT_RELEASE_WAIT = 0.5  # Time for OS to release socket after process termination


def _spawn_daemon_process(
    host: str,
    port: int,
    max_projects: int,
    enable_watch: bool,
) -> bool:
    """
    Spawn the daemon process without waiting for it to be ready.

    Args:
        host: Host to bind to
        port: Port to bind to
        max_projects: Maximum projects to cache
        enable_watch: Enable file watching

    Returns:
        True if process was spawned, False on spawn failure
    """
    # On Windows, use pythonw.exe to avoid console window
    if sys.platform == "win32":
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")
    else:
        python_exe = sys.executable

    cmd = [
        python_exe,
        "-m", "coden_retriever",
        "daemon", "run",
        "--daemon-host", host,
        "--daemon-port", str(port),
        "--max-projects", str(max_projects),
    ]

    if not enable_watch:
        cmd.append("--no-watch")

    log_file = get_log_file()

    try:
        # Use context manager for automatic cleanup
        # Child process inherits file descriptor during Popen(),
        # so closing parent's copy after Popen() returns is safe
        with open(log_file, "a") as log_f:
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=log_f,
                    stdin=subprocess.DEVNULL,
                    startupinfo=startupinfo,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW |
                        subprocess.DETACHED_PROCESS |
                        subprocess.CREATE_NEW_PROCESS_GROUP
                    ),
                    close_fds=False,
                )
            else:
                subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=log_f,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )

        return True

    except Exception as e:
        logger.debug(f"Failed to spawn daemon: {e}")
        return False


def start_daemon_async(
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    max_projects: int = 5,
    enable_watch: bool = True,
) -> bool:
    """
    Start the daemon asynchronously without waiting for it to be ready.

    This is optimized for fast startup - it spawns the daemon process and
    returns immediately without waiting for it to become responsive.

    Args:
        host: Host to bind to
        port: Port to bind to
        max_projects: Maximum projects to cache
        enable_watch: Enable file watching

    Returns:
        True if daemon was spawned (or already running), False on spawn failure
    """
    # Quick check if already running (short timeout for speed)
    client = DaemonClient(host=host, port=port, timeout=0.5)
    try:
        client.ping()
        return True
    except DaemonConnectionError:
        pass

    return _spawn_daemon_process(host, port, max_projects, enable_watch)


def start_daemon_silent(
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    max_projects: int = 5,
    enable_watch: bool = True,
) -> bool:
    """
    Start the daemon silently in the background.

    This starts the daemon as a completely detached process with no terminal
    window. Waits for daemon to become responsive before returning.

    Args:
        host: Host to bind to
        port: Port to bind to
        max_projects: Maximum projects to cache
        enable_watch: Enable file watching

    Returns:
        True if daemon started successfully, False otherwise
    """
    # Check if already running
    client = DaemonClient(host=host, port=port, timeout=1.0)
    try:
        client.ping()
        return True
    except DaemonConnectionError:
        pass

    if not _spawn_daemon_process(host, port, max_projects, enable_watch):
        return False

    # Wait for daemon to become responsive
    return _wait_for_daemon(host, port, timeout=DAEMON_START_TIMEOUT)


def _wait_for_daemon(
    host: str,
    port: int,
    timeout: float = DAEMON_START_TIMEOUT,
) -> bool:
    """
    Wait for daemon to become responsive.

    Args:
        host: Daemon host
        port: Daemon port
        timeout: Maximum time to wait

    Returns:
        True if daemon is responsive, False if timeout
    """
    client = DaemonClient(host=host, port=port, timeout=1.0)
    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            client.ping()
            return True
        except DaemonConnectionError:
            time.sleep(DAEMON_START_POLL_INTERVAL)

    return False


def ensure_daemon_running(
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    auto_start: bool = True,
) -> bool:
    """
    Ensure the daemon is running, starting it silently if needed.

    Also detects and recovers from zombie daemons (process exists but
    not responding).

    Args:
        host: Daemon host
        port: Daemon port
        auto_start: If True, start daemon if not running

    Returns:
        True if daemon is running (or was started), False otherwise
    """
    # Quick check via ping
    client = DaemonClient(host=host, port=port, timeout=1.0)
    try:
        client.ping()
        return True
    except DaemonConnectionError:
        pass

    # Ping failed - check if this is a zombie (process exists but not responding)
    running, pid = is_daemon_running()

    if running and pid:
        # ZOMBIE DETECTED: Process exists but doesn't respond to ping
        logger.warning(f"Zombie daemon detected (PID: {pid}), forcing termination...")

        if not _kill_daemon_process(pid, force=True):
            logger.error(f"Failed to kill zombie daemon (PID: {pid})")
            return False

        # Give OS time to release the port
        time.sleep(PORT_RELEASE_WAIT)

        # Verify the zombie is actually dead
        still_running, _ = is_daemon_running()
        if still_running:
            logger.error(f"Zombie daemon (PID: {pid}) survived force kill")
            return False

    # Not running (or zombie was killed) - start if allowed
    if auto_start:
        return start_daemon_silent(host=host, port=port)

    return False
