"""
Daemon server for coden-retriever.

Provides a long-running TCP server that keeps indices in memory
for fast search responses.

Uses Python's socketserver module for clean socket lifecycle management.
"""
import ctypes
import json
import logging
import os
import signal
import socketserver
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..cache import CacheManager, CachedIndices
from ..config import CENTRAL_CACHE_DIR
from ..constants import (
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DEFAULT_MAX_PROJECTS,
    DEFAULT_DAEMON_TIMEOUT,
)
from ..search import SearchEngine
from ..watcher import BatchedChanges, FileWatcher
from .handlers import create_handler_registry
from .project_cache import ProjectCache
from .protocol import (
    IDLE_CHECK_INTERVAL,
    MAX_MESSAGE_SIZE,
    MESSAGE_DELIMITER,
    RECV_BUFFER_SIZE,
    ErrorCode,
    Request,
    Response,
)


logger = logging.getLogger(__name__)


class DaemonTCPServer(socketserver.ThreadingTCPServer):
    """
    Custom ThreadingTCPServer with address reuse enabled.

    This subclass exists to cleanly enable SO_REUSEADDR without modifying
    the base class, which would affect all ThreadingTCPServer instances.
    """
    allow_reuse_address = True
    daemon_threads = True  # Handler threads are daemon threads (exit when main exits)
    daemon_instance: "DaemonServer | None" = None  # Set by DaemonServer.start()


class DaemonRequestHandler(socketserver.StreamRequestHandler):
    """
    Handles a single client connection.

    Instantiated automatically by the server for every incoming connection.
    Uses the message-delimiter protocol for framing JSON-RPC messages.
    """

    def handle(self) -> None:
        """
        Handle client connection. Reads messages until connection closes.

        Access the parent DaemonServer via self.server.daemon_instance
        to process messages and access shared state (ProjectCache).
        """
        daemon: "DaemonServer" = self.server.daemon_instance  # type: ignore
        daemon.update_activity()

        client_addr = self.client_address
        buffer = b""

        try:
            self.request.settimeout(DEFAULT_DAEMON_TIMEOUT)

            while True:
                try:
                    data = self.request.recv(RECV_BUFFER_SIZE)
                    if not data:
                        break

                    buffer += data

                    # Prevent DoS via unbounded buffer growth
                    if len(buffer) > MAX_MESSAGE_SIZE:
                        logger.warning(f"Client {client_addr} exceeded max message size, disconnecting")
                        break

                    while MESSAGE_DELIMITER in buffer:
                        message_bytes, buffer = buffer.split(MESSAGE_DELIMITER, 1)
                        response = daemon.process_message(message_bytes.decode("utf-8"))
                        self.request.sendall(response.to_bytes())

                except TimeoutError:
                    break

        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")


class DaemonServer:
    """
    TCP server daemon for coden-retriever.

    Keeps indices in memory and handles search requests via JSON-RPC.
    Optionally watches for file changes and updates indices automatically.

    This class focuses on:
    - Server lifecycle (start, stop, cleanup)
    - Socket handling and request routing
    - Project loading and caching
    - File watching coordination

    Request handling logic is delegated to specialized handlers in handlers.py.
    """

    def __init__(
        self,
        host: str = DEFAULT_DAEMON_HOST,
        port: int = DEFAULT_DAEMON_PORT,
        max_projects: int = DEFAULT_MAX_PROJECTS,
        idle_timeout: int | None = None,
        verbose: bool = False,
        enable_watch: bool = True,
    ):
        self.host = host
        self.port = port
        self.max_projects = max_projects
        self.idle_timeout = idle_timeout
        self.verbose = verbose
        self.enable_watch = enable_watch

        # Application state
        self._project_cache = ProjectCache(max_projects)
        self._start_time = time.time()  # Server start time for uptime tracking
        self._last_activity = time.time()
        self._activity_lock = threading.Lock()  # Protects _last_activity
        self._shutdown_event = threading.Event()
        self._update_lock = threading.Lock()

        # TCP server (initialized lazily in start())
        self._tcp_server: DaemonTCPServer | None = None

        # Create handler registry using Strategy pattern
        self._handler_registry = create_handler_registry(self)

    def start(self) -> None:
        """Start the daemon server."""
        try:
            self._tcp_server = DaemonTCPServer(
                (self.host, self.port),
                DaemonRequestHandler
            )
            # Attach self so handlers can access process_message and cache
            self._tcp_server.daemon_instance = self

            logger.info(f"Daemon server listening on {self.host}:{self.port}")
            print(f"Daemon started on {self.host}:{self.port}", file=sys.stderr)

            # Start idle monitor in background if timeout is set
            if self.idle_timeout:
                idle_thread = threading.Thread(target=self._idle_monitor, daemon=True)
                idle_thread.start()

            # Start the server loop (blocks until shutdown is called)
            self._tcp_server.serve_forever()

        except OSError as e:
            logger.error(f"Failed to start daemon: {e}")
            raise
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Stop the daemon server."""
        logger.info("Stopping daemon...")
        self._shutdown_event.set()
        # shutdown() must be called from a different thread than serve_forever()
        if self._tcp_server is not None:
            threading.Thread(target=self._tcp_server.shutdown).start()

    def update_activity(self) -> None:
        """Update last activity timestamp. Called by request handlers.

        Thread-safe: Uses _activity_lock to prevent race conditions
        with the idle monitor thread.
        """
        with self._activity_lock:
            self._last_activity = time.time()

    def process_message(self, message: str) -> Response:
        """Process a JSON-RPC message and return response."""
        try:
            request = Request.from_json(message)
        except json.JSONDecodeError as e:
            return Response.make_error(0, ErrorCode.PARSE_ERROR, f"Parse error: {e}")
        except Exception as e:
            return Response.make_error(0, ErrorCode.INVALID_REQUEST, f"Invalid request: {e}")

        handler = self._handler_registry.get(request.method)
        if handler is None:
            return Response.make_error(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {request.method}"
            )

        try:
            result = handler.handle(request.params)
            return Response.success(request.id, result)
        except Exception as e:
            logger.exception(f"Error handling {request.method}")
            return Response.make_error(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Internal error: {e}"
            )

    def _idle_monitor(self) -> None:
        """Monitor for idle timeout and shutdown if exceeded.

        Thread-safe: Uses _activity_lock when reading _last_activity.
        """
        while not self._shutdown_event.is_set():
            time.sleep(IDLE_CHECK_INTERVAL)

            if self.idle_timeout:
                with self._activity_lock:
                    idle_time = time.time() - self._last_activity
                if idle_time > self.idle_timeout:
                    logger.info(f"Idle timeout reached ({idle_time:.0f}s), shutting down")
                    self.stop()
                    return

    def _cleanup(self) -> None:
        """Clean up server resources."""
        # Stop all file watchers
        self._project_cache.stop_all_watchers()

        if self._tcp_server is not None:
            self._tcp_server.server_close()

        logger.info("Daemon server stopped")

    def _get_or_load_project(
        self,
        source_dir: str,
        enable_semantic: bool = False,
        model_path: str | None = None
    ) -> tuple[CachedIndices, SearchEngine]:
        """Get project from cache or load it.

        Args:
            source_dir: Path to the source directory.
            enable_semantic: Whether to enable semantic search.
            model_path: Optional path to the semantic model.

        Returns:
            Tuple of (CachedIndices, SearchEngine).

        Raises:
            ValueError: If source directory does not exist.
        """
        source_path = Path(source_dir).resolve()

        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Source directory not found: {source_dir}")

        cache_key = str(source_path)
        cached = self._project_cache.get(cache_key)

        if cached is not None:
            if cached.indices.has_semantic == enable_semantic:
                return cached.indices, cached.engine

        logger.info(f"Loading project: {source_path}")
        start_time = time.time()

        cache_manager = CacheManager(
            source_path,
            enable_semantic=enable_semantic,
            model_path=model_path,
            verbose=self.verbose,
        )

        indices = cache_manager.load_or_rebuild()
        engine = SearchEngine.from_cached_indices(indices, verbose=self.verbose)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Project loaded in {elapsed:.0f}ms: {len(indices.entities)} entities")

        # Create file watcher if enabled
        watcher = None
        if self.enable_watch:
            watcher = self._create_watcher(source_path, cache_key, enable_semantic, model_path)

        self._project_cache.put(
            cache_key,
            indices,
            engine,
            watcher=watcher,
            enable_semantic=enable_semantic,
            model_path=model_path,
        )
        return indices, engine

    def _create_watcher(
        self,
        source_path: Path,
        cache_key: str,
        enable_semantic: bool,
        model_path: str | None,
    ) -> FileWatcher:
        """Create and start a file watcher for a project."""
        def on_changes(changes: BatchedChanges) -> None:
            self._handle_file_changes(cache_key, changes, enable_semantic, model_path)

        watcher = FileWatcher(
            source_dir=source_path,
            callback=on_changes,
            debounce_ms=300,
            batch_window_ms=500,
        )
        watcher.start()
        logger.info(f"Started file watcher for: {source_path}")
        return watcher

    def _handle_file_changes(
        self,
        cache_key: str,
        changes: BatchedChanges,
        enable_semantic: bool,
        model_path: str | None,
    ) -> None:
        """Handle file changes detected by watcher."""
        with self._update_lock:
            project = self._project_cache.get(cache_key)
            if project is None:
                logger.warning(f"Project no longer cached: {cache_key}")
                return

            logger.info(f"Processing file changes for {cache_key}: {changes}")

            # Create incremental updater (lazy import to avoid networkx at startup)
            from ..watcher import IncrementalUpdater
            updater = IncrementalUpdater(
                source_dir=Path(cache_key),
                indices=project.indices,
                enable_semantic=enable_semantic,
                model_path=model_path,
            )

            # Apply changes
            try:
                new_indices, was_full_rebuild = updater.apply_changes(changes)

                # Update engine from new indices
                new_engine = SearchEngine.from_cached_indices(new_indices, verbose=self.verbose)

                # Update cache (preserves watcher)
                self._project_cache.update_indices(cache_key, new_indices, new_engine)

                if was_full_rebuild:
                    logger.info(f"Full rebuild completed for {cache_key}")
                else:
                    logger.info(f"Incremental update completed for {cache_key}")

            except Exception as e:
                logger.exception(f"Failed to update indices for {cache_key}: {e}")

    def _handle_ping(self, params: dict) -> dict:
        """Handle a ping request. (Backwards compatibility wrapper)"""
        return self._handler_registry["ping"].handle(params)

    def _handle_status(self, params: dict) -> dict:
        """Handle a status request. (Backwards compatibility wrapper)"""
        return self._handler_registry["status"].handle(params)

    def _handle_invalidate(self, params: dict) -> dict:
        """Handle a cache invalidation request. (Backwards compatibility wrapper)"""
        return self._handler_registry["invalidate"].handle(params)

    def _handle_shutdown(self, params: dict) -> dict:
        """Handle a shutdown request. (Backwards compatibility wrapper)"""
        return self._handler_registry["shutdown"].handle(params)

    def _handle_search(self, params: dict) -> dict:
        """Handle a search request. (Backwards compatibility wrapper)"""
        return self._handler_registry["search"].handle(params)

    def _handle_find(self, params: dict) -> dict:
        """Handle a find request. (Backwards compatibility wrapper)"""
        return self._handler_registry["find"].handle(params)

    def _handle_architectural_bottlenecks(self, params: dict) -> dict:
        """Handle architectural_bottlenecks request. (Backwards compatibility wrapper)"""
        return self._handler_registry["architectural_bottlenecks"].handle(params)

    def _handle_coupling_hotspots(self, params: dict) -> dict:
        """Handle coupling_hotspots request. (Backwards compatibility wrapper)"""
        return self._handler_registry["coupling_hotspots"].handle(params)

    def _handle_change_impact_radius(self, params: dict) -> dict:
        """Handle change_impact_radius request. (Backwards compatibility wrapper)"""
        return self._handler_registry["change_impact_radius"].handle(params)

    def _handle_trace_dependency(self, params: dict) -> dict:
        """Handle trace_dependency request. (Backwards compatibility wrapper)"""
        return self._handler_registry["trace_dependency"].handle(params)

    def _handle_debug_stacktrace(self, params: dict) -> dict:
        """Handle debug_stacktrace request. (Backwards compatibility wrapper)"""
        return self._handler_registry["debug_stacktrace"].handle(params)


def get_pid_file() -> Path:
    """Get the path to the daemon PID file."""
    home = Path.home()
    daemon_dir = home / CENTRAL_CACHE_DIR
    daemon_dir.mkdir(parents=True, exist_ok=True)
    return daemon_dir / "daemon.pid"


def get_log_file() -> Path:
    """Get the path to the daemon log file."""
    home = Path.home()
    daemon_dir = home / CENTRAL_CACHE_DIR
    daemon_dir.mkdir(parents=True, exist_ok=True)
    return daemon_dir / "daemon.log"


def is_daemon_running(pid_file: Path | None = None) -> tuple[bool, int | None]:
    """Check if daemon is running and return its PID."""
    pid_file = pid_file or get_pid_file()

    if not pid_file.exists():
        return False, None

    try:
        pid = int(pid_file.read_text().strip())

        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True, pid
            return False, None
        else:
            os.kill(pid, 0)
            return True, pid

    except (ValueError, OSError, FileNotFoundError):
        return False, None


def write_pid_file(pid_file: Path | None = None) -> None:
    """Write current PID to file."""
    pid_file = pid_file or get_pid_file()
    pid_file.write_text(str(os.getpid()))


def remove_pid_file(pid_file: Path | None = None) -> None:
    """Remove PID file."""
    pid_file = pid_file or get_pid_file()
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def run_daemon(
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    max_projects: int = DEFAULT_MAX_PROJECTS,
    idle_timeout: int | None = None,
    verbose: bool = False,
    foreground: bool = False,
    enable_watch: bool = True,
) -> int:
    """
    Run the daemon server.

    Args:
        host: Host address to bind to
        port: Port to bind to
        max_projects: Maximum number of projects to keep in memory
        idle_timeout: Seconds of idle time before auto-shutdown (None for no timeout)
        verbose: Enable verbose logging
        foreground: Run in foreground (for debugging)
        enable_watch: Enable automatic file watching for cache updates

    Returns:
        Exit code
    """
    pid_file = get_pid_file()

    running, existing_pid = is_daemon_running(pid_file)
    if running:
        print(f"Daemon is already running (PID: {existing_pid})", file=sys.stderr)
        return 1

    # Set up logging with rotation (10MB max, 5 backups)
    log_file = get_log_file()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler() if foreground else logging.NullHandler(),
        ],
        force=True,
    )

    write_pid_file(pid_file)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        remove_pid_file(pid_file)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        server = DaemonServer(
            host=host,
            port=port,
            max_projects=max_projects,
            idle_timeout=idle_timeout,
            verbose=verbose,
            enable_watch=enable_watch,
        )
        server.start()
        return 0

    except Exception as e:
        logger.exception(f"Daemon failed: {e}")
        return 1

    finally:
        remove_pid_file(pid_file)
