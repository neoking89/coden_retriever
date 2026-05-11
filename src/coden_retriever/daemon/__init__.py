"""
Daemon mode for coden-retriever.

Provides a long-running background process that keeps indices in memory
for sub-200ms response times. Includes automatic file watching for
real-time index updates.
"""

from .address import DaemonAddress
from .server import DaemonServer
from .client import (
    DaemonClient,
    ensure_daemon_running,
    start_daemon_async,
    start_daemon_silent,
    stop_daemon,
    try_daemon_search,
)
from .protocol import Request, Response, SearchParams

__all__ = [
    "DaemonAddress",
    "DaemonServer",
    "DaemonClient",
    "Request",
    "Response",
    "SearchParams",
    "ensure_daemon_running",
    "start_daemon_async",
    "start_daemon_silent",
    "stop_daemon",
    "try_daemon_search",
]
