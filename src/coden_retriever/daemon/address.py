"""Daemon network endpoint value object."""
from dataclasses import dataclass

from ..constants import DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT


@dataclass(frozen=True)
class DaemonAddress:
    """The host:port a daemon is reachable at.

    The two fields travel together at every transport site (DaemonClient,
    DaemonServer, all try_daemon_* helpers). Carrying them as one immutable
    value keeps signatures honest and lets the receiving entity hold the pair
    as a single attribute.
    """

    host: str = DEFAULT_DAEMON_HOST
    port: int = DEFAULT_DAEMON_PORT
