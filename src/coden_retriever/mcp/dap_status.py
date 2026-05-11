"""Canonical DAP client status results.

Replaces scattered `{"status": "...", ...}` dict literals across dap_client.py
with a single dataclass + per-shape factories. `to_dict()` drops None fields so
each shape emits exactly the keys it needs.
"""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DebugResult:
    """Status result for DAP client operations. Use classmethod factories."""

    status: str
    program: str | None = None
    host: str | None = None
    port: int | None = None
    stopped: bool | None = None
    reason: str | None = None
    file: str | None = None
    line: int | None = None
    exception: str | None = None
    output: list[str] | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def ready(cls, message: str) -> "DebugResult":
        return cls(status="ready", message=message)

    @classmethod
    def session_stopped(cls) -> "DebugResult":
        # "disconnected" distinguishes session teardown from stop_info() which
        # emits status="stopped" for breakpoint/step stops during an active
        # session. Callers checking status=="stopped" would otherwise conflate
        # the two.
        return cls(status="disconnected")

    @classmethod
    def attached(cls, host: str, port: int, stopped: bool) -> "DebugResult":
        return cls(status="attached", host=host, port=port, stopped=stopped)

    @classmethod
    def launched(
        cls,
        program: str,
        stopped: bool,
        reason: str | None = None,
        file: str | None = None,
        line: int | None = None,
    ) -> "DebugResult":
        return cls(
            status="launched",
            program=program,
            stopped=stopped,
            reason=reason,
            file=file,
            line=line,
        )

    @classmethod
    def stop_info(
        cls,
        reason: str | None,
        file: str | None,
        line: int | None,
        exception: str | None = None,
    ) -> "DebugResult":
        return cls(
            status="stopped",
            reason=reason,
            file=file,
            line=line,
            exception=exception,
        )

    @classmethod
    def terminated(cls, output: list[str]) -> "DebugResult":
        return cls(status="terminated", output=output)

    @classmethod
    def running(cls) -> "DebugResult":
        return cls(status="running")

    @classmethod
    def not_running(cls) -> "DebugResult":
        return cls(status="not_running")

    @classmethod
    def timeout(cls, message: str) -> "DebugResult":
        return cls(status="timeout", message=message)


def success_with(**payload: Any) -> dict[str, Any]:
    """Build `{"status": "success", **payload}` — for data-carrying success shapes."""
    return {"status": "success", **payload}
