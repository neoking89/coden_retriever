"""
JSON-RPC 2.0 protocol definitions for daemon communication.

Provides request/response models, serialization for IPC, and centralized constants.
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Any

PROTOCOL_VERSION = "1.0"

# Fallback defaults (used when config loading fails)
_DEFAULT_PORT = 19847
_DEFAULT_HOST = "127.0.0.1"
_SOCKET_TIMEOUT = 30.0
_CLIENT_TIMEOUT = 5.0
_MAX_PROJECTS_DEFAULT = 5

# Keep old names for backward compatibility
DEFAULT_PORT = _DEFAULT_PORT
DEFAULT_HOST = _DEFAULT_HOST
SOCKET_TIMEOUT = _SOCKET_TIMEOUT
CLIENT_TIMEOUT = _CLIENT_TIMEOUT
MAX_PROJECTS_DEFAULT = _MAX_PROJECTS_DEFAULT

IDLE_CHECK_INTERVAL = 10  # Check idle status every N seconds

# Buffer and limits
RECV_BUFFER_SIZE = 4096
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB max message size to prevent DoS

# Message delimiter for TCP streaming
MESSAGE_DELIMITER = b"\n\n"

# Windows process creation flags
WINDOWS_DETACHED_PROCESS = 0x00000008
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200


def get_daemon_defaults() -> tuple[str, int, float, int]:
    """Get daemon defaults from config or fallback to hardcoded values.

    Returns:
        Tuple of (host, port, socket_timeout, max_projects)
    """
    try:
        from ..config_loader import load_config
        config = load_config()
        return (
            config.daemon.host,
            config.daemon.port,
            config.daemon.socket_timeout,
            config.daemon.max_projects,
        )
    except ImportError:
        return _DEFAULT_HOST, _DEFAULT_PORT, _SOCKET_TIMEOUT, _MAX_PROJECTS_DEFAULT


@dataclass
class SearchParams:
    """Parameters for a search request."""
    source_dir: str
    query: str = ""
    enable_semantic: bool = False
    model_path: str | None = None
    limit: int = 20
    tokens: int | None = None
    show_deps: bool = False
    output_format: str = "tree"
    find_identifier: str | None = None
    map_mode: bool = False
    dir_tree: bool = True
    stats: bool = False
    reverse: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SearchParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GraphAnalysisParams:
    """Parameters for graph analysis requests (bottlenecks, hotspots, impact)."""

    source_dir: str
    limit: int = 20
    exclude_tests: bool = True
    token_limit: int = 4000

    # For architectural_bottlenecks
    min_betweenness: float = 0.001

    # For coupling_hotspots
    min_coupling_score: int = 10
    exclude_private: bool = False

    # For change_impact_radius
    symbol_name: str | None = None
    max_depth: int = 5
    min_importance: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GraphAnalysisParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TraceDependencyParams:
    """Parameters for trace_dependency_path requests."""

    source_dir: str
    start_identifier: str
    end_identifier: str | None = None
    direction: str = "downstream"  # "upstream", "downstream", "both"
    max_depth: int = 5
    limit_paths: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TraceDependencyParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StacktraceParams:
    """Parameters for debug_stacktrace requests."""

    source_dir: str
    stacktrace: str
    context_lines: int = 5
    show_dependencies: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StacktraceParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Request:
    """JSON-RPC 2.0 request."""
    id: int
    method: str
    params: dict = field(default_factory=dict)
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        return json.dumps({
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        })

    def to_bytes(self) -> bytes:
        """Serialize to bytes with delimiter."""
        return self.to_json().encode("utf-8") + MESSAGE_DELIMITER

    @classmethod
    def from_json(cls, data: str) -> "Request":
        obj = json.loads(data)
        return cls(
            id=obj.get("id", 0),
            method=obj["method"],
            params=obj.get("params", {}),
            jsonrpc=obj.get("jsonrpc", "2.0"),
        )


@dataclass
class Response:
    """JSON-RPC 2.0 response."""
    id: int
    result: Any = None
    error: dict | None = None
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        data = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error is not None:
            data["error"] = self.error
        else:
            data["result"] = self.result
        return json.dumps(data)

    def to_bytes(self) -> bytes:
        """Serialize to bytes with delimiter."""
        return self.to_json().encode("utf-8") + MESSAGE_DELIMITER

    @classmethod
    def from_json(cls, data: str) -> "Response":
        obj = json.loads(data)
        return cls(
            id=obj.get("id", 0),
            result=obj.get("result"),
            error=obj.get("error"),
            jsonrpc=obj.get("jsonrpc", "2.0"),
        )

    @classmethod
    def success(cls, id: int, result: Any) -> "Response":
        """Create a success response."""
        return cls(id=id, result=result, error=None)

    @classmethod
    def make_error(cls, id: int, code: int, message: str, data: Any = None) -> "Response":
        """Create an error response."""
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return cls(id=id, result=None, error=err)


class ErrorCode:
    """Standard and custom JSON-RPC error codes."""
    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes (server-defined, -32000 to -32099)
    SOURCE_NOT_FOUND = -32000
    INDEX_ERROR = -32001
    TIMEOUT = -32002
