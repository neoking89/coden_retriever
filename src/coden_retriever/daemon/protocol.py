"""
JSON-RPC 2.0 protocol definitions for daemon communication.

Provides request/response models, serialization for IPC, and centralized constants.
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from ..constants import (
    DEFAULT_CLONE_RESULT_LIMIT,
    DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    DEFAULT_CLONE_SEMANTIC_WEIGHT,
    DEFAULT_CLONE_SYNTACTIC_WEIGHT,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DEFAULT_DAEMON_TIMEOUT,
    DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD,
    DEFAULT_DEAD_CODE_RESULT_LIMIT,
    DEFAULT_ECHO_COMMENT_THRESHOLD,
    DEFAULT_HOTSPOT_RISK_THRESHOLD,
    DEFAULT_MAX_PROJECTS,
    DEFAULT_PROPAGATION_COST_THRESHOLD,
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    SENSITIVE_VALUE_DEFAULT_LIMIT,
    SENSITIVE_VALUE_DEFAULT_THRESHOLD,
    MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT,
    TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    TRAMP_DATA_DEFAULT_RESULT_LIMIT,
    TRAMP_DATA_MIN_GROUP_SIZE,
    DEFAULT_TOKEN_BUDGET,
)

PROTOCOL_VERSION = "1.0"

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
    """Get daemon defaults from config or fallback to centralized constants.

    Returns:
        Tuple of (host, port, daemon_timeout, max_projects)
    """
    try:
        from ..config_loader import load_config
        config = load_config()
        return (
            config.daemon.host,
            config.daemon.port,
            config.daemon.daemon_timeout,
            config.daemon.max_projects,
        )
    except ImportError:
        return DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT, DEFAULT_DAEMON_TIMEOUT, DEFAULT_MAX_PROJECTS


@dataclass
class SearchParams:
    """Parameters for a search request."""
    source_dir: str
    query: str = ""
    enable_semantic: bool = False
    model_path: str | None = None
    limit: int | None = 20
    tokens: int | None = None
    show_deps: bool = False
    output_format: str = "tree"
    find_identifier: str | None = None
    show_map: bool = False
    dir_tree: bool = True
    stats: bool = False
    reverse: bool = False
    # JSON-serialisable string ("static" | "simple"). Parsed to MapMode at the
    # consumer (handlers / SearchEngine.search) — the wire stays plain strings
    # to keep the protocol dict-friendly.
    map_mode: str = "static"

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
    token_limit: int = DEFAULT_TOKEN_BUDGET

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
class CloneDetectionParams:
    """Parameters for clone detection requests.

    Used to find semantically similar (cloned) functions across the codebase.
    Supports three modes: combined (default), semantic, and syntactic.

    Note: token_limit=None means no limit (CLI mode).
    MCP tools should pass an explicit limit (e.g., 4000) for LLM context.
    """

    source_dir: str

    mode: Literal["combined", "semantic", "syntactic"] = "combined"

    # Semantic parameters
    similarity_threshold: float = DEFAULT_CLONE_SEMANTIC_THRESHOLD

    # Syntactic parameters
    line_threshold: float = DEFAULT_SYNTACTIC_LINE_THRESHOLD
    func_threshold: float = DEFAULT_SYNTACTIC_FUNC_THRESHOLD
    min_shared_lines: int = 2

    # Shared parameters
    limit: int = DEFAULT_CLONE_RESULT_LIMIT
    exclude_tests: bool = True
    min_lines: int = 3
    token_limit: int | None = None

    # Score fusion weights (for combined mode harmonic mean)
    semantic_weight: float = DEFAULT_CLONE_SEMANTIC_WEIGHT
    syntactic_weight: float = DEFAULT_CLONE_SYNTACTIC_WEIGHT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CloneDetectionParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PropagationCostParams:
    """Parameters for propagation cost requests.

    Used to measure how changes ripple through the codebase (architectural coupling).
    Based on MacCormack et al. (2006) research on software architecture.

    Note: token_limit=None means no limit (CLI mode).
    MCP tools should pass an explicit limit (e.g., 4000) for LLM context.
    """

    source_dir: str
    include_breakdown: bool = True
    show_critical_paths: bool = True
    exclude_tests: bool = True
    token_limit: int | None = None
    approximate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PropagationCostParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FlagParams:
    """Parameters for code flagging requests.

    Used to insert [CODEN] comments above code objects based on analysis results.
    Supports flagging from hotspots, propagation cost, clone detection, echo comments, and dead code.
    """

    source_dir: str
    hotspots: bool = False
    propagation: bool = False
    clones: bool = False
    echo_comments: bool = False
    dead_code: bool = False
    tramp_data: bool = False
    risk_threshold: float = DEFAULT_HOTSPOT_RISK_THRESHOLD
    propagation_threshold: float = DEFAULT_PROPAGATION_COST_THRESHOLD
    clone_threshold: float = DEFAULT_CLONE_SEMANTIC_THRESHOLD
    echo_threshold: float = DEFAULT_ECHO_COMMENT_THRESHOLD
    dead_code_threshold: float = DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD
    dry_run: bool = False
    limit: int | None = None
    backup: bool = False
    verbose: bool = False
    exclude_tests: bool = True
    remove_comments: bool = False
    remove_dead_code: bool = False
    output_format: str = "tree"

    clone_mode: Literal["combined", "semantic", "syntactic"] = "combined"
    # Syntactic clone parameters
    line_threshold: float = DEFAULT_SYNTACTIC_LINE_THRESHOLD
    func_threshold: float = DEFAULT_SYNTACTIC_FUNC_THRESHOLD

    # Tramp data parameters
    min_occurrences: int = TRAMP_DATA_DEFAULT_MIN_OCCURRENCES
    min_group_size: int = TRAMP_DATA_MIN_GROUP_SIZE

    # Sensitive value parameters
    sensitive_values: bool = False
    sensitive_threshold: float = SENSITIVE_VALUE_DEFAULT_THRESHOLD
    replace_value: str | None = None
    sensitive_whitelist: list[str] | None = None

    # Magic constant parameters
    magic_constants: bool = False
    magic_constant_min_occurrences: int = MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES
    magic_constant_min_files: int = MAGIC_CONSTANT_DEFAULT_MIN_FILES

    def has_any_flag_enabled(self) -> bool:
        """Check if at least one analysis flag is enabled.

        Required because flag_code needs at least one analysis type to run.
        """
        return (
            self.hotspots or self.propagation or self.clones
            or self.echo_comments or self.dead_code or self.tramp_data
            or self.sensitive_values or self.magic_constants
        )

    def validate(self) -> str | None:
        """Validate flag parameters.

        Returns:
            Error message if validation fails, None if valid.
        """
        if not self.has_any_flag_enabled():
            return "At least one analysis flag (-H, -P, -C, -E, -D, -T, -S, or -K) is required"
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FlagParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FlagClearParams:
    """Parameters for clearing [CODEN] flags from code.

    Removes all [CODEN] comments inserted by the flag command.
    """

    source_dir: str
    dry_run: bool = False
    verbose: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FlagClearParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DeadCodeParams:
    """Parameters for dead code detection requests.

    Used to find functions with no incoming calls in the call graph.
    Confidence scoring reduces false positives for entry points.
    """

    source_dir: str
    confidence_threshold: float = DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD
    limit: int | None = DEFAULT_DEAD_CODE_RESULT_LIMIT
    exclude_tests: bool = True
    include_private: bool = False
    min_lines: int = 3
    token_limit: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DeadCodeParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrampDataParams:
    """Parameters for tramp data detection requests.

    Used to find parameter groups traveling together across functions.
    """

    source_dir: str
    min_occurrences: int = TRAMP_DATA_DEFAULT_MIN_OCCURRENCES
    limit: int | None = TRAMP_DATA_DEFAULT_RESULT_LIMIT
    exclude_tests: bool = True
    token_limit: int | None = None
    min_group_size: int = TRAMP_DATA_MIN_GROUP_SIZE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrampDataParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SensitiveValueParams:
    """Parameters for sensitive value detection requests.

    Used to find hardcoded secrets, credentials, and sensitive data in string literals.
    """

    source_dir: str
    confidence_threshold: float = SENSITIVE_VALUE_DEFAULT_THRESHOLD
    limit: int | None = SENSITIVE_VALUE_DEFAULT_LIMIT
    exclude_tests: bool = True
    token_limit: int | None = None
    replace_value: str | None = None
    whitelist: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SensitiveValueParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MagicConstantParams:
    """Parameters for magic constant detection requests.

    Used to find repeated literal values (magic numbers/strings) across files.
    """

    source_dir: str
    min_occurrences: int = MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES
    min_files: int = MAGIC_CONSTANT_DEFAULT_MIN_FILES
    limit: int | None = MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT
    exclude_tests: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MagicConstantParams":
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
