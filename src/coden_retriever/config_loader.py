"""Unified configuration loader for coden-retriever.

Provides a single source of truth for all user-configurable settings.
Priority: CLI args > config file > environment variables > hardcoded defaults

Configuration is stored at ~/.coden-retriever/settings.json
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Callable, Literal

from .agent._constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    LLAMACPP_DEFAULT_URL,
    OLLAMA_DEFAULT_URL,
)
from .agent.settings import GenerationSettings
from .constants import (
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DEFAULT_DAEMON_TIMEOUT,
    DEFAULT_FULL_SERVER_INSTRUCTIONS_TEMPLATE,
    DEFAULT_MAX_PROJECTS,
    DEFAULT_SEARCH_RESULT_LIMIT,
    DEFAULT_STARTER_QUESTIONS,
    DEFAULT_STUDY_PROMPT_TEMPLATE,
    DEFAULT_STUDY_TOOL_INSTRUCTIONS_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOOL_INSTRUCTIONS_TEMPLATE,
    DEFAULT_TOOL_ROUTER_PROMPT_TEMPLATE,
    FILE_PATH_PREFIX,
)
from .mcp.constants import DEFAULT_TOOL_TIMEOUT_S

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1


@dataclass
class SettingMeta:
    """Metadata for a user-configurable setting.

    Provides a single source of truth for setting descriptions,
    used by both /config display and tab completion.
    """
    key: str
    short_desc: str   # Brief description for tab completion
    long_desc: str    # Detailed description for /config display
    value_type: Literal["str", "int", "bool", "float"]  # Type of the setting value
    env_var: Optional[str] = None  # Environment variable name (None = no env override)


# Single source of truth for all user-facing setting metadata
SETTING_METADATA: dict[str, SettingMeta] = {
    "model": SettingMeta(
        "model",
        "LLM model identifier",
        "ollama:name, llamacpp:name, openai:name (official API), or name+base_url",
        "str",
        "CODEN_RETRIEVER_MODEL",
    ),
    "base_url": SettingMeta(
        "base_url",
        "API endpoint URL",
        "OpenAI-compatible endpoint (auto-detected for ollama/llamacpp)",
        "str",
        "CODEN_RETRIEVER_BASE_URL",
    ),
    "max_steps": SettingMeta(
        "max_steps",
        "Max tool calls per query",
        "Maximum tool calls per query",
        "int",
    ),
    "max_retries": SettingMeta(
        "max_retries",
        "Retry attempts for errors",
        "Retry attempts for MCP connections, agent validation, and malformed tool calls",
        "int",
    ),
    "debug": SettingMeta(
        "debug",
        "Enable debug logging",
        "Log prompts and tool calls to ~/.coden-retriever/",
        "bool",
    ),
    "tool_instructions": SettingMeta(
        "tool_instructions",
        "Include tool workflow guidance",
        "Add guidance to models how to use tools (helps weaker models)",
        "bool",
    ),
    "ask_tool_permission": SettingMeta(
        "ask_tool_permission",
        "Ask before executing tools",
        "Ask permission before executing each tool. Disable at own risk!",
        "bool",
    ),
    "dynamic_tool_filtering": SettingMeta(
        "dynamic_tool_filtering",
        "LLM-based tool routing",
        "LLM-based tool routing (show only relevant tools per query)",
        "bool",
    ),
    "tool_filter_model": SettingMeta(
        "tool_filter_model",
        "Model for tool routing",
        "LLM for tool filtering ('model'=sync with /model, or specific model string)",
        "str",
        "CODEN_RETRIEVER_TOOL_FILTER_MODEL",
    ),
    "temperature": SettingMeta(
        "temperature",
        "Model temperature (0-2)",
        "Controls randomness (0.0=deterministic, 1.0+=creative)",
        "float",
        "CODEN_RETRIEVER_TEMPERATURE",
    ),
    "max_tokens": SettingMeta(
        "max_tokens",
        "Max response tokens",
        "Maximum tokens in response (empty=model default)",
        "int",
        "CODEN_RETRIEVER_MAX_TOKENS",
    ),
    "timeout": SettingMeta(
        "timeout",
        "Request timeout (seconds)",
        "Timeout for model API requests (default: 120)",
        "float",
        "CODEN_RETRIEVER_TIMEOUT",
    ),
    "api_key": SettingMeta(
        "api_key",
        "API key override",
        "Custom API key (overrides OPENAI_API_KEY env var for custom endpoints)",
        "str",
        "CODEN_RETRIEVER_API_KEY",
    ),
    "host": SettingMeta(
        "host",
        "Daemon host address",
        "Host address for the daemon server (default: 127.0.0.1)",
        "str",
        "CODEN_RETRIEVER_DAEMON_HOST",
    ),
    "port": SettingMeta(
        "port",
        "Daemon port number",
        "Port for the daemon server (default: 19847)",
        "int",
        "CODEN_RETRIEVER_DAEMON_PORT",
    ),
    "daemon_timeout": SettingMeta(
        "daemon_timeout",
        "Socket timeout (seconds)",
        "Timeout for daemon socket operations (default: 30)",
        "float",
    ),
    "max_projects": SettingMeta(
        "max_projects",
        "Max cached projects",
        "Maximum number of projects to keep in daemon cache",
        "int",
    ),
    "auto_start": SettingMeta(
        "auto_start",
        "Auto-start the daemon on first use",
        "When false, CLI and MCP surfaces skip the daemon and run in-process. "
        "Useful for debugging, eval/benchmark runs, or keeping a deterministic "
        "cold path (default: true).",
        "bool",
        "CODEN_RETRIEVER_DAEMON_AUTO_START",
    ),
    "default_tokens": SettingMeta(
        "default_tokens",
        "Default token budget",
        "Default token budget for search results (default: 4000)",
        "int",
    ),
    "default_limit": SettingMeta(
        "default_limit",
        "Default result limit",
        "Default maximum number of search results (default: 20)",
        "int",
    ),
    "semantic_model_path": SettingMeta(
        "semantic_model_path",
        "Embedding model path",
        "Path to custom embedding model directory. "
        "Default: bundled all-MiniLM-L6-v2 (384-dim, INT8 ONNX).",
        "str",
        "CODEN_RETRIEVER_MODEL_PATH",
    ),
    "compaction_token_threshold": SettingMeta(
        "compaction_token_threshold",
        "Auto-compact history at N tokens",
        "Drop old tool-call groups when retained context reaches N tokens. "
        "0 disables compaction. Use /undo to recover the pre-compaction history.",
        "int",
        "CODEN_RETRIEVER_COMPACTION_THRESHOLD",
    ),
}

# Maps setting keys to their config section and attribute path
# Format: key -> (section, attr_name, sub_attr_name or None)
SETTING_LOCATIONS: dict[str, tuple[str, str, Optional[str]]] = {
    "model": ("model", "default", None),
    "base_url": ("model", "base_url", None),
    "temperature": ("model", "generation", "temperature"),
    "max_tokens": ("model", "generation", "max_tokens"),
    "timeout": ("model", "generation", "timeout"),
    "api_key": ("model", "generation", "api_key"),
    "max_steps": ("agent", "max_steps", None),
    "max_retries": ("agent", "max_retries", None),
    "debug": ("agent", "debug", None),
    "tool_instructions": ("agent", "tool_instructions", None),
    "ask_tool_permission": ("agent", "ask_tool_permission", None),
    "dynamic_tool_filtering": ("agent", "dynamic_tool_filtering", None),
    "tool_filter_model": ("agent", "tool_filter_model", None),
    "host": ("daemon", "host", None),
    "port": ("daemon", "port", None),
    "daemon_timeout": ("daemon", "daemon_timeout", None),
    "max_projects": ("daemon", "max_projects", None),
    "auto_start": ("daemon", "auto_start", None),
    "default_tokens": ("search", "default_tokens", None),
    "default_limit": ("search", "default_limit", None),
    "semantic_model_path": ("search", "semantic_model_path", None),
    "compaction_token_threshold": ("agent", "compaction_token_threshold", None),
}

# Validation constraints for settings
# Format: key -> (min_value, max_value, error_message) or None for no constraints
SETTING_CONSTRAINTS: dict[str, tuple[float, float, str]] = {
    "temperature": (0.0, 2.0, "must be between 0.0 and 2.0"),
    "timeout": (0.001, float("inf"), "must be greater than 0"),
    "daemon_timeout": (0.001, float("inf"), "must be greater than 0"),
    "max_steps": (1, float("inf"), "must be at least 1"),
    "max_retries": (0, float("inf"), "must be 0 or greater"),
    "max_tokens": (1, float("inf"), "must be at least 1"),
    "port": (1, 65535, "must be between 1 and 65535"),
    "max_projects": (1, float("inf"), "must be at least 1"),
    "default_tokens": (1, float("inf"), "must be at least 1"),
    "default_limit": (1, float("inf"), "must be at least 1"),
    "compaction_token_threshold": (0, float("inf"), "must be 0 or greater"),
}

# Per-setting step delta for <,> arrow-stepping in the /config picker.
# WHY per-key: a uniform step of 1 makes temperature unusable (0-2 range) and
# max_tokens tedious (typical values 2000-32000). Each delta is chosen so that
# ~10-30 presses can sweep the realistic range of that setting.
SETTING_STEPS: dict[str, float] = {
    "temperature": 0.1,      # 0.0-2.0 range, one-decimal feel
    "timeout": 5.0,          # seconds, typical 30-300
    "daemon_timeout": 5.0,   # seconds, same family as timeout
    "max_steps": 1,          # small integer knob
    "max_retries": 1,        # small integer knob
    "max_tokens": 500,       # typical 1000-32000, coarse but useful
    "default_tokens": 500,   # same shape as max_tokens
    "port": 1,               # 1-65535; user almost always wants ±1
    "max_projects": 1,       # small integer knob
    "default_limit": 1,      # search result count, small integer
    "compaction_token_threshold": 1000,  # tokens, useful sweep range 0-50000+
}

# Fallback deltas when a setting has no entry in SETTING_STEPS.
# WHY: chosen so "just works" defaults stay intuitive for any future setting.
DEFAULT_INT_STEP: int = 1
DEFAULT_FLOAT_STEP: float = 0.1

# String settings that must not be empty when explicitly set.
# api_key with empty string causes confusing downstream auth errors.
NON_EMPTY_STRING_SETTINGS: frozenset[str] = frozenset({"api_key"})

# Fields stored as line arrays in JSON for readability, keyed by config section.
# Add a new template by appending its field name to the right section's tuple
# (and adding the matching attribute on the corresponding dataclass).
_TEMPLATE_FIELDS: dict[str, tuple[str, ...]] = {
    "agent": (
        "system_prompt_template",
        "study_prompt_template",
        "tool_instructions_template",
        "study_tool_instructions_template",
        "tool_router_prompt_template",
    ),
    "mcp": (
        "full_server_instructions_template",
    ),
}

# Type parsers dispatch table - maps value_type to parser function
_TYPE_PARSERS: dict[str, Callable[[str], Any]] = {
    "bool": lambda v: v.lower() in ("true", "1", "yes"),
    "int": int,
    "float": float,
    "str": lambda v: v if v.lower() != "null" else None,
}


def resolve_template(template: str) -> str:
    """Return the template string, loading from disk if it starts with 'file:'.

    Raises OSError/ValueError on file: paths that can't be read.
    """
    if not template.startswith(FILE_PATH_PREFIX):
        return template

    path_str = template[len(FILE_PATH_PREFIX):].strip()
    if not path_str:
        raise ValueError("file: prefix requires a path")

    return Path(path_str).expanduser().resolve().read_text(encoding="utf-8")


def resolve_or_default(template: str, default: str, field_name: str) -> str:
    """Resolve template; on file:-load failure, log and fall back to default."""
    try:
        return resolve_template(template)
    except (OSError, ValueError) as exc:
        logger.warning("%s failed (%s), using default", field_name, exc)
        return default


def parse_config_value(key: str, value: str) -> tuple[bool, Any, str]:
    """Parse a string value to the appropriate type based on SETTING_METADATA.

    Args:
        key: The setting key.
        value: The string value to parse.

    Returns:
        Tuple of (success, parsed_value, error_message).
    """
    if key not in SETTING_METADATA:
        valid_keys = ", ".join(sorted(SETTING_METADATA.keys()))
        return False, None, f"Unknown key: {key}. Valid keys: {valid_keys}"

    meta = SETTING_METADATA[key]

    try:
        # Use dispatch table for type parsing
        parser = _TYPE_PARSERS.get(meta.value_type, _TYPE_PARSERS["str"])
        parsed = parser(value)
        return True, parsed, ""
    except ValueError as e:
        return False, None, f"Invalid {meta.value_type} value '{value}': {e}"


def validate_config_value(
    key: str, value: Any, check_paths: bool = False,
) -> tuple[bool, str]:
    """Validate a parsed config value against constraints.

    Args:
        key: The setting key.
        value: The parsed value to validate.
        check_paths: When True, validate that file path settings point to
            existing paths. Only enabled for interactive /config set, not
            during config file loading (paths may not exist on this machine).

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if value is None:
        return True, ""  # None values skip constraint validation

    # Reject empty strings for settings that require a value
    if key in NON_EMPTY_STRING_SETTINGS and isinstance(value, str) and value == "":
        return False, f"{key} must not be empty"

    # Validate that file paths exist when explicitly set by user
    if check_paths and key == "semantic_model_path" and isinstance(value, str):
        if not Path(value).exists():
            return False, f"semantic_model_path '{value}' does not exist"
        from .utils.embedding_validation import validate_embedding_model_dir
        ok, err = validate_embedding_model_dir(Path(value))
        if not ok:
            return False, err

    if key not in SETTING_CONSTRAINTS:
        return True, ""

    # Constraints require numeric comparison - reject non-numeric types
    if not isinstance(value, (int, float)):
        return False, f"{key} must be a number, got {type(value).__name__}"

    min_val, max_val, error_msg = SETTING_CONSTRAINTS[key]
    if not (min_val <= value <= max_val):
        return False, f"{key} {error_msg}"

    return True, ""


def read_config_value(config: "AppConfig", key: str) -> Any:
    """Read a value from the config at the appropriate location.

    Args:
        config: The AppConfig instance to read from.
        key: The setting key (must exist in SETTING_LOCATIONS).

    Returns:
        The current value of the setting.
    """
    section_name, attr_name, sub_attr = SETTING_LOCATIONS[key]
    section = getattr(config, section_name)

    if sub_attr:
        return getattr(getattr(section, attr_name), sub_attr)
    return getattr(section, attr_name)


def assign_config_value(config: "AppConfig", key: str, value: Any) -> None:
    """Assign a value to the config at the appropriate location.

    Args:
        config: The AppConfig instance to modify.
        key: The setting key.
        value: The value to assign (already parsed and validated).
    """
    section_name, attr_name, sub_attr = SETTING_LOCATIONS[key]
    section = getattr(config, section_name)

    if sub_attr:
        sub_obj = getattr(section, attr_name)
        setattr(sub_obj, sub_attr, value)
    else:
        setattr(section, attr_name, value)


def set_config_value(config: "AppConfig", key: str, value: str) -> tuple[bool, str]:
    """Parse, validate, and set a config value.

    Args:
        config: The AppConfig instance to modify.
        key: The setting key (e.g., "temperature").
        value: The string value to set.

    Returns:
        Tuple of (success, error_message). error_message is empty on success.
    """
    if key not in SETTING_LOCATIONS:
        if key in SETTING_METADATA:
            return False, f"Key '{key}' is not configurable via CLI"
        valid_keys = ", ".join(sorted(SETTING_METADATA.keys()))
        return False, f"Unknown key: {key}. Valid keys: {valid_keys}"

    # Parse
    success, parsed_value, error = parse_config_value(key, value)
    if not success:
        return False, error

    # Validate
    is_valid, error = validate_config_value(key, parsed_value)
    if not is_valid:
        return False, error

    # Assign
    assign_config_value(config, key, parsed_value)
    return True, ""


def get_config_dir() -> Path:
    """Get the cross-platform directory for configuration.

    Returns ~/.coden-retriever/ on all platforms (Linux, Windows, macOS).
    Creates the directory if it doesn't exist.
    """
    home = Path.home()
    config_dir = home / ".coden-retriever"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the path to the active config file.

    Returns the override set by set_config_file() when active, else the
    default ~/.coden-retriever/settings.json.
    """
    if _active_config_file is not None:
        return _active_config_file
    return get_config_dir() / "settings.json"


def set_config_file(path: Optional[Path]) -> None:
    """Override the active config file path for this process.

    Passing None clears the override. Clears the cached config so the next
    get_config() reloads from disk against the new target.
    """
    global _active_config_file, _cached_config
    _active_config_file = path
    _cached_config = None


def has_config_override() -> bool:
    """True iff set_config_file() established an explicit override."""
    return _active_config_file is not None


@dataclass
class ModelConfig:
    """Model and provider configuration.

    Contains two parts:
    - Provider settings: model identifier, base_url, provider_urls
    - Generation settings: temperature, max_tokens, timeout, api_key
    """

    default: str = "ollama:"
    base_url: Optional[str] = None
    provider_urls: dict[str, str] = field(
        default_factory=lambda: {
            "ollama": OLLAMA_DEFAULT_URL,
            "llamacpp": LLAMACPP_DEFAULT_URL,
        }
    )
    generation: GenerationSettings = field(default_factory=GenerationSettings)


# Tools disabled by default.
# Users can enable via /tools in --agent mode.
DEFAULT_DISABLED_TOOLS: list[str] = [
    "debug_server",  # IDE integration tool, less relevant for agents
]


@dataclass
class AgentConfig:
    """Agent behavior configuration."""

    max_steps: int = DEFAULT_MAX_STEPS
    max_retries: int = DEFAULT_MAX_RETRIES
    debug: bool = False
    disabled_tools: list[str] = field(default_factory=lambda: DEFAULT_DISABLED_TOOLS.copy())
    mcp_server_timeout: float = 30.0
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT_S
    tool_instructions: bool = False
    ask_tool_permission: bool = True
    dynamic_tool_filtering: bool = False
    tool_filter_model: Optional[str] = None
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE
    study_prompt_template: str = DEFAULT_STUDY_PROMPT_TEMPLATE
    tool_instructions_template: str = DEFAULT_TOOL_INSTRUCTIONS_TEMPLATE
    study_tool_instructions_template: str = DEFAULT_STUDY_TOOL_INSTRUCTIONS_TEMPLATE
    tool_router_prompt_template: str = DEFAULT_TOOL_ROUTER_PROMPT_TEMPLATE
    starter_questions: list[str] = field(
        default_factory=lambda: DEFAULT_STARTER_QUESTIONS.copy()
    )
    compaction_token_threshold: int = 0


@dataclass
class DaemonConfig:
    """Daemon server configuration."""

    host: str = DEFAULT_DAEMON_HOST
    port: int = DEFAULT_DAEMON_PORT
    daemon_timeout: float = DEFAULT_DAEMON_TIMEOUT
    max_projects: int = DEFAULT_MAX_PROJECTS
    auto_start: bool = True

    @property
    def address(self) -> "DaemonAddress":
        from .daemon.address import DaemonAddress
        return DaemonAddress(host=self.host, port=self.port)


@dataclass
class SearchDefaults:
    """Search defaults configuration (tokens, limits, model path).

    Note: This is distinct from pipeline.SearchConfig which defines
    the parameters for a single search execution.
    """

    default_tokens: int = DEFAULT_TOKEN_BUDGET
    default_limit: int = DEFAULT_SEARCH_RESULT_LIMIT
    semantic_model_path: Optional[str] = None


@dataclass
class MCPConfig:
    """MCP server protocol-level instruction string.

    Surfaced to MCP clients (Claude Code, etc.) at server handshake —
    not to the local agent's LLM.
    """

    full_server_instructions_template: str = DEFAULT_FULL_SERVER_INSTRUCTIONS_TEMPLATE


@dataclass
class AppConfig:
    """Root configuration container."""

    _version: int = CONFIG_VERSION
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    search: SearchDefaults = field(default_factory=SearchDefaults)
    mcp: MCPConfig = field(default_factory=MCPConfig)


def _config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert AppConfig to a JSON-serializable dictionary.

    Uses dataclasses.asdict for automatic serialization, then flattens
    the 'generation' sub-struct into 'model' to match the existing JSON schema.
    Multi-line prompt templates are split into line arrays for readability.
    """
    data = asdict(config)

    # Flatten 'generation' sub-struct into 'model' to match existing JSON schema
    # (generation params are stored at model.temperature, not model.generation.temperature)
    if "model" in data and "generation" in data["model"]:
        gen_data = data["model"].pop("generation")
        data["model"].update(gen_data)

    # Split prompt templates into line arrays for readable JSON
    for section_name, fields in _TEMPLATE_FIELDS.items():
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        for field_name in fields:
            val = section.get(field_name)
            if isinstance(val, str) and not val.startswith(FILE_PATH_PREFIX):
                section[field_name] = val.split("\n")

    return data


def _get_nested_value(data: dict, section: str, attr: str, sub_attr: Optional[str]) -> Any:
    """Safely get a nested value from config dict, handling the generation flattening.

    The JSON schema flattens generation params into model (e.g., model.temperature
    instead of model.generation.temperature), so we need to handle that mapping.

    Args:
        data: The config dictionary.
        section: Top-level section name (model, agent, daemon, search).
        attr: Attribute name within the section.
        sub_attr: Sub-attribute for nested dataclasses (e.g., generation.temperature).

    Returns:
        The value if found, None otherwise.
    """
    section_data = data.get(section, {})
    if not isinstance(section_data, dict):
        return None

    # Handle flattened generation params (stored at model level, not model.generation)
    if section == "model" and sub_attr:
        return section_data.get(sub_attr)

    return section_data.get(attr)


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    """Convert a dictionary to AppConfig using metadata-driven mapping.

    Uses SETTING_LOCATIONS as the source of truth for user-configurable settings,
    with special handling for internal settings not exposed via /config.
    """
    config = AppConfig()

    # Load user-configurable settings via SETTING_LOCATIONS (metadata-driven)
    for key, (section, attr, sub_attr) in SETTING_LOCATIONS.items():
        raw_val = _get_nested_value(data, section, attr, sub_attr)

        if raw_val is not None:
            is_valid, error = validate_config_value(key, raw_val)
            if is_valid:
                assign_config_value(config, key, raw_val)
            else:
                logger.warning(f"Config load error for '{key}': {error}")

    # Load internal settings with special handling (not in SETTING_METADATA)
    if "model" in data and isinstance(data["model"], dict):
        model_data = data["model"]
        if "provider_urls" in model_data and isinstance(model_data["provider_urls"], dict):
            config.model.provider_urls.update(model_data["provider_urls"])

    if "agent" in data and isinstance(data["agent"], dict):
        agent_data = data["agent"]
        # disabled_tools: None means use defaults, empty list means user enabled all
        saved_disabled = agent_data.get("disabled_tools")
        if saved_disabled is None:
            config.agent.disabled_tools = DEFAULT_DISABLED_TOOLS.copy()
        else:
            config.agent.disabled_tools = saved_disabled

        if "mcp_server_timeout" in agent_data:
            config.agent.mcp_server_timeout = agent_data["mcp_server_timeout"]

        if "tool_timeout" in agent_data:
            config.agent.tool_timeout = float(agent_data["tool_timeout"])

        # starter_questions: None/missing means use defaults, list overrides
        saved_questions = agent_data.get("starter_questions")
        if saved_questions is not None and isinstance(saved_questions, list):
            config.agent.starter_questions = saved_questions

    # Prompt templates: stored as line arrays in JSON, joined to strings
    for section_name, fields in _TEMPLATE_FIELDS.items():
        section_data = data.get(section_name)
        if not isinstance(section_data, dict):
            continue
        section_obj = getattr(config, section_name)
        for field_name in fields:
            raw = section_data.get(field_name)
            if isinstance(raw, list):
                setattr(section_obj, field_name, "\n".join(raw))
            elif isinstance(raw, str):
                setattr(section_obj, field_name, raw)

    return config


def _apply_env_overrides(config: AppConfig) -> None:
    """Apply environment variable overrides using SETTING_METADATA as a map.

    Uses the env_var field in SETTING_METADATA to determine which environment
    variables to check, with special handling for internal settings.
    """
    # Apply metadata-driven env overrides
    for key, meta in SETTING_METADATA.items():
        if not meta.env_var:
            continue

        env_val = os.environ.get(meta.env_var)
        if env_val is None:
            continue

        success, parsed_val, error = parse_config_value(key, env_val)
        if not success:
            logger.warning(f"Env override parse failed for {meta.env_var}: {error}")
            continue

        is_valid, v_error = validate_config_value(key, parsed_val)
        if not is_valid:
            logger.warning(f"Env override validation failed for {meta.env_var}: {v_error}")
            continue

        assign_config_value(config, key, parsed_val)

    # Handle internal env overrides not in SETTING_METADATA
    if env_mcp_timeout := os.environ.get("CODEN_RETRIEVER_MCP_TIMEOUT"):
        try:
            config.agent.mcp_server_timeout = float(env_mcp_timeout)
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_MCP_TIMEOUT: {env_mcp_timeout}")

    if env_tool_timeout := os.environ.get("CODEN_RETRIEVER_TOOL_TIMEOUT"):
        try:
            config.agent.tool_timeout = float(env_tool_timeout)
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_TOOL_TIMEOUT: {env_tool_timeout}")


def load_config() -> AppConfig:
    """Load configuration from disk with env override support.

    Priority: environment variables > config file > hardcoded defaults

    When an explicit override is active (set_config_file was called),
    a missing or unreadable file raises rather than silently falling back
    to defaults — the user named that file, so honoring it is load-bearing.

    Returns:
        AppConfig object with all settings.

    Raises:
        FileNotFoundError: override active and file missing.
        OSError / json.JSONDecodeError: override active and file unreadable.
    """
    config_file = get_config_file()
    strict = has_config_override()

    if not config_file.exists():
        if strict:
            raise FileNotFoundError(
                f"Config file not found: {config_file}. "
                f"Create it with: coden config new {config_file}"
            )
        config = AppConfig()
        _apply_env_overrides(config)
        return config

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = _dict_to_config(data)

        _apply_env_overrides(config)

        return config

    except (json.JSONDecodeError, OSError) as e:
        if strict:
            raise
        logger.warning(f"Could not load config: {e}, using defaults")
        config = AppConfig()
        _apply_env_overrides(config)
        return config


def save_config(config: AppConfig, path: Optional[Path] = None) -> bool:
    """Save configuration to disk.

    Args:
        config: The configuration to save.
        path: Optional explicit target path. When None, writes to the active
            config file (default or override). Used by `coden config new` to
            seed a fresh file without disturbing the active override.

    Returns:
        True if save was successful, False otherwise.
    """
    target = path if path is not None else get_config_file()

    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(_config_to_dict(config), f, indent=2)
        return True
    except OSError as e:
        logger.warning(f"Could not save config: {e}")
        return False


def get_default_config() -> AppConfig:
    """Get a fresh default configuration (without loading from disk)."""
    return AppConfig()


def reset_config() -> bool:
    """Reset configuration to defaults by removing the default config file.

    Always targets ~/.coden-retriever/settings.json — never the override
    set by set_config_file(). A future in-process /reset slash command
    must not be able to delete a user's custom config.

    Returns:
        True if reset was successful or file didn't exist, False otherwise.
    """
    config_file = get_config_dir() / "settings.json"

    if not config_file.exists():
        return True

    try:
        config_file.unlink()
        return True
    except OSError as e:
        logger.warning(f"Could not reset config: {e}")
        return False


# Singleton instance for caching
_cached_config: Optional[AppConfig] = None

# Override target for `coden -a --config <path>`. When set, get_config_file()
# returns this instead of the default ~/.coden-retriever/settings.json, and
# load_config() runs in strict mode (no silent fallback to defaults).
_active_config_file: Optional[Path] = None


def get_config() -> AppConfig:
    """Get the cached configuration (loads once, then returns cached).

    Use load_config() if you need to force a fresh load.
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reload_config() -> AppConfig:
    """Force reload configuration from disk, updating the cache."""
    global _cached_config
    _cached_config = load_config()
    return _cached_config


def daemon_enabled(args: Optional[argparse.Namespace] = None) -> bool:
    """Return True if the daemon should be attempted for this invocation.

    Precedence: `args.no_daemon` (CLI flag) > env CODEN_RETRIEVER_DAEMON_AUTO_START
    > config daemon.auto_start > default (True).

    CLI handlers pass the argparse Namespace; MCP callers omit it.
    The env var is folded into `get_config()` via `_apply_env_overrides`;
    this helper does not re-read it.
    """
    if args is not None and args.no_daemon:
        return False
    return get_config().daemon.auto_start


def get_semantic_model_path() -> str:
    """Get the semantic model path from config or use the default.

    This is a shared utility for clone detection and other semantic features.
    Returns the configured model path or falls back to the bundled default.

    Returns:
        Path to the semantic model directory.
    """
    default_model_path = str(
        Path(__file__).parent / "models" / "embeddings" / "minilm_onnx"
    )
    try:
        config = get_config()
        return config.search.semantic_model_path or default_model_path
    except Exception as e:
        logger.debug(f"Config load failed, using default model path: {e}")
        return default_model_path


def get_default_token_budget() -> int:
    try:
        return get_config().search.default_tokens
    except Exception as e:
        logger.debug(f"Config load failed, using DEFAULT_TOKEN_BUDGET: {e}")
        return DEFAULT_TOKEN_BUDGET
