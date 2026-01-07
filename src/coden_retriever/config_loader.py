"""Unified configuration loader for coden-retriever.

Provides a single source of truth for all user-configurable settings.
Priority: CLI args > config file > environment variables > hardcoded defaults

Configuration is stored at ~/.coden-retriever/settings.json
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Literal

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


# Single source of truth for all user-facing setting metadata
SETTING_METADATA: dict[str, SettingMeta] = {
    "model": SettingMeta(
        "model",
        "LLM model identifier",
        "ollama:name, llamacpp:name, openai:name (official API), or name+base_url",
        "str",
    ),
    "base_url": SettingMeta(
        "base_url",
        "API endpoint URL",
        "OpenAI-compatible endpoint (auto-detected for ollama/llamacpp)",
        "str",
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
        "Retry attempts for failed tool calls",
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
        "Filter tools by query semantics",
        "Filter tools based on query semantics (show relevant tools)",
        "bool",
    ),
    "tool_filter_threshold": SettingMeta(
        "tool_filter_threshold",
        "Tool filter similarity threshold",
        "Threshold (0-1) for dynamic_tool_filtering. Lower=more tools, Higher=fewer tools",
        "float",
    ),
    "temperature": SettingMeta(
        "temperature",
        "Model temperature (0-2)",
        "Controls randomness (0.0=deterministic, 1.0+=creative)",
        "float",
    ),
    "max_tokens": SettingMeta(
        "max_tokens",
        "Max response tokens",
        "Maximum tokens in response (empty=model default)",
        "int",
    ),
    "timeout": SettingMeta(
        "timeout",
        "Request timeout (seconds)",
        "Timeout for model API requests (default: 120)",
        "float",
    ),
    "api_key": SettingMeta(
        "api_key",
        "API key override",
        "Custom API key (overrides OPENAI_API_KEY env var for custom endpoints)",
        "str",
    ),
}


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
    """Get the path to the settings.json file."""
    return get_config_dir() / "settings.json"


@dataclass
class GenerationSettings:
    """Model generation parameters passed to pydantic-ai's ModelSettings.

    These settings control LLM behavior and are separate from provider config.

    Attributes:
        temperature: Controls randomness (0.0=deterministic, 1.0+=creative).
        max_tokens: Maximum response length (None=model default).
        timeout: Request timeout in seconds.
        api_key: API key override (used at provider level, not ModelSettings).
    """

    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    api_key: Optional[str] = None


def get_model_settings(generation: GenerationSettings) -> dict:
    """Convert GenerationSettings to pydantic-ai ModelSettings dict.

    This creates a dictionary compatible with pydantic-ai's ModelSettings TypedDict.
    Only includes non-None values to avoid overriding defaults.

    Args:
        generation: GenerationSettings instance.

    Returns:
        Dictionary suitable for Agent's model_settings parameter.
    """
    settings: dict = {
        "temperature": generation.temperature,
        "timeout": generation.timeout,
    }
    if generation.max_tokens is not None:
        settings["max_tokens"] = generation.max_tokens
    return settings


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
            "ollama": "http://localhost:11434/v1",
            "llamacpp": "http://localhost:8080/v1",
        }
    )
    generation: GenerationSettings = field(default_factory=GenerationSettings)


@dataclass
class AgentConfig:
    """Agent behavior configuration."""

    max_steps: int = 15
    max_retries: int = 3
    debug: bool = False
    disabled_tools: list[str] = field(default_factory=lambda: DEFAULT_DISABLED_TOOLS.copy())
    mcp_server_timeout: float = 30.0
    tool_instructions: bool = False
    ask_tool_permission: bool = True  
    dynamic_tool_filtering: bool = False
    tool_filter_threshold: float = 0.5


# Tools disabled by default.
# Users can enable via /tools in --agent mode.
DEFAULT_DISABLED_TOOLS = [
    "debug_server",  # IDE integration tool, less relevant for agents
]


@dataclass
class DaemonConfig:
    """Daemon server configuration."""

    host: str = "127.0.0.1"
    port: int = 19847
    socket_timeout: float = 30.0
    max_projects: int = 5


@dataclass
class SearchDefaults:
    """Search defaults configuration (tokens, limits, model path).

    Note: This is distinct from pipeline.SearchConfig which defines
    the parameters for a single search execution.
    """

    default_tokens: int = 4000
    default_limit: int = 20
    semantic_model_path: Optional[str] = None


@dataclass
class AppConfig:
    """Root configuration container."""

    _version: int = CONFIG_VERSION
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    search: SearchDefaults = field(default_factory=SearchDefaults)


def _config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert AppConfig to a JSON-serializable dictionary."""
    return {
        "_version": config._version,
        "model": {
            "default": config.model.default,
            "base_url": config.model.base_url,
            "provider_urls": config.model.provider_urls,
            "temperature": config.model.generation.temperature,
            "max_tokens": config.model.generation.max_tokens,
            "timeout": config.model.generation.timeout,
            "api_key": config.model.generation.api_key,
        },
        "agent": {
            "max_steps": config.agent.max_steps,
            "max_retries": config.agent.max_retries,
            "debug": config.agent.debug,
            "disabled_tools": config.agent.disabled_tools,
            "mcp_server_timeout": config.agent.mcp_server_timeout,
            "tool_instructions": config.agent.tool_instructions,
            "ask_tool_permission": config.agent.ask_tool_permission,
            "dynamic_tool_filtering": config.agent.dynamic_tool_filtering,
            "tool_filter_threshold": config.agent.tool_filter_threshold,
        },
        "daemon": {
            "host": config.daemon.host,
            "port": config.daemon.port,
            "socket_timeout": config.daemon.socket_timeout,
            "max_projects": config.daemon.max_projects,
        },
        "search": {
            "default_tokens": config.search.default_tokens,
            "default_limit": config.search.default_limit,
            "semantic_model_path": config.search.semantic_model_path,
        },
    }


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    """Convert a dictionary to AppConfig."""
    config = AppConfig()

    # Parse model section
    if "model" in data and isinstance(data["model"], dict):
        model_data = data["model"]
        config.model.default = model_data.get("default", config.model.default)
        config.model.base_url = model_data.get("base_url", config.model.base_url)
        if "provider_urls" in model_data and isinstance(model_data["provider_urls"], dict):
            config.model.provider_urls.update(model_data["provider_urls"])
        # Parse generation parameters with validation
        gen = config.model.generation
        raw_temp = model_data.get("temperature", gen.temperature)
        if isinstance(raw_temp, (int, float)) and 0.0 <= raw_temp <= 2.0:
            gen.temperature = float(raw_temp)
        else:
            logger.warning(f"Invalid temperature '{raw_temp}', using default {gen.temperature}")
        raw_max_tokens = model_data.get("max_tokens", gen.max_tokens)
        if raw_max_tokens is None or (isinstance(raw_max_tokens, int) and raw_max_tokens > 0):
            gen.max_tokens = raw_max_tokens
        else:
            logger.warning(f"Invalid max_tokens '{raw_max_tokens}', using default")
        raw_timeout = model_data.get("timeout", gen.timeout)
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            gen.timeout = float(raw_timeout)
        else:
            logger.warning(f"Invalid timeout '{raw_timeout}', using default {gen.timeout}")
        gen.api_key = model_data.get("api_key", gen.api_key)

    # Parse agent section
    if "agent" in data and isinstance(data["agent"], dict):
        agent_data = data["agent"]
        config.agent.max_steps = agent_data.get("max_steps", config.agent.max_steps)
        config.agent.max_retries = agent_data.get("max_retries", config.agent.max_retries)
        config.agent.debug = agent_data.get("debug", config.agent.debug)
        # Only use defaults if disabled_tools key doesn't exist (old config format)
        # An empty list means user explicitly enabled all tools - respect that
        saved_disabled = agent_data.get("disabled_tools")
        if saved_disabled is None:
            config.agent.disabled_tools = DEFAULT_DISABLED_TOOLS.copy()
        else:
            config.agent.disabled_tools = saved_disabled
        config.agent.mcp_server_timeout = agent_data.get("mcp_server_timeout", config.agent.mcp_server_timeout)
        config.agent.tool_instructions = agent_data.get("tool_instructions", config.agent.tool_instructions)
        config.agent.ask_tool_permission = agent_data.get("ask_tool_permission", config.agent.ask_tool_permission)
        config.agent.dynamic_tool_filtering = agent_data.get("dynamic_tool_filtering", config.agent.dynamic_tool_filtering)
        # Validate and load tool_filter_threshold with bounds checking
        raw_threshold = agent_data.get("tool_filter_threshold", config.agent.tool_filter_threshold)
        if isinstance(raw_threshold, (int, float)) and 0.0 <= raw_threshold <= 1.0:
            config.agent.tool_filter_threshold = float(raw_threshold)
        else:
            logger.warning(f"Invalid tool_filter_threshold '{raw_threshold}', using default {config.agent.tool_filter_threshold}")

    # Parse daemon section
    if "daemon" in data and isinstance(data["daemon"], dict):
        daemon_data = data["daemon"]
        config.daemon.host = daemon_data.get("host", config.daemon.host)
        config.daemon.port = daemon_data.get("port", config.daemon.port)
        config.daemon.socket_timeout = daemon_data.get("socket_timeout", config.daemon.socket_timeout)
        config.daemon.max_projects = daemon_data.get("max_projects", config.daemon.max_projects)

    # Parse search section
    if "search" in data and isinstance(data["search"], dict):
        search_data = data["search"]
        config.search.default_tokens = search_data.get("default_tokens", config.search.default_tokens)
        config.search.default_limit = search_data.get("default_limit", config.search.default_limit)
        config.search.semantic_model_path = search_data.get(
            "semantic_model_path", config.search.semantic_model_path
        )

    return config


def _apply_env_overrides(config: AppConfig) -> None:
    """Apply environment variable overrides to config (in-place)."""
    # Model overrides
    if env_model := os.environ.get("CODEN_RETRIEVER_MODEL"):
        config.model.default = env_model

    if env_base_url := os.environ.get("CODEN_RETRIEVER_BASE_URL"):
        config.model.base_url = env_base_url

    # Daemon overrides
    if env_port := os.environ.get("CODEN_RETRIEVER_DAEMON_PORT"):
        try:
            config.daemon.port = int(env_port)
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_DAEMON_PORT: {env_port}")

    if env_host := os.environ.get("CODEN_RETRIEVER_DAEMON_HOST"):
        config.daemon.host = env_host

    # Agent overrides
    if env_mcp_timeout := os.environ.get("CODEN_RETRIEVER_MCP_TIMEOUT"):
        try:
            config.agent.mcp_server_timeout = float(env_mcp_timeout)
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_MCP_TIMEOUT: {env_mcp_timeout}")

    # Search overrides
    if env_model_path := os.environ.get("CODEN_RETRIEVER_MODEL_PATH"):
        config.search.semantic_model_path = env_model_path

    # Tool filter threshold override
    if env_threshold := os.environ.get("CODEN_RETRIEVER_TOOL_FILTER_THRESHOLD"):
        try:
            threshold = float(env_threshold)
            if 0.0 <= threshold <= 1.0:
                config.agent.tool_filter_threshold = threshold
            else:
                logger.warning(f"Invalid CODEN_RETRIEVER_TOOL_FILTER_THRESHOLD: {env_threshold} (must be 0.0-1.0)")
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_TOOL_FILTER_THRESHOLD: {env_threshold}")

    # Model generation parameter overrides
    gen = config.model.generation
    if env_temperature := os.environ.get("CODEN_RETRIEVER_TEMPERATURE"):
        try:
            temp = float(env_temperature)
            if 0.0 <= temp <= 2.0:
                gen.temperature = temp
            else:
                logger.warning(f"Invalid CODEN_RETRIEVER_TEMPERATURE: {env_temperature} (must be 0.0-2.0)")
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_TEMPERATURE: {env_temperature}")

    if env_max_tokens := os.environ.get("CODEN_RETRIEVER_MAX_TOKENS"):
        try:
            max_tokens = int(env_max_tokens)
            if max_tokens > 0:
                gen.max_tokens = max_tokens
            else:
                logger.warning(f"Invalid CODEN_RETRIEVER_MAX_TOKENS: {env_max_tokens} (must be > 0)")
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_MAX_TOKENS: {env_max_tokens}")

    if env_timeout := os.environ.get("CODEN_RETRIEVER_TIMEOUT"):
        try:
            timeout = float(env_timeout)
            if timeout > 0:
                gen.timeout = timeout
            else:
                logger.warning(f"Invalid CODEN_RETRIEVER_TIMEOUT: {env_timeout} (must be > 0)")
        except ValueError:
            logger.warning(f"Invalid CODEN_RETRIEVER_TIMEOUT: {env_timeout}")

    if env_api_key := os.environ.get("CODEN_RETRIEVER_API_KEY"):
        gen.api_key = env_api_key


def load_config() -> AppConfig:
    """Load configuration from disk with env override support.

    Priority: environment variables > config file > hardcoded defaults

    Returns:
        AppConfig object with all settings.
    """
    config_file = get_config_file()

    if not config_file.exists():
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
        logger.warning(f"Could not load config: {e}, using defaults")
        config = AppConfig()
        _apply_env_overrides(config)
        return config


def save_config(config: AppConfig) -> bool:
    """Save configuration to disk.

    Args:
        config: The configuration to save.

    Returns:
        True if save was successful, False otherwise.
    """
    config_file = get_config_file()

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(_config_to_dict(config), f, indent=2)
        return True
    except OSError as e:
        logger.warning(f"Could not save config: {e}")
        return False


def get_default_config() -> AppConfig:
    """Get a fresh default configuration (without loading from disk)."""
    return AppConfig()


def reset_config() -> bool:
    """Reset configuration to defaults by removing the config file.

    Returns:
        True if reset was successful or file didn't exist, False otherwise.
    """
    config_file = get_config_file()

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
