"""
Constant definitions for coden-retriever.

Contains:
- Network constants (URLs, ports, hosts, timeouts)
- Invariant data sets used for filtering and classification

These are separated from config.py which contains tuning parameters.
"""

# =============================================================================
# Network Constants - Centralized URLs, ports, hosts, and timeouts
# =============================================================================

# Provider URLs (OpenAI-compatible endpoints)
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"
LLAMACPP_DEFAULT_URL = "http://localhost:8080/v1"

# Provider default API keys (for local servers that don't need real keys)
OLLAMA_DEFAULT_API_KEY = "ollama"
LLAMACPP_DEFAULT_API_KEY = "not-needed"

# Daemon server defaults
DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 19847
DEFAULT_DAEMON_TIMEOUT = 30.0
DEFAULT_CLIENT_TIMEOUT = 5.0
DEFAULT_HEAVY_ANALYSIS_TIMEOUT = 60.0  # For clone detection, propagation, etc.
DEFAULT_MAX_PROJECTS = 5

# Debug server defaults (debugpy)
DEFAULT_DEBUG_PORT = 5678

# Agent defaults
DEFAULT_MAX_RETRIES: int = 5

# =============================================================================
# Filtering and Classification Constants
# =============================================================================

# Ambiguous method names that should ONLY create edges when qualified lookup succeeds.
# These are common method names (like dict.get, list.append) that would create
# false positive edges to all 100+ methods with the same name if resolved by name only.
# When receiver is unknown, skip edge creation entirely for these names.
AMBIGUOUS_METHOD_NAMES: set[str] = {
    # Collection methods
    "get", "set", "put", "add", "remove", "pop", "push", "clear",
    "append", "extend", "insert", "update", "keys", "values", "items",
    # Lifecycle/initialization
    "__init__", "__new__", "__del__", "__enter__", "__exit__",
    # Common interface methods
    "read", "write", "close", "open", "flush", "seek",
    "send", "receive", "connect", "disconnect",
    "start", "stop", "run", "execute", "call",
    "load", "save", "dump", "parse",
    # Common property accessors
    "name", "value", "data", "result", "status", "type", "id",
}
