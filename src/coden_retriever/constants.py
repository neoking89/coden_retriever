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
# Architecture Analysis Thresholds (MacCormack et al., 2006)
# =============================================================================
# From "Exploring the Structure of Complex Software Designs"
# The study analyzed Linux kernel vs Mozilla codebase:
# - Linux (well-architected): PC ~10% - modular design with clear boundaries
# - Mozilla (pre-refactor): PC ~43% - high coupling, difficult to maintain

# 10%: Excellent - matches well-designed systems like Linux
PC_THRESHOLD_GOOD = 0.10
# 25%: Moderate - the midpoint indicating coupling warrants monitoring
PC_THRESHOLD_WARNING = 0.25
# 43%: Critical - matches pre-refactor Mozilla, needs action
PC_THRESHOLD_CRITICAL = 0.43

# =============================================================================
# Dead Code Detection Thresholds
# =============================================================================
# Confidence thresholds for dead code classification

# Skip dunder methods (__init__, __call__, etc.) - they are invoked by runtime
# These methods are called implicitly by Python/language constructs:
# - __init__ via ClassName(), __call__ via instance(), __enter__/__exit__ via 'with'
# - __getattr__ via attribute access, __iter__ via for loops, etc.
# Matching Vulture's approach: dunder methods are NEVER flagged as dead code.
DEAD_CODE_SKIP_DUNDER_METHODS = True

# Minimum confidence to include in results (default filter)
DEAD_CODE_MIN_CONFIDENCE = 0.30

# High confidence threshold - likely truly dead
DEAD_CODE_CONFIDENCE_HIGH = 0.80

# Medium confidence threshold - investigate further
DEAD_CODE_CONFIDENCE_MEDIUM = 0.50

# =============================================================================
# Dead Code Confidence Scoring Constants
# =============================================================================
# Values tuned for 90%+ accuracy based on empirical testing

# Base confidence: function with no callers is likely dead, but not certain
# 85% starting point leaves room for framework hooks, entry points, etc.
DEAD_CODE_BASE_CONFIDENCE = 0.85

# Private functions (_name) cannot be called externally, so more likely dead
DEAD_CODE_PRIVATE_BOOST = 0.10

# Decorated functions are almost always called externally by frameworks
# (e.g., @property, @mcp.tool, @kb.add, @registry.register)
DEAD_CODE_DECORATOR_PENALTY = 0.80

# Public module-level functions may be library exports or API endpoints
DEAD_CODE_PUBLIC_MODULE_PENALTY = 0.15

# Class methods may be called via instance (polymorphism, inheritance)
DEAD_CODE_METHOD_PENALTY = 0.20

# Entry point pattern: public module function that calls others but has no callers
# Structural detection: if a function has NO incoming calls but HAS outgoing calls,
# it's likely an entry point (main, run, handler) called externally by runtime/CLI
DEAD_CODE_ENTRY_POINT_PENALTY = 0.50

# =============================================================================
# Flag Insertion Constants
# =============================================================================
# Used by flag_code() to set default analysis limits

# Minimum function size to consider for flagging (avoids trivial getters/setters)
FLAG_MIN_LINES = 3

# Default limit for flag analysis results (prevents overwhelming output)
FLAG_ANALYSIS_LIMIT = 100

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
