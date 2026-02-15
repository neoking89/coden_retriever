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

# === CLI Daemon Constants ===
# Polling interval when waiting for daemon startup (seconds)
DAEMON_POLL_INTERVAL_SECONDS = 0.1
# Maximum polling attempts before giving up on daemon start
DAEMON_MAX_POLL_ATTEMPTS = 20
# Delay between stop and start operations during restart (seconds)
DAEMON_RESTART_DELAY_SECONDS = 0.5
# Timeout for cache clear operations (seconds)
DAEMON_CACHE_TIMEOUT_SECONDS = 5.0
# Default port for MCP HTTP server
MCP_DEFAULT_HTTP_PORT = 8000
# Output section separator width for formatted output
OUTPUT_SEPARATOR_WIDTH = 60
# Statistics separator width for detailed analysis output
STATS_SEPARATOR_WIDTH = 80

# === CLI Flag Constants ===
# Syntactic clone detection: line-level Jaccard similarity threshold
DEFAULT_SYNTACTIC_LINE_THRESHOLD = 0.70
# Syntactic clone detection: minimum percentage of matching lines required
DEFAULT_SYNTACTIC_FUNC_THRESHOLD = 0.50
# Time conversion factor for millisecond display
MILLISECONDS_PER_SECOND = 1000
# Percentage conversion factor for display formatting
PERCENT = 100

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
# Tramp Data Detection Thresholds
# =============================================================================

# Default minimum functions a param must appear in to be flagged
# 3 is the minimum meaningful threshold: appearing in 1-2 functions is normal
TRAMP_DATA_DEFAULT_MIN_OCCURRENCES = 3

# Minimum parameters in a group to report (a single param is not a group)
# 2 is the minimum: pairs like (host, port) are the smallest meaningful group
TRAMP_DATA_MIN_GROUP_SIZE = 2

# 20+ functions: param is passed everywhere -- strong refactoring signal
TRAMP_DATA_TIER_HIGH = 20
# 10+ functions: param crosses many boundaries -- worth investigating
TRAMP_DATA_TIER_MODERATE = 10
# 5+ functions: mild tramp data pattern -- may be intentional
TRAMP_DATA_TIER_LOW = 5

# Max functions to display per parameter in CLI card view
# 3 functions shown vertically per card; gives a clear sample without overwhelming output
TRAMP_DATA_MAX_FUNCTIONS_DISPLAY = 3

# Default result limit for MCP and CLI
# 50 results provide comprehensive overview without overwhelming LLM context windows or terminal output
TRAMP_DATA_DEFAULT_RESULT_LIMIT = 50

# Max results allowed in MCP context
# 500 is upper bound to prevent memory issues and token budget exhaustion in LLM interactions
TRAMP_DATA_MAX_RESULTS = 500

# Default result limit for search and find operations
# 20 balances relevance density with comprehensive coverage for most queries
DEFAULT_SEARCH_RESULT_LIMIT = 20

# =============================================================================
# Sensitive Value Detection Thresholds
# =============================================================================

# Best F1 (94.4%) from 5-fold stratified CV on 280-sample golden set
SENSITIVE_VALUE_DEFAULT_THRESHOLD = 0.35

# Default replacement text when redacting detected secrets
SENSITIVE_VALUE_DEFAULT_REPLACE = "***REDACTED***"

# Color tier boundaries for CLI display (confidence-based)
SENSITIVE_VALUE_TIER_HIGH = 0.80
SENSITIVE_VALUE_TIER_MODERATE = 0.50

# CLI result limits matching tramp data pattern
SENSITIVE_VALUE_DEFAULT_LIMIT = 50
SENSITIVE_VALUE_MAX_RESULTS = 500

# String length bounds for analysis (skip trivially short/long strings)
SENSITIVE_VALUE_MIN_STRING_LENGTH = 8
SENSITIVE_VALUE_MAX_STRING_LENGTH = 500

# Minimum lengths for hex/base64 heuristic checks
# 16 chars = 8 bytes hex-encoded, common for hashes/tokens
SENSITIVE_VALUE_MIN_HEX_LENGTH = 16
# 8 chars minimum to avoid false positives on short encoded strings
SENSITIVE_VALUE_MIN_BASE64_LENGTH = 8

# AST traversal depth for finding variable assignments
# 5 levels covers typical nesting: assignment -> expr_stmt -> block -> function -> class
SENSITIVE_VALUE_AST_MAX_DEPTH = 5

# String value preview truncation lengths for display
# 40 chars shows enough context without wrapping in most terminals
SENSITIVE_VALUE_PREVIEW_LENGTH = 40
# 28 chars fits the formatter table column width
SENSITIVE_VALUE_TABLE_DISPLAY_LENGTH = 28

# Base64 padding characters for validation
# Standard base64 uses "=" for padding to align to 4-byte boundaries
SENSITIVE_VALUE_BASE64_PADDING = "=="

# Formatter table dimensions (matching hotspots style for consistency)
# 110 chars matches the standard hotspot/dead-code table width
SENSITIVE_VALUE_TABLE_WIDTH = 110
# 30 chars for value preview column allows readable context
SENSITIVE_VALUE_VALUE_COLUMN_WIDTH = 30

# SVM-RBF classifier hyperparameters
# 1000 iterations ensures convergence on 99-sample training set
SENSITIVE_VALUE_CLASSIFIER_MAX_ITER = 1000
# C=5.0 optimal from grid search: Val-F1=0.957, perfect precision
SENSITIVE_VALUE_CLASSIFIER_REGULARIZATION = 5.0

# Feature extraction thresholds
# 2 distinct special chars required to distinguish passwords from CSS/format strings
SENSITIVE_VALUE_MIN_PASSWORD_SPECIAL_CHARS = 2
# 5 chars minimum to avoid incorrect plural stripping of short words like "bus", "gas"
SENSITIVE_VALUE_MIN_PLURAL_STRIP_LENGTH = 5
# 0.8 (80%) printable ratio threshold for base64 decoded text to be considered readable
# Above this threshold indicates safe config data, below indicates encrypted/random bytes
SENSITIVE_VALUE_BASE64_READABLE_RATIO = 0.8
# 2 minimum alphabetic characters required for all-caps phrase detection
# Avoids false positives on single-letter abbreviations or punctuation-only strings
SENSITIVE_VALUE_MIN_ALPHA_CHARS_FOR_CAPS = 2

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
