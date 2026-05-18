"""Constants used by the agent core: provider URLs, retry budgets, and the
error-detection keywords for the malformed-tool-call retry path.
"""


# Provider URLs (OpenAI-compatible endpoints)
OLLAMA_DEFAULT_URL: str = "http://localhost:11434/v1"
LLAMACPP_DEFAULT_URL: str = "http://localhost:8080/v1"

# llama-cpp-server default API key (no native LlamaCppProvider; relies on
# OpenAI-compatible endpoint that accepts a placeholder key).
LLAMACPP_DEFAULT_API_KEY: str = "not-needed"

# 30 tool calls: raised from 15 after observing that multi-file refactoring tasks
# (read N files, edit N files, verify) regularly hit the old limit mid-task.
DEFAULT_MAX_STEPS: int = 30

# 5 retries: enough to recover from transient provider 400s (malformed tool calls)
# while bounding runaway retry loops. Configurable via /config set max_retries.
DEFAULT_MAX_RETRIES: int = 5

# HTTP 400 Bad Request -- the status code providers return for malformed tool calls.
HTTP_BAD_REQUEST: int = 400

# The error type string providers use for invalid request payloads.
INVALID_REQUEST_ERROR_TYPE: str = "invalid_request_error"

# Prompt hint appended on retry to steer the model toward single tool calls.
SEQUENTIAL_TOOL_HINT: str = (
    "\n\nIMPORTANT: Call exactly ONE tool at a time. "
    "Do not combine multiple tool calls into a single request."
)

# Keywords that must appear in the error message body to confirm a tool-call 400,
# preventing over-broad retries on unrelated 400s (e.g. invalid model name, context overflow).
TOOL_CALL_ERROR_KEYWORDS: frozenset[str] = frozenset({"tool", "function", "tool_call"})

# H:M:S format for wall-clock timestamps shown alongside token usage.
WALL_CLOCK_FORMAT: str = "%H:%M:%S"
