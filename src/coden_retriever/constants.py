"""
Constant definitions for coden-retriever.

Contains:
- Network constants (URLs, ports, hosts, timeouts)
- Invariant data sets used for filtering and classification

These are separated from config.py which contains tuning parameters.
"""

from enum import Enum

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

# =============================================================================
# DAP Transport Tuning
# =============================================================================

# 4096: DAP messages are almost always <1KB; one recv call drains a whole
# response in the common case and avoids over-fragmenting the header parser.
DAP_READ_CHUNK_BYTES: int = 4096

# 0.5s: socket.recv timeout inside the blocking reader loop — keeps the thread
# responsive to the `is_running` stop predicate without burning CPU on polling.
DAP_READ_TIMEOUT_SECONDS: float = 0.5

# 0.2s: debugpy needs a moment after `listen()` before a connect succeeds;
# short enough that the overall connect-timeout (default 10s) still has budget
# for ~50 retries if the adapter is slow to come up.
DAP_CONNECT_RETRY_SLEEP_SECONDS: float = 0.2

# 2.0s: terminate → wait window before escalating to kill(). debugpy usually
# exits within milliseconds of SIGTERM; 2s covers Windows edge cases where the
# signal is emulated via CreateRemoteThread and can be slower.
DAP_PROCESS_TERMINATE_TIMEOUT: float = 2.0

# 1.0s: stderr drain-thread join timeout on session stop. Long enough for the
# thread to flush its last line, short enough not to stall session cleanup.
DAP_STDERR_DRAIN_JOIN_TIMEOUT: float = 1.0

# 10.0s: `debug_server` helper-process ready timeout. The helper prints a
# LISTENING line after binding its listener; 10s covers cold-start import
# cost on the first debugpy invocation of a session.
DAP_HELPER_READY_TIMEOUT: float = 10.0

# 5.0s: helper-process graceful-shutdown wait. After this the helper is
# SIGKILLed. Chosen to bound stop-session latency on a wedged helper.
DAP_HELPER_STOP_TIMEOUT: float = 5.0

# 0.5s: post-spawn pause before the first connect attempt to a launched
# socket-transport adapter. debugpy needs a moment to bind its listener after
# Popen returns, and attempting the connect immediately fans out into the
# DAP_CONNECT_RETRY_SLEEP_SECONDS retry loop for no reason.
DAP_ADAPTER_READY_WAIT_SECONDS: float = 0.5

# 3.0s: launch-request send-timeout (NOT the overall launch timeout).
# Several adapters defer the launch response until configurationDone is
# received — debugpy treats launch as "in handling" across the handshake
# and kotlin-debug-adapter outright blocks its own launch handler on it.
# DAPClient catches the TimeoutError so the deferred response routes
# through the reader later. The only reason we await at all is to keep
# the `seq` counter monotonic against `send_message`, so a short timeout
# is right here — 3s is plenty for every non-deferring adapter to ack.
LAUNCH_REQUEST_SHORT_TIMEOUT_SECONDS: float = 3.0

# Agent defaults
# 30 tool calls: raised from 15 after observing that multi-file refactoring tasks
# (read N files, edit N files, verify) regularly hit the old limit mid-task.
DEFAULT_MAX_STEPS: int = 30
# 5 retries: enough to recover from transient provider 400s (malformed tool calls)
# while bounding runaway retry loops. Configurable via /config set max_retries.
DEFAULT_MAX_RETRIES: int = 5

# H:M:S format for wall-clock timestamps shown alongside token usage
WALL_CLOCK_FORMAT: str = "%H:%M:%S"

# WHY "file:": allows referencing external template files in settings.json
# instead of inlining large prompt templates as JSON strings
FILE_PATH_PREFIX: str = "file:"

# Default system prompts — source of truth, shown in settings.json as line arrays.
# Follows the same pattern as DEFAULT_STARTER_QUESTIONS.
DEFAULT_SYSTEM_PROMPT_TEMPLATE: str = """
<role>
You are an Expert Code Analysis Agent. You are precise, methodical, and never hallucinate.
Your purpose is to help users understand, explore, and debug codebases using the available tools.
When you don't know something, you use tools to find out - you never invent code or file contents.
</role>

<environment>
Current Working Directory: {root_directory}
All tool calls MUST use absolute paths constructed from this directory.
Example: If a file is at "src/main.py", the absolute path is "{root_directory}/src/main.py"
</environment>

<constraints>
1. **Absolute Paths**: Construct absolute paths from the working directory for all tool calls.
2. **No Hallucination**: Never invent code or paths. If a tool fails, try different terms.
3. **Cite Sources**: Include file:line when referencing code.
4. **Token Budget**: Read specific line ranges, not full files.
5. **No Preamble**: Skip filler. Start directly with findings.
6. **Secrets**: Never output API keys, passwords, or credentials.
</constraints>

<reasoning_process>
Follow the ReAct loop for every query:
1. THOUGHT: Analyze what the user wants. Identify their intent category (exploration, lookup, reading, debugging, git analysis).
2. PLAN: Select the appropriate tool sequence based on intent.
3. ACTION: Execute ONE tool call. Wait for result before next tool.
4. OBSERVATION: Process result. If incomplete, return to THOUGHT for next tool.
5. ANSWER: When you have sufficient information, synthesize findings into a clear response with code citations.
</reasoning_process>

<directory_structure>
{directory_tree}
</directory_structure>
"""

DEFAULT_STUDY_PROMPT_TEMPLATE: str = """
<role>
You are an Interactive Coding Tutor. Teach through discovery, not lectures.
Guide users to understand code by asking focused questions with code citations.
</role>

<environment>
Working Directory: {root_directory}
Topic: {study_topic}
Use absolute paths for all tool calls.
</environment>

<experience_levels>
| Level | Signals | Approach |
|-------|---------|----------|
| EXPLORER | "first time", "new to this" | Define terms, analogies, slow pace, architecture first |
| LEARNER | "looked around", "still learning" | Connect concepts, medium pace, reinforce connections |
| PRACTITIONER | "know the basics", "want depth" | Jump to specifics, trade-offs, "why" questions |
| EXPERT | "I maintain this", "contributor" | Precise lookups, skip basics, edge cases |
</experience_levels>

<teaching_flow>
**Session start:** Greet warmly (one sentence), ask their experience level and goal. NO tool calls.

**Responding to answers:**
- Correct: Brief acknowledgment, go deeper, slightly harder follow-up
- Partial: Acknowledge correct part, guide to missing piece
- Stuck: No judgment, hint or simplify, easier follow-up

**Question types** (rotate based on level):
- RECALL: "What does X do?" (EXPLORER/LEARNER)
- COMPREHENSION: "How do X and Y connect?" (LEARNER/PRACTITIONER)
- APPLICATION: "Which component would you modify for [goal]?" (PRACTITIONER)
- ANALYSIS: "Why this design?" (PRACTITIONER/EXPERT)
- PREDICTION: "What if X changed?" (EXPERT)

**Session management:**
- Track: level, goal, topics covered, answer patterns
- Summarize progress when natural (not forced intervals)
- On exit: brief recap, key files, next steps
</teaching_flow>

<constraints>
1. **No hallucination**: Only cite `file:line` if verified by a tool call this turn.
2. **Memory**: Track what you've explored - don't re-fetch.
3. **Concise**: ~100-150 words per response. End with ONE question.
</constraints>

<directory_structure>
{directory_tree}
</directory_structure>
"""

# Starter questions shown on Tab in agent mode.
# Based on common developer onboarding challenges from:
# - https://mannes.tech/10-questions-codebase/
# - https://www.cortex.io/post/developer-onboarding-guide
# - https://trstringer.com/20-questions-for-new-software-team/
# 60 chars: keeps the completion menu readable on standard 80-col terminals
# without wrapping or truncating too aggressively on typical terminal widths.
STARTER_QUESTION_DISPLAY_LENGTH: int = 60

DEFAULT_STARTER_QUESTIONS: list[str] = [
    "How do I get started with this project? Walk me through setup, installation, and first steps.",
    "What does this project do? Give me a high-level overview of its purpose and main features.",
    "Explain the project structure. What are the main directories and what do they contain?",
    "Where are the main entry points of this application? How does it start?",
    "What architectural patterns does this codebase follow? (MVC, Clean Architecture, etc.)",
    "What are the key components/modules and how do they interact?",
    "How does data flow through the application? Trace a typical request/operation.",
    "What technologies and frameworks are used? Check package.json, requirements.txt, or similar.",
    "What are the main dependencies and what are they used for?",
    "How do I find where a specific function or class is defined?",
    "Where are the API endpoints defined? List them with their routes.",
    "Where are the database models/schemas defined? What entities exist?",
    "How do I build and run this project locally?",
    "How do I run the tests? What testing framework is used?",
    "How is the application configured? Where are config files and environment variables?",
    "What functions have high coupling and should be refactored? Find code with many callers AND dependencies.",
    "What are the architectural bottlenecks in this codebase? Find functions that many call paths flow through.",
    "If I change <insert function here>, what code will be affected? Show me the blast radius.",
    "Which parts of the codebase are most complex? Where should I focus code review efforts?",
]

# Tool workflow instructions appended to the system prompt when
# AgentConfig.tool_instructions is True. Helps weaker models route to the
# right tools by giving them an intent-keyed cheat sheet.
DEFAULT_TOOL_INSTRUCTIONS_TEMPLATE: str = """
<tool_workflow>
## Core Principle: BROAD TO NARROW
Always start with architectural overview, then narrow down to specifics.

## Intent-Based Workflows

For EXPLORATION ("How does X work?", "Explain architecture"):
1. code_map → code_search (mode="semantic") → read_source_range

For LOOKUP ("Find class X", "Who calls Y?"):
1. find_identifier → trace_dependency_path

For READING ("Show me the code"):
1. find_identifier → read_source_range (or read_source_ranges for multiple locations)

For DEBUGGING ("Fix error", stacktrace present):
1. debug_guide → debug_stacktrace → read_source_range → suggest fix

For FILE MODIFICATION ("Fix this", "Create file"):
1. read_source_range → edit_file (or write_file for new files)
2. On mistake: undo_file_change
3. To remove: delete_file

For GIT/HISTORY ("Who changed this?", "Why was this written?"):
1. find_hotspots (churn analysis) → code_evolution (function history)
2. git_history_context for line-level blame + commit messages

For REFACTORING ("What should I refactor?"):
1. coupling_hotspots → architectural_bottlenecks → read_source_range

For IMPACT ANALYSIS ("What breaks if I change X?"):
1. change_impact_radius → read_source_range

For CODE QUALITY ("Find issues", "Clean up code"):
1. detect_clones (mode: combined|semantic|syntactic)
2. detect_dead_code (unused functions)
3. detect_echo_comments (redundant comments)
4. propagation_cost (architecture health)
5. detect_tramp_data (parameters repeated across functions)
6. detect_sensitive_values (hardcoded secrets, API keys, credentials)
7. detect_magic_constants (repeated literal values across files)

For THIRD-PARTY CODE ("How does library X work?"):
1. check_python_virtual_env → get_python_package_path → code_search

For CODE FLAGGING ("Mark issues in source"):
1. flag_code (insert [CODEN] markers) → flag_clear (remove markers)

## Tool Selection

| You Need | Tool |
|----------|------|
| Exact symbol name | find_identifier |
| Conceptual search | code_search mode="semantic" |
| Literal text match | code_search mode="keyword" |
| Overview | code_map |
| Call paths/dependencies | trace_dependency_path |
| Refactoring targets | coupling_hotspots |
| Architectural risks | architectural_bottlenecks |
| Blast radius | change_impact_radius |
| Architecture health | propagation_cost |
| Duplicate code | detect_clones (combined/semantic/syntactic) |
| Unused code | detect_dead_code |
| Tramp data / coupling | detect_tramp_data |
| Hardcoded secrets/sensitive values | detect_sensitive_values |
| Useless comments | detect_echo_comments |
| Magic constants | detect_magic_constants |
| Parse error stacktrace | debug_stacktrace |
| Line-level blame | git_history_context |
| Churn analysis | find_hotspots |
| Function history | code_evolution |
| Virtual env check | check_python_virtual_env |
| Library source path | get_python_package_path |
| Create new file | write_file |
| Edit existing file | edit_file |
| Remove file | delete_file |
| Undo file change | undo_file_change |
| Flag code issues | flag_code |
| Clear [CODEN] markers | flag_clear |
| Read specific lines | read_source_range |
| Read multiple ranges | read_source_ranges |

## Rules
1. **Sequential workflow**: Wait for each result before proceeding
2. **Absolute paths**: Always use full paths
3. **Cite sources**: Format path/file.py:42
4. **On failure**: Try different terms, never repeat same failing query
5. **Efficiency**: Use read_source_range for specific lines, read_source_ranges for multiple locations

## When to STOP
Stop calling tools when:
- You have enough information to answer
- Modification succeeded (edit_file/write_file returned success)
- Found the requested code/symbol/file
</tool_workflow>

<debugging_strategy>
## Debugging
Use for runtime issues (wrong values, None mysteries). Skip for syntax/import errors.
Call debug_guide first if unsure which approach to use.

### Interactive Debug Session (Python only)
Tools: debug_session, debug_action, debug_eval, debug_variables, debug_stack, debug_breakpoint

Workflow:
1. debug_session action='launch', program='script.py', stop_on_entry=True
2. debug_breakpoint action='set', file_path='script.py', lines=[36]
3. debug_action action='continue' → auto-returns code, variables, stack
4. debug_eval expression='variable_name'
5. debug_action action='step_over' → auto-returns context
6. debug_session action='stop' (auto-removes source injections; pass cleanup_injections=False to keep)

debug_action action='pause' works on a running program (resolves MainThread automatically).
debug_variables(all_frames=True) dumps every frame's locals in one call.
For IDE attachment: debug_server to start a server, then attach from IDE.

### Source Injection (Python/JS/TS)
Tools: source_add_breakpoint, source_inject_trace, source_list_injections, source_remove_injections

| Extension | Breakpoint | Trace |
|-----------|------------|-------|
| .py | breakpoint() | print() |
| .js/.jsx/.mjs/.cjs | debugger; | console.log() |
| .ts/.tsx | debugger; | console.log() |

Use source_list_injections to see active injections. debug_session(action='stop') auto-cleans them.

### Stacktrace Analysis
Tool: debug_stacktrace - Parse any language stacktrace and map to local code.
Workflow: Paste stacktrace -> get user frames with source context -> read_source_range for details
</debugging_strategy>
"""

# Pedagogical addendum appended to DEFAULT_TOOL_INSTRUCTIONS_TEMPLATE when
# the agent runs in study/tutor mode.
DEFAULT_STUDY_TOOL_INSTRUCTIONS_TEMPLATE: str = """
<study_tool_strategy>
## Tool Selection by Experience Level

For EXPLORER level:
1. code_map → code_search → read_source_range
2. Avoid: Deep call graphs, analysis tools

For LEARNER level:
1. find_identifier → trace_dependency_path → code_search
2. Avoid: Overwhelming detail

For PRACTITIONER level:
1. find_identifier → trace_dependency_path → read_source_range
2. Code quality: detect_clones (mode=semantic), detect_dead_code

For EXPERT level:
1. trace_dependency_path → find_hotspots → code_evolution
2. Architecture: coupling_hotspots, architectural_bottlenecks, propagation_cost
3. Avoid: Architecture overviews

## Common Patterns

For "How does X work?":
find_identifier → read_source_range → trace_dependency_path

For "Where is X used?":
find_identifier → read_source_ranges (sample 2-3 callers)

For "What should I refactor?":
coupling_hotspots → detect_clones (mode=combined) → read_source_range
</study_tool_strategy>
"""

# System prompt for the LLM-based dynamic tool router (mcp/llm_tool_router.py).
# Recall over precision — false negatives cost more than false positives.
DEFAULT_TOOL_ROUTER_PROMPT_TEMPLATE: str = """\
You are a tool-selection router. Given a user's task and a catalog of \
available tools, return ONLY the names of tools that could be useful.

Rules:
- When uncertain, INCLUDE the tool — false negatives are costly, \
false positives are cheap
- Think about what steps are needed to complete the task, and which \
tools enable those steps
- A tool is relevant if it MIGHT be needed, not only if it's \
DEFINITELY needed
- Return tool names exactly as listed\
"""

# MCP server "instructions" surfaced to MCP clients (Claude Code, etc.).
DEFAULT_FULL_SERVER_INSTRUCTIONS_TEMPLATE: str = """\
Code intelligence server with tools for working with codebases.

Use tools to get real information - never make up code or file contents.
Each tool's description explains when to use it.
All paths must be absolute.

Debugging workflow: to set breakpoints by function name, use find_identifier to resolve
the name to a line number, then pass that line to debug_breakpoint.
On program termination, check breakpoint_summary to see which breakpoints were hit or missed."""

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
# Progress bar label shown during semantic embedding generation
SEMANTIC_INDEX_PROGRESS_LABEL = "Building semantic index"
# Formatter table/section separator width (used by all analysis formatters)
FORMATTER_WIDTH = 80
STATS_SEPARATOR_WIDTH = FORMATTER_WIDTH

# =============================================================================
# MCP Token Budget Constants
# =============================================================================
# Default token budget for MCP tool responses (fits comfortably in LLM context)
DEFAULT_TOKEN_BUDGET = 4000
# Maximum token limit allowed in MCP field validators (prevents OOM)
MAX_TOKEN_LIMIT = 100_000
# Minimum token limit allowed in MCP field validators (prevents useless results)
MIN_TOKEN_LIMIT = 100

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

# Sentinel status for results computed with the direct-edges approximation.
# Lives in constants.py rather than propagation_cost.py so the formatter can
# reference it without pulling in the daemon import chain.
PC_APPROXIMATE_STATUS = "APPROXIMATE"

# Beyond this node count, transitive closure is intractable and the analyzer
# requires opt-in via --approximate. O(N²) edges balloon to ~25M for 5,000 nodes.
PC_MAX_NODES_FOR_CLOSURE = 5000

# Dense graphs above this node count also fall back: closure cost is bounded by
# edges, not nodes alone.
PC_DENSE_GRAPH_NODE_THRESHOLD = 1000
PC_DENSE_GRAPH_DENSITY_THRESHOLD = 0.1

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
# Simple Map Mode (--simple) Constants
# =============================================================================

# Tiebreak divisor for `--simple` map ranking. The primary score is integer
# commits-per-file; we add `entity.line_count / DIVISOR` to break ties within a
# file so larger objects win over smaller ones. 1e6 keeps even the largest
# realistic line counts (<1e6 lines) below 1.0, so no entity's tiebreak can
# overpower a single additional commit on a peer file.
SIMPLE_MAP_LINE_TIEBREAK_DIVISOR: float = 1_000_000.0

# 1.0s: cheap probe to detect shallow/promisor clones before issuing any
# history-heavy command. This avoids `--simple` appearing hung while Git lazily
# fetches missing objects from a partial clone.
SIMPLE_MAP_HISTORY_PROBE_TIMEOUT_SECONDS: float = 1.0

# 5.0s: per-file blame budget for object-level `--simple` ranking. If one file's
# history exceeds this, we fail closed to line-count ranking rather than stall
# the terminal.
SIMPLE_MAP_BLAME_TIMEOUT_SECONDS: float = 5.0

# =============================================================================
# Magic Constant Detection Thresholds
# =============================================================================

# Minimum occurrences of a literal value to be flagged as a magic constant
# 3+ locations = naming opportunity; 1-2 is normal usage
MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES = 3

# Minimum distinct files a constant must appear in to be flagged
# 2 files means the constant crosses file boundaries — a strong naming signal
MAGIC_CONSTANT_DEFAULT_MIN_FILES = 2

# 10+ occurrences: constant is scattered everywhere — strong naming signal
MAGIC_CONSTANT_TIER_HIGH = 10
# 5+ occurrences: moderate repetition — worth investigating
MAGIC_CONSTANT_TIER_MODERATE = 5
# 3+ occurrences: mild repetition — may be intentional
MAGIC_CONSTANT_TIER_LOW = 3

# Default result limit matching other analysis flags
# 50 results provide comprehensive overview without overwhelming output
MAGIC_CONSTANT_DEFAULT_RESULT_LIMIT = 50

# Max results allowed in MCP context
# 500 is upper bound to prevent memory issues and token budget exhaustion
MAGIC_CONSTANT_MAX_RESULTS = 500

# Values universally considered non-magic (idiomatic in all languages)
# 0, 1, -1 are loop/boolean/sentinel idioms; empty strings are default inits
MAGIC_CONSTANT_TRIVIAL_VALUES: frozenset[str] = frozenset({
    "0", "1", "-1", "0.0", "1.0",
    '""', "''",
})

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
# Analysis Default Thresholds
# =============================================================================
# Canonical default values for analysis parameters used across CLI, MCP, daemon.
# ALL consumer code must import from here -- no hardcoded copies.

# Clone detection: semantic similarity for flagging near-duplicates
# 0.95 catches only high-confidence clones; lower values increase recall but add noise
# NOTE: These thresholds were calibrated with model2vec (256-dim). Now using
# MiniLM ONNX (384-dim) -- may need re-tuning in a follow-up.
DEFAULT_CLONE_SEMANTIC_THRESHOLD = 0.95

# Clone detection: combined-mode harmonic mean weights
# Semantic weighted higher because embedding similarity is more reliable
# than line-level Jaccard for detecting behavioral clones
DEFAULT_CLONE_SEMANTIC_WEIGHT = 0.65
# Complement to semantic weight (weights must sum to 1.0)
DEFAULT_CLONE_SYNTACTIC_WEIGHT = 0.35

# Clone detection: default result limit (comprehensive without overwhelming output)
DEFAULT_CLONE_RESULT_LIMIT = 50


class CloneCategory(str, Enum):
    """Clone pair classification categories.

    Inherits from str so the .value is directly usable as a dict value
    and in JSON serialization without calling .value explicitly.
    """

    EXACT = "EXACT"
    NEAR_CLONE = "NEAR-CLONE"
    SEMANTIC_STRUCTURAL = "SEMANTIC-STRUCTURAL"
    STRUCTURAL = "STRUCTURAL"
    SEMANTIC = "SEMANTIC"
    PARTIAL = "PARTIAL"


# =============================================================================
# Clone Classification Thresholds
# =============================================================================
# Used by _get_category / _get_combined_category in clone detectors.

# Near-perfect semantic match — functions are effectively identical
CLONE_EXACT_SEMANTIC_THRESHOLD = 0.9999
# Very high semantic similarity — minor textual differences
CLONE_NEAR_SEMANTIC_THRESHOLD = 0.98
# Strong syntactic overlap — high % of matching lines
CLONE_NEAR_SYNTACTIC_THRESHOLD = 0.80
# Moderate semantic with some syntactic — mixed signal
CLONE_SEM_STRUCT_SYNTACTIC_THRESHOLD = 0.50
# Minimum syntactic overlap for structural classification
CLONE_STRUCTURAL_SYNTACTIC_THRESHOLD = 0.70
# Minimum contiguous matching block size to detect structural clones
CLONE_MIN_BLOCK_SIZE = 5
# Exact-match threshold for syntactic-only mode
CLONE_EXACT_SYNTACTIC_THRESHOLD = 0.95

# === Clone Pair Filtering Heuristics ===
# Used by _is_intentional_pair() in semantic.py and combined.py
# to filter complementary method pairs (toggle, getter/setter, adapter).

# Functions with at most this many lines are "short" — high-similarity
# short functions are typically UI handlers, callbacks, or simple wrappers
CLONE_SHORT_FUNC_MAX_LINES = 5
# Cross-file reuse: identical names up to this line count are intentional
CLONE_REUSE_MAX_LINES = 10
# Body similarity at which short / same-class methods are filtered
CLONE_HIGH_SIM_THRESHOLD = 0.95
# Body similarity floor for remaining intentional-pair checks
CLONE_VERY_HIGH_SIM_THRESHOLD = 0.97
# Name similarity for identical-name cross-file reuse pattern
CLONE_IDENTICAL_NAME_SIM = 0.99
# Line-count ratio threshold for similarly-sized function detection
CLONE_LINE_RATIO_THRESHOLD = 0.7
# Name similarity above which same-file pairs are true duplicates (kept)
CLONE_SAME_FILE_NAME_SIM = 0.80
# Name similarity below which = unrelated names -> filtered as intentional
CLONE_UNRELATED_NAME_SIM = 0.85

# === Clone Action Suggestion Thresholds ===
# Used by _suggest_action() to recommend refactoring actions.

# Combined score above which suggests "CONSOLIDATE"
CLONE_CONSOLIDATE_SCORE = 0.95
# Syntactic percentage below which = "different implementation" review
CLONE_DIFFERENT_IMPL_SYN = 0.50
# Syntactic percentage above which = "line-by-line overlap" consolidation
CLONE_LINE_OVERLAP_SYN = 0.70

# Dead code detection: minimum confidence to report a function as unused
# 0.5 provides balanced precision/recall for functions without incoming calls
DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD = 0.5

# Dead code detection: default result limit
DEFAULT_DEAD_CODE_RESULT_LIMIT = 50

# Echo comment detection: minimum similarity to flag a comment as restating code
# 0.80 catches clear echo comments without flagging useful explanatory comments on constants
DEFAULT_ECHO_COMMENT_THRESHOLD = 0.80

# Echo comment severity thresholds — calibrated against MiniLM ONNX cosine similarity.
# CRITICAL: near-identical restatement, no value added whatsoever
ECHO_SEVERITY_CRITICAL = 0.95
# HIGH: strong restatement, comment paraphrases identifier with minimal extra info
ECHO_SEVERITY_HIGH = 0.90
# ELEVATED: moderate restatement, borderline useful
ECHO_SEVERITY_ELEVATED = 0.85

# Propagation cost: minimum internal coupling % to flag a module
# 25% is the midpoint between healthy modularity and concerning coupling
DEFAULT_PROPAGATION_COST_THRESHOLD = 0.25

# Hotspot risk: minimum risk score for flagging (raw score = coupling * log(complexity))
# 50.0 catches moderately-coupled complex functions without overwhelming results
DEFAULT_HOTSPOT_RISK_THRESHOLD = 50.0

# =============================================================================
# Shell Execution Constants (for `!` shell commands inside coden -a)
# =============================================================================

# 30s: long enough for realistic commands (grep, git log, Get-Process) but
# bounds runaway shells (accidental `cat`, `tail -f`, infinite loops).
SHELL_COMMAND_TIMEOUT: float = 30.0

# 50k chars per stream: matches MAX_FILE_INCLUDE_CHARS in file_reference.py
# so shell output can't blow up the LLM context window any more than inlined
# @file references already can.
SHELL_OUTPUT_MAX_SIZE: int = 50_000

# `@@` is not a primitive in bash / zsh / sh / fish / PowerShell 5 / PowerShell 7
# / cmd.exe, so it's safe to use as a client-side separator between the shell
# command and an optional follow-up query across every target shell.
SHELL_QUERY_SEPARATOR: str = "@@"

# POSIX standard exit code 127 means "command not found". Returned by every
# POSIX shell when the executable cannot be located, so we mirror it when the
# shell binary itself is missing (FileNotFoundError before any process runs).
SHELL_EXIT_COMMAND_NOT_FOUND: int = 127

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
    # Framework-level dispatchers — same-named hooks called on opaque
    # variable receivers (React `root.render(...)`, `instance.render(...)`)
    # would otherwise fan out across every same-named method in the repo.
    "render",
}
