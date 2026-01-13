<img src="images/readme/dog_logo.jpg" alt="Code Retriever Logo" width="400">

## Install & Run

```bash
pip install coden-retriever
```

**Requires Python 3.10-3.12.** Python 3.13+ is not supported because `tree-sitter-languages` (required for multi-language parsing) only provides wheels up to Python 3.12.

```bash
# Get a ranked map of a repo
coden /path/to/repo

# Top 50 results with stats
coden /path/to/repo --stats -n 50 -r

# Search for something
coden /path/to/repo --query "authentication"

# Find a specific symbol
coden /path/to/repo --find "UserAuth"

# Find refactoring hotspots (high coupling + complexity)
coden /path/to/repo --hotspots -n 20 --stats -r

# Detect code clones (find duplicate/similar functions)
coden /path/to/repo --clones --clone-threshold 0.90

# Clone detection modes: semantic-only or syntactic-only
coden /path/to/repo --clones --clone-semantic     # Similar behavior (embeddings)
coden /path/to/repo --clones --clone-syntactic    # Copy-paste detection (Jaccard)

# Analyze architecture health (propagation cost)
coden /path/to/repo --propagation --breakdown

# Find echo comments (comments that just restate the code)
coden /path/to/repo flag -E --dry-run
```

<img src="images/readme/coden_stats_reverzed.png" alt="Coden stats output showing directory tree and ranking metrics" width="700">

## The Problem

Codebases are not a flat collection of text files. It is extremely valueable to understand which files are key components and which ones are not. That is what this tool achieves: To help developers, as well as LLM's gain a strong mental model of the codebase.

**Note:** The first run of `$ coden` on a new codebase is slower because it parses everything and buils a call graph. Subsequent runs are cached.

## How It Works

We initially parse code with tree-sitter, build a call graph (functions, classes, methods as nodes; calls, imports, inheritance as edges), and then run two graph algorithms to find what matters:

**PageRank** finds the load-bearing code. If a function is called by many other important functions, it scores high. High PageRank means "if this breaks, a lot of things break."

**Betweenness Centrality** finds the bridges--code that sits between different parts of your system. These are the integration points, the places where module A talks to module B. High betweenness means "this is where different parts of the system meet."

We use these instead of simple text matching because structural dependencies matter. A file that is imported everywhere is more important than a file that happens to contain your search term five times.

| What You Are Looking At | PageRank | Betweenness | Example |
|------------------------|----------|-------------|---------|
| Core utility | High | Low | `Logger.log()` - heavily used, does not connect modules |
| Integration point | Medium | High | `APIGateway.route()` - bridges layers |
| Central hub | High | High | `Database.query()` - important AND connects many parts |

Results are ranked using Reciprocal Rank Fusion across:
- **BM25** - Keyword matching
- **Semantic similarity** - Conceptually similar code (enable with `--semantic`)
- **PageRank** - Structural importance
- **Betweenness** - Bridge detection

### Keyword vs Semantic Search

| Mode | When to Use |
|------|-------------|
| `--query "auth"` | You know the terminology |
| `--query "auth" --semantic` | You are asking a natural language question |

Semantic search uses a Model2Vec model distilled from [Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) that ships with the package.

## Supported Languages

**Support:** Python, Go, Rust, Java, C, C++, C#, Kotlin, Swift, Javascript/Typescript, PHP, Scala

## CLI Reference

```bash
coden /path/to/repo                          # Ranked map
coden /path/to/repo --query "auth"           # Keyword search
coden /path/to/repo --query "auth" --semantic # Semantic search
coden /path/to/repo --find "UserAuth"        # Find symbol
coden /path/to/repo --hotspots -n 20         # Top 20 refactoring hotspots
coden /path/to/repo -H --stats -r            # Hotspots with stats, reversed
coden /path/to/repo -C --clone-threshold 0.90      # Find code clones (90% similarity)
coden /path/to/repo -C --clone-semantic            # Semantic-only clone detection
coden /path/to/repo -C --clone-syntactic           # Syntactic-only clone detection
coden /path/to/repo -C --semantic-weight 0.65      # Adjust combined score weights
coden /path/to/repo -P --breakdown           # Architecture health analysis
coden /path/to/repo -P --critical-paths      # Show high-impact code paths
coden /path/to/repo --map --show-deps        # Show callers/callees
coden /path/to/repo --format json            # Output as json/markdown/xml
coden serve                                  # Start MCP server
coden serve --transport http --port 8000     # MCP over HTTP
coden -E --remove-comments --dry-run         # Preview echo comment removal
coden -E --remove-comments --backup          # Remove echoes entirely (with backup)
coden flag -C --dry-run                      # Preview clone flags
coden flag -E --dry-run                      # Preview echo comment flagging
coden flag -E --echo-threshold 0.85          # Flag echo comments (default threshold)
coden flag -E --echo-threshold 0.95          # Stricter: only near-identical echoes
coden flag -E --remove-comments --backup     # Alternative: remove via flag subcommand
coden flag -E --include-tests                # Include test files in analysis
coden flag -HPCE --backup                    # Flag all issues (hotspots, propagation, clones, echoes)
coden flag clear                             # Remove all [CODEN] comments
coden reset                                  # Reset everything (destructive!)
```

## Daemon Mode

If you are running repeated queries, the daemon keeps indices in memory so you do not pay startup costs every time.

```bash
coden daemon start                # Start background service
coden /path/to/repo -q "auth"     # Queries use daemon automatically
coden daemon status               # Check if running
coden daemon stop                 # Stop it
coden daemon restart              # Restart
coden daemon clear-cache          # Clear daemon cache
```

## Code Flagging

Add `[CODEN]` comments to mark problematic code or remove echo comments directly:

```bash
coden -E --remove-comments --dry-run # Preview echo comment removal (direct)
coden -E --remove-comments --backup  # Remove echo comments directly
coden flag -C --dry-run              # Preview clone flags
coden flag -E --dry-run              # Preview echo comment flags
coden flag -HPCE --backup            # Flag hotspots, propagation, clones, echoes
coden flag clear                     # Remove all [CODEN] comments
```

### Common Options

- **`--dry-run`**: Preview changes without modifying files. Always use this first to see what will be flagged.
- **`--backup`**: Create `.coden-backup` copies of files before modification (e.g., `file.py.coden-backup`).
- **`--include-tests`**: Include test files in analysis (excluded by default).
- **`--stats`**: Display summary statistics after flagging.

### Analysis Types & Thresholds

| Analysis Type | Flag | Threshold Option | Default | Description |
|---|---|---|---|---|
| **Hotspots** | `-H` | `--risk-threshold` | 50 | Min risk score (raw score, typically 50-200+) |
| **Propagation Cost** | `-P` | `--propagation-threshold` | 0.25 | Min internal coupling for modules (0-1) |
| **Code Clones** | `-C` | `--clone-threshold` | 0.95 | Min semantic similarity for clones (0-1) |
| **Echo Comments** | `-E` | `--echo-threshold` | 0.85 | Min similarity for echo detection (0-1) |

**Usage notes:**
- All threshold options work in both direct mode (`coden -H`) and flag mode (`coden flag -H`)
- `-H`: Filters hotspots with `risk_score >= threshold` (raw score = coupling × log(complexity))
- `-P`: Filters modules in breakdown with `internal_coupling >= threshold` (0-1 scale)
- `-C`: Filters clones with `similarity >= threshold` (0-1 scale)
- `-E`: Filters echo comments with `similarity >= threshold` (0-1 scale)

**Threshold ranges:** `-P`, `-C`, `-E` use 0-1 scale. Lower values are more permissive, higher values are stricter. `-H` uses raw risk scores (typically 50-200+).

### Code Clone Detection

Clone detection finds duplicate or near-duplicate functions that are candidates for refactoring. Three detection modes are available:

| Mode | Flag | Description | Best For |
|------|------|-------------|----------|
| **Combined** | (default) | Both semantic + syntactic | General use, balanced detection |
| **Semantic** | `--clone-semantic` | Model2Vec embeddings | Behaviorally similar functions |
| **Syntactic** | `--clone-syntactic` | Line-by-line Jaccard | Exact copy-paste detection |

#### Detection Modes Explained

**Combined mode** (default) uses a weighted harmonic mean of semantic and syntactic scores:
- Semantic weight: 0.65 (adjustable via `--semantic-weight`)
- Syntactic weight: 0.35 (adjustable via `--syntactic-weight`)

**Semantic mode** detects functions with similar behavior, even if structurally different. Uses Model2Vec embeddings to find:
- Async/sync variants of the same function
- Functions that do the same thing with different implementations
- Renamed copies with modified variable names

**Syntactic mode** detects copy-paste clones using line-by-line Jaccard similarity:
- `--line-threshold`: Min similarity per line (default: 0.70)
- `--func-threshold`: Min percentage of lines that must match (default: 0.50)

#### Usage Examples

```bash
# Default combined mode (recommended)
coden /path/to/repo -C --clone-threshold 0.90

# Semantic-only: find behaviorally similar functions
coden /path/to/repo -C --clone-semantic

# Syntactic-only: find copy-paste duplicates
coden /path/to/repo -C --clone-syntactic

# Adjust combined weights (more emphasis on semantic similarity)
coden /path/to/repo -C --semantic-weight 0.80 --syntactic-weight 0.20

# Syntactic with custom thresholds
coden /path/to/repo -C --clone-syntactic --line-threshold 0.80 --func-threshold 0.60

# Flag clones in code with [CODEN] comments
coden flag -C --backup
```

### Echo Comment Detection

Echo comments are comments that provide no additional value because they simply repeat what the code identifier already conveys. For example:

**Echo comments (redundant)**:
```python
# Calculate the total
def calculate_total(items):
    return sum(item.price for item in items)

# Process the payment
def process_payment(amount):
    ...
```

**Good comments (provide context)**:
```python
# Apply progressive discount based on customer lifetime value
# Tier 1: 0-10 purchases = 0%, Tier 2: 11-50 = 5%, Tier 3: 51+ = 10%
def calculate_discount_tier(purchases: int) -> float:
    ...
```

Echo detection uses:
- **Tree-sitter AST parsing** to extract ALL comments from your codebase
- **Semantic similarity analysis** (Model2Vec embeddings + cosine similarity) to compare comment text with code identifiers
- **Configurable threshold** to control strictness (0.95 = near-identical only, 0.75 = looser)

#### Usage Examples

```bash
# Preview echo comments (dry run)
coden flag -E --dry-run

# Flag echo comments with [CODEN] markers
coden flag -E --backup

# Remove echo comments entirely (no markers)
coden flag -E --remove-comments --backup

# Adjust threshold (stricter - only exact echoes)
coden flag -E --echo-threshold 0.95 --dry-run

# Adjust threshold (looser - catch more potential echoes)
coden flag -E --echo-threshold 0.75 --dry-run

# Include test files in analysis
coden flag -E --include-tests --dry-run

# Combine with other analysis types
coden flag -HPCE --backup  # All analyses at once

# Preview only top 10 issues (limit only works in dry-run mode)
coden flag -H --dry-run -n 10
```

#### Options Explained

**`--dry-run`**: Preview mode - shows what would be changed without modifying any files. Use this first to review results before making changes.

**`-n/--limit`**: Limit the number of results (default: 20). Use `-n -1` to show all results (may be slow for large repos). In flag mode, the limit only applies in dry-run preview - when actually flagging code (without `--dry-run`), all matching items will be flagged to ensure comprehensive coverage.

**`--backup`**: Creates a safety copy of each modified file with a `.coden-backup` extension before making changes. Recommended when removing comments or making bulk modifications. Example: modifying `src/utils.py` creates `src/utils.py.coden-backup`.

**`--remove-comments`**: Deletes detected echo comments entirely from the source files instead of flagging them with `[CODEN]` markers. Works with both `coden -E --remove-comments` and `coden flag -E --remove-comments`. Use with `--backup` for safety. Without this flag, echo comments are only flagged/displayed, not removed.

**`--include-tests`**: Include test files in the analysis (files matching `*test*.py`, `*spec.ts`, etc.). By default, test files are excluded since echo comments are more acceptable in tests where clarity is prioritized over conciseness.

**`--echo-threshold`**: Controls detection strictness (0.0-1.0 range):
- `0.95` = Very strict, only near-identical echoes (e.g., `# get user` -> `get_user()`)
- `0.85` = Default, balanced detection
- `0.75` = Looser, catches more potential echoes

#### Output Format

Flag mode displays a parameter header showing:
- **Active analysis types** and their thresholds (e.g., "Hotspots (risk >= 50.0)")
- **Preview limit** status (shows if `-n` is limiting results)
- **Warning message** if `-n` is used without `--dry-run`

Detected echo comments show:
- **File path and line number**
- **Similarity score** (0-100%)
- **Severity**: CRITICAL (>95%), HIGH (>90%), ELEVATED (>85%), MODERATE (<85%)
- **Comment text** and **associated code identifier**

## Caching

Indices are cached in `~/.coden-retriever/`.

```bash
coden cache list             # List cached projects
coden cache status           # Cache info for current directory
coden cache status /path     # Cache info for specific project
coden cache clear            # Clear cache for current directory
coden cache clear /path      # Clear cache for specific project
coden cache clear --all      # Clear everything
coden cache path             # Show cache directory
```

## Configuration

Settings live in `~/.coden-retriever/settings.json`.

```bash
coden config show        # Show all configuration
coden config path        # Show config file path
coden config reset       # Reset to defaults
coden config set <key> <value>  # Set a value
```

### Configuration Structure

```json
{
  "_version": 1,
  "model": {
    "default": "ollama:",
    "base_url": null,
    "provider_urls": {
      "ollama": "http://localhost:11434/v1",
      "llamacpp": "http://localhost:8080/v1"
    }
  },
  "agent": {
    "max_steps": 15,
    "max_retries": 5,
    "debug": false,
    "disabled_tools": ["debug_server"],
    "mcp_server_timeout": 30.0,
    "tool_instructions": false,
    "ask_tool_permission": true,
    "dynamic_tool_filtering": false,
    "tool_filter_threshold": 0.5
  },
  "daemon": {
    "host": "127.0.0.1",
    "port": 19847,
    "socket_timeout": 30.0,
    "max_projects": 5
  },
  "search": {
    "default_tokens": 4000,
    "default_limit": 20,
    "semantic_model_path": null
  }
}
```

### Config Values

```bash
# Model
coden config set model.default ollama:qwen2.5-coder
coden config set model.base_url http://localhost:11434/v1

# Agent
coden config set agent.max_steps 20
coden config set agent.debug true

# Daemon
coden config set daemon.port 8080
coden config set daemon.max_projects 10

# Search
coden config set search.default_tokens 8000
coden config set search.default_limit 50
```

### Environment Variables

These override the config file:

| Variable | What it does |
|----------|--------------|
| `CODEN_RETRIEVER_MODEL` | Override default model |
| `CODEN_RETRIEVER_BASE_URL` | Override base URL |
| `CODEN_RETRIEVER_DAEMON_PORT` | Override daemon port |
| `CODEN_RETRIEVER_DAEMON_HOST` | Override daemon host |
| `CODEN_RETRIEVER_MODEL_PATH` | Override semantic model path |
| `CODEN_RETRIEVER_MCP_TIMEOUT` | Override MCP server timeout |
| `CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS` | Enable dynamic tools (`1`, `true`, `yes`) |
| `CODEN_RETRIEVER_DISABLED_TOOLS` | Comma-separated tools to disable |
| `CODEN_RETRIEVER_TEMPERATURE` | Override model temperature (0.0-2.0) |
| `CODEN_RETRIEVER_MAX_TOKENS` | Override max response tokens |
| `CODEN_RETRIEVER_TIMEOUT` | Override request timeout (seconds) |

## Interactive Agent

<img src="images/readme/coden_agentic_mode.png" alt="Coden agent mode welcome screen" width="700">

Activate coden in agent mode and use an LLM to chat about your codebase.

```bash
coden -a                                           # Current directory
coden /path/to/repo --agent --model ollama:qwen2.5-coder  # With Ollama
coden /path/to/repo --agent --model llamacpp:      # With llama-cpp-server
```

**Supported model formats:**

| Format | Example | What it connects to |
|--------|---------|---------------------|
| `ollama:model` | `ollama:qwen2.5-coder:14b` | Ollama (localhost:11434) |
| `llamacpp:model` | `llamacpp:my-model` | llama-cpp-server (localhost:8080) |
| `openai:model` | `openai:gpt-4o` | OpenAI API (needs OPENAI_API_KEY) |
| `model` + `--base-url` | `my-model --base-url http://...` | Any OpenAI-compatible endpoint |

For vLLM, LM Studio, etc:
```bash
coden -a --model my-model-name --base-url http://localhost:8000/v1
```

Type `help` in agent mode to see available tools, or `menu`/`tools` for the interactive tool picker.

### Slash Commands

| Command | Aliases | What it does |
|---------|---------|--------------|
| `/help` | | Show commands |
| `/model [name]` | `/m` | Show/switch model |
| `/config` | | View/modify settings |
| `/tools` | `/t` | Tool picker |
| `/run` | `/r`, `/execute` | Tool wizard |
| `/study [topic]` | `/learn`, `/quiz` | Quiz mode |
| `/exit-study` | `/stop-study` | Exit quiz |
| `/debug [on\|off]` | `/d` | Toggle debug |
| `/cd [path]` | `/dir`, `/chdir` | Change directory |
| `/clear` | `/c` | Clear history |
| `/exit` | `/quit`, `/q` | Exit |
| `/cache` | | Cache management |
| `/cache-clear` | `/cc` | Clear current project cache |
| `/cache-list` | `/cl` | List cached projects |

In-agent config:
```
/config                    # Show settings
/config set model ollama:codellama
/config set max_steps 20
/config reset
```

## MCP Server

Transport options: `stdio` (default), `http`, `sse`, `streamable-http`

For VS Code, configure `.vscode/mcp.json`:

```json
{
  "servers": {
    "coden": {
      "command": "${workspaceFolder}/.venv/Scripts/python.exe",
      "args": ["${workspaceFolder}/coden.py", "serve"]
    }
  }
}
```

Reload VS Code (Ctrl+Shift+P -> "Developer: Reload Window").

### Tools

**Code Discovery**
- **code_map** - Architectural overview with dependencies. Start here.
- **code_search** - Keyword or semantic search.
- **coupling_hotspots** - Find refactoring targets (high coupling + complexity). CLI: `-H`
- **find_hotspots** - Git churn analysis (frequently changed files).
- **clone_detection** - Find duplicate functions (combined/semantic/syntactic modes). CLI: `-C`
- **propagation_cost** - Measure architecture health based on coupling. CLI: `-P`

**Graph Analysis**
- **change_impact_radius** - Blast radius analysis ("if I change this, what breaks?").
- **architectural_bottlenecks** - Find bridge functions with high betweenness centrality.

**Symbol Lookup**
- **find_identifier** - Find exact symbol definitions.
- **trace_dependency_path** - "If I change this, what breaks?"

**Code Inspection**
- **read_source_range** - Read specific lines from a file.
- **read_source_ranges** - Read multiple ranges at once.
- **git_history_context** - Git blame info.
- **code_evolution** - How code changed over time.

**File Editing**
- **write_file** - Create or overwrite files.
- **edit_file** - Surgical edits via SEARCH/REPLACE or AST-based SYMBOL targeting.
- **delete_file** - Remove files.
- **undo_file_change** - One-step undo per file.

**Debugging**
- **debug_stacktrace** - Analyze Python stack traces.
- **debug_session** - Manage DAP debug sessions.
- **debug_action** - Step, continue, etc.
- **debug_state** - Inspect variables, evaluate expressions.
- **add_breakpoint** - Inject breakpoints into source.
- **inject_trace** - Add trace/logging statements.
- **remove_injections** - Clean up injected debug code.
- **list_injections** - View active injections.

**Python Environment**
- **check_python_virtual_env** - Detect venvs.
- **get_python_package_path** - Locate installed packages.

**Dynamic Tools** (disabled by default)
- **create_dynamic_tool** - Create custom MCP tools at runtime.
- **remove_dynamic_tool** - Remove dynamic tools.

To enable dynamic tools:
```bash
export CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS=1
```

## Docker

### Build

```bash
docker build -t coden-retriever:latest .
```

### Usage

The `coden-docker` wrapper uses a persistent container:

```bash
cd /path/to/your/project
./coden-docker start .                  # Start container
./coden-docker .                        # Repository map
./coden-docker . --query "auth"         # Search
./coden-docker . --find "MyClass"       # Find symbol
./coden-docker -a                       # Agent mode
./coden-docker stop                     # Stop
```

First run builds the index. After that, the daemon keeps it in memory.

```bash
./coden-docker start [path]   # Start with workspace
./coden-docker stop           # Stop container
./coden-docker restart [path] # Restart with new workspace
./coden-docker status         # Container status
```

### MCP Server in Docker

```bash
docker run -d -p 8000:8000 --name coden-mcp coden-retriever
```

Available at `http://localhost:8000/mcp`, health check at `http://localhost:8000/health`.

### Docker Compose

```bash
docker compose up -d mcp-server
docker compose logs -f mcp-server
docker compose down
```

### Docker Environment Variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `CODEN_RETRIEVER_HOST` | `0.0.0.0` | MCP server bind address |
| `CODEN_RETRIEVER_PORT` | `8000` | MCP server port |
| `CODEN_RETRIEVER_DISABLED_TOOLS` | | Tools to disable |
| `CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS` | | Enable dynamic tools |

Health check:
```bash
curl http://localhost:8000/health
# {"status":"healthy","service":"CodenRetriever"}
```

### Agent Mode with Ollama in Docker

The container connects to host Ollama via `host.docker.internal`:

```bash
# On host
ollama serve

# In Docker
./coden-docker -a
# Then: /model ollama:qwen2.5-coder
```

## Troubleshooting

If you encounter problems, clearing the cache and stopping the daemon might help:

```bash
coden cache clear --all
coden daemon stop
```

### Full Reset

To reset everything at once, use the reset command:

```bash
coden reset
```

This performs all of the following in one step:
- Clears all project caches
- Stops the daemon
- Resets configuration to defaults

> **Warning:** This is a destructive operation. Your custom configuration settings will be lost and all cached indices will be deleted.
