<img src="images/readme/dog_logo.jpg" alt="Code Retriever Logo" width="400">

## Install & Run

```bash
pip install coden-retriever
```

Requires Python 3.10-3.12.

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
```

<img src="images/readme/coden_stats_reverzed.png" alt="Coden stats output showing directory tree and ranking metrics" width="700">

## The Problem

Codebases are not a flat collection of text files. It is extremely valueable to understand which files are key components and which ones are not. That is what this tool achieves: To help developers, as well as LLM's gain a strong mental model of the codebase.

**Note:** The first run of `$ coden` on a new codebase is slower because it parses everything and buils a call graph. Subsequent runs are cached.

## How It Works

We initially parse code with tree-sitter, build a call graph (functions, classes, methods as nodes; calls, imports, inheritance as edges), and then run two graph algorithms to find what matters:

**PageRank** finds the load-bearing code. If a function is called by many other important functions, it scores high. High PageRank means "if this breaks, a lot of things break."

**Betweenness Centrality** finds the bridges—code that sits between different parts of your system. These are the integration points, the places where module A talks to module B. High betweenness means "this is where different parts of the system meet."

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
coden /path/to/repo --map --show-deps        # Show callers/callees
coden /path/to/repo --format json            # Output as json/markdown/xml
coden serve                                  # Start MCP server
coden serve --transport http --port 8000     # MCP over HTTP
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
    "max_retries": 3,
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
- **coupling_hotspots** - Find refactoring targets (high coupling + complexity). CLI: `--hotspots` / `-H`
- **find_hotspots** - Git churn analysis (frequently changed files).

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
