<img src="images/readme/dog_logo.jpg" alt="Code Retriever Logo" width="400">

**Coden** provides you with an arsenal of code-analysis tools, allowing you to gain unique insights into codebases.

## Why use Coden?

- **Find and reduce AI slop.**
- **Build a mental model faster** with external codebases.
- **Get an unique set of MCP tools.** Let LLMs look at code and find sensitive values, find refactor-targets, debug, etc.
- **Numbers of other reasons.** Feel encouraged to discover them for yourself.

```bash
pip install coden-retriever
```

Python 3.10–3.12. The semantic model ships inside the wheel.

---

## A 30-second tour

Get a ranked map of any repo, with stats:

```bash
coden /path/to/repo --stats -n 50 -r
```

<img src="images/readme/coden_stats_reverzed.png" alt="Coden stats output: directory tree with PageRank / betweenness ranking" width="700">

Audit architecture across ten languages, including multi-module workspaces (Cargo, Maven, Go, .NET, npm):

```bash
coden architecture /path/to/repo
```

Coden's chat harness works with Ollama, llama.cpp, and any OpenAI-compatible endpoint.
Chat with the codebase using a local model:

```bash
coden -a --model ollama:gemma4:31b
```

<img src="images/readme/coden_agentic_mode.png" alt="Coden agent mode welcome screen" width="700">

---

## How it ranks code

Coden builds a call graph (functions, classes, and methods as nodes; calls, imports, and inheritance as edges) and runs two graph algorithms over it.

**PageRank** finds load-bearing code. A function called by many important functions scores high: *if this breaks, a lot of things break*.

**Betweenness centrality** finds the bridges. Functions that sit between subsystems, where module A talks to module B: *this is where the parts of the system meet*.

| Role | PageRank | Betweenness | Example |
|---|---|---|---|
| Core utility | High | Low | `Logger.log()` — heavily used, doesn't connect modules |
| Integration point | Medium | High | `APIGateway.route()` — bridges layers |
| Central hub | High | High | `Database.query()` — important *and* connects many parts |

The default `coden /path` view fuses BM25, semantic similarity, PageRank, and betweenness via Reciprocal Rank Fusion. Add `--query "auth"` to bias toward keywords, `--semantic` for natural-language queries.

For a fast architectural glance that skips the graph entirely, `--map-mode simple` ranks by per-file git commit count and uses a separate on-disk cache. Warm runs come back in about 88 ms.

The first run on a new repo is slower; Coden parses everything and builds the graph. Subsequent runs hit the cache.

---

## Feature tour

### Search and find

```bash
coden /path/to/repo -q "authentication"                  # Keyword search
coden /path/to/repo -q "how does auth work" --semantic   # Natural language
coden /path/to/repo --find "UserAuth"                    # Exact symbol lookup
coden /path/to/repo --hotspots -n 20                     # Refactoring targets
```

### Architecture audit

Five drift checks (cycles, kitchen-sink files, oversized files, shallow packages, in-function imports) across **Python, JavaScript, TypeScript, Go, Rust, Java, Kotlin, PHP, C#, Scala**. Workspace manifests (`Cargo.toml`, `go.work`, `pom.xml`, `.sln`, `package.json` workspaces) are auto-discovered; cross-module call edges and cycles surface in the unified report.

```bash
coden architecture /path/to/repo
coden architecture /path/to/repo --top 20 --exclude tests,vendor
coden architecture /path/to/repo --format json
```

<details>
<summary>How each language defines a "package"</summary>

The audit's load-bearing concept is the package, and ten languages disagree on what one is:

- **Python**: directory containing `__init__.py`, or a PEP 420 namespace dir with `.py` files.
- **JavaScript**: directory with `package.json`, or any `index.{js,mjs,cjs}`-rooted subtree.
- **TypeScript**: same as JS plus `index.{ts,tsx}` / `.d.ts` entries; `tsconfig.json` `paths` resolved.
- **C#**: the namespace declared by `.cs` files; identity comes from the namespace header.
- **Go**: directory holding `.go` files; module boundary from `go.mod`.
- **Java**: directory under `src/main/java/` containing `.java` files.
- **Kotlin**: directory whose first `.kt` file declares a `package` header.
- **PHP**: directory whose `.php` files declare a `namespace` header.
- **Rust**: `mod.rs` / sibling-`.rs` tree under a `Cargo.toml`.
- **Scala**: directory whose first `.scala` file has a `package_clause`.

**Not supported by `coden architecture`:** C, C++, and Bash have no language-level package construct. Gradle and sbt multi-project layouts fall back to single-root audit.

</details>

### Code flagging

Eight detectors that surface smells and (optionally) write `[CODEN]` comments inline next to the offending code, or remove the issue entirely:

| Detector | Flag | Finds |
|---|---|---|
| Hotspots | `-H` | High coupling × complexity (refactoring targets) |
| Propagation | `-P` | Internal module coupling |
| Clones | `-C` | Duplicate / near-duplicate functions (semantic + syntactic) |
| Echo comments | `-E` | Comments that restate the code identifier |
| Dead code | `-D` | Functions with no callers |
| Tramp data | `-T` | Parameter groups passed through many functions |
| Sensitive values | `-S` | Hardcoded secrets, API keys, credentials |
| Magic constants | `-K` | Repeated literal values that should be named |

```bash
coden flag -E --dry-run          # Preview echo comments
coden flag -HPCETSK --backup     # Flag every issue type, with .coden-backup files
coden flag -S --replace --backup # Replace secrets with ***REDACTED***
coden flag clear                 # Remove all [CODEN] comments
```

`--dry-run` previews. `--backup` writes `.coden-backup` files next to anything it modifies. `--include-tests` reaches into test files (excluded by default).

<details>
<summary>Per-detector thresholds and modes</summary>

| Flag | Threshold option | Default | Notes |
|---|---|---|---|
| `-H` | `--risk-threshold` | 50 | Raw score = coupling × log(complexity) |
| `-P` | `--propagation-threshold` | 0.25 | 0–1 scale |
| `-C` | `--clone-threshold` | 0.95 | 0–1 scale, semantic + syntactic blend |
| `-E` | `--echo-threshold` | 0.80 | 0–1 scale |
| `-D` | `--dead-code-threshold` | 0.50 | 0–1 scale |
| `-T` | `--min-occurrences` | 3 | Integer; min functions a group must appear in |
| `-S` | `--sensitive-threshold` | 0.35 | 0–1 scale; classifier on entropy + known prefixes |
| `-K` | `--min-constant-occurrences` / `--min-constant-files` | 3 / 2 | Integer counts |

**Clone modes.** `-C` defaults to a weighted blend (semantic 0.65, syntactic 0.35). `--clone-semantic` uses MiniLM ONNX embeddings only, finding async/sync variants and renamed copies. `--clone-syntactic` uses line-by-line Jaccard for copy-paste duplicates. Weights adjustable via `--semantic-weight` / `--syntactic-weight`.

**Sensitive values.** By default Coden only scans source files. `--whitelist "*.env" "*.json" "*.yaml"` adds text-file scanning on top.

**Dead code.** Automatically skipped: dunder methods, runtime-called functions (`init`, `constructor`), test functions, and functions under 3 lines. `--remove-dead-code` deletes them entirely (destructive; use `--backup`).

**Tramp data.** `--min-group-size` (default 2) requires at least N parameters in a co-occurring group before reporting.

</details>

### Interactive agent

`coden -a` opens a chat against your codebase. Built on Pydantic-AI, local-LLM-first (Ollama / llama.cpp), with OpenAI as an option. A few features worth knowing about:

- **`/undo`**: resume from any prior tool call, with branches preserved. Useful when an agent step went sideways and you want to fork from before it.
- **`/run`**: execute an MCP tool directly, inspect the raw result, then continue the conversation with the agent.
- **`!cmd @@ query`**: pipe a shell command's output straight into the agent prompt.
- **`/config`**: interactive picker with inline editing.

```bash
coden -a                                            # Current directory, default model
coden -a --model ollama:gemma4:31b                  # Ollama
coden -a --model openai:gpt-4o                      # OpenAI (needs OPENAI_API_KEY)
coden -a --model my-model --base-url http://localhost:8000/v1  # vLLM, LM Studio, etc.
```

<details>
<summary>Slash commands</summary>

| Command | Aliases | What it does |
|---|---|---|
| `/help` | | Show commands |
| `/model [name]` | `/m` | Show or switch model |
| `/config` | | Interactive settings picker |
| `/tools` | `/t` | Enable/disable which MCP tools the agent sees |
| `/run` | `/r`, `/execute` | Run an MCP tool directly, see its raw output |
| `/study [topic]` | `/learn`, `/quiz` | Quiz mode |
| `/undo` | | Resume from any prior tool call |
| `/copy` | `/cp` | Copy last response to clipboard |
| `/cd [path]` | `/dir`, `/chdir` | Change directory |
| `/debug [on\|off]` | `/d` | Toggle debug logging |
| `/clear` | `/c` | Clear history |
| `/cache` / `/cache-clear` / `/cache-list` | `/cc`, `/cl` | Cache management |
| `/exit` | `/quit`, `/q` | Exit |

</details>

### MCP server

`coden serve` exposes Coden's tools over the Model Context Protocol, with stdio as the default transport and HTTP / SSE / streamable-HTTP as options. 25+ tools across code search, architecture audit, all eight detectors, file editing, DAP-based debugging, and git history.

For VS Code, drop this into `.vscode/mcp.json`:

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

<details>
<summary>Tool catalogue</summary>

**Code discovery**: `code_map`, `code_search`, `architecture`, `coupling_hotspots`, `find_hotspots`, `clone_detection`, `propagation_cost`, `detect_dead_code`, `detect_tramp_data`, `detect_sensitive_values`, `detect_magic_constants`.

**Graph analysis**: `change_impact_radius`, `architectural_bottlenecks`.

**Symbol lookup**: `find_identifier`, `trace_dependency_path`.

**Code inspection**: `read_source_range`, `read_source_ranges`, `git_history_context`, `code_evolution`.

**File editing**: `write_file`, `edit_file` (SEARCH/REPLACE or AST symbol targeting), `delete_file`, `undo_file_change`.

**Debugging**: `debug_stacktrace`, `debug_session`, `debug_action`, `debug_state`, `add_breakpoint`, `inject_trace`, `remove_injections`, `list_injections`.

**Python environment**: `check_python_virtual_env`, `get_python_package_path`.

**Dynamic tools** (off by default): `create_dynamic_tool`, `remove_dynamic_tool`. Enable with `CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS=1`.

</details>

---

## Daemon and caching

Repeated queries are faster with the daemon, which keeps indices in memory:

```bash
coden daemon start       # Background service
coden daemon status
coden daemon stop
```

Caches live in `~/.coden-retriever/`. The full call-graph cache and the simple-mode lite cache coexist in disjoint files, so you can run both `coden src` and `coden src --map-mode simple` on the same repo without conflict.

```bash
coden cache list         # List cached projects
coden cache clear        # Clear caches for the current directory
coden cache clear --all  # Clear everything
```

**Skip the daemon for a single invocation** with `--no-daemon` on `coden <root> -q ...`, `coden flag add`, or `coden flag clear`. The flag forces the in-process direct path without stopping a running daemon. For a process-wide switch, set `daemon.auto_start = false` in the config or export `CODEN_RETRIEVER_DAEMON_AUTO_START=false`; both CLI and MCP surfaces then honor it. Precedence is `--no-daemon` > env var > config > default (daemon on).

---

## Configuration

Settings live in `~/.coden-retriever/settings.json`. Inspect and tweak them inline:

```bash
coden config show
coden config set model.default ollama:gemma4:31b
coden config set agent.max_steps 20
coden config set daemon.port 8080
```

<details>
<summary>All settings and environment variables</summary>

**Top-level keys:** `model`, `agent`, `system_prompt`, `daemon`, `search`.

Common overrides:

```bash
coden config set model.default ollama:gemma4:31b
coden config set model.base_url http://localhost:11434/v1
coden config set model.generation.temperature 0.5
coden config set model.generation.api_key sk-...
coden config set agent.max_steps 20
coden config set agent.debug true
coden config set daemon.port 8080
coden config set search.default_tokens 8000
coden config set search.default_limit 50
```

**Environment variables** override the config file:

| Variable | Effect |
|---|---|
| `CODEN_RETRIEVER_MODEL` | Default model |
| `CODEN_RETRIEVER_BASE_URL` | Base URL |
| `CODEN_RETRIEVER_DAEMON_PORT` / `_HOST` | Daemon address |
| `CODEN_RETRIEVER_DAEMON_AUTO_START` | `false` / `0` / `no` disables the daemon for CLI and MCP; the in-process direct path is used instead. Default `true`. |
| `CODEN_RETRIEVER_MODEL_PATH` | Semantic model path |
| `CODEN_RETRIEVER_MCP_TIMEOUT` | MCP timeout |
| `CODEN_RETRIEVER_ENABLE_DYNAMIC_TOOLS` | Enable dynamic tools (`1`, `true`, `yes`) |
| `CODEN_RETRIEVER_DISABLED_TOOLS` | Comma-separated tool names |
| `CODEN_RETRIEVER_TEMPERATURE` / `_MAX_TOKENS` / `_TIMEOUT` | Generation knobs |

</details>

---

## Docker

All Docker assets live under `docker/`. Build from the repo root:

```bash
docker build -t coden-retriever:latest -f docker/Dockerfile .
```

The `coden-docker` wrapper runs a persistent container that holds the daemon:

```bash
./docker/coden-docker start .              # Start with current directory as workspace
./docker/coden-docker .                    # Ranked map
./docker/coden-docker . --query "auth"     # Search
./docker/coden-docker -a                   # Agent mode
./docker/coden-docker stop
```

For the MCP server as a long-lived service:

```bash
docker run -d -p 8000:8000 --name coden-mcp coden-retriever
# Available at http://localhost:8000/mcp ; health at /health
```

<details>
<summary>Compose and agent-with-host-Ollama</summary>

```bash
docker compose -f docker/docker-compose.yml up -d mcp-server
docker compose -f docker/docker-compose.yml logs -f mcp-server
docker compose -f docker/docker-compose.yml down
```

For agent mode using host Ollama, the container reaches `host.docker.internal`:

```bash
ollama serve                      # On host
./docker/coden-docker -a          # In Docker, then /model ollama:qwen2.5-coder
```

**Docker environment variables:** `CODEN_RETRIEVER_HOST` (bind, default `0.0.0.0`), `CODEN_RETRIEVER_PORT` (default `8000`), plus the standard `CODEN_RETRIEVER_DISABLED_TOOLS` / `_ENABLE_DYNAMIC_TOOLS`.

</details>

---

## Troubleshooting

When something is off (stale results, daemon stuck, config corrupted), the hammer is:

```bash
coden reset    # Clear all caches, stop daemon, reset config. Destructive.
```

<details>
<summary>Common issues</summary>

**Ollama context window too small.** Ollama defaults to 4096 tokens — too small for agent mode (the system prompt and tool schemas alone exceed it; you'll see truncated responses). Set `OLLAMA_CONTEXT_LENGTH=64000` before starting Ollama:

- Windows: `setx OLLAMA_CONTEXT_LENGTH 64000`, then restart the Ollama tray app.
- macOS / Linux: `export OLLAMA_CONTEXT_LENGTH=64000` before `ollama serve`.

Verify with `ollama ps`; the `CONTEXT` column should show `64000`. Some models pin `num_ctx` in their Modelfile and override this; in that case use a custom Modelfile with `PARAMETER num_ctx 64000`.

**Debug prerequisites.** `coden debug-availability` checks whether DAP debuggers are reachable for each supported language. Add a language to narrow it down (`coden debug-availability python`), or `--format json` for machine-readable output.

**Bash output looks thin.** Bash has no class or method concept, so `code_map` and `find_identifier` only surface function definitions and command calls for `.sh` sources. That's by design.

</details>
