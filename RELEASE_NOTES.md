# Release Notes - Version 2.0.0

**Release Date:** May 11, 2026

## Breaking Changes

- **Unified install — optional extras removed.** All features (semantic search, MCP server, interactive agent) are now bundled into the base package. There are no more `[semantic]`, `[mcp]`, `[agent]`, or `[all]` extras.
  - `pip install coden-retriever` now installs the complete tool.
  - Users with `pip install 'coden-retriever[semantic]'` (or other extras) should switch to plain `pip install -U coden-retriever`. The extras are gone and pip will warn/error.
- **Bundled semantic model.** A MiniLM ONNX embedding model now ships inside the wheel under `coden_retriever/models/embeddings/minilm_onnx/`. Semantic search, clone detection, and echo comments work out of the box with no extra download step. The embedding backend switched from `model2vec` to `onnxruntime`.
- **New required core dependencies:** `onnxruntime`, `scipy`, `pathspec`, `fastmcp>=3.0`, `pydantic-ai-slim[mcp,openai]`. Install size grows accordingly — this is the trade-off for the single-install experience.

## New Features

- **Magic Constant Detection (`-K`)**: Find repeated literal values (numbers, strings) scattered across files that should be named constants.
  - `coden /path -K` — scan for magic constants
  - `--min-constant-occurrences` (default: 3) — minimum number of occurrences to flag
  - `--min-constant-files` (default: 2) — minimum distinct files the literal appears in
  - `--constant-types` — filter by literal type (numeric, string, all)
  - Supported in flag mode: `coden flag -K --dry-run` / `coden flag -K --backup`

- **`debug-availability` subcommand**: Check whether a language has a working debug adapter on the current machine.
  - `coden debug-availability` — list adapters across all supported languages
  - `coden debug-availability python` — check one language
  - `coden debug-availability cpp --format json` — machine-readable output
  - improved debugging support for all supported languages

## Improvements

- AST-based constant extraction (numeric + string literals, including default parameter values) shared between magic-constant and sensitive-value detection — fewer false positives from keywords and structural tokens.
- Semantic embeddings now load from a bundled ONNX model — first run no longer requires network access.

## Migration Guide

Coming from 1.4.x:

```bash
pip install -U coden-retriever
```

If previously installed an extra, drop the bracket suffix — everything is now in the base package. No CLI flags were removed; existing scripts keep working.

---

# Release Notes - Version 1.4.0

**Release Date:** February 15, 2026

## New Features

- **Whitelist Scanning (`--whitelist`)**: Scan text files (`.env`, `.json`, `.yaml`, `.toml`, etc.) for sensitive values alongside source code
  - `coden /path -S --whitelist "*.env" "*.json"`

## Improvements

- Improved sensitive value detection accuracy

---

# Release Notes - Version 1.3.1

**Release Date:** February 13, 2026

## Bug Fix

- **detect_sensitive_values** MCP-tool had a bug, which caused indefinite hanging on execution. 
This bug is now fixed.

---

# Release Notes - Version 1.3.0

**Release Date:** February 11, 2026

## New Features

- **Tramp Data Detection (`-T`)**: Identify parameter groups that travel together across many functions, revealing opportunities to refactor into configuration objects
  - Uses frequent pair mining and greedy group expansion to find co-occurring parameters
  - `--min-occurrences` to set minimum function count (default: 3)
  - `--min-group-size` to set minimum parameters per group (default: 2)
  - Supports flag mode: `coden flag -T --dry-run` / `coden flag -T --backup`

- **Sensitive Value Detection (`-S`)**: Find hardcoded secrets, API keys, credentials, and other sensitive strings using an ML classifier (LogisticRegression) trained on entropy, character patterns, and known secret prefixes
  - `--sensitive-threshold` to adjust confidence (default: 0.35, range: 0-1)
  - Replace mode: `coden flag -S --replace --backup` replaces secrets with `***REDACTED***`
  - Custom placeholder: `coden flag -S --replace "HIDDEN" --backup`
  - Supports flag mode and dry-run previews

- **Bash Language Support**: Bash/shell scripts are now parsed and analyzed alongside other supported languages

## MCP Server

- New tools: `detect_tramp_data` and `detect_sensitive_values` available via `coden serve`

## Dependency Changes

- `scikit-learn` and `numpy` are now core dependencies (required for sensitive value classifier)
- `numpy` removed from `[semantic]` extras (no longer duplicated)

## Other Improvements

- Combined flag shorthand updated from `-HPCED` to `-HPCETS` to include tramp data and sensitive values
- Propagation analysis (`-P`) now respects the `--include-tests` flag
- Internal refactoring of flag command validation and formatting

---

# Release Notes - Version 1.2.0

**Release Date:** January 18, 2026

## New Features

- **Dead Code Detection (`-D`)**: Find potentially unused functions and methods in your codebase
  - Use `--dead-code-threshold` to adjust confidence level (default: 0.5)
  - In flag mode: use `--remove-dead-code` to delete dead code instead of flagging

## Installation Changes

- **Modular Installation**: Base install is now significantly lighter. Heavy dependencies are now optional:
  - `pip install coden-retriever` — Base install (search, hotspots, propagation analysis)
  - `pip install 'coden-retriever[semantic]'` — Adds semantic search, clone detection, echo comments
  - `pip install 'coden-retriever[mcp]'` — Adds MCP server (`coden serve`)
  - `pip install 'coden-retriever[agent]'` — Adds interactive agent (`coden agent`)
  - `pip install 'coden-retriever[all]'` — All features

---

# Release Notes - Version 1.1.0

**Release Date:** January 13, 2026

## Bug Fixes

- Fixed bug where LLM would format tools incorrectly in `--agent` mode
- Other minor bugfixes

## New Features

- **Clone Detection (`-C`)**: Detect duplicate code patterns in your codebase
- **Echo Comment Detection (`-E`)**: Find comments that simply repeat what the code does
- Both features support flag modes for flexible integration

## Other Improvements

- Multi-editor support via `CODEN_EDITOR` environment variable (VSCode, PyCharm, IntelliJ, Sublime), allowing users to click directly on code as hyperlink from CLI

