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

