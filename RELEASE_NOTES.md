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

