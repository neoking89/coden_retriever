#!/usr/bin/env python
"""Entry point for MCP server and CLI.

This module provides a convenient entry point for running the coden_retriever
package as a standalone script. It delegates to the main entry point in
the package's __main__ module.

Usage:
    python coden.py [options]

Example:
    python coden.py --help
"""
import sys

from coden_retriever.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
