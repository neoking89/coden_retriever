"""
Constant definitions for coden-retriever.

Contains invariant data sets used for filtering and classification.
These are separated from config.py which contains tuning parameters.
"""

# Functions that are utility sinks (high in-degree, low informational value)
UTILITY_NAMES: set[str] = {
    "print", "println", "printf", "eprint", "eprintln",
    "log", "debug", "info", "warn", "error", "trace", "fatal",
    "console", "assert", "panic", "exit", "len", "str", "int",
    "float", "bool", "list", "dict", "set", "tuple", "range",
    "open", "close", "read", "write", "append", "extend",
    "get", "set", "has", "delete", "remove", "pop", "push",
    "toString", "valueOf", "hasOwnProperty", "getElementById",
    "querySelector", "addEventListener", "setTimeout", "setInterval",
}

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

# Directories to skip during indexing
SKIP_DIRS: set[str] = {
    "venv", "env", ".venv", ".env",
    "node_modules", "bower_components",
    ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "target", "out", "bin", "obj",
    "vendor", "third_party", "external", "deps",
    ".idea", ".vscode", ".vs",
    "coverage", ".coverage", "htmlcov",
    ".tox", ".nox",
}

# Files to skip during indexing
SKIP_FILES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "Cargo.lock",
    "go.sum", "composer.lock", "Gemfile.lock",
}

# Important files to always consider
IMPORTANT_FILES: set[str] = {
    "README.md", "README.txt", "README.rst", "README",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "Cargo.toml", "go.mod", "pom.xml",
    "Makefile", "CMakeLists.txt", "Dockerfile",
    "main.py", "app.py", "index.js", "index.ts", "main.go",
    "main.rs", "Main.java", "Program.cs",
}
