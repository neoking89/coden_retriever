"""
Dynamic Tools Management.

Handles creation, removal, and registration of user-created dynamic MCP tools.
Tools are stored in ~/.coden-retriever/dynamic_tools.py for cross-platform persistence.
"""
import ast
import asyncio
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import Field

logger = logging.getLogger(__name__)

# Markers for dynamic tools section in the generated file
_MARKER_START = "# --- DYNAMIC TOOLS START ---"
_MARKER_END = "# --- DYNAMIC TOOLS END ---"


class DynamicToolsManager:
    """Manages dynamic MCP tools with thread-safe operations and caching."""

    def __init__(self):
        """Initialize the Dynamic Tools Manager."""
        # MCP server instance for dynamic tool registration
        self._mcp_server_instance = None
        # Asyncio lock to serialize dynamic tool writes and prevent file corruption
        self._dynamic_tools_write_lock = asyncio.Lock()
        # Cached dynamic tools module
        self._dynamic_tools_module = None
        # Reserved tool names that cannot be used for dynamic tools
        self._reserved_tool_names = {
            "code_search",
            "code_map",
            "find_identifier",
            "trace_dependency_path",
            "check_python_virtual_env",
            "get_python_package_path",
            "create_dynamic_tool",
            "remove_dynamic_tool",
            "register_tools",
        }

    @property
    def reserved_tool_names(self) -> set[str]:
        """Get the set of reserved tool names."""
        return self._reserved_tool_names.copy()

    def set_mcp_server_instance(self, mcp) -> None:
        """Set the MCP server instance for dynamic tool registration."""
        self._mcp_server_instance = mcp

    def get_mcp_server_instance(self):
        """Get the current MCP server instance."""
        return self._mcp_server_instance


# Global singleton instance
_manager = DynamicToolsManager()


def get_dynamic_tools_dir() -> Path:
    """Get the cross-platform directory for dynamic tools.

    Returns ~/.coden-retriever/ on all platforms (Linux, Windows, macOS).
    Creates the directory if it doesn't exist.
    """
    home = Path.home()
    dynamic_tools_dir = home / ".coden-retriever"
    dynamic_tools_dir.mkdir(parents=True, exist_ok=True)
    return dynamic_tools_dir


def get_dynamic_tools_file() -> Path:
    """Get the path to the dynamic_tools.py file."""
    return get_dynamic_tools_dir() / "dynamic_tools.py"


def _ensure_dynamic_tools_file() -> None:
    """Ensure dynamic_tools.py exists with proper structure."""
    dynamic_tools_file = get_dynamic_tools_file()

    if not dynamic_tools_file.exists():
        template = f'''"""
Dynamic MCP Tools.

This file contains dynamically created MCP tools. Tools are automatically
created and removed via the create_dynamic_tool and remove_dynamic_tool functions.

This file is located in ~/.coden-retriever/ for cross-platform compatibility.
"""
from __future__ import annotations

import inspect
import sys
from collections.abc import Callable

{_MARKER_START}
# Dynamic tools will be added below this marker

{_MARKER_END}


def get_dynamic_tool_functions() -> list[Callable[..., object]]:
    """Return all dynamic tool functions defined in this module.

    A dynamic tool function is any function defined in this module that has
    the attribute `_is_dynamic_tool = True`.
    """
    current_module = sys.modules[__name__]
    tool_functions: list[Callable[..., object]] = []

    for name, obj in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("_") or name == "get_dynamic_tool_functions":
            continue
        if getattr(obj, "_is_dynamic_tool", False):
            tool_functions.append(obj)

    return tool_functions
'''
        dynamic_tools_file.write_text(template, encoding="utf-8")
        logger.info(f"Created dynamic tools file at {dynamic_tools_file}")


def _load_dynamic_tools_module():
    """Load the dynamic_tools module from ~/.coden-retriever/."""
    _ensure_dynamic_tools_file()
    dynamic_tools_file = get_dynamic_tools_file()

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("dynamic_tools", dynamic_tools_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dynamic tools from {dynamic_tools_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_tools"] = module
    spec.loader.exec_module(module)
    _manager._dynamic_tools_module = module
    return module


def _get_dynamic_tools_module():
    """Get or load the dynamic tools module."""
    if _manager._dynamic_tools_module is None:
        return _load_dynamic_tools_module()
    return _manager._dynamic_tools_module


def _extract_and_validate_tool(code_str: str) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Parse and validate a single top-level function definition (sync or async)."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Code must contain exactly one top-level function definition. Import statements must be placed inside the function-body.")

    function_def: ast.FunctionDef | ast.AsyncFunctionDef = tree.body[0]

    if function_def.decorator_list:
        raise ValueError("Tool function must not use decorators.")

    docstring = ast.get_docstring(function_def)
    if not docstring:
        raise ValueError("Tool function must include a docstring.")

    # Require full type hints (args + return)
    all_args = list(function_def.args.posonlyargs) + list(function_def.args.args) + list(function_def.args.kwonlyargs)
    if any(arg.annotation is None for arg in all_args):
        raise ValueError("All tool parameters must have type hints.")
    if function_def.returns is None:
        raise ValueError("Tool function must have a return type hint.")

    return function_def.name, function_def


def _insert_into_dynamic_tools_file(dynamic_tools_path: Path, tool_name: str, code_str: str) -> None:
    """Insert tool code between the START/END markers in dynamic_tools.py."""
    content = dynamic_tools_path.read_text(encoding="utf-8")

    start_idx = content.find(_MARKER_START)
    end_idx = content.find(_MARKER_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise RuntimeError("dynamic_tools.py markers not found or malformed.")

    # Check for duplicates using AST parsing
    block_start = start_idx + len(_MARKER_START)
    block_content = content[block_start:end_idx]
    if block_content.strip():
        try:
            tree = ast.parse(block_content)
            existing_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            if tool_name in existing_functions:
                raise ValueError(f"Tool '{tool_name}' already exists.")
        except SyntaxError:
            pass  # If block is unparseable, allow insertion

    # Insert just before END marker
    insertion_point = end_idx
    normalized_code = code_str.strip("\n") + "\n"
    tool_block = f"\n\n{normalized_code}\n{tool_name}._is_dynamic_tool = True\n"
    new_content = content[:insertion_point] + tool_block + content[insertion_point:]
    dynamic_tools_path.write_text(new_content, encoding="utf-8")


def _remove_from_dynamic_tools_file(dynamic_tools_path: Path, tool_name: str) -> None:
    """Remove a tool definition from dynamic_tools.py using AST parsing."""
    content = dynamic_tools_path.read_text(encoding="utf-8")

    start_idx = content.find(_MARKER_START)
    end_idx = content.find(_MARKER_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise RuntimeError("dynamic_tools.py markers not found or malformed.")

    # Check if tool exists
    dynamic_tools = _get_dynamic_tools_module()
    existing = {f.__name__ for f in dynamic_tools.get_dynamic_tool_functions()}
    if tool_name not in existing:
        raise ValueError(f"Tool '{tool_name}' not found in dynamic tools.")

    # Parse block and remove target function + its marker
    block_start = start_idx + len(_MARKER_START)
    block_content = content[block_start:end_idx]

    try:
        tree = ast.parse(block_content)
    except SyntaxError as e:
        raise RuntimeError(f"Failed to parse dynamic tools block: {e}") from e

    # Filter out the target function and its marker statement
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == tool_name:
            continue
        # Skip marker: tool_name._is_dynamic_tool = True
        if isinstance(node, ast.Assign):
            if (len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Attribute) and
                isinstance(node.targets[0].value, ast.Name) and
                node.targets[0].value.id == tool_name and
                node.targets[0].attr == "_is_dynamic_tool"):
                continue
        new_body.append(node)

    tree.body = new_body
    new_block_content = ast.unparse(tree) if tree.body else ""

    # Preserve some spacing
    if new_block_content:
        new_block_content = "\n" + new_block_content + "\n"

    new_content = content[:block_start] + new_block_content + content[end_idx:]
    dynamic_tools_path.write_text(new_content, encoding="utf-8")


def _extract_imports_from_file(source_content: str) -> dict[str, str]:
    """Extract top-level imports from a Python file.

    Only extracts module-level imports, not imports inside functions.
    Returns a dict mapping imported names to their import statements.
    E.g., {"Path": "from pathlib import Path", "os": "import os"}
    """
    try:
        tree = ast.parse(source_content)
    except SyntaxError:
        return {}

    imports = {}
    # Only iterate over top-level statements, not nested ones
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                # For "import x.y.z", use the first part as the usable name
                top_name = name.split(".")[0]
                imports[top_name] = ast.unparse(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == "*":
                        # Can't reliably handle star imports
                        continue
                    imports[name] = ast.unparse(node)
    return imports


def _find_used_names_in_function(func_code: str) -> set[str]:
    """Find all names used (referenced) in a function's body.

    Returns a set of all Name nodes that are loaded (read, not assigned).
    """
    try:
        tree = ast.parse(func_code)
    except SyntaxError:
        return set()

    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For attribute access like "os.path", get the root name "os"
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                used_names.add(current.id)
    return used_names


def _inject_imports_into_function(func_code: str, import_statements: list[str]) -> str:
    """Inject import statements at the beginning of a function body.

    Places imports right after the function signature and docstring.
    """
    if not import_statements:
        return func_code

    try:
        tree = ast.parse(func_code)
    except SyntaxError:
        return func_code

    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return func_code

    func_def = tree.body[0]
    func_lines = func_code.splitlines()

    # Detect indentation from the first line of function body
    body_indent = "    "  # Default 4 spaces
    if func_def.body:
        first_body_line_no = func_def.body[0].lineno - 1
        if first_body_line_no < len(func_lines):
            line = func_lines[first_body_line_no]
            stripped = line.lstrip()
            if stripped:
                body_indent = line[: len(line) - len(stripped)]

    # Check if there's a docstring
    docstring_end_line = 0
    if (func_def.body and isinstance(func_def.body[0], ast.Expr) and
            isinstance(func_def.body[0].value, ast.Constant) and
            isinstance(func_def.body[0].value.value, str)):
        docstring_end_line = func_def.body[0].end_lineno or 0

    # Find insertion point (after signature and docstring)
    if docstring_end_line > 0:
        insert_after_line = docstring_end_line
    else:
        # Insert after function signature
        insert_after_line = func_def.lineno
        # Handle multi-line signatures
        for i, line in enumerate(func_lines[func_def.lineno - 1:], start=func_def.lineno):
            if line.rstrip().endswith(":"):
                insert_after_line = i
                break

    # Build import block
    import_block = "\n".join(body_indent + stmt for stmt in import_statements)

    # Insert imports
    result_lines = (
        func_lines[:insert_after_line] +
        [import_block] +
        func_lines[insert_after_line:]
    )
    return "\n".join(result_lines)


async def _resolve_symbol_to_code(
    root_directory: str,
    identifier: str,
) -> str:
    """Resolve a symbol identifier to its source code.

    Automatically detects external imports used by the function and injects
    them into the function body (since module-level imports aren't allowed
    in dynamic tools).

    Args:
        root_directory: Absolute path to project root
        identifier: Symbol in format "relative/path.py::symbol_name"

    Returns:
        The source code of the identified function with imports injected

    Raises:
        ValueError: If identifier format is invalid or symbol not found
    """
    from coden_retriever.parsers.tree_sitter_parser import RepoParser

    # Parse identifier
    parts = identifier.split("::")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid identifier format: '{identifier}'. "
            "Expected 'path/to/file.py::function_name'."
        )

    rel_path, symbol_name = parts
    file_path = Path(root_directory) / rel_path

    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    # Read file content
    source_content = file_path.read_text(encoding="utf-8")

    # Parse file to find entity
    parser = RepoParser()
    entities, _ = await asyncio.to_thread(parser.parse_file, str(file_path), source_content)

    # Find matching entity (function only - classes/methods not supported for dynamic tools)
    matching = [e for e in entities if e.name == symbol_name and e.entity_type == "function"]

    if not matching:
        available = [e.name for e in entities if e.entity_type == "function"]
        raise ValueError(
            f"Function '{symbol_name}' not found in {rel_path}. "
            f"Available functions: {available[:10]}"
        )

    if len(matching) > 1:
        # Multiple matches - take the first one
        logger.warning(f"Multiple functions named '{symbol_name}' found, using first match.")

    entity = matching[0]

    # Extract source code
    lines = source_content.splitlines()
    code_lines = lines[entity.line_start - 1 : entity.line_end]
    func_code = "\n".join(code_lines)

    # Auto-detect and inject required imports
    file_imports = _extract_imports_from_file(source_content)
    used_names = _find_used_names_in_function(func_code)

    # Find which imports are actually used by the function
    required_imports = []
    for name in used_names:
        if name in file_imports:
            import_stmt = file_imports[name]
            if import_stmt not in required_imports:
                required_imports.append(import_stmt)

    # Inject imports into function body if needed
    if required_imports:
        func_code = _inject_imports_into_function(func_code, required_imports)
        logger.info(f"Injected {len(required_imports)} import(s) into function '{symbol_name}'")

    return func_code


def _clear_dynamic_tools_file(dynamic_tools_path: Path) -> int:
    """Clear all tools from dynamic_tools.py, return count of tools removed."""
    content = dynamic_tools_path.read_text(encoding="utf-8")

    start_idx = content.find(_MARKER_START)
    end_idx = content.find(_MARKER_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise RuntimeError("dynamic_tools.py markers not found or malformed.")

    # Count existing tools
    dynamic_tools = _get_dynamic_tools_module()
    tool_count = len(dynamic_tools.get_dynamic_tool_functions())

    # Clear the block between markers
    block_start = start_idx + len(_MARKER_START)
    new_block_content = "\n# Dynamic tools will be added below this marker\n\n"
    new_content = content[:block_start] + new_block_content + content[end_idx:]
    dynamic_tools_path.write_text(new_content, encoding="utf-8")

    return tool_count


async def create_dynamic_tool(
    code: Annotated[str | None, Field(description="The Python code for the tool. Must include the function definition with type hints and docstring.")] = None,
    identifier: Annotated[str | None, Field(description="Symbol identifier in format 'relative/path.py::function_name'. Use this instead of 'code' to create a tool from an existing function in the codebase.")] = None,
    root_directory: Annotated[str | None, Field(description="Absolute path to project root. Required when using 'identifier'.")] = None,
) -> dict[str, Any]:
    """Creates a new tool dynamically.
    The code (in python) must define a function. The function name is inferred from the code.
    The function will be added to dynamic_tools.py and registered with the server.

    WHEN TO USE:
    - When you need a utility function that is missing from the standard tools
    - When you want to create a reusable helper for complex calculations or data processing
    - When you need to persist a custom tool across server restarts
    - When you want to turn an existing function from the codebase into a tool

    IMPORTANT:
    - Module-level imports are not allowed; place import statements inside the function body.
    - Provide either 'code' OR ('identifier' + 'root_directory'), not both.

    EXAMPLE USAGE (JSON):

    **Example 1**: Create from full code (Fibonacci):
    ```json
    {
        "code": "def calculate_fibonacci(n: int) -> int:\\n    \\"\\"\\"Calculates the nth Fibonacci number.\\"\\"\\"\\n    if n <= 1:\\n        return n\\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
    }
    ```

    **Example 2**: Create from full code (list files):
    ```json
    {
        "code": "def list_files_in_directory(directory: str) -> list[str]:\\n    \\"\\"\\"Lists all files in the given directory.\\"\\"\\"\\n    import os\\n    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]"
    }
    ```

    **Example 3**: Create from existing function in codebase:
    ```json
    {
        "identifier": "src/utils/helpers.py::format_date",
        "root_directory": "/path/to/project"
    }
    ```
    """
    try:
        # Validate input - must have exactly one of: code OR (identifier + root_directory)
        if code and identifier:
            return {"error": "Provide either 'code' or 'identifier', not both."}

        if identifier:
            if not root_directory:
                return {"error": "'root_directory' is required when using 'identifier'."}
            if "::" not in identifier:
                return {"error": "Invalid identifier format. Use 'path/to/file.py::function_name'."}
            # Resolve identifier to code
            try:
                code = await _resolve_symbol_to_code(root_directory, identifier)
            except ValueError as e:
                return {"error": str(e)}

        if not code:
            return {"error": "Provide either 'code' or 'identifier' with 'root_directory'."}

        tool_name, _ = _extract_and_validate_tool(code)
        if tool_name in _manager.reserved_tool_names:
            return {"error": f"Tool name '{tool_name}' is reserved."}

        dynamic_tools_path = get_dynamic_tools_file()

        async with _manager._dynamic_tools_write_lock:
            await asyncio.to_thread(
                _insert_into_dynamic_tools_file,
                dynamic_tools_path,
                tool_name,
                code,
            )

            # Reload the dynamic tools module
            importlib.invalidate_caches()
            dynamic_tools = _load_dynamic_tools_module()

        func = getattr(dynamic_tools, tool_name, None)
        if func is None or not callable(func) or not getattr(func, "_is_dynamic_tool", False):
            return {"error": f"Tool '{tool_name}' was written but could not be loaded."}

        mcp_instance = _manager.get_mcp_server_instance()
        if mcp_instance is None:
            return {
                "status": "saved",
                "name": tool_name,
                "message": f"Tool saved to {dynamic_tools_path}; server instance not available for immediate registration.",
            }

        mcp_instance.tool()(func)
        return {
            "status": "success",
            "name": tool_name,
            "message": f"Tool '{tool_name}' created and registered. Saved to {dynamic_tools_path}",
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Failed to create dynamic tool: {e}")
        return {"error": f"Failed to create dynamic tool: {e}"}


async def remove_dynamic_tool(
    tool_name: Annotated[
        str | None,
        Field(description="Name of specific tool to remove. Required unless remove_all=True.")
    ] = None,
    remove_all: Annotated[
        bool,
        Field(description="Set to True to remove ALL dynamic tools at once")
    ] = False,
) -> dict[str, Any]:
    """Remove dynamically created tool(s).

    Removes tool definition(s) from ~/.coden-retriever/dynamic_tools.py and reloads the module.
    Note: Tools cannot be unregistered from the current MCP session, but will
    not be loaded on the next server restart.

    WHEN TO USE:
    - tool_name: to remove a specific tool created with create_dynamic_tool
    - remove_all=True: to clear all dynamic tools and start fresh

    EXAMPLE USAGE (JSON):
    Remove specific: {"tool_name": "calculate_fibonacci"}
    Remove all: {"remove_all": true}
    """
    try:
        if not tool_name and not remove_all:
            return {"error": "Specify tool_name or set remove_all=True"}

        dynamic_tools_path = get_dynamic_tools_file()

        if remove_all:
            async with _manager._dynamic_tools_write_lock:
                tool_count = await asyncio.to_thread(
                    _clear_dynamic_tools_file,
                    dynamic_tools_path,
                )
                importlib.invalidate_caches()
                _load_dynamic_tools_module()

            return {
                "status": "success",
                "count": tool_count,
                "message": f"Cleared {tool_count} dynamic tool(s). They remain registered in current session but won't load on restart.",
            }

        # Remove specific tool - tool_name is guaranteed set (checked at line 614)
        name = cast(str, tool_name)
        if name in _manager.reserved_tool_names:
            return {"error": f"Cannot remove built-in tool '{name}'."}

        async with _manager._dynamic_tools_write_lock:
            await asyncio.to_thread(
                _remove_from_dynamic_tools_file,
                dynamic_tools_path,
                name,
            )
            importlib.invalidate_caches()
            _load_dynamic_tools_module()

        return {
            "status": "success",
            "name": name,
            "message": f"Tool '{name}' removed. It remains registered in current session but won't load on restart.",
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Failed to remove dynamic tool: {e}")
        return {"error": f"Failed to remove dynamic tool: {e}"}


def set_mcp_server_instance(mcp) -> None:
    """Set the MCP server instance for dynamic tool registration."""
    _manager.set_mcp_server_instance(mcp)


def load_and_register_dynamic_tools(mcp) -> None:
    """Load and register all dynamic tools on the given MCP instance."""
    try:
        dynamic_tools = _get_dynamic_tools_module()
        for tool in dynamic_tools.get_dynamic_tool_functions():
            try:
                mcp.tool()(tool)
                logger.info(f"Registered dynamic tool: {tool.__name__}")
            except Exception as e:
                logger.error(f"Failed to register dynamic tool {tool.__name__}: {e}")
    except Exception as e:
        logger.warning(f"Could not load dynamic tools: {e}")


def register_dynamic_tools(mcp, disabled_tools: set[str] | None = None) -> None:
    """Register dynamic tools MCP tools on the given FastMCP instance.

    Registers the following tools:
    - create_dynamic_tool: Create custom MCP tools at runtime
    - remove_dynamic_tool: Remove dynamic tool(s) by name or all at once

    Also loads and registers any user-created dynamic tools from ~/.coden-retriever/dynamic_tools.py

    Args:
        mcp: FastMCP instance to register tools on.
        disabled_tools: Optional set of tool names to skip registration.
    """
    disabled = disabled_tools or set()

    # Set server instance for dynamic tools module
    set_mcp_server_instance(mcp)

    # Register dynamic tool management tools
    for func in [
        create_dynamic_tool,
        remove_dynamic_tool,
    ]:
        if func.__name__ not in disabled:
            mcp.tool()(func)

    # Register user-created dynamic tools from ~/.coden-retriever/
    load_and_register_dynamic_tools(mcp)
