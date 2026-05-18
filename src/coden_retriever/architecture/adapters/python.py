"""Python adapter: tree-sitter only, no `ast`, no regex.

A "package" here is a directory directly under the *effective* audit root
that contains `__init__.py`. The effective root may differ from the
user-supplied `root` argument: if `root` itself has no `__init__.py` and
contains exactly one direct child that does, the adapter auto-descends into
that child. This makes `coden architecture src/` work the same as
`coden architecture src/<pkg>/` for the standard `src/`-layout repo.

Files at the effective root itself (e.g. `__main__.py`, `config_loader.py`)
are analyzed but get `package=None` — included in n_files/total_loc/
oversized/in-function-imports totals, excluded from the package-level graph.

The outer skeleton (file walk, layout cache, parser caching, LOC counting,
`analyze_file` / `package_public_facade` orchestration) lives in
`BaseTreeSitterAdapter`. This module owns only what is genuinely Python-
specific.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.protocol import (
    FileAnalysis,
    ImportRef,
    InFunctionImport,
)
from ._base import BaseTreeSitterAdapter, _node_text

ANON_FUNCTION_NAME = "<anonymous>"
# Why: lambdas don't have a `name` field in tree-sitter — but we still want
# to attribute imports inside them. Used as a sentinel in InFunctionImport.


class PythonAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for Python."""

    LANGUAGE = "python"
    EXTENSIONS = frozenset({".py", ".pyw"})
    INDEX_BASENAMES = ("__init__.py",)
    LINE_COMMENT_PREFIXES = ("#",)

    def _compute_effective_root(self, root: Path) -> Path:
        """Auto-descend `src/<pkg>/` layouts: drop one wrapper level if it has no `__init__.py`."""
        if (root / "__init__.py").exists():
            return root
        try:
            children = list(root.iterdir())
        except OSError:
            return root
        pkg_children = [
            c for c in children
            if c.is_dir() and (c / "__init__.py").exists()
        ]
        if len(pkg_children) == 1:
            return pkg_children[0]
        return root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Direct subdirectories of `effective_root` that contain `__init__.py`, sorted by name.

        For Python the package "name" is always the directory basename.
        """
        try:
            sub = list(effective_root.iterdir())
        except OSError:
            return ()
        pairs = [
            (c, c.name) for c in sub
            if c.is_dir() and (c / "__init__.py").exists()
        ]
        return tuple(sorted(pairs, key=lambda p: p[1]))

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Walk the parsed module, return (stmt_count, imports, in_function_imports)."""
        audit_root_name = effective_root.name
        file_module_parts = _file_module_parts(file, effective_root, audit_root_name)

        top_imports: list[ImportRef] = []
        in_func_imports: list[InFunctionImport] = []
        statement_count = _ImportWalker(
            source_bytes=source_bytes,
            file_module_parts=file_module_parts,
            project_packages=self._cache_package_names,
            audit_root_name=audit_root_name,
        ).walk(tree.root_node, top_imports, in_func_imports)

        return statement_count, tuple(top_imports), tuple(in_func_imports)

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """`__all__` if declared, else non-underscore top-level defs+classes.

        `__all__` membership wins over discovery: if `__all__` names a symbol
        with no matching `def`/`class` at top level, it's still public with
        `param-count = 0` (matches the prior tuple-flattening behavior).
        """
        root_node = tree.root_node
        all_list = _extract_all(root_node, source_bytes)
        public_functions = _extract_top_level_public_functions(root_node, source_bytes)
        public_classes = _extract_top_level_public_classes(root_node, source_bytes)

        if all_list is not None:
            return {name: public_functions.get(name, 0) for name in all_list}
        merged: dict[str, int] = dict(public_functions)
        for name in public_classes:
            merged.setdefault(name, 0)
        return merged


def _file_module_parts(file: Path, root: Path, audit_root_name: str) -> tuple[str, ...]:
    """Return the dotted-path parts of the package CONTAINING `file`."""
    try:
        rel = file.relative_to(root)
    except ValueError:
        return ()
    return (audit_root_name, *rel.parts[:-1])


class _ImportWalker:
    """Walks a tree-sitter Python tree once, collecting imports + in-function imports."""

    def __init__(
        self,
        source_bytes: bytes,
        file_module_parts: tuple[str, ...],
        project_packages: frozenset[str],
        audit_root_name: str,
    ) -> None:
        self._source = source_bytes
        self._file_module_parts = file_module_parts
        self._project_packages = project_packages
        self._audit_root_name = audit_root_name
        self._function_stack: list[str] = []
        self._statement_count = 0

    def walk(
        self,
        root_node: Any,
        top_imports: list[ImportRef],
        in_func_imports: list[InFunctionImport],
    ) -> int:
        """Recursive DFS. Returns the count of top-level import STATEMENTS."""
        self._visit(root_node, top_imports, in_func_imports)
        return self._statement_count

    def _visit(
        self,
        node: Any,
        top_imports: list[ImportRef],
        in_func_imports: list[InFunctionImport],
    ) -> None:
        nt = node.type
        if nt in ("function_definition", "async_function_definition", "lambda"):
            fn_name = self._function_name(node)
            self._function_stack.append(fn_name)
            try:
                for child in node.children:
                    self._visit(child, top_imports, in_func_imports)
            finally:
                self._function_stack.pop()
            return

        if nt == "import_statement":
            self._handle_import(node, top_imports, in_func_imports)
            return

        if nt == "import_from_statement":
            self._handle_import_from(node, top_imports, in_func_imports)
            return

        for child in node.children:
            self._visit(child, top_imports, in_func_imports)

    def _function_name(self, node: Any) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return ANON_FUNCTION_NAME
        return _node_text(name_node, self._source)

    def _handle_import(
        self,
        node: Any,
        top_imports: list[ImportRef],
        in_func_imports: list[InFunctionImport],
    ) -> None:
        """`import a, b.c [as d]` — one statement, 1+ targets."""
        in_function = bool(self._function_stack)
        if not in_function:
            self._statement_count += 1
        line = node.start_point[0] + 1
        for module_text in _iter_import_targets(node, self._source):
            target_pkg = self._resolve(level=0, module_name=module_text)
            if in_function:
                in_func_imports.append(InFunctionImport(
                    line=line,
                    function=self._function_stack[-1],
                    import_text=_node_text(node, self._source).strip(),
                    target_package=target_pkg,
                ))
            else:
                top_imports.append(ImportRef(
                    target_module=module_text,
                    target_package=target_pkg,
                    line=line,
                ))

    def _handle_import_from(
        self,
        node: Any,
        top_imports: list[ImportRef],
        in_func_imports: list[InFunctionImport],
    ) -> None:
        """`from .x.y import a, b` — one statement, one source module."""
        in_function = bool(self._function_stack)
        if not in_function:
            self._statement_count += 1
        line = node.start_point[0] + 1
        module_node = node.child_by_field_name("module_name")
        level, module_text = _parse_from_module(module_node, self._source)
        target_pkg = self._resolve(level=level, module_name=module_text)
        display_module = ("." * level) + (module_text or "")
        if in_function:
            in_func_imports.append(InFunctionImport(
                line=line,
                function=self._function_stack[-1],
                import_text=_node_text(node, self._source).strip(),
                target_package=target_pkg,
            ))
        else:
            top_imports.append(ImportRef(
                target_module=display_module,
                target_package=target_pkg,
                line=line,
            ))

    def _resolve(self, level: int, module_name: str | None) -> str | None:
        """Resolve `from <level-dots><module_name> import ...` to a project package or None."""
        if level == 0:
            if not module_name:
                return None
            parts = module_name.split(".")
            if (
                parts[0] == self._audit_root_name
                and len(parts) >= 2
                and parts[1] in self._project_packages
            ):
                return parts[1]
            if parts[0] in self._project_packages:
                return parts[0]
            return None
        # Relative import. Drop (level - 1) segments from file_module_parts, append module.
        if level - 1 >= len(self._file_module_parts):
            return None
        base = self._file_module_parts[: len(self._file_module_parts) - (level - 1)]
        suffix = tuple(module_name.split(".")) if module_name else ()
        full = base + suffix
        if (
            len(full) >= 2
            and full[0] == self._audit_root_name
            and full[1] in self._project_packages
        ):
            return full[1]
        return None


def _iter_import_targets(node: Any, source: bytes) -> list[str]:
    """Extract the dotted names from an `import_statement` node."""
    targets: list[str] = []
    for child in node.named_children:
        if child.type == "dotted_name":
            targets.append(_node_text(child, source))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            if name_node is not None and name_node.type == "dotted_name":
                targets.append(_node_text(name_node, source))
    return targets


def _parse_from_module(node: Any | None, source: bytes) -> tuple[int, str | None]:
    """Parse the `module_name` field of an `import_from_statement` → (level, dotted_name)."""
    if node is None:
        return 0, None
    if node.type == "dotted_name":
        return 0, _node_text(node, source)
    if node.type == "relative_import":
        level = 0
        module_text: str | None = None
        for child in node.children:
            if child.type == "import_prefix":
                level = len(_node_text(child, source).strip())
            elif child.type == "dotted_name":
                module_text = _node_text(child, source)
        return level, module_text
    return 0, None


def _extract_all(root_node: Any, source: bytes) -> list[str] | None:
    """Return the list of strings assigned to `__all__` at module top, or None."""
    for child in root_node.children:
        if child.type != "expression_statement":
            continue
        for inner in child.children:
            if inner.type != "assignment":
                continue
            left = inner.child_by_field_name("left")
            right = inner.child_by_field_name("right")
            if left is None or right is None:
                continue
            if left.type != "identifier" or _node_text(left, source) != "__all__":
                continue
            return _extract_string_list(right, source)
    return None


def _extract_string_list(node: Any, source: bytes) -> list[str] | None:
    """Extract literal-string elements from a `list` or `tuple` node, or None on failure."""
    if node.type not in ("list", "tuple"):
        return None
    out: list[str] = []
    for child in node.named_children:
        if child.type != "string":
            return None
        content = _string_content(child, source)
        if content is None:
            return None
        out.append(content)
    return out


def _string_content(string_node: Any, source: bytes) -> str | None:
    """Return the text inside a Python `string` node's quotes, or None on f-strings/binary."""
    has_content = False
    pieces: list[str] = []
    for child in string_node.children:
        if child.type == "string_content":
            pieces.append(_node_text(child, source))
            has_content = True
        elif child.type == "interpolation":
            return None
    return "".join(pieces) if has_content else ""


def _extract_top_level_public_functions(root_node: Any, source: bytes) -> dict[str, int]:
    """Map each top-level non-underscore `def` to its parameter count."""
    out: dict[str, int] = {}
    for child in root_node.children:
        fn_node = _unwrap_decorated(child)
        if fn_node is None or fn_node.type not in ("function_definition", "async_function_definition"):
            continue
        name_node = fn_node.child_by_field_name("name")
        if name_node is None:
            continue
        name = _node_text(name_node, source)
        if name.startswith("_"):
            continue
        params_node = fn_node.child_by_field_name("parameters")
        param_count = len(params_node.named_children) if params_node is not None else 0
        out[name] = param_count
    return out


def _extract_top_level_public_classes(root_node: Any, source: bytes) -> set[str]:
    """Names of top-level non-underscore classes."""
    out: set[str] = set()
    for child in root_node.children:
        cls_node = _unwrap_decorated(child)
        if cls_node is None or cls_node.type != "class_definition":
            continue
        name_node = cls_node.child_by_field_name("name")
        if name_node is None:
            continue
        name = _node_text(name_node, source)
        if not name.startswith("_"):
            out.add(name)
    return out


def _unwrap_decorated(node: Any) -> Any | None:
    """Return the inner def/class of a `decorated_definition`, else `node` itself."""
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "async_function_definition", "class_definition"):
                return child
        return None
    return node
