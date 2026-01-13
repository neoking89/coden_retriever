"""
Tree-sitter parser module.

Parses source files using Tree-sitter and extracts code entities.
"""
import logging
from pathlib import Path
from typing import Any

from ..language import LANGUAGE_MAP, LANGUAGE_QUERIES, LanguageLoader
from ..models import CodeEntity

logger = logging.getLogger(__name__)

# Stub statement types (language-agnostic AST patterns)
# These node types represent explicit stub/placeholder constructs
STUB_STATEMENT_TYPES: frozenset[str] = frozenset({
    "pass_statement",      # Python: pass
    "raise_statement",     # Python: raise
    "throw_statement",     # JS/Java/C++: throw
    "macro_invocation",    # Rust: todo!(), unimplemented!(), panic!()
})

# AST node types that increase cyclomatic complexity by language
COMPLEXITY_NODES: dict[str, set[str]] = {
    "python": {
        "if_statement", "elif_clause", "for_statement", "while_statement",
        "except_clause", "with_statement", "match_statement", "case_clause",
        "conditional_expression",  # ternary: a if b else c
        "boolean_operator",  # and/or add decision points
    },
    "javascript": {
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "do_statement", "switch_case", "catch_clause", "ternary_expression",
        "binary_expression",  # && and || add decision points
    },
    "typescript": {
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "do_statement", "switch_case", "catch_clause", "ternary_expression",
        "binary_expression",
    },
    "go": {
        "if_statement", "for_statement", "expression_switch_statement",
        "type_switch_statement", "select_statement", "expression_case",
        "type_case", "default_case",
    },
    "java": {
        "if_statement", "for_statement", "enhanced_for_statement",
        "while_statement", "do_statement", "switch_expression",
        "switch_block_statement_group", "catch_clause", "ternary_expression",
    },
    "rust": {
        "if_expression", "for_expression", "while_expression", "loop_expression",
        "match_arm", "if_let_expression", "while_let_expression",
    },
    "cpp": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "case_statement", "catch_clause", "conditional_expression",
    },
    "c": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "case_statement", "conditional_expression",
    },
}


class RepoParser:
    """Parses source files using Tree-sitter."""

    def __init__(self):
        self._loader = LanguageLoader()
        self._parsers: dict[str, Any] = {}
        self._queries: dict[str, Any] = {}

    def _get_parser(self, lang_name: str) -> Any | None:
        """Get or create a parser for the given language."""
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        language = self._loader.load(lang_name)
        if not language:
            return None

        if lang_name not in LANGUAGE_QUERIES:
            logger.debug(f"No query defined for language: {lang_name}")
            return None

        # Lazy import to avoid circular import issues
        try:
            from tree_sitter import Parser, Query
        except ImportError:
            logger.error("tree-sitter not installed")
            return None

        try:
            try:
                parser = Parser(language)
            except TypeError:
                parser = Parser()
                parser.set_language(language)

            # Use language.query() method (preferred in newer tree-sitter)
            try:
                query = language.query(LANGUAGE_QUERIES[lang_name])
            except AttributeError:
                # Fallback for older versions
                query = Query(language, LANGUAGE_QUERIES[lang_name])

            self._parsers[lang_name] = parser
            self._queries[lang_name] = query
            return parser

        except Exception as e:
            logger.warning(f"Failed to initialize parser for {lang_name}: {e}")
            return None

    def parse_file(
        self,
        file_path: str,
        source_code: str
    ) -> tuple[list[CodeEntity], list[tuple[int, str, str, str | None]]]:
        """Parse a source file and extract entities and references.

        Returns:
            Tuple of (entities, references) where references are
            (line, target_name, ref_type, receiver) tuples.
            receiver is None for simple function calls, or the object name
            for method calls (e.g., 'cache' in 'cache.get()').
        """
        ext = Path(file_path).suffix.lower()
        lang_name = LANGUAGE_MAP.get(ext)

        if not lang_name:
            return [], []

        parser = self._get_parser(lang_name)
        if not parser:
            return [], []

        try:
            source_bytes = source_code.encode("utf-8")
            tree = parser.parse(source_bytes)

            query = self._queries[lang_name]

            # Try different API methods for compatibility across tree-sitter versions
            captures_list = []
            try:
                # Newer API: query.captures() returns list of (node, capture_name) tuples
                raw_captures = query.captures(tree.root_node)
                # Handle different return formats
                if raw_captures:
                    if isinstance(raw_captures, dict):
                        # Some versions return {capture_name: [nodes]}
                        for capture_name, nodes in raw_captures.items():
                            if not isinstance(nodes, list):
                                nodes = [nodes]
                            for node in nodes:
                                captures_list.append((node, capture_name))
                    elif isinstance(raw_captures, list):
                        if raw_captures and isinstance(raw_captures[0], tuple):
                            if len(raw_captures[0]) == 2:
                                # Format: [(node, capture_name), ...]
                                captures_list = list(raw_captures)
                            else:
                                # Format might be [(node, capture_name, extra...), ...]
                                captures_list = [(item[0], item[1]) for item in raw_captures]
            except (AttributeError, TypeError):
                # Fallback: try matches() API
                try:
                    matches = query.matches(tree.root_node)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            pattern_idx, capture_dict = match[0], match[1]
                            if isinstance(capture_dict, dict):
                                for capture_name, nodes in capture_dict.items():
                                    if not isinstance(nodes, list):
                                        nodes = [nodes]
                                    for node in nodes:
                                        captures_list.append((node, capture_name))
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Parse error in {file_path}: {e}")
            return [], []

        if not captures_list:
            return [], []

        entities: list[CodeEntity] = []
        references: list[tuple[int, str, str, str | None]] = []
        seen_entities: set[tuple[str, int]] = set()
        body_ranges: dict[tuple[int, int], tuple[int, int]] = {}

        # First pass: collect body ranges
        for node, capture_name in captures_list:
            if capture_name.startswith("body."):
                parent = node.parent
                if parent:
                    def_range = (parent.start_point.row, parent.end_point.row)
                    body_ranges[def_range] = (node.start_point.row, node.end_point.row)

        # Second pass: extract entities and references
        for node, capture_name in captures_list:
            try:
                text = node.text.decode("utf-8", errors="replace") if node.text else ""

                if capture_name.startswith("def."):
                    # FIX: Skip variable definitions.
                    # Local variables (e.g., 'result', 'data') create high-degree nodes
                    # that distort the dependency graph and waste token budget.
                    if capture_name == "def.variable":
                        continue

                    entity = self._extract_entity(
                        node, text, capture_name,
                        file_path, lang_name, source_bytes,
                        seen_entities, body_ranges
                    )
                    if entity:
                        entities.append(entity)

                elif capture_name.startswith("ref."):
                    ref_type = capture_name.split(".")[1]
                    line = node.start_point.row + 1

                    # Handle method_call: extract receiver and method name
                    if ref_type == "method_call":
                        receiver, method = self._extract_method_call_parts(node, lang_name)
                        if method:
                            references.append((line, method, "call", receiver))
                    else:
                        references.append((line, text, ref_type, None))

            except Exception as e:
                logger.debug(f"Error processing node in {file_path}: {e}")
                continue

        return entities, references


    def _extract_method_call_parts(
        self,
        node: Any,
        lang_name: str
    ) -> tuple[str | None, str | None]:
        """Extract receiver and method name from an attribute/member expression node.

        For Python: cache.get() -> ('cache', 'get')
        For JS/TS: obj.method() -> ('obj', 'method')
        For chained: a.b.c() -> ('b', 'c')  # immediate receiver only

        Returns:
            Tuple of (receiver, method_name). Either may be None if extraction fails.
        """
        try:
            receiver = None
            method = None

            if lang_name == "python":
                # Python attribute node structure
                obj_node = None
                attr_node = None
                for child in node.children:
                    if hasattr(child, 'type'):
                        if child.type == "identifier":
                            if obj_node is None:
                                obj_node = child
                            attr_node = child
                        elif child.type == "attribute":
                            # Chained: a.b.c - the nested attribute becomes our object
                            obj_node = child
                        elif child.type in ("call", "subscript"):
                            obj_node = child

                if attr_node and attr_node.text:
                    method = attr_node.text.decode("utf-8", errors="replace")
                if obj_node and obj_node.text and obj_node != attr_node:
                    receiver_text = obj_node.text.decode("utf-8", errors="replace")
                    if '.' in receiver_text:
                        receiver = receiver_text.split('.')[-1]
                    else:
                        receiver = receiver_text

            elif lang_name in ("javascript", "typescript"):
                # JS/TS member_expression
                for child in node.children:
                    if hasattr(child, 'type'):
                        if child.type == "property_identifier":
                            if child.text:
                                method = child.text.decode("utf-8", errors="replace")
                        elif child.type == "identifier":
                            if child.text:
                                receiver = child.text.decode("utf-8", errors="replace")
                        elif child.type == "member_expression":
                            for subchild in child.children:
                                if hasattr(subchild, 'type') and subchild.type == "property_identifier":
                                    if subchild.text:
                                        receiver = subchild.text.decode("utf-8", errors="replace")
                                    break

            elif lang_name == "go":
                # Go selector_expression
                for child in node.children:
                    if hasattr(child, 'type'):
                        if child.type == "field_identifier":
                            if child.text:
                                method = child.text.decode("utf-8", errors="replace")
                        elif child.type == "identifier":
                            if child.text:
                                receiver = child.text.decode("utf-8", errors="replace")

            elif lang_name in ("rust", "cpp"):
                # Rust/C++ field_expression
                for child in node.children:
                    if hasattr(child, 'type'):
                        if child.type == "field_identifier":
                            if child.text:
                                method = child.text.decode("utf-8", errors="replace")
                        elif child.type == "identifier":
                            if child.text:
                                receiver = child.text.decode("utf-8", errors="replace")

            else:
                # Fallback: parse from text
                if node.text:
                    text = node.text.decode("utf-8", errors="replace")
                    if '.' in text:
                        parts = text.split('.')
                        method = parts[-1]
                        if len(parts) >= 2:
                            receiver = parts[-2]

            # Skip self/this/super - can't resolve without type analysis
            if receiver in ('self', 'this', 'super', 'cls'):
                receiver = None

            return receiver, method

        except Exception as e:
            logger.debug(f"Error extracting method call parts: {e}")
            return None, None

    def _is_stub_node(self, node: Any) -> bool:
        """Detect stub/interface methods using Tree-sitter AST.

        Language-agnostic: analyzes the 'body' node structure.
        A method is a stub if:
        - No body (abstract/interface)
        - Empty body (0 statements)
        - Single statement that is a stub pattern (pass, raise, throw, empty return)

        NOT a stub if:
        - Has multiple statements
        - Has a return with a value (return 42, return True, etc.)
        """
        try:
            # Traverse up to find the node with a 'body' field
            # This handles languages like C++ where the captured node is nested
            # (e.g., identifier inside function_declarator inside function_definition)
            body = None
            current = node
            max_depth = 5  # Prevent infinite loops
            for _ in range(max_depth):
                body = current.child_by_field_name('body')
                if body is not None:
                    break
                if current.parent is None:
                    break
                current = current.parent

            if not body:
                return True  # No body = abstract/interface

            # Count named children that are not extras (comments/whitespace)
            statements = [c for c in body.children if c.is_named and not c.is_extra]

            # Skip leading string literal (docstring pattern across languages)
            if statements and statements[0].type in ("expression_statement", "string"):
                first = statements[0]
                if first.type == "string" or (
                    first.children and
                    all(c.type == "string" or not c.is_named for c in first.children)
                ):
                    statements = statements[1:]

            # Empty body = stub
            if len(statements) == 0:
                return True

            # Multiple statements = not a stub
            if len(statements) > 1:
                return False

            # Single statement - check if it's a stub pattern
            stmt = statements[0]
            stmt_type = stmt.type

            # Use module-level constant for stub statement types
            if stmt_type in STUB_STATEMENT_TYPES:
                return True

            # Check for ellipsis (Python: ...)
            if stmt_type == "expression_statement":
                for child in stmt.children:
                    if child.type == "ellipsis":
                        return True
                    # Rust macro_invocation inside expression_statement
                    if child.type == "macro_invocation":
                        return True

            # Return statement: stub only if no value
            if stmt_type == "return_statement":
                # Check if there's a value (any named child that's not the return keyword)
                value_children = [c for c in stmt.children if c.is_named]
                return len(value_children) == 0

            # Other single statements with actual logic = not a stub
            return False

        except Exception:
            return False

    def _extract_entity(
        self,
        node: Any,
        name: str,
        capture_name: str,
        file_path: str,
        lang_name: str,
        source_bytes: bytes,
        seen: set[tuple[str, int]],
        body_ranges: dict[tuple[int, int], tuple[int, int]]
    ) -> CodeEntity | None:
        """Extract a CodeEntity from an AST node."""
        def_node = node.parent
        if not def_node:
            return None

        start_line = def_node.start_point.row + 1
        end_line = def_node.end_point.row + 1

        key = (file_path, start_line)
        if key in seen:
            return None
        seen.add(key)

        try:
            block_bytes = source_bytes[def_node.start_byte:def_node.end_byte]
            source = block_bytes.decode("utf-8", errors="replace")
        except Exception:
            source = ""

        def_range = (def_node.start_point.row, def_node.end_point.row)
        body_range = body_ranges.get(def_range)
        body_start = body_range[0] + 1 if body_range else None
        body_end = body_range[1] + 1 if body_range else None

        docstring = self._extract_docstring(def_node, lang_name, source_bytes)
        entity_type = capture_name.split(".")[1]

        parent_class = None
        current = def_node.parent
        while current:
            if current.type in ("class_definition", "class_declaration", "class_specifier"):
                for child in current.children:
                    if child.type in ("identifier", "type_identifier", "name"):
                        parent_class = child.text.decode("utf-8", errors="replace") if child.text else None
                        break
                break
            current = current.parent

        # Calculate cyclomatic complexity for functions/methods only
        complexity = None
        is_stub = False
        if entity_type in ("function", "method"):
            complexity = self._calculate_cyclomatic_complexity(def_node, lang_name, source_bytes)
            is_stub = self._is_stub_node(def_node)

        return CodeEntity(
            name=name,
            entity_type=entity_type,
            file_path=file_path,
            language=lang_name,
            line_start=start_line,
            line_end=end_line,
            source_code=source,
            body_start=body_start,
            body_end=body_end,
            docstring=docstring,
            parent_class=parent_class,
            cyclomatic_complexity=complexity,
            is_stub=is_stub,
        )

    def _extract_docstring(
        self,
        node: Any,
        lang_name: str,
        source_bytes: bytes
    ) -> str | None:
        """Extract docstring from a definition node if present."""
        try:
            if lang_name == "python":
                for child in node.children:
                    if child.type == "block":
                        for stmt in child.children:
                            if stmt.type == "expression_statement":
                                for expr in stmt.children:
                                    if expr.type == "string":
                                        text = source_bytes[expr.start_byte:expr.end_byte]
                                        doc = text.decode("utf-8", errors="replace").strip('"\'')
                                        doc = doc.strip('"\'')
                                        return doc[:500] if len(doc) > 500 else doc
                        break
            elif lang_name in ("javascript", "typescript", "java", "cpp", "c"):
                prev = node.prev_sibling
                if prev and prev.type == "comment":
                    text = source_bytes[prev.start_byte:prev.end_byte]
                    doc = text.decode("utf-8", errors="replace")
                    return doc[:500] if len(doc) > 500 else doc
        except Exception:
            pass
        return None

    def _calculate_cyclomatic_complexity(
        self,
        node: Any,
        lang_name: str,
        source_bytes: bytes
    ) -> int:
        """Calculate cyclomatic complexity for a function/method node.

        Cyclomatic complexity = E - N + 2P where:
        - E = edges, N = nodes, P = connected components
        For a single function, this simplifies to: 1 + number of decision points

        Decision points are: if, elif, for, while, except, case, ternary, and/or, etc.
        """
        complexity_node_types = COMPLEXITY_NODES.get(lang_name, set())
        if not complexity_node_types:
            return 1  # Default complexity for unsupported languages

        count = 0

        def walk_tree(current_node: Any) -> None:
            nonlocal count
            node_type = current_node.type

            # Count decision points
            if node_type in complexity_node_types:
                # Special handling for boolean operators (and/or/&&/||)
                if node_type == "boolean_operator":
                    # Each and/or adds a decision point
                    count += 1
                elif node_type == "binary_expression":
                    # Only count && and || operators, not arithmetic
                    # Look for the operator child node to avoid over-counting
                    try:
                        for child in current_node.children:
                            if child.type in ("&&", "||"):
                                count += 1
                                break
                    except Exception:
                        pass
                else:
                    count += 1

            # Recursively process children
            for child in current_node.children:
                walk_tree(child)

        try:
            walk_tree(node)
        except Exception as e:
            logger.debug(f"Error calculating complexity: {e}")

        # Base complexity is 1, plus all decision points
        return 1 + count
