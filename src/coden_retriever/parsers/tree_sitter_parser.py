"""
Tree-sitter parser module.

Parses source files using Tree-sitter and extracts code entities.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..language import LANGUAGE_MAP, LANGUAGE_QUERIES, LanguageLoader
from ..models import CodeEntity

logger = logging.getLogger(__name__)

# 500 chars balances retaining useful context against embedding API token
# limits (typically 512 tokens).  Longer docstrings rarely add retrieval value.
MAX_DOCSTRING_LENGTH = 500

# Maximum AST depth when searching for body nodes in malformed ASTs
MAX_BODY_SEARCH_DEPTH = 5

# Stub statement types (language-agnostic AST patterns)
# These node types represent explicit stub/placeholder constructs
STUB_STATEMENT_TYPES: frozenset[str] = frozenset({
    "pass_statement",      # Python: pass
    "raise_statement",     # Python: raise
    "throw_statement",     # JS/Java/C++: throw
    "macro_invocation",    # Rust: todo!(), unimplemented!(), panic!()
})

# AST node types that indicate decorators/annotations (language-agnostic)
# These are standard node type names used by tree-sitter grammars
DECORATOR_NODE_TYPES: frozenset[str] = frozenset({
    "decorator",            # Python, JavaScript, TypeScript
    "decorated_definition", # Python (parent node containing decorators)
    "annotation",           # Java, Kotlin, Scala
    "marker_annotation",    # Java (annotation without arguments)
    "attribute",            # C#, Rust (#[...])
    "attribute_item",       # Rust
    "attribute_list",       # C#
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
        "match_arm",
        # tree-sitter-rust parses if-let/while-let as regular if/while
        # with a let_condition child, so they are already counted above
    },
    "cpp": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "case_statement", "catch_clause", "conditional_expression",
    },
    "c": {
        "if_statement", "for_statement", "while_statement", "do_statement",
        "case_statement", "conditional_expression",
    },
    "bash": {
        "if_statement", "elif_clause", "for_statement",
        "c_style_for_statement", "while_statement", "case_statement",
        "case_item",
        "test_command",  # [ ] and [[ ]]
        "binary_expression",  # && and ||
    },
}


def _extract_python_method_call(node: Any) -> tuple[str | None, str | None]:
    """Extract receiver and method from Python attribute node."""
    obj_node = None
    attr_node = None
    for child in node.children:
        if hasattr(child, 'type'):
            if child.type == "identifier":
                if obj_node is None:
                    obj_node = child
                attr_node = child
            elif child.type == "attribute":
                obj_node = child
            elif child.type in ("call", "subscript"):
                obj_node = child

    method = None
    receiver = None
    if attr_node and attr_node.text:
        method = attr_node.text.decode("utf-8", errors="replace")
    if obj_node and obj_node.text and obj_node != attr_node:
        receiver_text = obj_node.text.decode("utf-8", errors="replace")
        receiver = receiver_text.split('.')[-1] if '.' in receiver_text else receiver_text

    return receiver, method


def _extract_js_ts_method_call(node: Any) -> tuple[str | None, str | None]:
    """Extract receiver and method from JS/TS member_expression."""
    method = None
    receiver = None
    for child in node.children:
        if hasattr(child, 'type'):
            if child.type == "property_identifier" and child.text:
                method = child.text.decode("utf-8", errors="replace")
            elif child.type == "identifier" and child.text:
                receiver = child.text.decode("utf-8", errors="replace")
            elif child.type == "member_expression":
                for subchild in child.children:
                    if hasattr(subchild, 'type') and subchild.type == "property_identifier":
                        if subchild.text:
                            receiver = subchild.text.decode("utf-8", errors="replace")
                        break
    return receiver, method


def _extract_go_method_call(node: Any) -> tuple[str | None, str | None]:
    """Extract receiver and method from Go selector_expression."""
    method = None
    receiver = None
    for child in node.children:
        if hasattr(child, 'type'):
            if child.type == "field_identifier" and child.text:
                method = child.text.decode("utf-8", errors="replace")
            elif child.type == "identifier" and child.text:
                receiver = child.text.decode("utf-8", errors="replace")
    return receiver, method


def _extract_rust_cpp_method_call(node: Any) -> tuple[str | None, str | None]:
    """Extract receiver and method from Rust/C++ field_expression."""
    method = None
    receiver = None
    for child in node.children:
        if hasattr(child, 'type'):
            if child.type == "field_identifier" and child.text:
                method = child.text.decode("utf-8", errors="replace")
            elif child.type == "identifier" and child.text:
                receiver = child.text.decode("utf-8", errors="replace")
    return receiver, method


def _extract_fallback_method_call(node: Any) -> tuple[str | None, str | None]:
    """Fallback: parse receiver and method from node text."""
    if not node.text:
        return None, None
    text = node.text.decode("utf-8", errors="replace")
    if '.' not in text:
        return None, None
    parts = text.split('.')
    method = parts[-1]
    receiver = parts[-2] if len(parts) >= 2 else None
    return receiver, method


# Node types that represent identifier definitions (not usages)
# When an identifier appears as a child of these nodes, it's being DEFINED
DEFINITION_PARENT_TYPES: frozenset[str] = frozenset({
    # Function/method definitions
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "function_item", "arrow_function",
    # Class definitions
    "class_definition", "class_declaration", "struct_item", "enum_item",
    "trait_item", "interface_declaration", "type_alias_declaration",
    # Parameters
    "parameter", "parameters", "typed_parameter", "typed_default_parameter",
    "default_parameter", "formal_parameters", "required_parameter",
    # Variable definitions (LHS of assignment)
    "assignment", "augmented_assignment", "variable_declarator",
    "short_var_declaration", "let_declaration", "const_declaration",
    "variable_assignment",  # Bash: MY_VAR="value"
    # Import definitions
    "import_statement", "import_from_statement", "import_declaration",
    "use_declaration", "aliased_import",
    # For loop variable
    "for_statement", "for_in_statement", "for_in_clause",
    # Exception handling
    "except_clause", "catch_clause",
    # Comprehension variables
    "list_comprehension", "dictionary_comprehension", "set_comprehension",
    "generator_expression",
})

# Node types that ARE identifiers (language-specific names)
IDENTIFIER_NODE_TYPES: frozenset[str] = frozenset({
    "identifier",           # Python, JS, TS, Go, Rust, Java, etc.
    "property_identifier",  # JS/TS for object properties
    "field_identifier",     # Go, Rust for struct fields
    "type_identifier",      # Many languages for type names
    "simple_identifier",    # Kotlin, Swift
    "name",                 # PHP
    "shorthand_property_identifier",  # JS destructuring
    "variable_name",        # Bash: $VAR references and assignments
})

# Language keywords/builtins excluded from identifier usage tracking
# because they are never user-defined names
_BUILTIN_NAMES: frozenset[str] = frozenset({
    "self", "this", "super", "cls", "None",
    "True", "False", "null", "undefined",
    "true", "false", "nil",
})


@dataclass
class ExtractionContext:
    """Shared state for a single file's entity and reference extraction pass."""

    file_path: str
    lang_name: str
    source_bytes: bytes
    seen_entities: set[tuple[str, int]] = field(default_factory=set)
    body_ranges: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    references: list[tuple[int, str, str, str | None]] = field(default_factory=list)


class RepoParser:
    """Parses source files using Tree-sitter."""

    def __init__(self) -> None:
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
                parser.set_language(language)  # type: ignore[attr-defined]

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
            captures_list = self._build_captures_list(query, tree.root_node)
        except Exception as e:
            logger.debug(f"Parse error in {file_path}: {e}")
            return [], []

        if not captures_list:
            return [], []

        return self._extract_entities_and_references(
            captures_list, file_path, lang_name, source_bytes
        )

    def _build_captures_list(
        self,
        query: Any,
        root_node: Any
    ) -> list[tuple[Any, str]]:
        """Build (node, capture_name) tuples from tree-sitter query results.

        Handles different tree-sitter API versions and return formats.
        """
        captures_list: list[tuple[Any, str]] = []
        try:
            raw_captures = query.captures(root_node)
            if raw_captures:
                if isinstance(raw_captures, dict):
                    for capture_name, nodes in raw_captures.items():
                        if not isinstance(nodes, list):
                            nodes = [nodes]
                        for node in nodes:
                            captures_list.append((node, capture_name))
                elif isinstance(raw_captures, list):
                    if raw_captures and isinstance(raw_captures[0], tuple):
                        if len(raw_captures[0]) == 2:
                            captures_list = list(raw_captures)
                        else:
                            captures_list = [
                                (item[0], item[1]) for item in raw_captures
                            ]
        except (AttributeError, TypeError):
            captures_list = self._build_captures_from_matches(query, root_node)
        return captures_list

    def _build_captures_from_matches(
        self,
        query: Any,
        root_node: Any
    ) -> list[tuple[Any, str]]:
        """Fallback: build captures via the matches() API for older tree-sitter."""
        captures_list: list[tuple[Any, str]] = []
        try:
            matches = query.matches(root_node)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    capture_dict = match[1]
                    if isinstance(capture_dict, dict):
                        for capture_name, nodes in capture_dict.items():
                            if not isinstance(nodes, list):
                                nodes = [nodes]
                            for node in nodes:
                                captures_list.append((node, capture_name))
        except Exception as e:
            logger.debug("Failed to build captures from matches: %s", e)
        return captures_list

    def _extract_entities_and_references(
        self,
        captures_list: list[tuple[Any, str]],
        file_path: str,
        lang_name: str,
        source_bytes: bytes
    ) -> tuple[list[CodeEntity], list[tuple[int, str, str, str | None]]]:
        """Process captured AST nodes into entities and references."""
        entities: list[CodeEntity] = []
        ctx = ExtractionContext(
            file_path=file_path,
            lang_name=lang_name,
            source_bytes=source_bytes,
            body_ranges=self._collect_body_ranges(captures_list),
        )

        for node, capture_name in captures_list:
            try:
                text = node.text.decode("utf-8", errors="replace") if node.text else ""

                if capture_name.startswith("def."):
                    # Skip variable definitions: local variables create high-degree
                    # nodes that distort the dependency graph and waste token budget.
                    if capture_name == "def.variable":
                        continue
                    entity = self._extract_entity(node, text, capture_name, ctx)
                    if entity:
                        entities.append(entity)

                elif capture_name.startswith("ref."):
                    self._process_reference(node, text, capture_name, ctx)

            except Exception as e:
                logger.debug(f"Error processing node in {file_path}: {e}")
                continue

        return entities, ctx.references

    def _collect_body_ranges(
        self,
        captures_list: list[tuple[Any, str]]
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """Map definition ranges to their body ranges from captured nodes."""
        body_ranges: dict[tuple[int, int], tuple[int, int]] = {}
        for node, capture_name in captures_list:
            if capture_name.startswith("body."):
                parent = node.parent
                if parent:
                    def_range = (parent.start_point.row, parent.end_point.row)
                    body_ranges[def_range] = (node.start_point.row, node.end_point.row)
        return body_ranges

    def _process_reference(
        self,
        node: Any,
        text: str,
        capture_name: str,
        ctx: ExtractionContext
    ) -> None:
        """Process a single ref.* capture into reference tuples."""
        ref_type = capture_name.split(".")[1]
        line = node.start_point.row + 1

        if ref_type == "method_call":
            receiver, method = self._extract_method_call_parts(node, ctx.lang_name)
            if method:
                ctx.references.append((line, method, "call", receiver))
        else:
            ctx.references.append((line, text, ref_type, None))

        # Bash source/. commands are import equivalents
        if ctx.lang_name == "bash" and ref_type == "call" and text in ("source", "."):
            import_target = self._extract_bash_source_target(node)
            if import_target:
                ctx.references.append((line, import_target, "import", None))

    def extract_identifier_usages(
        self,
        file_path: str,
        source_code: str
    ) -> set[str]:
        """Extract all identifier names that are USED (referenced) in a file.

        This implements Vulture-style name tracking: any identifier that appears
        in a non-definition context is considered "used". This catches patterns
        that call-graph analysis misses:
        - Dictionary values: {"key": func}
        - Callbacks: sort(key=func)
        - Assignments: handler = func
        - List literals: [func1, func2]

        Returns:
            Set of identifier names that are referenced (not just defined).
        """
        ext = Path(file_path).suffix.lower()
        lang_name = LANGUAGE_MAP.get(ext)

        if not lang_name:
            return set()

        parser = self._get_parser(lang_name)
        if not parser:
            return set()

        try:
            source_bytes = source_code.encode("utf-8")
            tree = parser.parse(source_bytes)
        except Exception as e:
            logger.debug(f"Parse error in {file_path}: {e}")
            return set()

        return self._collect_identifier_usages(tree.root_node, file_path)

    @staticmethod
    def _is_definition_context(node: Any) -> bool:
        """Check if this identifier is being defined rather than referenced."""
        parent = node.parent
        if not parent:
            return False

        if parent.type in DEFINITION_PARENT_TYPES:
            if parent.type in ("assignment", "augmented_assignment"):
                left = parent.child_by_field_name("left")
                if left and node.start_byte >= left.start_byte and node.end_byte <= left.end_byte:
                    return True
                return False
            return True

        if parent.type in ("function_definition", "function_declaration",
                          "method_definition", "method_declaration",
                          "function_item", "class_definition",
                          "class_declaration", "struct_item", "enum_item",
                          "trait_item", "interface_declaration"):
            name_node = parent.child_by_field_name("name")
            if name_node and node.id == name_node.id:
                return True

        if parent.type in ("parameter", "typed_parameter",
                          "typed_default_parameter", "default_parameter",
                          "required_parameter"):
            return True

        if parent.type == "variable_declarator":
            name_node = parent.child_by_field_name("name")
            if name_node and node.id == name_node.id:
                return True

        if parent.type in ("for_in_clause", "for_in_statement"):
            left = parent.child_by_field_name("left")
            if left and node.start_byte >= left.start_byte and node.end_byte <= left.end_byte:
                return True

        return False

    def _collect_identifier_usages(
        self,
        root_node: Any,
        file_path: str
    ) -> set[str]:
        """Walk AST and collect all non-definition identifier references."""
        used_names: set[str] = set()

        def walk(node: Any) -> None:
            if node.type in IDENTIFIER_NODE_TYPES and node.text:
                name = node.text.decode("utf-8", errors="replace")
                if not self._is_definition_context(node) and name not in _BUILTIN_NAMES:
                    used_names.add(name)
            for child in node.children:
                walk(child)

        try:
            walk(root_node)
        except Exception as e:
            logger.debug(f"Error walking AST in {file_path}: {e}")

        return used_names

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
            if lang_name == "python":
                receiver, method = _extract_python_method_call(node)
            elif lang_name in ("javascript", "typescript"):
                receiver, method = _extract_js_ts_method_call(node)
            elif lang_name == "go":
                receiver, method = _extract_go_method_call(node)
            elif lang_name in ("rust", "cpp"):
                receiver, method = _extract_rust_cpp_method_call(node)
            else:
                receiver, method = _extract_fallback_method_call(node)

            # Skip self/this/super - can't resolve without type analysis
            if receiver in ('self', 'this', 'super', 'cls'):
                receiver = None

            return receiver, method

        except Exception as e:
            logger.debug(f"Error extracting method call parts: {e}")
            return None, None

    def _extract_bash_source_target(self, command_name_node: Any) -> str | None:
        """Extract the file path from a bash source/. command.

        In bash, 'source ./lib.sh' and '. ./lib.sh' load external scripts.
        The command_name node's parent is the command node; sibling children
        are the arguments.
        """
        try:
            cmd_node = command_name_node.parent
            if cmd_node is None:
                return None
            for child in cmd_node.children:
                if child.type != "command_name" and child.is_named:
                    text = child.text.decode("utf-8", errors="replace") if child.text else ""
                    if text:
                        return text.strip("'\"")
        except Exception as e:
            logger.debug("Failed to extract bash source target: %s", e)
        return None

    def _has_decorator_or_annotation(self, def_node: Any) -> bool:
        """Detect if a function/method definition has decorators/annotations.

        Language-agnostic: checks for standard tree-sitter node type names
        that represent decorators/annotations across different languages.

        Important: Only checks IMMEDIATE parent or preceding sibling to avoid
        false positives from other decorated definitions elsewhere in the file.
        """
        try:
            parent = def_node.parent
            if parent is None:
                return False

            # Check immediate parent for decorated_definition wrapper
            # This is the standard pattern in Python for decorated functions
            if parent.type in DECORATOR_NODE_TYPES:
                return True

            # Find the immediate preceding sibling (not all preceding siblings)
            # This handles Java/C#/Rust where annotations are siblings
            prev_sibling = None
            for child in parent.children:
                if child.start_byte >= def_node.start_byte:
                    break
                prev_sibling = child

            if prev_sibling is not None:
                if prev_sibling.type in DECORATOR_NODE_TYPES:
                    return True
                # Java/Kotlin: annotations in modifiers block
                if prev_sibling.type == "modifiers":
                    for mod_child in prev_sibling.children:
                        if mod_child.type in DECORATOR_NODE_TYPES:
                            return True

        except (AttributeError, TypeError):
            # AST node access can fail with various attribute/type errors
            pass
        return False

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
            for _ in range(MAX_BODY_SEARCH_DEPTH):
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

            # Bash: colon (:) is the no-op equivalent of Python's pass
            if stmt_type == "command":
                name_node = stmt.child_by_field_name("name")
                if name_node and name_node.text:
                    cmd_text = name_node.text.decode("utf-8", errors="replace")
                    if cmd_text == ":":
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

        except Exception as e:
            logger.debug("Failed to check stub status: %s", e)
            return False

    def _extract_arrow_function_name(self, arrow_node: Any, line: int) -> str:
        """Extract name for arrow function from parent variable assignment.

        Arrow functions get names from variable declarations:
        - const myFunc = () => {...}  -> "myFunc"
        - let handler = x => x        -> "handler"
        - obj.method = () => {}       -> "<arrow@line>"  (no good name)
        - (() => {})()                -> "<arrow@line>"  (IIFE)

        Returns:
            Function name if found, otherwise "<arrow@line:col>" format.
        """
        try:
            parent = arrow_node.parent
            if parent and parent.type == "variable_declarator":
                # const myFunc = () => {...}
                name_node = parent.child_by_field_name("name")
                if name_node and name_node.text:
                    return name_node.text.decode("utf-8", errors="replace")

            # Also check for assignment_expression (e.g., myFunc = () => {...})
            if parent and parent.type == "assignment_expression":
                left = parent.child_by_field_name("left")
                if left and left.type == "identifier" and left.text:
                    return left.text.decode("utf-8", errors="replace")

        except (AttributeError, TypeError):
            pass

        # Fallback: use <arrow@line> format for anonymous arrow functions
        return f"<arrow@{line}>"

    def _extract_entity(
        self,
        node: Any,
        name: str,
        capture_name: str,
        ctx: ExtractionContext
    ) -> CodeEntity | None:
        """Extract a CodeEntity from an AST node."""
        def_node = node.parent
        if not def_node:
            return None

        start_line = def_node.start_point.row + 1
        end_line = def_node.end_point.row + 1

        # Handle arrow functions: extract name from parent variable_declarator
        # Arrow functions don't have inline names, they get names via assignment
        # e.g., `const myFunc = () => {...}` -> name is "myFunc"
        # Note: For arrow functions, the query captures the arrow_function node itself
        # (not a child identifier), so we check node.type, not def_node.type
        if node.type == "arrow_function":
            arrow_line = node.start_point.row + 1
            name = self._extract_arrow_function_name(node, arrow_line)

        key = (ctx.file_path, start_line)
        if key in ctx.seen_entities:
            return None
        ctx.seen_entities.add(key)

        try:
            block_bytes = ctx.source_bytes[def_node.start_byte:def_node.end_byte]
            source = block_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("Failed to extract entity source: %s", e)
            source = ""

        def_range = (def_node.start_point.row, def_node.end_point.row)
        body_range = ctx.body_ranges.get(def_range)
        body_start = body_range[0] + 1 if body_range else None
        body_end = body_range[1] + 1 if body_range else None

        docstring = self._extract_docstring(def_node, ctx.lang_name, ctx.source_bytes)
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
        is_decorated = False
        if entity_type in ("function", "method"):
            complexity = self._calculate_cyclomatic_complexity(
                def_node, ctx.lang_name, ctx.source_bytes
            )
            is_stub = self._is_stub_node(def_node)
            is_decorated = self._has_decorator_or_annotation(def_node)

        return CodeEntity(
            name=name,
            entity_type=entity_type,
            file_path=ctx.file_path,
            language=ctx.lang_name,
            line_start=start_line,
            line_end=end_line,
            source_code=source,
            body_start=body_start,
            body_end=body_end,
            docstring=docstring,
            parent_class=parent_class,
            cyclomatic_complexity=complexity,
            is_stub=is_stub,
            is_decorated=is_decorated,
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
                                        return doc[:MAX_DOCSTRING_LENGTH] if len(doc) > MAX_DOCSTRING_LENGTH else doc
                        break
            elif lang_name in ("javascript", "typescript", "java", "cpp", "c", "bash"):
                prev = node.prev_sibling
                if prev and prev.type == "comment":
                    text = source_bytes[prev.start_byte:prev.end_byte]
                    doc = text.decode("utf-8", errors="replace")
                    return doc[:MAX_DOCSTRING_LENGTH] if len(doc) > MAX_DOCSTRING_LENGTH else doc
        except Exception as e:
            logger.debug("Failed to extract docstring: %s", e)
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
                    except Exception as e:
                        logger.debug("Failed to check binary operator: %s", e)
                else:
                    count += 1

            # Bash list nodes contain command-level && / || that create branching
            # execution paths (e.g. cmd1 && cmd2 || cmd3).  These are separate from
            # binary_expression operators inside [[ ]] tests, which are already counted.
            if node_type == "list" and lang_name == "bash":
                for child in current_node.children:
                    if child.type in ("&&", "||"):
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
