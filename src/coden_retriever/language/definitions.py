"""
Language definitions module.

Contains language-to-extension mappings and Tree-sitter queries.
"""
from pathlib import Path
from typing import Dict

# Language file extension mapping
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c++": "cpp", ".hpp": "cpp", ".hxx": "cpp",
    ".c": "c", ".h": "c",
    ".php": "php",
    ".cs": "c_sharp",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "bash", ".bash": "bash",
}


def language_for_path(file_path: str | Path) -> str | None:
    """Resolve a file path to its tree-sitter language name, or None if unsupported."""
    return LANGUAGE_MAP.get(Path(file_path).suffix.lower())

# Tree-sitter queries for each language
# NOTE: ref.method_call captures the FULL attribute/member node so we can extract
# both the receiver (object) and method name for qualified resolution.
# This reduces false positive coupling on common method names like .get(), .set()
LANGUAGE_QUERIES: Dict[str, str] = {
    "python": """
        (class_definition name: (identifier) @def.class body: (block) @body.class)
        (function_definition name: (identifier) @def.function body: (block) @body.function)
        (call function: (identifier) @ref.call)
        (call function: (attribute) @ref.method_call)
        (import_from_statement module_name: (dotted_name) @ref.import)
        (import_statement name: (dotted_name) @ref.import)
        (argument_list (identifier) @ref.usage)
        (assignment left: (identifier) @def.variable)
        (typed_parameter type: (_) @ref.type)
        (typed_default_parameter type: (_) @ref.type)
        (function_definition return_type: (_) @ref.type)
    """,
    "javascript": """
        (class_declaration name: (identifier) @def.class body: (class_body) @body.class)
        (function_declaration name: (identifier) @def.function body: (statement_block) @body.function)
        (method_definition name: (property_identifier) @def.method body: (statement_block) @body.method)
        (arrow_function body: (_) @body.function) @def.function
        (variable_declarator name: (identifier) @def.variable)
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (member_expression) @ref.method_call)
        (import_statement source: (string) @ref.import)
    """,
    "typescript": """
        (class_declaration name: (type_identifier) @def.class body: (class_body) @body.class)
        (function_declaration name: (identifier) @def.function body: (statement_block) @body.function)
        (method_definition name: (property_identifier) @def.method body: (statement_block) @body.method)
        (interface_declaration name: (type_identifier) @def.class body: (object_type) @body.class)
        (type_alias_declaration name: (type_identifier) @def.class)
        (arrow_function body: (_) @body.function) @def.function
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (member_expression) @ref.method_call)
        (import_statement source: (string) @ref.import)
        (required_parameter type: (_) @ref.type)
        (optional_parameter type: (_) @ref.type)
        (public_field_definition type: (_) @ref.type)
        (property_signature type: (_) @ref.type)
        (variable_declarator type: (_) @ref.type)
        (function_declaration return_type: (_) @ref.type)
        (method_signature return_type: (_) @ref.type)
        (method_definition return_type: (_) @ref.type)
        (arrow_function return_type: (_) @ref.type)
    """,
    "go": """
        (function_declaration name: (identifier) @def.function body: (block) @body.function)
        (method_declaration name: (field_identifier) @def.method body: (block) @body.method)
        (type_declaration (type_spec name: (type_identifier) @def.class))
        (type_declaration (type_spec name: (type_identifier) @def.class type: (struct_type) @body.class))
        (type_declaration (type_spec name: (type_identifier) @def.class type: (interface_type) @body.class))
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (selector_expression) @ref.method_call)
        (import_declaration (import_spec path: (interpreted_string_literal) @ref.import))
        (parameter_declaration type: (_) @ref.type)
        (variadic_parameter_declaration type: (_) @ref.type)
        (field_declaration type: (_) @ref.type)
        (function_declaration result: (_) @ref.type)
        (method_declaration result: (_) @ref.type)
    """,
    "rust": """
        (function_item name: (identifier) @def.function body: (block) @body.function)
        (struct_item name: (type_identifier) @def.class body: (field_declaration_list) @body.class)
        (enum_item name: (type_identifier) @def.class body: (enum_variant_list) @body.class)
        (impl_item type: (type_identifier) @ref.usage body: (declaration_list) @body.class)
        (trait_item name: (type_identifier) @def.class body: (declaration_list) @body.class)
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (field_expression) @ref.method_call)
        (call_expression function: (scoped_identifier name: (identifier) @ref.call))
        (use_declaration argument: (scoped_identifier) @ref.import)
        (parameter type: (_) @ref.type)
        (function_item return_type: (_) @ref.type)
        (field_declaration type: (_) @ref.type)
        (let_declaration type: (_) @ref.type)
    """,
    "java": """
        (class_declaration name: (identifier) @def.class body: (class_body) @body.class)
        (interface_declaration name: (identifier) @def.class body: (interface_body) @body.class)
        (method_declaration name: (identifier) @def.method body: (block) @body.method)
        (constructor_declaration name: (identifier) @def.method body: (constructor_body) @body.method)
        (method_invocation name: (identifier) @ref.call)
        (import_declaration (scoped_identifier) @ref.import)
        (formal_parameter type: (_) @ref.type)
        (method_declaration type: (_) @ref.type)
        (field_declaration type: (_) @ref.type)
        (local_variable_declaration type: (_) @ref.type)
    """,
    "cpp": """
        (function_definition declarator: (function_declarator declarator: (identifier) @def.function) body: (compound_statement) @body.function)
        (function_definition declarator: (function_declarator declarator: (qualified_identifier name: (identifier) @def.function)) body: (compound_statement) @body.function)
        (class_specifier name: (type_identifier) @def.class body: (field_declaration_list) @body.class)
        (struct_specifier name: (type_identifier) @def.class body: (field_declaration_list) @body.class)
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (field_expression) @ref.method_call)
        (preproc_include path: (_) @ref.import)
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @def.function) body: (compound_statement) @body.function)
        (struct_specifier name: (type_identifier) @def.class body: (field_declaration_list) @body.class)
        (enum_specifier name: (type_identifier) @def.class body: (enumerator_list) @body.class)
        (call_expression function: (identifier) @ref.call)
        (preproc_include path: (_) @ref.import)
    """,
    "php": """
        (class_declaration name: (name) @def.class body: (declaration_list) @body.class)
        (interface_declaration name: (name) @def.class body: (declaration_list) @body.class)
        (trait_declaration name: (name) @def.class body: (declaration_list) @body.class)
        (function_definition name: (name) @def.function body: (compound_statement) @body.function)
        (method_declaration name: (name) @def.method body: (compound_statement) @body.method)
        (function_call_expression function: (name) @ref.call)
        (member_call_expression name: (name) @ref.call)
        (namespace_use_clause (qualified_name) @ref.import)
    """,
    "c_sharp": """
        (class_declaration name: (identifier) @def.class body: (declaration_list) @body.class)
        (interface_declaration name: (identifier) @def.class body: (declaration_list) @body.class)
        (struct_declaration name: (identifier) @def.class body: (declaration_list) @body.class)
        (method_declaration name: (identifier) @def.method body: (block) @body.method)
        (constructor_declaration name: (identifier) @def.method body: (block) @body.method)
        (invocation_expression function: (identifier) @ref.call)
        (using_directive (qualified_name) @ref.import)
        (parameter type: (_) @ref.type)
        (method_declaration type: (_) @ref.type)
        (variable_declaration type: (_) @ref.type)
        (property_declaration type: (_) @ref.type)
    """,
    "kotlin": """
        (class_declaration (type_identifier) @def.class (class_body) @body.class)
        (object_declaration (type_identifier) @def.class (class_body) @body.class)
        (function_declaration (simple_identifier) @def.function (function_body) @body.function)
        (call_expression (simple_identifier) @ref.call)
        (import_header (identifier) @ref.import)
        (parameter (user_type) @ref.type)
        (function_declaration (user_type) @ref.type)
        (class_parameter (user_type) @ref.type)
        (variable_declaration (user_type) @ref.type)
    """,
    "scala": """
        (class_definition name: (identifier) @def.class body: (template_body) @body.class)
        (object_definition name: (identifier) @def.class body: (template_body) @body.class)
        (trait_definition name: (identifier) @def.class body: (template_body) @body.class)
        (function_definition (identifier) @def.function body: (_) @body.function)
        (function_declaration (identifier) @def.function)
        (call_expression function: (identifier) @ref.call)
        (import_declaration (identifier) @ref.import)
        (parameter type: (_) @ref.type)
        (class_parameter type: (_) @ref.type)
        (function_definition return_type: (_) @ref.type)
        (function_declaration return_type: (_) @ref.type)
        (val_definition type: (_) @ref.type)
        (var_definition type: (_) @ref.type)
    """,
    "bash": """
        (function_definition name: (word) @def.function body: (compound_statement) @body.function)
        (command name: (command_name) @ref.call)
    """,
}
