"""AST node type definitions for literal constants across languages.

Centralized mapping of tree-sitter node types for numeric and string literals.
Derived empirically from each language's tree-sitter grammar.
"""

# Numeric literal node types per language
NUMERIC_LITERAL_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"integer", "float"}),
    "javascript": frozenset({"number"}),
    "typescript": frozenset({"number"}),
    "go": frozenset({"int_literal", "float_literal"}),
    "rust": frozenset({"integer_literal", "float_literal"}),
    "java": frozenset({
        "decimal_integer_literal", "hex_integer_literal",
        "octal_integer_literal", "binary_integer_literal",
        "decimal_floating_point_literal",
    }),
    "c": frozenset({"number_literal"}),
    "cpp": frozenset({"number_literal"}),
    "c_sharp": frozenset({"integer_literal", "real_literal"}),
    "kotlin": frozenset({"integer_literal", "real_literal"}),
    "php": frozenset({"integer", "float"}),
    "scala": frozenset({"integer_literal", "floating_point_literal"}),
    "bash": frozenset({"number"}),
}

# String literal node types per language
STRING_LITERAL_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"string"}),
    "javascript": frozenset({"string", "template_string"}),
    "typescript": frozenset({"string", "template_string"}),
    "go": frozenset({"interpreted_string_literal", "raw_string_literal"}),
    "rust": frozenset({"string_literal"}),
    "java": frozenset({"string_literal"}),
    "c": frozenset({"string_literal"}),
    "cpp": frozenset({"string_literal"}),
    "c_sharp": frozenset({"string_literal"}),
    "ruby": frozenset({"string"}),
    "kotlin": frozenset({"string_literal"}),
    "php": frozenset({"string", "encapsed_string"}),
    "scala": frozenset({"string"}),
    "bash": frozenset({"string", "raw_string"}),
}
