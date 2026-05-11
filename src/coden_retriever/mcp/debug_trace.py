"""
Debug Trace Tools for MCP.

Provides tools for adding breakpoints and trace statements to source code
across every language with a registered DAP adapter — Python, JS/TS, Go,
Rust, C/C++, C#, Java, Kotlin, Ruby, PHP, Lua, Bash, R, PowerShell.

Supports both DAP (Debug Adapter Protocol) integration and source code
injection. Trace lines use language-native idioms (Rust dbg!(), Ruby
inspect, PowerShell ConvertTo-Json, etc.) and write to stderr.

`source_inject_region_trace` additionally wraps a line range in a flow
toggle (`set -x`/`set +x` or `Set-PSDebug -Trace 1`/`0`) for bash and
powershell.

Works with VS Code, PyCharm, and other DAP-compatible IDEs.
"""
import asyncio
import importlib.util
import keyword
import logging
import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from ..constants import (
    DAP_HELPER_READY_TIMEOUT,
    DAP_HELPER_STOP_TIMEOUT,
    DEFAULT_DEBUG_PORT,
)
from .adapters.registry import REGISTRY
from .debug_errors import (
    debug_error,
    dependency_missing_error,
    file_not_found_error,
    line_out_of_range_error,
    unsupported_extension_error,
)
from .debug_injection_store import (
    Breakpoint,
    InjectedTrace,
    get_debug_session_manager,
)
from .debug_recovery import mark_debug_server_started

# debugpy is intentionally not imported in-process; the helper subprocess owns it.
_DEBUGPY_AVAILABLE = importlib.util.find_spec("debugpy") is not None

logger = logging.getLogger(__name__)

# Language-agnostic marker tag for injected trace statements
# The comment prefix (# or //) is added per-language
_TRACE_MARKER = "[DEBUG_TRACE]"

# WHY 8: uuid4().hex is 32 hex chars — 8 is enough entropy (~4B values) to keep
# per-session collision probability negligible while keeping injection IDs
# short enough to paste into `source_remove_injections(injection_id=...)` by
# hand. Longer prefixes (16, 32) hurt readability in MCP responses.
_INJECTION_ID_HEX_LENGTH = 8

# JavaScript/TypeScript reserved words and contextual keywords.
# WHY: Python's `keyword` module only covers Python; we still need to reject
# JS reserved words so caller-supplied names cannot smuggle keywords into
# generated template literals.
_JS_RESERVED_WORDS: frozenset[str] = frozenset({
    "break", "case", "catch", "class", "const", "continue", "debugger",
    "default", "delete", "do", "else", "enum", "export", "extends", "false",
    "finally", "for", "function", "if", "import", "in", "instanceof", "new",
    "null", "return", "super", "switch", "this", "throw", "true", "try",
    "typeof", "var", "void", "while", "with", "yield", "let", "static",
    "implements", "interface", "package", "private", "protected", "public",
    "await", "async",
})


class TraceKind(StrEnum):
    """Selects which language-specific trace-line emitter to use.

    Each kind maps to one entry in `_TRACE_GENERATORS` and one in
    `_LOGPOINT_GENERATORS`. The split lets us keep per-language emission
    under 30 lines per case rather than growing one giant if/else chain
    as we add languages.
    """

    PYTHON_FSTRING = "python_fstring"
    JS_TEMPLATE_LITERAL = "js_template_literal"
    GO_FPRINTF = "go_fprintf"
    RUST_DBG = "rust_dbg"
    RUBY_PUTS_INSPECT = "ruby_puts_inspect"
    JAVA_PRINTF = "java_printf"
    KOTLIN_PRINTLN = "kotlin_println"
    CSHARP_INTERP = "csharp_interp"
    PHP_ERROR_LOG = "php_error_log"
    LUA_IO_STDERR = "lua_io_stderr"
    BASH_PRINTF = "bash_printf"
    R_CAT_STDERR = "r_cat_stderr"
    POWERSHELL_WRITE_ERROR = "powershell_write_error"
    C_FPRINTF = "c_fprintf"


@dataclass(frozen=True)
class LanguageConfig:
    """Language-specific syntax for debug-trace and breakpoint injection.

    `breakpoint_stmt` is the bare statement that pauses execution
    (e.g. `breakpoint()`, `debugger;`); empty string means the language has
    no in-source breakpoint and `source_add_breakpoint` must reject mode='source'.

    `conditional_template` wraps `breakpoint_stmt` with a conditional guard;
    must contain literal `{cond}` and `{bp}` placeholders. Empty when the
    language has no breakpoint.
    """

    name: str
    comment_prefix: str
    trace_kind: TraceKind
    breakpoint_stmt: str = ""
    conditional_template: str = ""


# Supported file extensions mapped to their language configurations.
# Extensions are stored lowercase with leading dot — matched via Path.suffix.lower().
_PY = LanguageConfig(
    name="python",
    comment_prefix="#",
    trace_kind=TraceKind.PYTHON_FSTRING,
    breakpoint_stmt="breakpoint()",
    conditional_template="if {cond}: {bp}",
)
_JS = LanguageConfig(
    name="javascript",
    comment_prefix="//",
    trace_kind=TraceKind.JS_TEMPLATE_LITERAL,
    breakpoint_stmt="debugger;",
    conditional_template="if ({cond}) {{ {bp} }}",
)
_TS = LanguageConfig(
    name="typescript",
    comment_prefix="//",
    trace_kind=TraceKind.JS_TEMPLATE_LITERAL,
    breakpoint_stmt="debugger;",
    conditional_template="if ({cond}) {{ {bp} }}",
)
_BASH = LanguageConfig(
    name="bash",
    comment_prefix="#",
    trace_kind=TraceKind.BASH_PRINTF,
    # No native breakpoint — bashdb attaches via DAP; source mode rejects.
    breakpoint_stmt="",
    conditional_template="",
)
_POWERSHELL = LanguageConfig(
    name="powershell",
    comment_prefix="#",
    trace_kind=TraceKind.POWERSHELL_WRITE_ERROR,
    breakpoint_stmt="Wait-Debugger",
    conditional_template="if ({cond}) {{ {bp} }}",
)
_RUBY = LanguageConfig(
    name="ruby",
    comment_prefix="#",
    trace_kind=TraceKind.RUBY_PUTS_INSPECT,
    breakpoint_stmt="binding.irb",
    # Ruby modifier-if: `expr if cond`. Parens around bp keep precedence safe
    # for chained statements like `x = 1; (binding.irb) if cond`.
    conditional_template="({bp}) if {cond}",
)
_R = LanguageConfig(
    name="r",
    comment_prefix="#",
    trace_kind=TraceKind.R_CAT_STDERR,
    breakpoint_stmt="browser()",
    conditional_template="if ({cond}) {bp}",
)
_JAVA = LanguageConfig(
    name="java",
    comment_prefix="//",
    trace_kind=TraceKind.JAVA_PRINTF,
    # No native breakpoint — JDB attaches via DAP/JDWP; source mode rejects.
    breakpoint_stmt="",
    conditional_template="",
)
_KOTLIN = LanguageConfig(
    name="kotlin",
    comment_prefix="//",
    trace_kind=TraceKind.KOTLIN_PRINTLN,
    breakpoint_stmt="",
    conditional_template="",
)
_LUA = LanguageConfig(
    name="lua",
    comment_prefix="--",
    trace_kind=TraceKind.LUA_IO_STDERR,
    breakpoint_stmt="",
    conditional_template="",
)
_GO = LanguageConfig(
    name="go",
    comment_prefix="//",
    trace_kind=TraceKind.GO_FPRINTF,
    breakpoint_stmt="",
    conditional_template="",
)
_RUST = LanguageConfig(
    name="rust",
    comment_prefix="//",
    trace_kind=TraceKind.RUST_DBG,
    breakpoint_stmt="",
    conditional_template="",
)
_CSHARP = LanguageConfig(
    name="csharp",
    comment_prefix="//",
    trace_kind=TraceKind.CSHARP_INTERP,
    # Fully qualified — caller does not need a `using System.Diagnostics;`.
    breakpoint_stmt="System.Diagnostics.Debugger.Break();",
    conditional_template="if ({cond}) {{ {bp} }}",
)
_PHP = LanguageConfig(
    name="php",
    comment_prefix="//",
    trace_kind=TraceKind.PHP_ERROR_LOG,
    breakpoint_stmt="",
    conditional_template="",
)
_C = LanguageConfig(
    name="c",
    comment_prefix="//",
    trace_kind=TraceKind.C_FPRINTF,
    breakpoint_stmt="",
    conditional_template="",
)
SUPPORTED_EXTENSIONS: dict[str, LanguageConfig] = {
    ".py": _PY,
    ".js": _JS,
    ".jsx": _JS,
    ".ts": _TS,
    ".tsx": _TS,
    ".mjs": _JS,
    ".cjs": _JS,
    ".sh": _BASH,
    ".bash": _BASH,
    ".ps1": _POWERSHELL,
    ".rb": _RUBY,
    ".r": _R,
    ".java": _JAVA,
    ".kt": _KOTLIN,
    ".kts": _KOTLIN,
    ".lua": _LUA,
    ".go": _GO,
    ".rs": _RUST,
    ".cs": _CSHARP,
    ".php": _PHP,
    ".c": _C,
    ".cc": _C,
    ".cpp": _C,
    ".cxx": _C,
    ".h": _C,
    ".hpp": _C,
}


def _get_language_config(file_path: Path) -> LanguageConfig | None:
    """Get language configuration based on file extension."""
    return SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())


# Module-local handle to the process-wide DebugSessionManager singleton.
# `DebugSessionManager` the class lives in `debug_injection_store` (M4); only
# this handle lives here because the source-injection MCP tools defined below
# are the primary consumers.
_manager = get_debug_session_manager()


def _generate_id(prefix: str = "dbg") -> str:
    """Generate a unique ID for breakpoints/traces."""
    return f"{prefix}-{uuid.uuid4().hex[:_INJECTION_ID_HEX_LENGTH]}"


async def _create_backup(file_path: Path) -> Path:
    """Create a backup of the file before modification.

    WHY the existence check: a second injection into the same file would
    otherwise overwrite the pristine .debugbak with the already-modified
    content, making the backup useless for restoring the original.
    """

    def _backup_sync() -> Path:
        backup_path = file_path.with_suffix(file_path.suffix + ".debugbak")
        if not backup_path.exists():
            content = file_path.read_text(encoding="utf-8")
            backup_path.write_text(content, encoding="utf-8")
        return backup_path

    return await asyncio.to_thread(_backup_sync)


async def _read_file_lines(file_path: Path) -> list[str]:
    """Read file and return lines with preserved endings."""

    def _read_sync() -> list[str]:
        content = file_path.read_text(encoding="utf-8")
        return content.splitlines(keepends=True)

    return await asyncio.to_thread(_read_sync)


async def _write_file_lines(file_path: Path, lines: list[str]) -> None:
    """Write lines back to file.

    `splitlines(keepends=True)` leaves the final list entry without a trailing
    newline when the source file lacks one. After an insertion that pushes
    that entry into a non-final position, joining without a separator would
    glue the original last line onto the inserted line and destroy both on
    cleanup. Pad any non-final line missing a terminator before joining.
    """

    def _write_sync() -> None:
        for i in range(len(lines) - 1):
            if not lines[i].endswith(("\n", "\r")):
                lines[i] = lines[i] + "\n"
        content = "".join(lines)
        file_path.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_write_sync)


def _get_indentation(line: str) -> str:
    """Extract the indentation from a line."""
    stripped = line.lstrip()
    if not stripped:
        return ""
    return line[: len(line) - len(stripped)]


def _escape_for_python_fstring(s: str) -> str:
    """Escape a string for safe inclusion in a Python f-string with double quotes.

    Escapes backslashes, double quotes, curly braces, and newline characters to
    prevent syntax errors and unintended format string interpolation. Newlines
    (LF/CR) must be escaped because the generated statement is a single source
    line — a raw newline would break the enclosing `print(f"...")` mid-string.
    """
    # Escape backslashes first, then double quotes, curly braces, and
    # finally CR/LF so the generated f-string stays on one source line.
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_js_template_literal(s: str) -> str:
    """Escape a string for safe inclusion in a JavaScript template literal.

    Escapes backslashes, backticks, `${` interpolation markers, and CR/LF so
    the generated `console.log(`...`);` statement stays a single source line
    and cannot be broken by caller-controlled input.
    """
    # Escape backslashes first, then backticks, then ${, then CR/LF so the
    # generated template literal cannot be broken across source lines.
    return (
        s.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_bash_single_quote(s: str) -> str:
    """Escape a string for safe inclusion in a single-quoted bash printf format.

    Single-quoted strings can't contain literal single quotes, so we use the
    standard `'\\''` close-escape-reopen trick. Backslashes are doubled because
    printf interprets `\\n` etc. as escape sequences. `%` is doubled because it
    introduces printf format specifiers — caller-controlled `%s` would otherwise
    consume one of our supplied arguments.
    """
    return (
        s.replace("\\", "\\\\")
        .replace("%", "%%")
        .replace("'", "'\\''")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_powershell_double_quote(s: str) -> str:
    """Escape a string for safe inclusion in a PowerShell double-quoted string.

    PowerShell uses backtick (`) as its escape character and expands `$var` /
    `$(...)` inside double-quoted strings. Order matters: existing user
    backticks must be doubled BEFORE we add escape backticks for $/quote/etc.,
    otherwise our escape backticks would be doubled too.
    """
    return (
        s.replace("`", "``")
        .replace('"', '`"')
        .replace("$", "`$")
        .replace("\r", "`r")
        .replace("\n", "`n")
    )


def _escape_for_ruby_double_quote(s: str) -> str:
    """Escape a string for safe inclusion in a Ruby double-quoted string.

    Neutralizes `#{...}` interpolation by escaping `#{` to `\\#{`. A bare `#`
    is harmless — only `#{` opens an interpolation, so we don't need to
    escape every `#`.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("#{", "\\#{")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_r_double_quote(s: str) -> str:
    """Escape a string for safe inclusion in an R sprintf double-quoted format.

    R has no string interpolation, so we only escape backslashes, double
    quotes, and `%` (which would otherwise be read by sprintf as a format
    specifier and consume one of our trailing arguments).
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_java_string(s: str) -> str:
    """Escape a string for safe inclusion in a Java/printf double-quoted string.

    Java's `System.err.printf` consumes `%` as a format specifier, so we
    double user `%` chars even though Java string literals don't otherwise
    treat `%` specially.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_kotlin_double_quote(s: str) -> str:
    """Escape a string for safe inclusion in a Kotlin double-quoted string.

    Kotlin's `$x` and `${expr}` interpolate inside double-quoted strings;
    escaping `$` to `\\$` neutralizes both forms.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_lua_double_quote(s: str) -> str:
    """Escape a string for safe inclusion in a Lua string.format double-quoted format.

    Lua has no interpolation in string literals, but `string.format` reads
    `%` as a format specifier — same hazard as R/Bash printf.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_go_string(s: str) -> str:
    """Escape a string for inclusion in a Go fmt.Fprintf format string."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_rust_string(s: str) -> str:
    """Escape a string for inclusion in a Rust eprintln! format string.

    Rust's format strings consume `{` and `}` for argument placeholders;
    user braces must be doubled to `{{` / `}}` exactly like Python f-strings.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_csharp_interp(s: str) -> str:
    """Escape a string for safe inclusion in a C# `$"..."` interpolated string."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_php_double_quote(s: str) -> str:
    """Escape a string for safe inclusion in a PHP sprintf double-quoted format.

    PHP interpolates `$var` inside double-quoted strings, so we escape `$`
    in addition to the usual sprintf-format `%` doubling.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "\\$")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_for_c_string(s: str) -> str:
    """Escape a string for inclusion in a C/C++ printf format string."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _validate_variable_name(var: str, language: str) -> dict[str, Any] | None:
    """Return a structured error dict if `var` is not a safe identifier for `language`.

    The generated trace code embeds `var` directly into f-string / template-literal
    interpolation (e.g. `f"{var}={{{var}!r}}"`). Without validation a caller-supplied
    name like ``__import__('os').system('id')`` becomes executable code at runtime.
    Returns ``None`` when the name is safe; otherwise a ``debug_error`` payload
    suitable for returning to the MCP caller.
    """
    if not isinstance(var, str) or not var.isidentifier():
        return debug_error(
            "invalid_parameter",
            f"Variable name {var!r} is not a valid identifier",
            "Pass plain identifier names (e.g. ['x', 'result']), not expressions",
        )
    if keyword.iskeyword(var):
        return debug_error(
            "invalid_parameter",
            f"Variable name {var!r} is a Python reserved keyword",
            "Use a non-keyword identifier",
        )
    # JS and TS share the same reserved-word set; Python names are allowed in
    # a Python file but must not collide with JS reserved words for JS/TS files.
    if language in ("javascript", "typescript") and var in _JS_RESERVED_WORDS:
        return debug_error(
            "invalid_parameter",
            f"Variable name {var!r} is a JavaScript/TypeScript reserved word",
            "Use a non-reserved identifier",
        )
    return None


@dataclass(frozen=True)
class _TraceParts:
    """Inputs collected by `_generate_trace_statement` before emitter dispatch."""

    config: LanguageConfig
    variables: list[str] | None
    message: str | None
    file_name: str
    line_number: int
    include_timestamp: bool
    include_location: bool
    marker: str


# Each trace emitter takes a `_TraceParts` and returns the source line to inject.
# The dispatch dict (`_TRACE_GENERATORS`) is populated as new languages are added.
TraceEmitter = Callable[["_TraceParts"], str]


def _gen_trace_python(parts: "_TraceParts") -> str:
    """Python: print(f"[TRACE] ts file:line msg | x={x!r}, y={y!r}")."""
    segments: list[str] = []
    if parts.include_timestamp:
        segments.append("{__import__('datetime').datetime.now().isoformat()}")
    if parts.include_location:
        segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        segments.append(_escape_for_python_fstring(parts.message))
    if parts.variables:
        var_parts = [f"{v}={{{v}!r}}" for v in parts.variables]
        segments.append(f"| {', '.join(var_parts)}")

    prefix = "[TRACE] " if segments else "[TRACE]"
    inner = " ".join(segments)
    return f'print(f"{prefix}{inner}")  {parts.marker}'


def _gen_trace_js(parts: "_TraceParts") -> str:
    """JavaScript/TypeScript: console.log(`[TRACE] ${ts} file:line msg | x=${JSON.stringify(x)}`)."""
    segments: list[str] = []
    if parts.include_timestamp:
        segments.append("${new Date().toISOString()}")
    if parts.include_location:
        segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        segments.append(_escape_for_js_template_literal(parts.message))
    if parts.variables:
        var_parts = [f"{v}=${{JSON.stringify({v})}}" for v in parts.variables]
        segments.append(f"| {', '.join(var_parts)}")

    prefix = "[TRACE] " if segments else "[TRACE]"
    inner = " ".join(segments)
    return f"console.log(`{prefix}{inner}`);  {parts.marker}"


def _gen_trace_bash(parts: "_TraceParts") -> str:
    """Bash: printf '[TRACE] file:line msg | x=%s\n' "${x}" >&2."""
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        fmt_segments.append("%s")
        arg_segments.append('"$(date -Iseconds)"')
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_bash_single_quote(parts.message))
    if parts.variables:
        var_specs = []
        for v in parts.variables:
            var_specs.append(f"{v}=%s")
            arg_segments.append(f'"${{{v}}}"')
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (" " + " ".join(arg_segments)) if arg_segments else ""
    return f"printf '{prefix}{fmt_inner}\\n'{args} >&2  {parts.marker}"


def _gen_trace_powershell(parts: "_TraceParts") -> str:
    """PowerShell: [Console]::Error.WriteLine("[TRACE] file:line msg | x=$($x | ConvertTo-Json -Compress)")."""
    segments: list[str] = []
    if parts.include_timestamp:
        segments.append("$((Get-Date).ToString('o'))")
    if parts.include_location:
        segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        segments.append(_escape_for_powershell_double_quote(parts.message))
    if parts.variables:
        # `-Depth 5` because the default 2 truncates nested objects in JSON output.
        var_parts = [
            f"{v}=$(${v} | ConvertTo-Json -Compress -Depth 5)"
            for v in parts.variables
        ]
        segments.append(f"| {', '.join(var_parts)}")

    prefix = "[TRACE] " if segments else "[TRACE]"
    inner = " ".join(segments)
    return f'[Console]::Error.WriteLine("{prefix}{inner}")  {parts.marker}'


def _gen_trace_ruby(parts: "_TraceParts") -> str:
    """Ruby: STDERR.puts "[TRACE] file:line msg | x=#{x.inspect}"."""
    segments: list[str] = []
    if parts.include_timestamp:
        segments.append("#{Time.now.iso8601}")
    if parts.include_location:
        segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        segments.append(_escape_for_ruby_double_quote(parts.message))
    if parts.variables:
        var_parts = [f"{v}=#{{{v}.inspect}}" for v in parts.variables]
        segments.append(f"| {', '.join(var_parts)}")

    prefix = "[TRACE] " if segments else "[TRACE]"
    inner = " ".join(segments)
    return f'STDERR.puts "{prefix}{inner}"  {parts.marker}'


def _gen_trace_r(parts: "_TraceParts") -> str:
    """R: cat(sprintf("[TRACE] %s file:line msg | x=%s\\n", ts, deparse(x)), file=stderr())."""
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        fmt_segments.append("%s")
        arg_segments.append('format(Sys.time(), "%Y-%m-%dT%H:%M:%S")')
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_r_double_quote(parts.message))
    if parts.variables:
        var_specs = []
        for v in parts.variables:
            var_specs.append(f"{v}=%s")
            arg_segments.append(f"deparse({v})")
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'cat(sprintf("{prefix}{fmt_inner}\\n"{args}), file=stderr())  {parts.marker}'


def _gen_trace_java(parts: "_TraceParts") -> str:
    """Java: System.err.printf("[TRACE] file:line msg | x=%s%n", x);."""
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        fmt_segments.append("%s")
        arg_segments.append("java.time.Instant.now()")
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_java_string(parts.message))
    if parts.variables:
        var_specs = []
        for v in parts.variables:
            var_specs.append(f"{v}=%s")
            arg_segments.append(v)
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'System.err.printf("{prefix}{fmt_inner}%n"{args});  {parts.marker}'


def _gen_trace_kotlin(parts: "_TraceParts") -> str:
    """Kotlin: System.err.println("[TRACE] file:line msg | x=$x")."""
    segments: list[str] = []
    if parts.include_timestamp:
        segments.append("${java.time.Instant.now()}")
    if parts.include_location:
        segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        segments.append(_escape_for_kotlin_double_quote(parts.message))
    if parts.variables:
        var_parts = [f"{v}=${v}" for v in parts.variables]
        segments.append(f"| {', '.join(var_parts)}")

    prefix = "[TRACE] " if segments else "[TRACE]"
    inner = " ".join(segments)
    return f'System.err.println("{prefix}{inner}")  {parts.marker}'


def _gen_trace_lua(parts: "_TraceParts") -> str:
    """Lua: io.stderr:write(string.format("[TRACE] %s file:line msg | x=%s\\n", os.date(...), tostring(x)))."""
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        fmt_segments.append("%s")
        arg_segments.append('os.date("%Y-%m-%dT%H:%M:%S")')
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_lua_double_quote(parts.message))
    if parts.variables:
        var_specs = []
        for v in parts.variables:
            var_specs.append(f"{v}=%s")
            arg_segments.append(f"tostring({v})")
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'io.stderr:write(string.format("{prefix}{fmt_inner}\\n"{args}))  {parts.marker}'


def _gen_trace_go(parts: "_TraceParts") -> str:
    """Go: fmt.Fprintf(os.Stderr, "[TRACE] %s file:line msg | x=%+v\\n", time.Now(), x)."""
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        fmt_segments.append("%s")
        arg_segments.append("time.Now().Format(time.RFC3339)")
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_go_string(parts.message))
    if parts.variables:
        var_specs = []
        for v in parts.variables:
            var_specs.append(f"{v}=%+v")
            arg_segments.append(v)
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'fmt.Fprintf(os.Stderr, "{prefix}{fmt_inner}\\n"{args})  {parts.marker}'


def _gen_trace_rust(parts: "_TraceParts") -> str:
    """Rust: dbg!(&x, &y); when only variables are requested (idiomatic, auto file:line),
    else eprintln!("[TRACE] file:line msg | x={:?}", x);.
    """
    # The plain "show me these variables" call collapses to the language-native
    # `dbg!()` macro — it auto-prints file:line via the macro, takes refs to
    # avoid moving non-Copy values, and is the idiom Rust users would write
    # by hand.
    if (
        parts.variables
        and not parts.message
        and not parts.include_timestamp
    ):
        refs = ", ".join(f"&{v}" for v in parts.variables)
        return f"dbg!({refs});  {parts.marker}"

    fmt_segments: list[str] = []
    arg_segments: list[str] = []
    if parts.include_timestamp:
        # std-only: no zero-dep timestamp string. Emit unix-epoch seconds — the
        # caller can add chrono manually if richer formatting is wanted.
        fmt_segments.append("{}")
        arg_segments.append(
            "std::time::UNIX_EPOCH.elapsed().map_or(0, |d| d.as_secs())"
        )
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_rust_string(parts.message))
    if parts.variables:
        var_specs = [f"{v}={{:?}}" for v in parts.variables]
        for v in parts.variables:
            arg_segments.append(v)
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'eprintln!("{prefix}{fmt_inner}"{args});  {parts.marker}'


def _gen_trace_csharp(parts: "_TraceParts") -> str:
    """C#: Console.Error.WriteLine($"[TRACE] file:line msg | x={x}");."""
    segments: list[str] = []
    if parts.include_timestamp:
        segments.append("{System.DateTime.UtcNow:O}")
    if parts.include_location:
        segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        segments.append(_escape_for_csharp_interp(parts.message))
    if parts.variables:
        var_parts = [f"{v}={{{v}}}" for v in parts.variables]
        segments.append(f"| {', '.join(var_parts)}")

    prefix = "[TRACE] " if segments else "[TRACE]"
    inner = " ".join(segments)
    return f'Console.Error.WriteLine($"{prefix}{inner}");  {parts.marker}'


def _gen_trace_php(parts: "_TraceParts") -> str:
    """PHP: fwrite(STDERR, sprintf("[TRACE] %s file:line msg | x=%s\\n", date(...), var_export($x, true)));."""
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        fmt_segments.append("%s")
        arg_segments.append("date(DATE_ATOM)")
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_php_double_quote(parts.message))
    if parts.variables:
        var_specs = []
        for v in parts.variables:
            var_specs.append(f"{v}=%s")
            arg_segments.append(f"var_export(${v}, true)")
        fmt_segments.append(f"| {', '.join(var_specs)}")

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'fwrite(STDERR, sprintf("{prefix}{fmt_inner}\\n"{args}));  {parts.marker}'


def _gen_trace_c(parts: "_TraceParts") -> str:
    """C/C++: fprintf(stderr, "[TRACE] file:line msg\\n");

    No `variables` interpolation — C has no generic format specifier and
    typed variables can't be auto-formatted without per-type knowledge.
    Documented as an explicit non-goal in the tool docstring.
    """
    fmt_segments: list[str] = []
    arg_segments: list[str] = []

    if parts.include_timestamp:
        # Casting time_t to long for printf-portable formatting; user code
        # must already include <time.h> / <ctime>.
        fmt_segments.append("%ld")
        arg_segments.append("(long)time(NULL)")
    if parts.include_location:
        fmt_segments.append(f"{parts.file_name}:{parts.line_number}")
    if parts.message:
        fmt_segments.append(_escape_for_c_string(parts.message))

    prefix = "[TRACE] " if fmt_segments else "[TRACE]"
    fmt_inner = " ".join(fmt_segments)
    args = (", " + ", ".join(arg_segments)) if arg_segments else ""
    return f'fprintf(stderr, "{prefix}{fmt_inner}\\n"{args});  {parts.marker}'


_TRACE_GENERATORS: dict[TraceKind, TraceEmitter] = {
    TraceKind.PYTHON_FSTRING: _gen_trace_python,
    TraceKind.JS_TEMPLATE_LITERAL: _gen_trace_js,
    TraceKind.BASH_PRINTF: _gen_trace_bash,
    TraceKind.POWERSHELL_WRITE_ERROR: _gen_trace_powershell,
    TraceKind.RUBY_PUTS_INSPECT: _gen_trace_ruby,
    TraceKind.R_CAT_STDERR: _gen_trace_r,
    TraceKind.JAVA_PRINTF: _gen_trace_java,
    TraceKind.KOTLIN_PRINTLN: _gen_trace_kotlin,
    TraceKind.LUA_IO_STDERR: _gen_trace_lua,
    TraceKind.GO_FPRINTF: _gen_trace_go,
    TraceKind.RUST_DBG: _gen_trace_rust,
    TraceKind.CSHARP_INTERP: _gen_trace_csharp,
    TraceKind.PHP_ERROR_LOG: _gen_trace_php,
    TraceKind.C_FPRINTF: _gen_trace_c,
}


# Each logpoint emitter is a tuple-style callable: (config, escaped_msg, marker) -> str.
# Logpoints are simpler than full traces (just a message), so a 3-arg signature
# suffices instead of building a full _TraceParts.
LogpointEmitter = Callable[[LanguageConfig, str, str], str]


def _gen_logpoint_python(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'print(f"[LOGPOINT] {escaped_msg}")  {marker}'


def _gen_logpoint_js(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f"console.log(`[LOGPOINT] {escaped_msg}`);  {marker}"


def _gen_logpoint_bash(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f"printf '[LOGPOINT] {escaped_msg}\\n' >&2  {marker}"


def _gen_logpoint_powershell(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'[Console]::Error.WriteLine("[LOGPOINT] {escaped_msg}")  {marker}'


def _gen_logpoint_ruby(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'STDERR.puts "[LOGPOINT] {escaped_msg}"  {marker}'


def _gen_logpoint_r(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'cat(sprintf("[LOGPOINT] {escaped_msg}\\n"), file=stderr())  {marker}'


def _gen_logpoint_java(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'System.err.printf("[LOGPOINT] {escaped_msg}%n");  {marker}'


def _gen_logpoint_kotlin(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'System.err.println("[LOGPOINT] {escaped_msg}")  {marker}'


def _gen_logpoint_lua(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'io.stderr:write(string.format("[LOGPOINT] {escaped_msg}\\n"))  {marker}'


def _gen_logpoint_go(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'fmt.Fprintf(os.Stderr, "[LOGPOINT] {escaped_msg}\\n")  {marker}'


def _gen_logpoint_rust(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'eprintln!("[LOGPOINT] {escaped_msg}");  {marker}'


def _gen_logpoint_csharp(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'Console.Error.WriteLine($"[LOGPOINT] {escaped_msg}");  {marker}'


def _gen_logpoint_php(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'fwrite(STDERR, sprintf("[LOGPOINT] {escaped_msg}\\n"));  {marker}'


def _gen_logpoint_c(config: LanguageConfig, escaped_msg: str, marker: str) -> str:
    return f'fprintf(stderr, "[LOGPOINT] {escaped_msg}\\n");  {marker}'


_LOGPOINT_GENERATORS: dict[TraceKind, LogpointEmitter] = {
    TraceKind.PYTHON_FSTRING: _gen_logpoint_python,
    TraceKind.JS_TEMPLATE_LITERAL: _gen_logpoint_js,
    TraceKind.BASH_PRINTF: _gen_logpoint_bash,
    TraceKind.POWERSHELL_WRITE_ERROR: _gen_logpoint_powershell,
    TraceKind.RUBY_PUTS_INSPECT: _gen_logpoint_ruby,
    TraceKind.R_CAT_STDERR: _gen_logpoint_r,
    TraceKind.JAVA_PRINTF: _gen_logpoint_java,
    TraceKind.KOTLIN_PRINTLN: _gen_logpoint_kotlin,
    TraceKind.LUA_IO_STDERR: _gen_logpoint_lua,
    TraceKind.GO_FPRINTF: _gen_logpoint_go,
    TraceKind.RUST_DBG: _gen_logpoint_rust,
    TraceKind.CSHARP_INTERP: _gen_logpoint_csharp,
    TraceKind.PHP_ERROR_LOG: _gen_logpoint_php,
    TraceKind.C_FPRINTF: _gen_logpoint_c,
}


# Each escape helper protects the corresponding emitter's interpolation syntax —
# Python f-strings escape `{}`, JS template literals escape `${}` and backticks.
EscapeFn = Callable[[str], str]


_ESCAPE_FUNCTIONS: dict[TraceKind, EscapeFn] = {
    TraceKind.PYTHON_FSTRING: _escape_for_python_fstring,
    TraceKind.JS_TEMPLATE_LITERAL: _escape_for_js_template_literal,
    TraceKind.BASH_PRINTF: _escape_for_bash_single_quote,
    TraceKind.POWERSHELL_WRITE_ERROR: _escape_for_powershell_double_quote,
    TraceKind.RUBY_PUTS_INSPECT: _escape_for_ruby_double_quote,
    TraceKind.R_CAT_STDERR: _escape_for_r_double_quote,
    TraceKind.JAVA_PRINTF: _escape_for_java_string,
    TraceKind.KOTLIN_PRINTLN: _escape_for_kotlin_double_quote,
    TraceKind.LUA_IO_STDERR: _escape_for_lua_double_quote,
    TraceKind.GO_FPRINTF: _escape_for_go_string,
    TraceKind.RUST_DBG: _escape_for_rust_string,
    TraceKind.CSHARP_INTERP: _escape_for_csharp_interp,
    TraceKind.PHP_ERROR_LOG: _escape_for_php_double_quote,
    TraceKind.C_FPRINTF: _escape_for_c_string,
}


def _generate_trace_statement(
    config: LanguageConfig,
    variables: list[str] | None,
    message: str | None,
    file_path: str,
    line_number: int,
    include_timestamp: bool,
    include_location: bool,
) -> str:
    """Generate a trace print/console.log statement for the language in `config`."""
    escape = _ESCAPE_FUNCTIONS[config.trace_kind]
    parts = _TraceParts(
        config=config,
        variables=variables,
        message=message,
        file_name=escape(Path(file_path).name),
        line_number=line_number,
        include_timestamp=include_timestamp,
        include_location=include_location,
        marker=f"{config.comment_prefix} {_TRACE_MARKER}",
    )
    return _TRACE_GENERATORS[config.trace_kind](parts)


def _generate_breakpoint_code(
    config: LanguageConfig,
    condition: str | None = None,
    log_message: str | None = None,
) -> str:
    """Generate breakpoint code for the language in `config`.

    Languages without a native in-source breakpoint statement (empty
    `breakpoint_stmt`) reach here only via `source_add_breakpoint`, which
    rejects them up front — so any call here is guaranteed to have one.
    """
    marker = f"{config.comment_prefix} {_TRACE_MARKER}"

    if log_message:
        escape = _ESCAPE_FUNCTIONS[config.trace_kind]
        return _LOGPOINT_GENERATORS[config.trace_kind](config, escape(log_message), marker)

    if condition:
        wrapped = config.conditional_template.format(
            cond=condition, bp=config.breakpoint_stmt
        )
        return f"{wrapped}  {marker}"

    return f"{config.breakpoint_stmt}  {marker}"


async def source_add_breakpoint(
    file_path: Annotated[
        str,
        Field(description=(
            "Absolute path to a source file in any DAP-supported language: "
            ".py .js .jsx .ts .tsx .mjs .cjs .go .rs .rb .java .kt .kts "
            ".cs .php .lua .sh .bash .r .ps1 .c .cc .cpp .cxx .h .hpp"
        )),
    ],
    line_number: Annotated[
        int,
        Field(description="Line number (1-based) where to set the breakpoint", ge=1),
    ],
    mode: Annotated[
        Literal["dap", "source", "auto"],
        Field(
            description=(
                "'source' injects breakpoint()/debugger; into code, "
                "'dap' uses Debug Adapter Protocol (requires active session; "
                "only languages with a registered DAP adapter), "
                "'auto' tries DAP first then falls back to source"
            )
        ),
    ] = "auto",
    condition: Annotated[
        str | None,
        Field(description="Optional condition expression (e.g., 'x > 10')"),
    ] = None,
    log_message: Annotated[
        str | None,
        Field(
            description=(
                "If provided, creates a logpoint that logs this message "
                "instead of breaking execution"
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Inject a native breakpoint statement into source code (MODIFIES THE FILE).

    This is SOURCE INJECTION — it edits the file to add a language-native
    breakpoint statement. For DAP runtime breakpoints during an active
    debug session, use `debug_breakpoint` instead.

    LANGUAGES WITH A NATIVE BREAKPOINT STATEMENT (mode='source' supported):
    - Python (.py): breakpoint()
    - JS/TS (.js, .jsx, .mjs, .cjs, .ts, .tsx): debugger;
    - Ruby (.rb): binding.irb
    - R (.r): browser()
    - C# (.cs): System.Diagnostics.Debugger.Break();
    - PowerShell (.ps1): Wait-Debugger

    LANGUAGES WITHOUT A NATIVE BREAKPOINT STATEMENT (returns
    no_native_breakpoint error — use debug_session/debug_breakpoint):
    - Go, Rust, Java, Kotlin, PHP, Lua, Bash, C/C++

    WHEN TO USE:
    - Persistent breakpoints that trigger when the script runs normally
    - Using an external debugger (VS Code, PyCharm, browser DevTools)
    - For languages without a registered DAP adapter, or when the
      modification needs to persist across runs

    NOTE: This modifies the source file. Use source_remove_injections to clean up.
    """
    try:
        path = Path(file_path).resolve()

        # Validate file exists
        if not path.exists():
            return file_not_found_error(file_path)
        if not path.is_file():
            return debug_error("not_a_file", f"Not a file: {file_path}")

        # Get language configuration
        config = _get_language_config(path)
        if not config:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            return unsupported_extension_error(path.suffix, supported)

        # Read file
        lines = await _read_file_lines(path)
        if line_number > len(lines):
            return line_out_of_range_error(line_number, len(lines))

        # Determine actual mode
        actual_mode: Literal["dap", "source"] = "source"
        # Match _get_language_config: extensions are registered lowercase,
        # but the filesystem may hand us mixed case (e.g. Script.PY on Windows).
        ext = path.suffix.lower()
        if mode == "dap":
            # DAP mode requires a registered adapter for the file's extension.
            # Languages without one fall back to source-injection via mode='source'.
            if REGISTRY.get_by_extension(ext) is None:
                return debug_error(
                    "dap_adapter_not_registered",
                    f"No DAP adapter registered for extension '{ext}'. "
                    f"Use mode='source' for {config.name} instead.",
                )
            # Check if DAP session is active
            if not _manager.is_dap_active():
                return debug_error(
                    "no_session",
                    "No active DAP session. Use debug_session first, or use mode='source'",
                )
            actual_mode = "dap"
        elif mode == "auto":
            # Prefer DAP only when a session is already running for an adapter
            # registered to this extension; otherwise fall back to source.
            if _manager.is_dap_active() and REGISTRY.get_by_extension(ext) is not None:
                actual_mode = "dap"
            else:
                actual_mode = "source"

        bp_id = _generate_id("bp")

        if actual_mode == "source":
            # Reject source-mode for languages with no in-source breakpoint
            # statement (e.g. bash, go, java). The DAP path via debug_breakpoint
            # is the only way to pause those languages.
            if not config.breakpoint_stmt:
                return debug_error(
                    "no_native_breakpoint",
                    f"Language '{config.name}' has no in-source breakpoint statement.",
                    f"Start a DAP session with debug_session(language='{config.name}', "
                    f"program=...) and use debug_breakpoint instead.",
                )
            # Create backup
            await _create_backup(path)

            # Get the target line and its indentation
            target_line = lines[line_number - 1]
            indent = _get_indentation(target_line)

            # Generate breakpoint code for the language
            bp_code = _generate_breakpoint_code(config, condition, log_message)

            # Insert breakpoint before the target line
            new_line = f"{indent}{bp_code}\n"
            lines.insert(line_number - 1, new_line)

            # Write back
            await _write_file_lines(path, lines)

            # Store breakpoint
            bp = Breakpoint(
                id=bp_id,
                file_path=str(path),
                line_number=line_number,
                mode="source",
                condition=condition,
                log_message=log_message,
                original_line=target_line.rstrip("\n\r"),
            )
            await _manager.add_breakpoint(bp)

            return {
                "status": "success",
                "breakpoint_id": bp_id,
                "mode": "source",
                "language": config.name,
                "file": str(path),
                "line": line_number,
                "injected_code": bp_code,
                "message": f"Breakpoint added at {path.name}:{line_number}",
            }

        else:
            return debug_error(
                "dap_not_supported",
                "DAP line-specific breakpoints require source injection.",
                "Use mode='source' instead",
            )

    except Exception as e:
        logger.exception("Failed to add breakpoint: %s", e)
        return debug_error("unexpected_error", str(e))


async def source_remove_injections(
    injection_type: Annotated[
        Literal["breakpoint", "trace", "all"],
        Field(
            description=(
                "'breakpoint': remove injected breakpoints; "
                "'trace': remove injected trace statements; "
                "'all': remove both breakpoints and traces"
            )
        ),
    ] = "all",
    injection_id: Annotated[
        str | None,
        Field(description="Specific injection ID to remove (breakpoint or trace ID)"),
    ] = None,
    file_path: Annotated[
        str | None,
        Field(description="Remove all injections from this file"),
    ] = None,
    remove_all: Annotated[
        bool,
        Field(description="Remove all injections of the specified type"),
    ] = False,
) -> dict[str, Any]:
    """Remove injected breakpoints and/or trace statements from source files.

    Works for every language source_inject_trace and source_inject_region_trace
    can target — Python, JS/TS, Go, Rust, Ruby, Java, Kotlin, C#, PHP, Lua,
    Bash, R, PowerShell, C/C++. Identifies injected lines by the
    [DEBUG_TRACE] marker regardless of comment style (#, //, --).

    WHEN TO USE:
    - To clean up after using source_add_breakpoint or source_inject_trace
    - To remove injected code from source files
    - Use injection_type='all' to clean up everything at once

    NOTE: This is for source-injected code. DAP breakpoints (from debug_breakpoint)
    are automatically cleared when the debug session ends.
    """
    try:
        if not injection_id and not file_path and not remove_all:
            return debug_error(
                "missing_parameter",
                "Specify injection_id, file_path, or remove_all=True",
                "Provide injection_id, file_path, or set remove_all=True",
            )

        removed_breakpoints = []
        removed_traces = []
        errors = []

        # Determine which types to process
        process_breakpoints = injection_type in ("breakpoint", "all")
        process_traces = injection_type in ("trace", "all")

        # Collect items to remove
        breakpoints_to_remove: list[Breakpoint] = []
        traces_to_remove: list[InjectedTrace] = []

        if process_breakpoints:
            if remove_all:
                breakpoints_to_remove = await _manager.get_breakpoints()
            elif file_path:
                breakpoints_to_remove = await _manager.get_breakpoints(file_path)
            elif injection_id and injection_id.startswith("bp-"):
                for b in await _manager.get_breakpoints():
                    if b.id == injection_id:
                        breakpoints_to_remove = [b]
                        break

        if process_traces:
            if remove_all:
                traces_to_remove = await _manager.get_traces()
            elif file_path:
                traces_to_remove = await _manager.get_traces(file_path)
            elif injection_id and injection_id.startswith("trace-"):
                for t in await _manager.get_traces():
                    if t.id == injection_id:
                        traces_to_remove = [t]
                        break

        if not breakpoints_to_remove and not traces_to_remove:
            return {"status": "success", "removed": 0, "message": "No injections found"}

        # Group all items by file for efficient processing
        files_to_process: dict[str, dict[str, list]] = {}

        for bp in breakpoints_to_remove:
            if bp.file_path not in files_to_process:
                files_to_process[bp.file_path] = {"breakpoints": [], "traces": []}
            files_to_process[bp.file_path]["breakpoints"].append(bp)

        for trace in traces_to_remove:
            if trace.file_path not in files_to_process:
                files_to_process[trace.file_path] = {"breakpoints": [], "traces": []}
            files_to_process[trace.file_path]["traces"].append(trace)

        async def _purge_items(
            items: dict[str, list],
            removed_breakpoints: list[str],
            removed_traces: list[str],
        ) -> None:
            """Remove breakpoints and traces from manager state and record their IDs."""
            for bp in items["breakpoints"]:
                await _manager.remove_breakpoint(bp.id)
                removed_breakpoints.append(bp.id)
            for trace in items["traces"]:
                await _manager.remove_trace(trace.id)
                removed_traces.append(trace.id)

        # Process each file
        for fpath, items in files_to_process.items():
            path = Path(fpath)

            # Handle missing files
            if not path.exists():
                await _purge_items(items, removed_breakpoints, removed_traces)
                continue

            try:
                try:
                    lines = await _read_file_lines(path)

                    # Remove all lines containing our trace marker (from bottom up)
                    indices_to_remove = [
                        i for i, line in enumerate(lines) if _TRACE_MARKER in line
                    ]
                    for idx in reversed(indices_to_remove):
                        lines.pop(idx)

                    await _write_file_lines(path, lines)
                except Exception as e:
                    errors.append(f"{fpath}: {e}")
            finally:
                # Purge state even on I/O failure: leaving entries in-memory
                # would otherwise cause the same unreadable/unwritable file to
                # be re-processed on every subsequent call (phantom entries).
                # Callers inspect `errors` to see which files still hold markers.
                await _purge_items(items, removed_breakpoints, removed_traces)

        result: dict[str, Any] = {
            "status": "success",
            "removed_breakpoints": len(removed_breakpoints),
            "removed_traces": len(removed_traces),
            "total_removed": len(removed_breakpoints) + len(removed_traces),
            "removed_ids": removed_breakpoints + removed_traces,
        }
        if errors:
            result["errors"] = errors

        return result

    except Exception as e:
        logger.exception("Failed to remove injections: %s", e)
        return debug_error("unexpected_error", str(e))


async def source_list_injections(
    injection_type: Annotated[
        Literal["breakpoint", "trace", "all"],
        Field(
            description=(
                "'breakpoint': list only breakpoints; "
                "'trace': list only traces; "
                "'all': list both"
            )
        ),
    ] = "all",
    file_path: Annotated[
        str | None,
        Field(description="Filter injections by file path"),
    ] = None,
    prune: Annotated[
        bool,
        Field(description="If True, delete entries whose files no longer exist before listing"),
    ] = False,
) -> dict[str, Any]:
    """List all source-injected breakpoints and/or traces.

    WHEN TO USE:
    - To see what is injected in source files
    - To get IDs for source_remove_injections
    - With prune=True to clean up stale entries from deleted files
    """
    try:
        pruned_counts: tuple[int, int] | None = None
        if prune:
            pruned_counts = await _manager.prune_missing()

        # Always include all keys for consistent response structure
        result: dict[str, Any] = {
            "status": "success",
            "dap_session_active": _manager.is_dap_active(),
            "breakpoints": [],
            "breakpoint_count": 0,
            "traces": [],
            "trace_count": 0,
        }
        if pruned_counts is not None:
            result["pruned"] = {
                "breakpoints": pruned_counts[0],
                "traces": pruned_counts[1],
            }

        if injection_type in ("breakpoint", "all"):
            breakpoints = await _manager.get_breakpoints(file_path)
            result["breakpoints"] = [
                {
                    "id": bp.id,
                    "file": bp.file_path,
                    "line": bp.line_number,
                    "mode": bp.mode,
                    "condition": bp.condition,
                    "log_message": bp.log_message,
                    "created_at": bp.created_at,
                }
                for bp in breakpoints
            ]
            result["breakpoint_count"] = len(breakpoints)

        if injection_type in ("trace", "all"):
            traces = await _manager.get_traces(file_path)
            result["traces"] = [
                {
                    "id": t.id,
                    "file": t.file_path,
                    "line": t.line_number,
                    "trace_code": t.trace_code,
                    "created_at": t.created_at,
                }
                for t in traces
            ]
            result["trace_count"] = len(traces)

        return result

    except Exception as e:
        logger.exception("Failed to list injections: %s", e)
        return debug_error("unexpected_error", str(e))


async def source_inject_trace(
    file_path: Annotated[
        str,
        Field(description=(
            "Absolute path to a source file in any DAP-supported language: "
            ".py .js .jsx .ts .tsx .mjs .cjs .go .rs .rb .java .kt .kts "
            ".cs .php .lua .sh .bash .r .ps1 .c .cc .cpp .cxx .h .hpp"
        )),
    ],
    line_number: Annotated[
        int,
        Field(
            description="Line number (1-based) after which to inject the trace",
            ge=1,
        ),
    ],
    variables: Annotated[
        list[str] | None,
        Field(description="List of variable names to log (e.g., ['x', 'y', 'result'])"),
    ] = None,
    message: Annotated[
        str | None,
        Field(description="Custom message to include in the trace output"),
    ] = None,
    include_timestamp: Annotated[
        bool,
        Field(description="Include timestamp in trace output"),
    ] = True,
    include_location: Annotated[
        bool,
        Field(description="Include file:line in trace output"),
    ] = True,
) -> dict[str, Any]:
    """Inject a stderr trace statement into source code to log variable values (modifies the file).

    Each language uses its idiomatic logging form:
    - Python (.py): print(f"[TRACE] ... | x={x!r}")
    - JS/TS (.js, .jsx, .mjs, .cjs, .ts, .tsx): console.log(`[TRACE] ... | x=${{JSON.stringify(x)}}`)
    - Go (.go): fmt.Fprintf(os.Stderr, "[TRACE] ... | x=%+v\\n", x)  (caller must import fmt+os+time)
    - Rust (.rs): dbg!(&x, &y) when only `variables` are given (auto file:line); else eprintln!
    - Ruby (.rb): STDERR.puts "[TRACE] ... | x=#{{x.inspect}}"
    - Java (.java): System.err.printf("[TRACE] ... | x=%s%n", x)
    - Kotlin (.kt, .kts): System.err.println("[TRACE] ... | x=$x")
    - C# (.cs): Console.Error.WriteLine($"[TRACE] ... | x={{x}}")
    - PHP (.php): fwrite(STDERR, sprintf("[TRACE] ... | x=%s\\n", var_export($x, true)))
    - Lua (.lua): io.stderr:write(string.format("[TRACE] ... | x=%s\\n", tostring(x)))
    - Bash (.sh, .bash): printf '[TRACE] ... | x=%s\\n' "${{x}}" >&2
    - R (.r): cat(sprintf("[TRACE] ... | x=%s\\n", deparse(x)), file=stderr())
    - PowerShell (.ps1): [Console]::Error.WriteLine("[TRACE] ... | x=$($x | ConvertTo-Json -Compress)")
    - C/C++ (.c, .cc, .cpp, .cxx, .h, .hpp): fprintf(stderr, "[TRACE] message\\n")
      — message-only; `variables` are ignored because C has no generic format specifier.

    WHEN TO USE:
    - When you want to see variable values by running the script normally (not in debugger)
    - When you want to trace execution flow with print/console.log statements
    - Quick alternative to full debugging

    WHEN NOT TO USE:
    - For interactive debugging (any language with a registered DAP adapter) —
      use debug_session + debug_eval instead.
    - For region/flow tracing in bash or powershell — use source_inject_region_trace.

    NOTE: This modifies the source file. Use source_remove_injections to clean up.
    """
    try:
        path = Path(file_path).resolve()

        # Validate file exists
        if not path.exists():
            return file_not_found_error(file_path)
        if not path.is_file():
            return debug_error("not_a_file", f"Not a file: {file_path}")

        # Get language configuration
        config = _get_language_config(path)
        if not config:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            return unsupported_extension_error(path.suffix, supported)

        # Read file
        lines = await _read_file_lines(path)
        if line_number > len(lines):
            return line_out_of_range_error(line_number, len(lines))

        # Validate caller-supplied variable names BEFORE we mutate anything on
        # disk: the generator embeds them into executable code, so invalid names
        # must fail fast to prevent code injection.
        if variables:
            for var in variables:
                error = _validate_variable_name(var, config.name)
                if error is not None:
                    return error

        # Create backup
        await _create_backup(path)

        # Get the target line and its indentation
        target_line = lines[line_number - 1]
        indent = _get_indentation(target_line)

        # Generate trace statement for the language
        trace_code = _generate_trace_statement(
            config=config,
            variables=variables,
            message=message,
            file_path=str(path),
            line_number=line_number,
            include_timestamp=include_timestamp,
            include_location=include_location,
        )

        # Insert trace after the target line
        new_line = f"{indent}{trace_code}\n"
        lines.insert(line_number, new_line)

        # Write back
        await _write_file_lines(path, lines)

        # Store trace
        trace_id = _generate_id("trace")
        trace = InjectedTrace(
            id=trace_id,
            file_path=str(path),
            line_number=line_number + 1,  # Actual line where trace was inserted
            trace_code=trace_code,
            original_line=target_line.rstrip("\n\r"),
        )
        await _manager.add_trace(trace)

        return {
            "status": "success",
            "trace_id": trace_id,
            "language": config.name,
            "file": str(path),
            "line": line_number + 1,
            "injected_code": trace_code,
            "message": f"Trace added after {path.name}:{line_number}",
        }

    except Exception as e:
        logger.exception("Failed to inject trace: %s", e)
        return debug_error("unexpected_error", str(e))


# Region-trace toggle pairs: language extension -> (on_stmt, off_stmt).
# Only included for languages whose native flow tracer is a one-statement
# toggle (no callback function required). Python's sys.settrace, Ruby's
# TracePoint, and Lua's debug.sethook all need callbacks and are deferred
# until there's a demonstrated need.
_REGION_TOGGLES: dict[str, tuple[str, str]] = {
    ".sh": ("set -x", "set +x"),
    ".bash": ("set -x", "set +x"),
    ".ps1": ("Set-PSDebug -Trace 1", "Set-PSDebug -Trace 0"),
}


async def source_inject_region_trace(
    file_path: Annotated[
        str,
        Field(description="Absolute path to a .sh / .bash / .ps1 file"),
    ],
    start_line: Annotated[
        int,
        Field(description="Line number (1-based) to start tracing BEFORE", ge=1),
    ],
    end_line: Annotated[
        int,
        Field(description="Line number (1-based) to stop tracing AFTER", ge=1),
    ],
) -> dict[str, Any]:
    """Wrap a line range in a language-native flow-trace toggle (modifies the file).

    Bash (.sh/.bash) → `set -x` / `set +x`. PowerShell (.ps1) →
    `Set-PSDebug -Trace 1` / `Set-PSDebug -Trace 0`. Both pairs are
    free, zero-install, and emit one trace line per executed source line
    when the script runs normally.

    WHEN TO USE:
    - "Why did this script behave wrong?" — flow tracing is faster than
      stepping through a debugger for short scripts.
    - You only have a Linux/Unix shell and don't want to install bashdb
      or the PowerShell DAP adapter.

    NOTE: Both toggles are GLOBAL — they affect subshells and called
    functions, not just the lexical region. Use source_remove_injections
    to clean up.

    For other languages, use source_inject_trace (single-line) or
    debug_session (interactive).
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return file_not_found_error(file_path)
        if not path.is_file():
            return debug_error("not_a_file", f"Not a file: {file_path}")

        ext = path.suffix.lower()
        toggle = _REGION_TOGGLES.get(ext)
        if toggle is None:
            supported = ", ".join(sorted(_REGION_TOGGLES.keys()))
            return unsupported_extension_error(path.suffix, supported)

        if end_line < start_line:
            return debug_error(
                "invalid_parameter",
                f"end_line ({end_line}) must be >= start_line ({start_line})",
                "Pass end_line greater than or equal to start_line",
            )

        config = _get_language_config(path)
        # Both region-trace languages are also in SUPPORTED_EXTENSIONS, so
        # config is non-None — assert keeps the type-checker happy.
        assert config is not None
        marker = f"{config.comment_prefix} {_TRACE_MARKER}"

        lines = await _read_file_lines(path)
        if end_line > len(lines):
            return line_out_of_range_error(end_line, len(lines))

        await _create_backup(path)

        on_stmt, off_stmt = toggle
        # Preserve the indentation of the surrounding lines so the toggles
        # land flush with neighbouring code rather than at column 0 inside
        # a nested function.
        on_indent = _get_indentation(lines[start_line - 1])
        off_indent = _get_indentation(lines[end_line - 1])
        on_line = f"{on_indent}{on_stmt}  {marker}\n"
        off_line = f"{off_indent}{off_stmt}  {marker}\n"

        # Two inserts: ON before start_line, OFF after end_line. The OFF
        # insertion index accounts for the line ON just added.
        lines.insert(start_line - 1, on_line)
        lines.insert(end_line + 1, off_line)
        await _write_file_lines(path, lines)

        group_id = _generate_id("region")
        on_trace = InjectedTrace(
            id=_generate_id("trace"),
            file_path=str(path),
            line_number=start_line,
            trace_code=on_line.strip(),
            original_line=lines[start_line].rstrip("\n\r"),
            group_id=group_id,
        )
        off_trace = InjectedTrace(
            id=_generate_id("trace"),
            file_path=str(path),
            line_number=end_line + 2,
            trace_code=off_line.strip(),
            original_line=lines[end_line].rstrip("\n\r"),
            group_id=group_id,
        )
        await _manager.add_trace(on_trace)
        await _manager.add_trace(off_trace)

        return {
            "status": "success",
            "group_id": group_id,
            "trace_ids": [on_trace.id, off_trace.id],
            "language": config.name,
            "file": str(path),
            "start_line": start_line,
            "end_line": end_line + 2,
            "injected_on": on_line.strip(),
            "injected_off": off_line.strip(),
            "message": (
                f"Region trace added at {path.name} lines "
                f"{start_line}..{end_line + 2}"
            ),
        }

    except Exception as e:
        logger.exception("Failed to inject region trace: %s", e)
        return debug_error("unexpected_error", str(e))


async def debug_server(
    action: Annotated[
        Literal["start", "stop", "status"],
        Field(
            description=(
                "'start': start debug server for IDE attachment; "
                "'stop': stop the debug server; "
                "'status': check if server is running"
            )
        ),
    ],
    port: Annotated[
        int,
        Field(description="Port for debugpy to listen on (only used with action='start')", ge=1024, le=65535),
    ] = DEFAULT_DEBUG_PORT,
    wait_for_client: Annotated[
        bool,
        Field(description="Block until a debugger client connects (only used with action='start')"),
    ] = False,
) -> dict[str, Any]:
    """Control a debug server for external IDE attachment (VS Code, PyCharm).

    WHEN TO USE:
    - 'start': when user wants to debug with their IDE
    - 'stop': to stop the debug server
    - 'status': to check if server is running

    WHEN NOT TO USE:
    - For interactive debugging controlled by this tool - use debug_launch instead

    After starting, connect IDE debugger to localhost:<port>.
    """
    try:
        # Clear stale handle before any branch reads is_dap_active() so
        # status/stop don't report a zombie process that has already exited.
        _refresh_dap_server_liveness()

        if action == "status":
            return {
                "status": "running" if _manager.is_dap_active() else "not_running",
                "port": _manager.dap_port(),
            }

        if action == "stop":
            if not _manager.is_dap_active():
                return {"status": "not_running", "message": "No debug session is active"}
            await _stop_dap_server_process()
            return {
                "status": "stopped",
                "message": "Debug server subprocess terminated; port released.",
            }

        # action == "start"
        if not _DEBUGPY_AVAILABLE:
            return dependency_missing_error("debugpy", "pip install debugpy")

        if _manager.is_dap_active():
            active_port = _manager.dap_port()
            return {
                "status": "already_running",
                "port": active_port,
                "message": f"Debug session already active on port {active_port}",
            }

        result = await _start_dap_server_process(port=port, wait_for_client=wait_for_client)
        if "error" in result:
            return debug_error("debug_server_failed", result["error"])
        return result

    except Exception as e:
        logger.exception("Failed to %s debug server: %s", action, e)
        return debug_error("unexpected_error", str(e))


def _refresh_dap_server_liveness() -> None:
    """Reset the manager state when the helper subprocess has exited on its own."""
    proc = _manager.dap_server_process()
    if proc is not None and proc.poll() is not None:
        _manager.stop_server()


async def _start_dap_server_process(
    port: int, wait_for_client: bool
) -> dict[str, Any]:
    """Spawn the out-of-process debugpy helper and block until it reports LISTENING."""
    helper_path = Path(__file__).parent / "_debugpy_server.py"
    cmd = [sys.executable, str(helper_path), str(port), "1" if wait_for_client else "0"]

    def _spawn() -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            close_fds=os.name != "nt",
        )

    proc = await asyncio.to_thread(_spawn)

    try:
        ready_line = await asyncio.wait_for(
            asyncio.to_thread(_read_line, proc), timeout=DAP_HELPER_READY_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.terminate()
        await asyncio.to_thread(proc.wait, DAP_HELPER_STOP_TIMEOUT)
        return {"error": f"debug_server helper did not report ready within {DAP_HELPER_READY_TIMEOUT}s"}

    if not ready_line or not ready_line.startswith("LISTENING"):
        stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")
        proc.terminate()
        await asyncio.to_thread(proc.wait, DAP_HELPER_STOP_TIMEOUT)
        return {"error": f"debug_server helper failed to listen: {ready_line or stderr.strip()}"}

    _manager.start_server(proc, port)
    mark_debug_server_started()

    if wait_for_client:
        connected = await asyncio.to_thread(_read_line, proc)
        if not connected or not connected.startswith("CONNECTED"):
            return {"error": f"debug_server helper reported unexpected status after wait_for_client: {connected}"}
        return {
            "status": "connected",
            "port": port,
            "message": f"Debugger connected on port {port}",
        }

    return {
        "status": "listening",
        "port": port,
        "message": f"Debug server listening on port {port}. Connect your IDE debugger to localhost:{port}",
        "vscode_launch_config": {
            "name": "Python: Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {"host": "localhost", "port": port},
        },
    }


def _read_line(proc: subprocess.Popen[bytes]) -> str:
    """Read one line from the helper subprocess stdout. Blocking; run via to_thread."""
    if proc.stdout is None:
        return ""
    raw = proc.stdout.readline()
    return raw.decode("utf-8", errors="replace").strip()


async def _stop_dap_server_process() -> None:
    """Terminate the helper subprocess and clear manager state."""
    proc = _manager.stop_server()
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        # terminate() on an already-reaped pid raises OSError on Linux /
        # ProcessLookupError on Windows — either way the handle is dead,
        # so the subsequent wait() will return immediately.
        pass
    try:
        await asyncio.to_thread(proc.wait, DAP_HELPER_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            # Same race as terminate() above: process may have exited between
            # terminate() and kill(). Fall through to the final wait().
            pass
        await asyncio.to_thread(proc.wait, DAP_HELPER_STOP_TIMEOUT)


async def cleanup_all_source_injections() -> dict[str, Any]:
    """Remove every tracked source injection from disk in one call.

    Used by debug_session(action='stop', cleanup_injections=True) to avoid
    leaving breakpoint() / debugger; / print() lines in source files after
    a debug session ends. Equivalent to source_remove_injections(injection_type='all', remove_all=True).
    """
    return await source_remove_injections(
        injection_type="all",
        injection_id=None,
        file_path=None,
        remove_all=True,
    )
