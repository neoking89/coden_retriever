"""Java adapter via `microsoft/java-debug` running inside Eclipse JDT LSP.

java-debug is NOT a standalone process — it lives inside the Eclipse JDT
language server, which exposes a `workspace/executeCommand(
"vscode.java.startDebugSession")` LSP call that returns a random local
port for java-debug's DAP socket. Phase 6 obtains that port in
`prepare_launch` via `_lsp_bridge.request_debug_port`; the DAPClient
bypass (Phase 6 C1) then dials the returned port directly without
spawning a subprocess (argv is `[]`).

Three env vars govern the JDTLS side:
- `JDTLS_SOCKET` — host:port of an already-running JDTLS; skips spawn.
- `JDTLS_COMMAND` — argv override for spawning JDTLS.
- `JDTLS_WORKSPACE` — workspace path fallback when neither
  `cfg.extras["jdtls_workspace"]` nor `cfg.cwd` is set.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from ...language.loader import LanguageLoader
from ...language.parser_utils import get_or_create_parser
from . import _lsp_bridge
from .availability import (
    DebugDependencyStatus,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

logger = logging.getLogger(__name__)

_JAVA_BINARY = "java"
_JDTLS_BINARY = "jdtls"
# Cold-start on a fresh workspace is typically ~10s without bundles; with a
# `java-debug` bundle loaded via `initializationOptions.bundles` JDTLS needs
# to start Eclipse's OSGi runtime and activate the bundle before
# `executeCommand("vscode.java.startDebugSession")` resolves. 90s provides
# headroom. Override via cfg.extras["lsp_startup_timeout_seconds"].
_DEFAULT_LSP_TIMEOUT_SECONDS = 90.0
_ADAPTER_ID = "java"
_INSTALL_HINT = (
    "Install Eclipse JDT Language Server (`jdtls`). macOS: `brew install jdtls`. "
    "Linux/Windows: download from https://projects.eclipse.org/projects/eclipse.jdt.ls/releases "
    "and set JDTLS_COMMAND to the launch argv plus JDTLS_DEBUG_BUNDLES to the "
    "java-debug plugin jars, OR set JDTLS_SOCKET=host:port to point at an "
    "already-running server with java-debug enabled."
)
_JAVA_RUNTIME_HINT = "Install Java and ensure `java` is on PATH"
_JAVA_LANGUAGE = "java"
# Lazily constructed: `LanguageLoader()` performs tree-sitter grammar probing
# on instantiation and that cost shouldn't fall on imports of non-Java modules.
_JAVA_LOADER: LanguageLoader | None = None


def _get_java_loader() -> LanguageLoader:
    """Return the module-singleton ``LanguageLoader``, creating it on first use."""
    global _JAVA_LOADER
    if _JAVA_LOADER is None:
        _JAVA_LOADER = LanguageLoader()
    return _JAVA_LOADER


def _has_spawnable_jdtls() -> bool:
    """True when this process can launch JDTLS itself."""
    return bool(os.environ.get("JDTLS_COMMAND") or shutil.which(_JDTLS_BINARY))


def _has_configured_debug_bundles() -> bool:
    """True when java-debug bundles are configured for the JDTLS launch path."""
    return bool(_lsp_bridge._resolve_debug_bundles())


def _has_java_debug_transport() -> bool:
    """True when either an external debug-capable JDTLS exists or we can launch one."""
    if os.environ.get("JDTLS_SOCKET"):
        return True
    return _has_spawnable_jdtls() and _has_configured_debug_bundles()


class JavaAdapter(DebugAdapter):
    """microsoft/java-debug DAP adapter, coordinated via Eclipse JDT LSP.

    Socket transport; `build_launch_argv` returns `[]` because java-debug
    is never spawned as our child process — `prepare_launch` asks the
    running JDTLS for java-debug's port and the DAPClient dispatch branch
    (Phase 6 C1) dials that port directly.
    """

    name = "java"
    file_extensions = (".java",)
    transport_type = "socket"
    adapter_id = _ADAPTER_ID
    program_is_class_name = True

    def __init__(self) -> None:
        super().__init__()
        # Per-instance parser cache. Module-level mutable globals are a
        # code-review smell and make multi-instance lifetime bugs (e.g.
        # stale parsers after a tree-sitter upgrade) harder to reason
        # about. `get_or_create_parser` populates this dict in place.
        self._parser_cache: dict[str, Any] = {}

    def detect_installed(self) -> tuple[bool, str]:
        if _has_java_debug_transport():
            return (True, "")
        return (False, _INSTALL_HINT)

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            resolver_dependency_status(
                kind="runtime",
                name="Java runtime",
                install_hint=_JAVA_RUNTIME_HINT,
                resolver=lambda: os.environ.get("JDTLS_SOCKET") or shutil.which(_JAVA_BINARY),
            ),
            resolver_dependency_status(
                kind="debugger",
                name="JDTLS / java-debug",
                install_hint=_INSTALL_HINT,
                resolver=_has_java_debug_transport,
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        # Empty argv signals "no subprocess to spawn" to DAPClient. The
        # actual port comes from prepare_launch via _lsp_bridge.
        return []

    async def prepare_launch(self, cfg: LaunchConfig) -> int | None:
        """Ask the running JDTLS for java-debug's socket port via LSP."""
        workspace = self._resolve_workspace(cfg)
        timeout = cfg.extras.get("lsp_startup_timeout_seconds", _DEFAULT_LSP_TIMEOUT_SECONDS)
        return await _lsp_bridge.request_debug_port(workspace=workspace, timeout=timeout)

    def _resolve_workspace(self, cfg: LaunchConfig) -> str:
        """Five-step fallback.

        Order: cfg.extras → JDTLS_WORKSPACE env → parent dir of a `.java`
        program → cfg.cwd → os.getcwd(). Promoting the program's parent
        above cfg.cwd matters because the matrix test runner invokes the
        adapter without setting cfg.cwd, and os.getcwd() is usually the
        whole project root — JDTLS would then try to index every file in
        the repo, delaying (or failing) the debug-session handshake. When
        the user passes a bare class name instead of a file path, this
        branch is skipped and we fall through to cfg.cwd/os.getcwd() as
        before.
        """
        from_extras = cfg.extras.get("jdtls_workspace")
        if from_extras:
            return str(from_extras)
        from_env = os.environ.get("JDTLS_WORKSPACE")
        if from_env:
            return from_env
        if cfg.program.endswith(".java"):
            src = Path(cfg.program)
            if src.exists():
                return str(src.parent)
        if cfg.cwd:
            return cfg.cwd
        return os.getcwd()

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        # java-debug wants `args` as a single space-separated string, not a
        # list — a long-standing adapter-specific quirk documented upstream.
        main_class, derived_classpaths = self._resolve_main_class_and_classpaths(cfg)
        # java-debug silently refuses to emit `initialized` when cwd is an
        # empty string — empirically observed against microsoft/java-debug
        # 0.53.1. Fall back to the program's parent when cfg.cwd is unset.
        body: dict[str, Any] = {
            "mainClass": main_class,
            "args": " ".join(cfg.args),
            "cwd": cfg.cwd or self._default_cwd(cfg),
            "stopOnEntry": cfg.stop_on_entry,
        }
        if derived_classpaths and "classPaths" not in cfg.extras:
            body["classPaths"] = derived_classpaths
        # java-debug's evaluate request throws java.lang.IllegalStateException
        # when launch never declared a projectName — expression evaluation
        # looks up source-scoped symbols via JDTLS's project index. For
        # standalone-file debug sessions we use the parent dir name as a
        # stable project identifier; this matches how VS Code's Java
        # extension labels unmanaged folders.
        if "projectName" not in cfg.extras:
            body["projectName"] = self._default_project_name(cfg)
        for key in ("classPaths", "modulePaths", "vmArgs", "env", "projectName"):
            value = cfg.extras.get(key)
            if value is not None:
                body[key] = value
        return body

    def _default_project_name(self, cfg: LaunchConfig) -> str:
        if cfg.program.endswith(".java"):
            src = Path(cfg.program)
            if src.exists():
                return src.parent.name
        return "default"

    def _default_cwd(self, cfg: LaunchConfig) -> str:
        if cfg.program.endswith(".java"):
            src = Path(cfg.program)
            if src.exists():
                return str(src.parent)
        return os.getcwd()

    def _resolve_main_class_and_classpaths(
        self, cfg: LaunchConfig,
    ) -> tuple[str, list[str]]:
        """Return (mainClass, derived_classPaths) from cfg.program.

        If `cfg.program` is a `.java` file path, parse the first class
        declaration and package name via tree-sitter.
        Otherwise trust `cfg.program` as a fully-qualified class name and
        return an empty classpath list (caller may still override via
        `cfg.extras["classPaths"]`).
        """
        explicit = cfg.extras.get("mainClass")
        program = cfg.program
        if not program.endswith(".java"):
            return (str(explicit) if explicit else program, [])
        src = Path(program)
        package_name, class_name = _extract_java_package_and_main_class(
            src, self._parser_cache,
        )
        if explicit:
            main_class = str(explicit)
        elif package_name:
            main_class = f"{package_name}.{class_name}"
        else:
            main_class = class_name
        return main_class, _derive_java_classpaths(src, package_name)

    async def handle_reverse_request(
        self,
        command: str,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        # java-debug emits runInTerminal when console != "internal". We stay
        # in "internal" mode; acknowledging with an empty body keeps the
        # adapter happy without requiring a real terminal spawn. Phase 6 does
        # not implement external-terminal support for Java.
        if command == "runInTerminal":
            return (True, {})
        return await super().handle_reverse_request(command, arguments)


def _node_text(source_bytes: bytes, node: Any) -> str:
    """Decode a tree-sitter node slice from the original source bytes."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_java_package_and_main_class(
    src: Path, parser_cache: dict[str, Any],
) -> tuple[str | None, str]:
    """Return `(package_name, class_name)` from a Java source file.

    Falls back to `(None, src.stem)` when parsing is unavailable or the file
    does not expose a top-level class declaration.
    """
    try:
        source_bytes = src.read_bytes()
    except OSError:
        return None, src.stem

    parser = get_or_create_parser(_JAVA_LANGUAGE, _get_java_loader(), parser_cache)
    if parser is None:
        return None, src.stem

    try:
        tree = parser.parse(source_bytes)
    except Exception as exc:
        logger.debug("java tree-sitter parse failed for %s", src, exc_info=exc)
        return None, src.stem

    package_name: str | None = None
    classes: list[str] = []
    for child in tree.root_node.named_children:
        if child.type == "package_declaration" and package_name is None:
            for grandchild in child.named_children:
                if grandchild.type in {"scoped_identifier", "identifier"}:
                    package_name = _node_text(source_bytes, grandchild)
                    break
        elif child.type == "class_declaration":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                classes.append(_node_text(source_bytes, name_node))

    # Java enforces that a file's public top-level type's name equals the
    # filename stem, so prefer the class that matches src.stem over the
    # first-encountered one (which may be a package-private helper).
    stem = src.stem
    if stem in classes:
        class_name = stem
    elif classes:
        class_name = classes[0]
    else:
        class_name = stem
    return package_name, class_name


def _derive_java_classpaths(src: Path, package_name: str | None) -> list[str]:
    """Derive a classpath root from the source layout when it is unambiguous."""
    if package_name is None:
        return [str(src.parent)]

    package_parts = package_name.split(".")
    if len(package_parts) > len(src.parent.parts):
        return []
    if list(src.parent.parts[-len(package_parts):]) != package_parts:
        return []

    classpath_root = src.parent
    for _ in package_parts:
        classpath_root = classpath_root.parent
    return [str(classpath_root)]


REGISTRY.register(JavaAdapter())
