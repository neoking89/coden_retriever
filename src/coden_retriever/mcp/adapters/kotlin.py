"""Kotlin adapter via `fwcd/kotlin-debug-adapter`.

Standalone JVM-hosted DAP bridge; transport is stdio. The upstream release
zip unpacks to `adapter/bin/kotlin-debug-adapter` (POSIX) and
`adapter/bin/kotlin-debug-adapter.bat` (Windows) — names pinned by
`adapter/build.gradle.kts::tasks.startScripts.applicationName` upstream.

`program` in the launch body is the Kotlin main class (e.g., `"HelloKt"` for
a top-level `fun main()` compiled from `hello.kt`), NOT a file path — the
Kotlin convention differs from JVM-ecosystem adapters like java-debug.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .availability import (
    DebugDependencyStatus,
    binary_dependency_status,
    resolver_dependency_status,
)
from .base import DebugAdapter, LaunchConfig
from .registry import REGISTRY

_JAVA_BINARY = "java"
_WRAPPER_POSIX = "kotlin-debug-adapter"
_WRAPPER_WINDOWS = "kotlin-debug-adapter.bat"
_ENV_OVERRIDE = "KOTLIN_DEBUG_ADAPTER"
_ADAPTER_ID = "kotlin"
_INSTALL_HINT = (
    "Install: download adapter.zip from "
    "https://github.com/fwcd/kotlin-debug-adapter/releases, unpack, and "
    "place adapter/bin/kotlin-debug-adapter on PATH "
    f"(or set {_ENV_OVERRIDE} to the absolute wrapper path)."
)
_JAVA_RUNTIME_HINT = "Install Java and ensure `java` is on PATH"


class KotlinAdapter(DebugAdapter):
    """fwcd/kotlin-debug-adapter DAP adapter for Kotlin."""

    name = "kotlin"
    file_extensions = (".kt", ".kts")
    transport_type = "stdio"
    adapter_id = _ADAPTER_ID
    program_is_class_name = True
    # kotlin-debug-adapter merges the configurationDone response into the
    # launch response — it never emits a standalone configurationDone one,
    # so awaiting here times out. Fire-and-forget matches observed behaviour.
    configuration_done_fire_and_forget = True

    def _resolve_wrapper(self) -> str | None:
        override = os.environ.get(_ENV_OVERRIDE)
        if override and os.path.isfile(override):
            return override
        candidate = _WRAPPER_WINDOWS if sys.platform == "win32" else _WRAPPER_POSIX
        return shutil.which(candidate)

    def detect_installed(self) -> tuple[bool, str]:
        if self._resolve_wrapper() is None:
            return (False, _INSTALL_HINT)
        return (True, "")

    def dependency_statuses(self) -> tuple[DebugDependencyStatus, ...]:
        return (
            binary_dependency_status(
                binary=_JAVA_BINARY,
                kind="runtime",
                name="Java runtime",
                install_hint=_JAVA_RUNTIME_HINT,
            ),
            resolver_dependency_status(
                kind="debugger",
                name="kotlin-debug-adapter",
                install_hint=_INSTALL_HINT,
                resolver=lambda: self._resolve_wrapper(),
            ),
        )

    def build_launch_argv(self, cfg: LaunchConfig, port: int | None = None) -> list[str]:
        wrapper = self._resolve_wrapper()
        if wrapper is None:
            raise RuntimeError(_INSTALL_HINT)
        # Win32 CreateProcess(shell=False) cannot exec .bat wrappers directly;
        # route through an absolute cmd.exe resolved via %ComSpec% (bare "cmd"
        # is not on PATH under Popen's exec path).
        if sys.platform == "win32":
            comspec = os.environ.get("ComSpec") or r"C:\Windows\System32\cmd.exe"
            return [comspec, "/c", wrapper]
        return [wrapper]

    def launch_request_args(self, cfg: LaunchConfig) -> dict[str, Any]:
        project_root, main_class = self._resolve_project_and_main(cfg)
        body: dict[str, Any] = {
            "projectRoot": project_root,
            "mainClass": main_class,
            "args": list(cfg.args),
            "stopOnEntry": cfg.stop_on_entry,
        }
        for key in ("classPath", "vmArguments", "modulePaths"):
            value = cfg.extras.get(key)
            if value is not None:
                body[key] = value
        return body

    def _resolve_project_and_main(self, cfg: LaunchConfig) -> tuple[str, str]:
        """Derive (projectRoot, mainClass) from cfg.program when it's a .kt source.

        Kotlin's top-level `fun main()` in `fixture.kt` compiles to a class
        named `FixtureKt` (file-name stem + `Kt`). Matches kotlinc's default
        class-naming convention. Callers can override via cfg.extras.

        For projectRoot, walk up any `src/main/{kotlin,java}` or `src/test/...`
        ancestor so the path points at the Gradle/Maven project root (where
        kotlin-debug-adapter's `ProjectClassesResolver` probes for
        `build/classes/{kotlin,java}/{main,test}` and `target/classes`).
        Without this, a source at `.../src/main/kotlin/fixture.kt` would
        yield projectRoot = `.../src/main/kotlin/`, the classes dir would
        not be discovered, and `setBreakpoints` would verify against a
        source the debuggee VM never loads — breakpoints never hit.
        """
        explicit_root = cfg.extras.get("project_root") or cfg.cwd
        explicit_main = cfg.extras.get("main_class")
        if cfg.program.endswith((".kt", ".kts")):
            src = Path(cfg.program)
            project_root = (
                str(explicit_root) if explicit_root
                else self._climb_to_project_root(src)
            )
            main_class = str(explicit_main) if explicit_main else self._file_facade_class_name(src.stem)
            return project_root, main_class
        project_root = str(explicit_root) if explicit_root else os.getcwd()
        main_class = str(explicit_main) if explicit_main else cfg.program
        return project_root, main_class

    @staticmethod
    def _file_facade_class_name(stem: str) -> str:
        """Mirror kotlinc's `JvmAbi.getDefaultJvmInternalName` facade naming.

        kotlinc only upper-cases the first character of the file stem and
        preserves the rest verbatim (`myHandler.kt` -> `MyHandlerKt`,
        `URLHandler.kt` -> `URLHandlerKt`). Python's `str.capitalize()`
        also lower-cases every trailing character, so it diverges on
        camelCase / PascalCase / ALL-CAPS stems and produces a class
        name the JVM class loader cannot resolve.
        """
        if not stem:
            return "Kt"
        return f"{stem[0].upper()}{stem[1:]}Kt"

    @staticmethod
    def _climb_to_project_root(src: Path) -> str:
        """Walk up past `src/{main,test}/{kotlin,java}` or `src/<sourceSet>`.

        kotlin-debug-adapter's `ProjectClassesResolver` probes
        `<projectRoot>/build/classes/{kotlin,java}/{main,test}`,
        `<projectRoot>/build/resources/main`, `<projectRoot>/target/classes`,
        `<projectRoot>/target/test-classes`. These live at the Gradle/Maven
        project root, not under `src/main/kotlin`. Returns the source's
        parent when no recognisable source-set ancestor exists.
        """
        parent = src.parent
        parts = parent.parts
        _source_set_langs = ("kotlin", "java")
        _source_sets = ("main", "test")
        # Climb past `.../src/<set>/<lang>/` → drop last three components.
        if (
            len(parts) >= 3
            and parts[-1] in _source_set_langs
            and parts[-2] in _source_sets
            and parts[-3] == "src"
        ):
            return str(Path(*parts[:-3]))
        # Fallback: `.../src/<set>/` (language defaulted) — drop last two.
        if (
            len(parts) >= 2
            and parts[-1] in _source_sets
            and parts[-2] == "src"
        ):
            return str(Path(*parts[:-2]))
        return str(parent)


REGISTRY.register(KotlinAdapter())
