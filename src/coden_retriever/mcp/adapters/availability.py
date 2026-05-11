"""Structured preflight reports for debug adapter prerequisites."""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

DependencyKind: TypeAlias = Literal["runtime", "debugger", "platform"]
Resolver: TypeAlias = Callable[[], object | None]


@dataclass(frozen=True)
class DebugDependencyStatus:
    """One prerequisite needed to make debugging work for an adapter."""

    kind: DependencyKind
    name: str
    installed: bool
    detail: str
    install_hint: str = ""


@dataclass(frozen=True)
class DebugAvailability:
    """Structured answer for "can this language be debugged here?"."""

    language: str
    can_debug: bool
    reason: str
    dependencies: tuple[DebugDependencyStatus, ...]


def missing_dependency_status(
    *,
    kind: DependencyKind,
    name: str,
    install_hint: str,
    detail: str | None = None,
) -> DebugDependencyStatus:
    """Create a missing-dependency result with a stable default message."""

    return DebugDependencyStatus(
        kind=kind,
        name=name,
        installed=False,
        detail=detail or f"{name} is not installed",
        install_hint=install_hint,
    )


def resolver_dependency_status(
    *,
    kind: DependencyKind,
    name: str,
    install_hint: str,
    resolver: Resolver,
    detail: str | None = None,
) -> DebugDependencyStatus:
    """Probe one prerequisite using a cheap resolver function."""

    is_installed = bool(resolver())
    if is_installed:
        return DebugDependencyStatus(
            kind=kind,
            name=name,
            installed=True,
            detail="",
            install_hint="",
        )
    return missing_dependency_status(
        kind=kind,
        name=name,
        install_hint=install_hint,
        detail=detail,
    )


def binary_dependency_status(
    *,
    binary: str,
    kind: DependencyKind,
    name: str,
    install_hint: str,
    detail: str | None = None,
) -> DebugDependencyStatus:
    """Probe a PATH-resolved executable dependency."""

    return resolver_dependency_status(
        kind=kind,
        name=name,
        install_hint=install_hint,
        resolver=lambda: shutil.which(binary),
        detail=detail,
    )


def module_dependency_status(
    *,
    module_name: str,
    kind: DependencyKind,
    name: str,
    install_hint: str,
    detail: str | None = None,
) -> DebugDependencyStatus:
    """Probe an importable Python module dependency."""

    return resolver_dependency_status(
        kind=kind,
        name=name,
        install_hint=install_hint,
        resolver=lambda: importlib.util.find_spec(module_name),
        detail=detail,
    )


def availability_from_dependencies(
    language: str,
    dependencies: tuple[DebugDependencyStatus, ...],
) -> DebugAvailability:
    """Collapse dependency statuses into one caller-facing preflight answer."""

    for dependency in dependencies:
        if not dependency.installed:
            return DebugAvailability(
                language=language,
                can_debug=False,
                reason=dependency.detail,
                dependencies=dependencies,
            )
    return DebugAvailability(
        language=language,
        can_debug=True,
        reason=f"{language} debugging is available",
        dependencies=dependencies,
    )