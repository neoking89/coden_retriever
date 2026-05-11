"""Adapter registry with name-indexed and extension-indexed lookup.

Shape mirrors `agent/commands.py` `CommandRegistry` — a thin dict wrapper
with explicit registration and no magic. A module-level `REGISTRY` singleton
is populated at import time by each `adapters/<lang>.py` module.
"""
from __future__ import annotations

from .availability import DebugAvailability
from .base import DebugAdapter

_ALL_LANGUAGES = "all"


class AdapterRegistry:
    """Dual-indexed store of `DebugAdapter` instances.

    Name conflicts and extension conflicts both raise `ValueError` — the
    refined plan rules out silent overrides and back-compat shims. If two
    adapters legitimately compete for the same extension (lldb-dap vs
    CodeLLDB for .rs), resolution is Phase 4 work.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, DebugAdapter] = {}
        self._by_ext: dict[str, DebugAdapter] = {}
        self._canonical_names: set[str] = set()

    def register(self, adapter: DebugAdapter) -> DebugAdapter:
        if adapter.name in self._by_name:
            raise ValueError(f"Adapter name already registered: {adapter.name!r}")
        for alias in adapter.language_aliases:
            if alias in self._by_name:
                owner = self._by_name[alias].name
                raise ValueError(
                    f"Language alias {alias!r} already registered to adapter {owner!r}"
                )
        for ext in adapter.file_extensions:
            if ext in self._by_ext:
                owner = self._by_ext[ext].name
                raise ValueError(
                    f"Extension {ext!r} already registered to adapter {owner!r}"
                )
        self._by_name[adapter.name] = adapter
        self._canonical_names.add(adapter.name)
        for alias in adapter.language_aliases:
            self._by_name[alias] = adapter
        for ext in adapter.file_extensions:
            self._by_ext[ext] = adapter
        return adapter

    def get_by_name(self, name: str) -> DebugAdapter | None:
        return self._by_name.get(name)

    def get_by_extension(self, ext: str) -> DebugAdapter | None:
        return self._by_ext.get(ext)

    def names(self) -> list[str]:
        """Return canonical adapter names only (aliases excluded).

        The MCP resolver uses this to build the "Known languages" hint, where
        aliases would add noise without identity value.
        """
        return sorted(self._canonical_names)

    def all(self) -> list[DebugAdapter]:
        return [self._by_name[name] for name in sorted(self._canonical_names)]

    def debug_availability(
        self, language: str = _ALL_LANGUAGES,
    ) -> DebugAvailability | tuple[DebugAvailability, ...]:
        """Preflight whether debugging is possible for one language or all."""

        if language == _ALL_LANGUAGES:
            return tuple(adapter.debug_availability() for adapter in self.all())

        adapter = self.get_by_name(language)
        if adapter is None:
            known_languages = ", ".join(self.names()) or "(none)"
            return DebugAvailability(
                language=language,
                can_debug=False,
                reason=(
                    f"No adapter registered for language '{language}'. "
                    f"Known languages: {known_languages}"
                ),
                dependencies=(),
            )
        return adapter.debug_availability()


REGISTRY = AdapterRegistry()


def check_debug_availability(
    language: str = _ALL_LANGUAGES,
) -> DebugAvailability | tuple[DebugAvailability, ...]:
    """Convenience wrapper around the process-global adapter registry."""

    return REGISTRY.debug_availability(language)
