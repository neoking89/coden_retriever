"""Persistent storage for DAP breakpoint presets.

Saves/loads breakpoint configurations to ~/.coden-retriever/dap_breakpoints.json
so they survive stop/launch cycles. Separate from debug_trace.py's source
injection state — this is for DAP-level breakpoints only.
"""
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Reserved preset name for auto-restore on launch (unscoped fallback for
# legacy callers that don't pass a program path).
AUTO_RESTORE_PRESET = "_auto"


def _auto_restore_key(program: str | None) -> str:
    """Preset key for a program's auto-restore set, or the unscoped fallback."""
    if not program:
        return AUTO_RESTORE_PRESET
    return f"{AUTO_RESTORE_PRESET}::{Path(program).resolve()}"

# State file name within ~/.coden-retriever/
STATE_FILENAME = "dap_breakpoints.json"


@dataclass
class BreakpointConfig:
    """A persistable breakpoint configuration (not tied to a live session)."""

    file: str
    line: int
    condition: str | None = None
    log_message: str | None = None


class BreakpointStore:
    """Persists DAP breakpoint presets to disk.

    Presets are named collections of breakpoint configs. The special "_auto"
    preset is used for auto-restore: saved on stop, applied on next launch.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or (Path.home() / ".coden-retriever")
        self._presets: dict[str, list[BreakpointConfig]] = {}
        self._loaded = False

    @property
    def _state_file(self) -> Path:
        return self._state_dir / STATE_FILENAME

    async def _load(self) -> None:
        """Load presets from disk (once)."""
        if self._loaded:
            return

        def _load_sync() -> dict[str, list[BreakpointConfig]]:
            if not self._state_file.exists():
                return {}
            try:
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                result: dict[str, list[BreakpointConfig]] = {}
                for name, configs in data.items():
                    result[name] = [BreakpointConfig(**cfg) for cfg in configs]
                return result
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"Failed to load breakpoint presets: {e}")
                return {}

        self._presets = await asyncio.to_thread(_load_sync)
        self._loaded = True

    async def _save(self) -> None:
        """Write presets to disk."""

        def _save_sync(presets: dict[str, list[dict]]) -> None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(presets, indent=2), encoding="utf-8"
            )

        serialized = {
            name: [asdict(cfg) for cfg in configs]
            for name, configs in self._presets.items()
        }
        await asyncio.to_thread(_save_sync, serialized)

    async def save_preset(
        self, name: str, breakpoints: list[BreakpointConfig],
    ) -> dict[str, Any]:
        """Save a named breakpoint preset."""
        await self._load()
        self._presets[name] = list(breakpoints)
        await self._save()
        return {"status": "success", "preset": name, "count": len(breakpoints)}

    async def load_preset(self, name: str) -> list[BreakpointConfig] | None:
        """Load a named preset. Returns None if not found."""
        await self._load()
        configs = self._presets.get(name)
        return list(configs) if configs else None

    async def list_presets(self) -> list[str]:
        """List all saved preset names."""
        await self._load()
        return sorted(self._presets.keys())

    async def delete_preset(self, name: str) -> bool:
        """Delete a preset. Returns True if it existed."""
        await self._load()
        if name in self._presets:
            del self._presets[name]
            await self._save()
            return True
        return False

    async def save_auto_restore(
        self, dap_breakpoints: dict[str, list], program: str | None = None,
    ) -> None:
        """Save current DAP breakpoints as the auto-restore preset.

        Args:
            dap_breakpoints: DAPClient.breakpoints.by_file dict (file -> list[DebugBreakpoint])
            program: program path to scope the preset by, so sessions on
                different programs don't leak breakpoints into each other.
        """
        configs = []
        for bps in dap_breakpoints.values():
            for bp in bps:
                configs.append(BreakpointConfig(
                    file=bp.file, line=bp.line,
                    condition=bp.condition, log_message=bp.log_message,
                ))
        if configs:
            await self.save_preset(_auto_restore_key(program), configs)

    async def get_auto_restore(
        self, program: str | None = None,
    ) -> list[BreakpointConfig] | None:
        """Load the auto-restore preset, scoped by program if provided.

        Falls back to the unscoped legacy preset only when a program-scoped
        preset is not found, to keep older saved state loadable.
        """
        scoped = await self.load_preset(_auto_restore_key(program))
        if scoped:
            return scoped
        if program:
            return await self.load_preset(AUTO_RESTORE_PRESET)
        return None


# Module-level singleton
_store: BreakpointStore | None = None


def get_breakpoint_store() -> BreakpointStore:
    """Get or create the global BreakpointStore."""
    global _store
    if _store is None:
        _store = BreakpointStore()
    return _store
