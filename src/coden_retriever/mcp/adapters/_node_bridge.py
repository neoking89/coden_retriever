"""Shared resolver for Node.js-based DAP bridge adapters (Lua, Bash, PHP).

Three Phase-5 Tier-2 adapters are not native DAP binaries — they're
`node <bridge>.js` wrappers bundled as npm packages. The resolution
pattern is identical across all three: check an adapter-specific env
var first, then walk npm's global-install prefix for a known relative
path. Consolidated here so the shape stays one-source-of-truth per CLAUDE.md
DRY.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

NODE_BINARY = "node"
NPM_BINARY = "npm"
# 1-second cap keeps `detect_installed()` fast even when `npm` is present
# but slow (e.g., cold cache on Windows). Longer waits are paid at launch
# time if they're actually going to succeed.
_NPM_PREFIX_QUERY_TIMEOUT_SECONDS = 1.0


def resolve_bridge_script(
    env_var: str,
    relative_paths: tuple[str, ...],
) -> str | None:
    """Locate a Node-bundled DAP adapter script.

    Precedence:
      1. `env_var` pointing at an absolute file.
      2. Any `relative_path` joined to `npm config get prefix`.
      3. None — caller surfaces a `dependency_missing_error` hint.
    """
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).is_file():
        return env_path
    prefix = _npm_global_prefix()
    if prefix is None:
        return None
    for rel in relative_paths:
        candidate = Path(prefix) / rel
        if candidate.is_file():
            return str(candidate)
    return None


def _npm_global_prefix() -> str | None:
    """Return `npm config get prefix` output, or None if npm is absent.

    Any non-zero exit, timeout, or missing-binary case returns None so
    callers treat it uniformly as "bridge not discoverable" and fall
    through to the install-hint path.
    """
    if shutil.which(NPM_BINARY) is None:
        return None
    try:
        result = subprocess.run(
            [NPM_BINARY, "config", "get", "prefix"],
            capture_output=True, text=True,
            timeout=_NPM_PREFIX_QUERY_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    prefix = (result.stdout or "").strip()
    if not prefix or prefix.lower() == "undefined":
        return None
    return prefix
