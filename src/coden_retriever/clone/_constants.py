"""Token-budget primitives shared across clone-detection modes."""

from __future__ import annotations

from typing import Any

TOKEN_OVERHEAD_CLONES = 200
TOKEN_PER_CLONE_PAIR = 80
TOKEN_PER_COMBINED_CLONE_PAIR = 100


def clone_pair_text(pair: dict[str, Any]) -> str:
    """Build the per-pair text used for token-budget estimation."""
    e1, e2 = pair["entity1"], pair["entity2"]
    return f"{e1['name']} {e1['file']} {e2['name']} {e2['file']} {pair['suggested_action']}"
