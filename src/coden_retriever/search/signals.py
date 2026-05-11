"""Ranking signal registry.

Each `Signal` declares its compute function, per-mode weights, optional class
aggregation strategy, and optional `--stats` column. Adding a new signal means
appending one entry to `SIGNALS`; nothing else in the engine needs editing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from ..config import Config

if TYPE_CHECKING:
    from .engine import SearchEngine

logger = logging.getLogger(__name__)

Mode = Literal["map", "query_bm25", "query_semantic"]


def sqrt_sum(values: list[float]) -> float:
    return math.sqrt(sum(values))


def max_plus_sum(values: list[float]) -> float:
    return max(values) + sum(values)


@dataclass(frozen=True)
class Signal:
    name: str
    compute: Callable[["SearchEngine", str], dict[str, float]]
    weights: dict[Mode, float]
    aggregate: Callable[[list[float]], float] | None = None
    column: tuple[str, str] | None = None


def _semantic(eng: "SearchEngine", query: str) -> dict[str, float]:
    if not eng._semantic_index:
        return {}
    try:
        return eng._semantic_index.score_all(query)
    except Exception as e:
        logger.warning(f"Semantic scoring failed: {e}. Returning empty.")
        return {}


SIGNALS: list[Signal] = [
    Signal("bm25",
           lambda e, q: e._bm25.score_all(q) if q else e._bm25.static_idf_score_all(),
           {"map": Config.MAP_WEIGHT_BM25, "query_bm25": Config.WEIGHT_BM25},
           column=("BM25", "{:<8.2f}")),
    Signal("semantic", _semantic,
           {"query_semantic": Config.WEIGHT_SEMANTIC},
           column=("Sem", "{:<6.3f}")),
    Signal("pr",
           lambda e, q: e._get_pagerank(e._bm25.score_all(q) if q else {}),
           {"map": Config.MAP_WEIGHT_PAGERANK,
            "query_bm25": Config.WEIGHT_PAGERANK_BM25,
            "query_semantic": Config.WEIGHT_PAGERANK_SEMANTIC},
           sqrt_sum, ("PR", "{:<8.5f}")),
    Signal("bt",
           lambda e, q: e._centrality.betweenness,
           {"map": Config.MAP_WEIGHT_BETWEENNESS,
            "query_bm25": Config.WEIGHT_BETWEENNESS_BM25,
            "query_semantic": Config.WEIGHT_BETWEENNESS_SEMANTIC},
           max_plus_sum, ("BT", "{:<8.5f}")),
    Signal("dispatcher",
           lambda e, q: e._get_dispatcher_scores(),
           {"map": Config.MAP_WEIGHT_DISPATCHER},
           sqrt_sum, ("Disp", "{:<8.2f}")),
    Signal("entry",
           lambda e, q: e._get_entry_scores(),
           {"map": Config.MAP_WEIGHT_ENTRY},
           sqrt_sum, ("Entry", "{:<8.5f}")),
    Signal("type_ref",
           lambda e, q: e._centrality.type_pagerank,
           {"map": Config.MAP_WEIGHT_TYPE_REF,
            "query_bm25": Config.WEIGHT_TYPE_REF_BM25,
            "query_semantic": Config.WEIGHT_TYPE_REF_SEMANTIC},
           sqrt_sum, ("Type", "{:<8.5f}")),
]


def signals_for_mode(mode: Mode) -> list[Signal]:
    return [s for s in SIGNALS if mode in s.weights]
