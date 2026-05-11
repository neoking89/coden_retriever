"""Score domain models for the search/ranking pipeline."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..search.signals import Signal
    from .entities import CodeEntity


@dataclass
class CentralityCache:
    """Index-time graph centrality snapshot."""

    pagerank: dict[str, float] = field(default_factory=dict)
    betweenness: dict[str, float] = field(default_factory=dict)
    type_pagerank: dict[str, float] = field(default_factory=dict)


@dataclass
class RankingSignals:
    """Per-query signal scores keyed by signal name. One entry per active
    `Signal`; absent keys mean the signal did not run for this mode."""

    by_signal: dict[str, dict[str, float]] = field(default_factory=dict)

    def aggregate_to_classes(
        self,
        signals: list["Signal"],
        entities: dict[str, "CodeEntity"],
        dampening: float,
    ) -> "RankingSignals":
        """Roll up method/function scores into parent classes per the signal's
        own aggregator. Returns a new instance; the original is not mutated."""
        class_lookup: dict[tuple[str, str], str] = {
            (e.name, e.file_path): nid
            for nid, e in entities.items()
            if e.entity_type == "class"
        }
        out = {name: dict(scores) for name, scores in self.by_signal.items()}

        for sig in signals:
            if sig.aggregate is None:
                continue
            scores = self.by_signal.get(sig.name)
            if not scores:
                continue
            contrib: dict[str, list[float]] = defaultdict(list)
            for nid, e in entities.items():
                if e.entity_type not in ("method", "function") or not e.parent_class:
                    continue
                pid = class_lookup.get((e.parent_class, e.file_path))
                if pid and nid in scores:
                    contrib[pid].append(scores[nid])
            agg = out.setdefault(sig.name, {})
            for cid, vals in contrib.items():
                agg[cid] = agg.get(cid, 0.0) + dampening * sig.aggregate(vals)

        return RankingSignals(by_signal=out)

    def components_for(self, node_id: str) -> dict[str, float]:
        return {name: scores.get(node_id, 0.0) for name, scores in self.by_signal.items()}
