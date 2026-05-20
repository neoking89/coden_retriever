"""Package-level import graph: build, find cycles, fan-out, kitchen-sinks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import (
    KITCHEN_SINK_FANOUT,
    KITCHEN_SINK_FILES,
    KITCHEN_SINK_LOC,
)
from .protocol import FileAnalysis

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class Cycle:
    """A simple cycle in the package-level import graph."""
    members: tuple[str, ...]
    workaround_count: int


@dataclass(frozen=True)
class KitchenSinkFacts:
    """A package that breaches both the size AND fan-out thresholds."""
    name: str
    body_loc: int
    files: int
    fan_out: int


def build_package_graph(file_analyses: list[FileAnalysis]) -> nx.DiGraph:
    """Build a DiGraph: nodes are package names; edges are cross-package imports.

    Includes BOTH module-top imports AND in-function imports as edges — matches
    tach's behavior. A cycle hidden behind function-level imports (the classic
    "move it inside a function to silence ImportError" workaround) still
    appears as a cycle in this graph; the workaround count attached to each
    `Cycle` tells the user how many of its edges are lazy.
    """
    import networkx as nx
    graph: nx.DiGraph = nx.DiGraph()
    for fa in file_analyses:
        if fa.package is None:
            continue
        graph.add_node(fa.package)
        for imp in fa.imports:
            if imp.target_package is None or imp.target_package == fa.package:
                continue
            graph.add_edge(fa.package, imp.target_package)
        for ifi in fa.in_function_imports:
            if ifi.target_package is None or ifi.target_package == fa.package:
                continue
            graph.add_edge(fa.package, ifi.target_package)
    return graph


def find_cycles(
    graph: nx.DiGraph,
    file_analyses: list[FileAnalysis],
) -> list[Cycle]:
    """Return one `Cycle` per strongly-connected component of size > 1.

    SCCs collapse the explosion of "simple cycles" Johnson's algorithm yields
    for entangled groups (a 5-node SCC has 84 simple cycles but only one
    real entanglement). Each SCC = one architectural problem to fix.
    """
    import networkx as nx
    cycles: list[Cycle] = []
    for scc in nx.strongly_connected_components(graph):
        if len(scc) <= 1:
            continue
        members = tuple(sorted(scc))
        workaround = _count_workaround_imports(set(scc), file_analyses)
        cycles.append(Cycle(members=members, workaround_count=workaround))
    cycles.sort(key=lambda c: (len(c.members), c.members))
    return cycles


def _count_workaround_imports(
    cycle_members: set[str],
    file_analyses: list[FileAnalysis],
) -> int:
    """Count in-function imports whose source and target packages are both in the cycle."""
    count = 0
    for fa in file_analyses:
        if fa.package not in cycle_members:
            continue
        for ifi in fa.in_function_imports:
            if (
                ifi.target_package in cycle_members
                and ifi.target_package != fa.package
            ):
                count += 1
    return count


def package_fan_out(graph: nx.DiGraph) -> dict[str, int]:
    """Return out-degree per node — efferent coupling at the package level."""
    return {node: graph.out_degree(node) for node in graph.nodes}


def find_kitchen_sinks(
    file_analyses: list[FileAnalysis],
    fan_out_map: dict[str, int],
) -> list[KitchenSinkFacts]:
    """Identify packages crossing the kitchen-sink size AND fan-out thresholds."""
    files_per_pkg: dict[str, int] = defaultdict(int)
    loc_per_pkg: dict[str, int] = defaultdict(int)
    for fa in file_analyses:
        if fa.package is None:
            continue
        files_per_pkg[fa.package] += 1
        loc_per_pkg[fa.package] += fa.loc

    results: list[KitchenSinkFacts] = []
    for pkg, files in files_per_pkg.items():
        body_loc = loc_per_pkg[pkg]
        fan_out = fan_out_map.get(pkg, 0)
        size_breach = body_loc > KITCHEN_SINK_LOC or files > KITCHEN_SINK_FILES
        if size_breach and fan_out > KITCHEN_SINK_FANOUT:
            results.append(KitchenSinkFacts(
                name=pkg,
                body_loc=body_loc,
                files=files,
                fan_out=fan_out,
            ))

    results.sort(key=lambda k: (-k.body_loc, -k.files, k.name))
    return results
