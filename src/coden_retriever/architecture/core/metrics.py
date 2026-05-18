"""Per-package depth ratio + shallow rule + in-function-import aggregation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .constants import (
    IN_FUNCTION_TOP_PACKAGES,
    SHALLOW_DEPTH_RATIO,
    SHALLOW_MIN_BODY,
)
from .protocol import FileAnalysis, PackageFacade


@dataclass(frozen=True)
class PackageMetric:
    """Aggregated metrics for one package — used by the shallow rule."""
    name: str
    files: int
    body_loc: int
    public_symbols: int
    public_params: int
    interface_area: int
    depth_ratio: float


@dataclass(frozen=True)
class InFunctionStats:
    """Total + top-package breakdown of imports living inside function bodies."""
    total: int
    by_package: tuple[tuple[str, int], ...]
    elsewhere: int


def compute_package_metrics(
    file_analyses: list[FileAnalysis],
    facades: dict[str, PackageFacade],
) -> list[PackageMetric]:
    """Combine per-file aggregates with adapter-reported facades into PackageMetrics."""
    files_per_pkg: dict[str, int] = defaultdict(int)
    loc_per_pkg: dict[str, int] = defaultdict(int)
    for fa in file_analyses:
        if fa.package is None:
            continue
        files_per_pkg[fa.package] += 1
        loc_per_pkg[fa.package] += fa.loc

    metrics: list[PackageMetric] = []
    for pkg, files in files_per_pkg.items():
        facade = facades.get(pkg, PackageFacade(public_symbols=(), public_params=0))
        public_n = len(facade.public_symbols)
        interface_area = max(public_n + facade.public_params, 1)
        body_loc = loc_per_pkg[pkg]
        metrics.append(PackageMetric(
            name=pkg,
            files=files,
            body_loc=body_loc,
            public_symbols=public_n,
            public_params=facade.public_params,
            interface_area=interface_area,
            depth_ratio=round(body_loc / interface_area, 2),
        ))
    metrics.sort(key=lambda m: m.name)
    return metrics


def find_shallow_packages(metrics: list[PackageMetric]) -> list[PackageMetric]:
    """Packages with `depth_ratio < SHALLOW_DEPTH_RATIO AND body_loc >= SHALLOW_MIN_BODY`."""
    results = [
        m for m in metrics
        if m.depth_ratio < SHALLOW_DEPTH_RATIO and m.body_loc >= SHALLOW_MIN_BODY
    ]
    results.sort(key=lambda m: (m.depth_ratio, m.name))
    return results


def count_in_function_imports(file_analyses: list[FileAnalysis]) -> InFunctionStats:
    """Total + top-K-package breakdown of in-function imports.

    Files outside any package (package=None) contribute to `total` and to
    `elsewhere`, never to `by_package`.
    """
    total = 0
    by_pkg: dict[str, int] = defaultdict(int)
    nopkg_total = 0
    for fa in file_analyses:
        n = len(fa.in_function_imports)
        if n == 0:
            continue
        total += n
        if fa.package is None:
            nopkg_total += n
        else:
            by_pkg[fa.package] += n

    sorted_pkgs = sorted(by_pkg.items(), key=lambda kv: (-kv[1], kv[0]))
    top = sorted_pkgs[:IN_FUNCTION_TOP_PACKAGES]
    elsewhere = sum(n for _, n in sorted_pkgs[IN_FUNCTION_TOP_PACKAGES:]) + nopkg_total
    return InFunctionStats(
        total=total,
        by_package=tuple(top),
        elsewhere=elsewhere,
    )
