"""Audit orchestrator: pick adapter, walk files, run all five rules, build `Report`."""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ...language import LANGUAGE_MAP, language_for_path
from ...utils.source_walker import iter_source_files, path_hits_excludes
from ..adapters._stub import StubAdapter
from ..adapters.csharp import CSharpAdapter
from ..adapters.go import GoAdapter
from ..adapters.java import JavaAdapter
from ..adapters.javascript import JavaScriptAdapter
from ..adapters.kotlin import KotlinAdapter
from ..adapters.php import PhpAdapter
from ..adapters.python import PythonAdapter
from ..adapters.rust import RustAdapter
from ..adapters.scala import ScalaAdapter
from ..adapters.typescript import TypeScriptAdapter
from .files import OversizedFile, find_oversized_files
from .messages import multi_module_warning_text, unsupported_language_message
from .graph import (
    Cycle,
    KitchenSinkFacts,
    build_package_graph,
    find_cycles,
    find_kitchen_sinks,
    package_fan_out,
)
from .metrics import (
    InFunctionStats,
    PackageMetric,
    compute_package_metrics,
    count_in_function_imports,
    find_shallow_packages,
)
from .protocol import FileAnalysis, LanguageAdapter, PackageFacade


@dataclass(frozen=True)
class Report:
    """Five-section audit report; consumed by `output.render_text` / `render_json`.

    `layout_warning` is non-`None` when the adapter walked fewer source files
    than exist under the audit root — the v1 multi-module / workspace
    explicit-fail signal. See `messages.multi_module_warning_text`.

    `n_modules` is 0 for single-module audits and >=1 for workspace audits
    (Cargo `[workspace]`, Maven `<modules>`, `go.work`, `.sln`). Drives the
    optional `· N modules ·` segment in the stat line.
    """
    language: str
    n_packages: int
    n_files: int
    total_loc: int
    cycles: tuple[Cycle, ...]
    kitchen_sinks: tuple[KitchenSinkFacts, ...]
    oversized_files: tuple[OversizedFile, ...]
    shallow_packages: tuple[PackageMetric, ...]
    in_function_stats: InFunctionStats
    layout_warning: str | None
    n_modules: int = 0


_ADAPTERS: dict[str, LanguageAdapter] = {
    "python": PythonAdapter(),
    "javascript": JavaScriptAdapter(),
    "typescript": TypeScriptAdapter(),
    "c_sharp": CSharpAdapter(),
    "go": GoAdapter(),
    "java": JavaAdapter(),
    "kotlin": KotlinAdapter(),
    "php": PhpAdapter(),
    "rust": RustAdapter(),
    "scala": ScalaAdapter(),
    "stub": StubAdapter(),
}


def run_audit(
    root: Path,
    lang: str | None,
    top: int,
    excludes: tuple[str, ...],
) -> tuple[Report | None, str | None]:
    """Audit `root` and return `(report, error_message)`.

    `report=None` means we couldn't audit (no adapter for the detected
    language) — the handler should print `error_message` to stderr and exit 0.
    """
    if lang is not None:
        adapter = _ADAPTERS.get(lang)
        if adapter is None:
            return None, unsupported_language_message(lang)
    else:
        detected = _detect_language(root)
        if detected is None:
            return None, "no source files detected under given path"
        adapter = _ADAPTERS.get(detected)
        if adapter is None:
            return None, unsupported_language_message(detected)

    package_roots = adapter.package_roots(root)
    effective_root = package_roots[0].parent if package_roots else root
    module_count = getattr(adapter, "_cache_module_count", 0)
    dropped_out_of_root = getattr(adapter, "_cache_dropped_out_of_root_count", 0)
    _emit_workspace_tripwire(module_count, dropped_out_of_root)
    files = adapter.project_files(root, excludes)
    analyses = [adapter.analyze_file(f, root) for f in files]
    facades = _collect_facades(analyses, adapter)
    layout_warning = (
        None
        if module_count > 0
        else _detect_layout_warning(root, adapter, excludes, len(files))
    )
    # Why: `layout_warning` is for unsupported parent layouts (Gradle / sbt
    # parents that fall through to single-module mode). When the adapter
    # already engaged workspace mode (`module_count > 0`), members were
    # explicitly enumerated; any extra files on disk (polyglot/aggregator
    # modules, examples/, tools/) are intentionally out of scope and
    # already surfaced via the stderr tripwire + adapter-level INFO logs.
    report = _build_report(
        adapter.LANGUAGE, analyses, facades, top, effective_root, layout_warning,
        n_modules=module_count,
    )
    return report, None


def _emit_workspace_tripwire(n_modules: int, n_dropped: int) -> None:
    """Surface workspace counts on stderr so silent undercounts are visible.

    Why stderr-print, not `logger.info`: `__main__.py:112` only configures
    `logging.basicConfig` under `if __name__ == "__main__"`. The architecture
    handler does not call `basicConfig`, so a logger call silently drops in
    production CLI runs. Stderr print works without configuration coupling
    and matches the existing pattern at `cli/handlers/architecture.py:21`.
    """
    if n_modules <= 0:
        return
    print(
        f"architecture audit: {n_modules} modules walked; "
        f"{n_dropped} members dropped out-of-root",
        file=sys.stderr,
    )


def _detect_language(root: Path) -> str | None:
    """Pick the most common LANGUAGE_MAP-mapped language under `root`."""
    counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in LANGUAGE_MAP:
            continue
        lang = language_for_path(path)
        if lang is not None:
            counts[lang] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _collect_facades(
    analyses: list[FileAnalysis],
    adapter: LanguageAdapter,
) -> dict[str, PackageFacade]:
    """Ask the adapter for one `PackageFacade` per discovered package."""
    seen: dict[str, Path] = {}
    for fa in analyses:
        if fa.package is None or fa.package_root is None:
            continue
        seen.setdefault(fa.package, fa.package_root)
    return {
        pkg: adapter.package_public_facade(root)
        for pkg, root in seen.items()
    }


def _build_report(
    language: str,
    analyses: list[FileAnalysis],
    facades: dict[str, PackageFacade],
    top: int,
    effective_root: Path,
    layout_warning: str | None,
    n_modules: int,
) -> Report:
    """Run all five rules and apply the per-section `top` cap.

    `effective_root` is the adapter's resolved namespace root (post auto-descend);
    passing it to `find_oversized_files` ensures display paths are identical
    whether the user invoked with the wrapper or the inner directory.
    """
    graph = build_package_graph(analyses)
    cycles = find_cycles(graph, analyses)
    fan_out = package_fan_out(graph)
    kitchen_sinks = find_kitchen_sinks(analyses, fan_out)
    oversized = find_oversized_files(analyses, effective_root)
    metrics = compute_package_metrics(analyses, facades)
    shallow = find_shallow_packages(metrics)
    in_func = count_in_function_imports(analyses)

    return Report(
        language=language,
        n_packages=graph.number_of_nodes(),
        n_files=len(analyses),
        total_loc=sum(fa.loc for fa in analyses),
        cycles=tuple(cycles[:top]),
        kitchen_sinks=tuple(kitchen_sinks[:top]),
        oversized_files=tuple(oversized[:top]),
        shallow_packages=tuple(shallow[:top]),
        in_function_stats=in_func,
        layout_warning=layout_warning,
        n_modules=n_modules,
    )


def _detect_layout_warning(
    root: Path,
    adapter: LanguageAdapter,
    excludes: tuple[str, ...],
    walked: int,
) -> str | None:
    """Compare adapter-walked file count vs tree-wide ext-matching file count.

    Returns warning text iff the adapter scanned a strict subset — the v1
    multi-module / workspace explicit-fail symptom (e.g. Maven `<modules>`
    parent → walked=0; Cargo workspace → walked counts only the root crate).

    Tree count honors the same gitignore + SKIP_DIRS rules as the adapter
    walker, plus the user-supplied `excludes` (so `--exclude=tests` doesn't
    false-positive). Adapters without `EXTENSIONS` (none today, but the
    Protocol doesn't mandate it) skip the check.
    """
    extensions = getattr(adapter, "EXTENSIONS", frozenset())
    if not extensions:
        return None
    exclude_parts = {e for e in excludes if e}
    detected = 0
    for path, _stat in iter_source_files(root):
        if path.suffix.lower() not in extensions:
            continue
        if path_hits_excludes(path, root, exclude_parts):
            continue
        detected += 1
    if walked >= detected:
        return None
    return multi_module_warning_text(walked=walked, detected=detected)
