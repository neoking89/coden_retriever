"""
Cache layout: declarative file/version map for one cache flavor.

Lifecycle plumbing in `CacheManager` (manifest read/write, version check,
per-flavor clear) operates on a `CacheLayout` instance instead of hardcoded
constants, so adding a second flavor (e.g. a "lite" map-mode cache) is a new
layout value plus its own load/save methods — not a parallel branch through
every plumbing method.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheLayout:
    """Names every artifact that belongs to one cache flavor.

    The manifest is tracked separately because lifecycle plumbing handles it
    distinctly from the artifact bundle (created by `_create_manifest`,
    validated against `version`, written last in `_save_cache`).
    """
    version: str
    manifest_file: str
    artifact_files: tuple[str, ...]


# Static-flavor artifact filenames. Named so `CacheManager._load_cached` /
# `_save_cache` can reference each artifact by role without inline-duplicating
# the string that also appears in `STATIC_LAYOUT.artifact_files`.
STATIC_MANIFEST_FILE = "manifest.json"
STATIC_ENTITIES_FILE = "entities.pkl"
STATIC_EMBEDDINGS_FILE = "embeddings.npy"
STATIC_NODE_IDS_FILE = "node_ids.json"
STATIC_BM25_FILE = "bm25_index.pkl"
STATIC_GRAPH_FILE = "graph.pkl"
STATIC_TYPE_GRAPH_FILE = "type_graph.pkl"
STATIC_CENTRALITY_FILE = "centrality.pkl"
STATIC_USED_NAMES_BY_FILE_FILE = "used_names_by_file.pkl"

# Bumping `version` forces a full rebuild of the static cache on next load.
# History: "6" landed when used_names_by_file was added to CachedIndices.
STATIC_LAYOUT = CacheLayout(
    version="6",
    manifest_file=STATIC_MANIFEST_FILE,
    artifact_files=(
        STATIC_ENTITIES_FILE,
        STATIC_EMBEDDINGS_FILE,
        STATIC_NODE_IDS_FILE,
        STATIC_BM25_FILE,
        STATIC_GRAPH_FILE,
        STATIC_TYPE_GRAPH_FILE,
        STATIC_CENTRALITY_FILE,
        STATIC_USED_NAMES_BY_FILE_FILE,
    ),
)

# Lite-flavor artifact filenames. Disjoint from the static set so both flavors
# coexist in one project's cache_dir without stomping each other.
LITE_MANIFEST_FILE = "lite_manifest.json"
LITE_ENTITIES_FILE = "lite_entities.pkl"
LITE_CHANGE_COUNT_FILE = "lite_change_count.pkl"

# Bumping `version` forces a full rebuild of the lite cache on next load. Bump
# whenever LiteCachedIndices shape or _create_lite_manifest schema changes.
LITE_LAYOUT = CacheLayout(
    version="1",
    manifest_file=LITE_MANIFEST_FILE,
    artifact_files=(
        LITE_ENTITIES_FILE,
        LITE_CHANGE_COUNT_FILE,
    ),
)


def manifest_path(cache_dir: Path, layout: CacheLayout) -> Path:
    return cache_dir / layout.manifest_file


def artifact_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / name


def all_paths(cache_dir: Path, layout: CacheLayout) -> Iterable[Path]:
    """Every file the layout owns under `cache_dir` (manifest + artifacts).

    Used by the per-flavor clear path: deletes only this flavor's files,
    leaving any coexisting flavor's cache intact.
    """
    yield manifest_path(cache_dir, layout)
    for name in layout.artifact_files:
        yield artifact_path(cache_dir, name)
