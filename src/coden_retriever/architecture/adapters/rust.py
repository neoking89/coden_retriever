"""Rust adapter: tree-sitter only.

A "package" here is any directory under the effective crate-source root that
contains at least one `.rs` file directly (excluding `target/`, hidden dirs,
and `node_modules/`). Single-crate packages are named by their POSIX-relative
path from the effective root — `pkg_a`, `internal/auth/middleware`. Workspace
packages carry the crate name as a `::`-separated prefix — `clap_builder`,
`clap_builder::parser`, `clap_builder::error::format`. Files directly at the
effective root (`lib.rs`, `main.rs`, …) attribute to `package=None` —
included in n_files/total_loc/oversized totals, excluded from the
package-level graph (mirrors Python/JS/TS/Go).

The effective root depends on the layout:

1. `root/Cargo.toml` declares `[workspace] members = [...]` → effective =
   `root`. Each declared member is walked under its own `src/` (or crate
   root if absent); cross-crate `use` paths resolve via `::`-join.
2. `root/` has `Cargo.toml` + `src/` → effective = `root/src/` (single crate).
3. `root/` has `Cargo.toml` but no `src/` → effective = `root` (rare).
4. `root/` has no `Cargo.toml` but a single direct child does → recurse the
   same rule on that child (mirrors Go's one-level `go.mod` descent).

Import resolution: three branches —
  - **`use crate::...`** in single-crate mode strips the prefix, joins with
    `/`, and tries longest-first against registered package names.
  - **`use crate::...`** in workspace mode replaces `crate` with the file's
    owning crate name, joins with `::`, and tries longest-first.
  - **`use <sibling_crate>::...`** in workspace mode joins with `::` and
    tries longest-first against the qualified package set.

Stdlib (`std::`, `core::`, `alloc::`), relative (`self::`, `super::`), and
external-crate imports all attribute to `target_package = None`. Brace-list
items resolve **independently** — `use crate::{pkg_a, pkg_b}` emits two refs
against two different packages, not one ref against the bare `crate` prefix.

Public-symbol extraction aggregates fully-`pub` top-level items across every
`.rs` file in the package directory. `pub(crate)`, `pub(super)`, and
`pub(in path)` are excluded — they're not crate-external API. Captured item
kinds: `function_item`, `struct_item`, `enum_item`, `trait_item`, `type_item`,
`const_item`, `static_item`. `mod_item` is intentionally excluded (structural,
not API surface — Python doesn't include submodules in its facade either).

`in_function_imports` is always `()` — Rust permits in-function `use`
declarations but the base's `_walk_top_level_imports` only sees top-level
statements, mirroring the Go policy.

What v1 deliberately does NOT do:

- `path = "../foo"` workspace members that resolve outside the workspace
  root — silently dropped.
- `[patch]` / `[replace]` workspace overrides — ignored.
- `default-members` selectors — every declared `members` entry is walked.
- `extern crate <name>;` legacy statements (rare in Rust 2018+).
- `#[cfg(...)]`-gated modules — every file parsed regardless of build flags.
- `pub(crate)`, `pub(super)`, `pub(in path)` — restricted visibility is not
  part of the crate-external facade.
- Procedural macros / `macro_rules!` — not counted as public symbols.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.protocol import (
    ImportRef,
    InFunctionImport,
)
from ._base import (
    BaseTreeSitterAdapter,
    _node_text,
)


_RUST_SCAN_SKIP: frozenset[str] = frozenset({
    "target", "node_modules",
})
# Why: `target/` is Cargo's build output (compiled artifacts, incremental
# caches) — never source. `node_modules/` is filtered for crates that ship a
# JS toolchain alongside their Rust source (Tauri apps, wasm-bindgen demos).
# Hidden dirs (`.git`, `.github`) are filtered separately via leading-dot.

_CARGO_MANIFEST: str = "Cargo.toml"
# Why: the well-known Cargo manifest filename. Its presence marks a crate
# root; the audit descends into the crate's `src/` from there.

_SRC_DIRNAME: str = "src"
# Why: Cargo convention places crate sources under `<crate-root>/src/`. The
# adapter descends one extra step when both the manifest and this directory
# exist, so `lib.rs`/`main.rs` get treated as effective-root-level files
# (`package=None`) and module dirs become packages.

_CRATE_PREFIX: str = "crate"
# Why: the keyword that marks an in-crate absolute import path. The v1 import
# resolver only emits internal edges for paths that begin with this keyword,
# mirroring Go's "module-rooted absolute imports only" policy.

_USE_LIST_REF_NODE_TYPES: frozenset[str] = frozenset({
    "identifier", "scoped_identifier", "use_as_clause",
    "use_wildcard", "scoped_use_list", "self",
})
# Why: the named-child node types that count as one ref inside a
# `use_list` (`{a, b, foo::Bar, *, c as alias, self}`). Anything else
# (punctuation, comments) is structural and does not contribute a ref.

_PATH_SEGMENT_NODE_TYPES: frozenset[str] = frozenset({
    "identifier", "crate", "self", "super",
})
# Why: tree-sitter-rust represents path components as either nested
# `scoped_identifier` nodes (which we recurse into) or leaf tokens of
# these types. Anything else in a `scoped_identifier` (`::`, etc.) is
# structural noise that does not contribute a segment.

_PUBLIC_ITEM_NODE_TYPES: frozenset[str] = frozenset({
    "function_item", "struct_item", "enum_item", "trait_item",
    "type_item", "const_item", "static_item",
})
# Why: the top-level item kinds whose `pub` form contributes to the package
# facade. `mod_item` and `impl_item` are intentionally absent — they are
# structural, not API surface.

_WORKSPACE_TABLE: str = "workspace"
# Why: top-level table in `Cargo.toml` whose presence marks the file as a
# workspace declaration (with or without an accompanying `[package]` table).

_WORKSPACE_MEMBERS_KEY: str = "members"
# Why: the list of crate path globs inside `[workspace]`. Empty/missing
# means "not a workspace declaration we treat as one" — fall through to
# single-crate mode.

_WORKSPACE_PACKAGE_TABLE: str = "workspace.package"
# Why: Cargo 1.64+ inheritance source. A member declaring `name.workspace =
# true` resolves its crate name from the root's `[workspace.package].name`.
# Absent → fall back to `crate_root.name`.

_DOT_GLOB_LITERAL: str = "."
# Why: `members = ["."]` is the Cargo idiom for "the workspace root is also
# a member crate." `Path(root).glob(".")` returns zero matches in CPython,
# so the dot literal needs explicit identity handling before glob expansion.


@dataclass(frozen=True)
class _RustWorkspaceMember:
    """One declared Cargo workspace member resolved on disk."""
    crate_root: Path        # dir containing the member's Cargo.toml
    src_root: Path          # crate_root/src/ if exists else crate_root
    crate_name: str         # resolved per _resolve_crate_name fallback chain


@dataclass(frozen=True)
class _RustResolverCtx:
    """Bundle of resolver inputs threaded through the `use`-walker chain.

    `workspace_crate_names` is `frozenset()` and `owning_crate` is `None`
    in single-crate mode — the resolver then follows the legacy
    `crate::`-strip-and-`/`-join behavior verbatim.
    """
    project_packages: frozenset[str]
    workspace_crate_names: frozenset[str] = frozenset()
    owning_crate: str | None = None


class RustAdapter(BaseTreeSitterAdapter):
    """`LanguageAdapter` implementation for Rust crates."""

    LANGUAGE = "rust"
    EXTENSIONS = frozenset({".rs"})
    INDEX_BASENAMES = ()
    LINE_COMMENT_PREFIXES = ("//",)

    def __init__(self) -> None:
        super().__init__()
        self._cache_workspace_members: tuple[_RustWorkspaceMember, ...] = ()
        self._cache_module_count: int = 0
        self._cache_workspace_crate_names: frozenset[str] = frozenset()
        self._cache_dropped_out_of_root_count: int = 0

    def _compute_effective_root(self, root: Path) -> Path:
        """Auto-descend through one optional wrapper, then into `src/` if present.

        Workspace exception: a root `Cargo.toml` declaring `[workspace]
        members = [...]` keeps `root` as the effective root — descent into
        `src/` only happens for single-crate layouts.
        """
        if _read_cargo_workspace(root / _CARGO_MANIFEST) is not None:
            return root
        crate_root = _find_crate_root(root)
        if crate_root is None:
            return root
        src_dir = crate_root / _SRC_DIRNAME
        if src_dir.is_dir():
            return src_dir
        return crate_root

    def _discover_package_roots(
        self, effective_root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        """Return `(dir, name)` pairs for every Rust module dir under `effective_root`.

        Writes `self._cache_workspace_members` as a side effect — see the
        plan's "Critical ordering" section. `_post_layout` reads that cache
        AFTER this method returns to populate derived state.

        Single-crate mode: any directory containing ≥1 `.rs` file directly
        (not recursively — subdirs are independent packages). Files at
        `effective_root` itself attribute to `None` via the base's
        `_find_package` walk.

        Workspace mode: each declared `members` entry is walked under its
        own `src/` (or crate root if absent). Names are qualified with the
        crate name (`crate_a::foo::bar`); the crate root itself registers
        as the bare crate name. Child dirs carrying their own `Cargo.toml`
        but NOT in the declared members list are skipped (defends against
        vendored crates living inside a member).
        """
        globs = _read_cargo_workspace(effective_root / _CARGO_MANIFEST)
        if globs is not None:
            inherited = _read_workspace_package_name(
                effective_root / _CARGO_MANIFEST,
            )
            members, dropped = _expand_workspace_members(
                effective_root, globs, inherited,
            )
            self._cache_workspace_members = members
            self._cache_dropped_out_of_root_count = dropped
            return _discover_workspace_pairs(members)
        self._cache_workspace_members = ()
        self._cache_dropped_out_of_root_count = 0
        return _discover_single_crate_pairs(effective_root)

    def _post_layout(self, effective_root: Path, audit_root: Path) -> None:
        """Populate derived caches read by downstream resolver calls."""
        del effective_root, audit_root
        self._cache_module_count = len(self._cache_workspace_members)
        self._cache_workspace_crate_names = frozenset(
            m.crate_name for m in self._cache_workspace_members
        )

    def _walk_imports(
        self,
        tree: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> tuple[int, tuple[ImportRef, ...], tuple[InFunctionImport, ...]]:
        """Collect every top-level `use_declaration`. One declaration = one statement."""
        stmt_count, imports = self._walk_top_level_imports(
            tree, source_bytes, file, effective_root,
            per_stmt=self._imports_for_top_level_statement,
        )
        return stmt_count, imports, ()

    def _imports_for_top_level_statement(
        self,
        node: Any,
        source_bytes: bytes,
        file: Path,
        effective_root: Path,
    ) -> list[ImportRef]:
        """Refs produced by ONE top-level statement (only `use_declaration` matters)."""
        del effective_root
        if node.type != "use_declaration":
            return []
        body = _use_declaration_body(node)
        if body is None:
            return []
        line = node.start_point[0] + 1
        ctx = _RustResolverCtx(
            project_packages=self._cache_package_names,
            workspace_crate_names=self._cache_workspace_crate_names,
            owning_crate=_owning_crate_for(
                file, self._cache_workspace_members,
            ),
        )
        return _refs_from_use_body(body, source_bytes, line, ctx)

    def _collect_public_symbols(
        self,
        tree: Any,
        source_bytes: bytes,
    ) -> dict[str, int]:
        """Fully-`pub` top-level items in one Rust file.

        The base aggregates these across every file returned by
        `_facade_source_files`. `pub(crate)` and friends are excluded by
        `_is_fully_pub`.
        """
        symbols: dict[str, int] = {}
        for child in tree.root_node.children:
            _collect_public_item(child, source_bytes, symbols)
        return symbols

    def _facade_source_files(self, package_root: Path) -> tuple[Path, ...]:
        """Every `.rs` file directly in `package_root`, sorted by name.

        Override because Rust has no single-index convention — `mod.rs`
        (Rust 2015) and sibling-file (`pkg/foo.rs` + `pkg/foo/`, Rust 2018+)
        coexist. Sub-dirs are independent packages so the walk is
        non-recursive.
        """
        try:
            entries = sorted(package_root.iterdir(), key=lambda p: p.name)
        except OSError:
            return ()
        return tuple(
            e for e in entries
            if e.is_file() and e.suffix.lower() == ".rs"
        )


def _find_crate_root(root: Path) -> Path | None:
    """Locate the directory holding `Cargo.toml`, with one-level descent."""
    if (root / _CARGO_MANIFEST).is_file():
        return root
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    cargo_children = [
        c for c in children
        if c.is_dir() and (c / _CARGO_MANIFEST).is_file()
    ]
    if len(cargo_children) == 1:
        return cargo_children[0]
    return None


def _dir_has_rust_files(entries: list[Path]) -> bool:
    """True if any entry in `entries` is a regular `.rs` file."""
    return any(
        e.is_file() and e.suffix.lower() == ".rs"
        for e in entries
    )


def _read_cargo_toml(cargo_toml: Path) -> dict[str, Any] | None:
    """Parse a `Cargo.toml` via stdlib `tomllib`. Returns `None` on any error."""
    if not cargo_toml.is_file():
        return None
    try:
        return tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None


def _read_cargo_workspace(cargo_toml: Path) -> tuple[str, ...] | None:
    """Return `[workspace] members` globs, or `None` for non-workspace manifests."""
    data = _read_cargo_toml(cargo_toml)
    if data is None:
        return None
    workspace = data.get(_WORKSPACE_TABLE)
    if not isinstance(workspace, dict):
        return None
    members = workspace.get(_WORKSPACE_MEMBERS_KEY)
    if not isinstance(members, list) or not members:
        return None
    return tuple(str(m) for m in members if isinstance(m, str))


def _read_workspace_package_name(cargo_toml: Path) -> str | None:
    """Return `[workspace.package].name`, or `None`. Cargo 1.64+ inheritance source."""
    data = _read_cargo_toml(cargo_toml)
    if data is None:
        return None
    workspace = data.get(_WORKSPACE_TABLE)
    if not isinstance(workspace, dict):
        return None
    package = workspace.get("package")
    if not isinstance(package, dict):
        return None
    name = package.get("name")
    return name if isinstance(name, str) else None


def _resolve_crate_name(
    member_cargo_toml: Path,
    workspace_inherited_name: str | None,
    fallback: str,
) -> str:
    """Resolve a member crate's name via the Cargo inheritance fallback chain.

    Order: literal `[package].name` → workspace-inherited (when the member
    declares `name.workspace = true` and the root has `[workspace.package]
    .name`) → caller-provided fallback (usually the crate dir basename).
    """
    data = _read_cargo_toml(member_cargo_toml)
    if data is None:
        return fallback
    package = data.get("package")
    if not isinstance(package, dict):
        return fallback
    name = package.get("name")
    if isinstance(name, str):
        return name
    if isinstance(name, dict) and name.get("workspace") is True:
        if workspace_inherited_name is not None:
            return workspace_inherited_name
    return fallback


def _expand_workspace_members(
    workspace_root: Path,
    globs: tuple[str, ...],
    workspace_inherited_name: str | None,
) -> tuple[tuple[_RustWorkspaceMember, ...], int]:
    """Expand workspace glob entries into resolved members; count out-of-root drops.

    Returns `(members, dropped_count)`. Members are deduped by `crate_root`
    and sorted by `crate_name`. Out-of-root entries (`members = ["../foo"]`)
    are silently dropped — out of scope per the ticket; the count surfaces
    via the workspace tripwire so silent undercounts are visible.
    """
    seen: dict[Path, _RustWorkspaceMember] = {}
    dropped = 0
    for raw in globs:
        for crate_root in _glob_workspace_entry(workspace_root, raw):
            if not _is_within(crate_root, workspace_root):
                dropped += 1
                continue
            if crate_root in seen:
                continue
            member = _build_workspace_member(
                crate_root, workspace_inherited_name,
            )
            if member is not None:
                seen[crate_root] = member
    return (
        tuple(sorted(seen.values(), key=lambda m: m.crate_name)),
        dropped,
    )


def _glob_workspace_entry(workspace_root: Path, raw: str) -> list[Path]:
    """Expand one `members = [...]` entry into candidate crate-root dirs.

    `_DOT_GLOB_LITERAL` ("`.`") is the Cargo idiom for "workspace root is
    also a member" — explicit handling because `Path.glob(".")` returns
    zero matches in CPython.
    """
    if raw == _DOT_GLOB_LITERAL:
        candidates = [workspace_root]
    else:
        try:
            candidates = list(workspace_root.glob(raw))
        except (OSError, ValueError):
            return []
    return [
        c for c in candidates
        if c.is_dir() and (c / _CARGO_MANIFEST).is_file()
    ]


def _is_within(candidate: Path, root: Path) -> bool:
    """True iff `candidate` resolves under `root`."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _build_workspace_member(
    crate_root: Path, workspace_inherited_name: str | None,
) -> _RustWorkspaceMember | None:
    """Construct a `_RustWorkspaceMember` from a crate-root directory."""
    src_dir = crate_root / _SRC_DIRNAME
    src_root = src_dir if src_dir.is_dir() else crate_root
    crate_name = _resolve_crate_name(
        crate_root / _CARGO_MANIFEST,
        workspace_inherited_name,
        crate_root.name,
    )
    return _RustWorkspaceMember(
        crate_root=crate_root, src_root=src_root, crate_name=crate_name,
    )


def _discover_single_crate_pairs(
    effective_root: Path,
) -> tuple[tuple[Path, str], ...]:
    """DFS for `(dir, posix-relative-name)` pairs under one crate's source root."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [effective_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if cur != effective_root and _dir_has_rust_files(entries):
            rel = cur.relative_to(effective_root).as_posix()
            pairs.append((cur, rel))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _RUST_SCAN_SKIP:
                continue
            stack.append(entry)
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _discover_workspace_pairs(
    members: tuple[_RustWorkspaceMember, ...],
) -> tuple[tuple[Path, str], ...]:
    """Discover `(dir, qualified-name)` pairs across every workspace member.

    Each member contributes its crate root (under the bare crate name) plus
    every sub-directory containing `.rs` files (qualified `<crate>::<sub>`).
    Child manifests NOT in the declared members list are skipped to defend
    against vendored crates living inside a member's tree.
    """
    member_roots: frozenset[Path] = frozenset(m.crate_root for m in members)
    pairs: list[tuple[Path, str]] = []
    for member in members:
        pairs.append((member.src_root, member.crate_name))
        pairs.extend(
            _walk_member_packages(member, member_roots),
        )
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _walk_member_packages(
    member: _RustWorkspaceMember,
    member_roots: frozenset[Path],
) -> list[tuple[Path, str]]:
    """DFS one workspace member's source tree, yielding qualified pairs."""
    pairs: list[tuple[Path, str]] = []
    stack: list[Path] = [member.src_root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        if cur != member.src_root and _dir_has_rust_files(entries):
            rel = cur.relative_to(member.src_root).as_posix()
            pairs.append((cur, _qualify_rust_package_name(member.crate_name, rel)))
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in _RUST_SCAN_SKIP:
                continue
            if _is_unregistered_nested_crate(entry, member_roots):
                continue
            stack.append(entry)
    return pairs


def _is_unregistered_nested_crate(
    entry: Path, member_roots: frozenset[Path],
) -> bool:
    """True iff `entry` carries its own `Cargo.toml` and is NOT a declared member."""
    if not (entry / _CARGO_MANIFEST).is_file():
        return False
    return entry not in member_roots


def _qualify_rust_package_name(crate: str, sub_rel: str) -> str:
    """Convert a per-crate POSIX-relative path into the qualified workspace name."""
    if not sub_rel or sub_rel == ".":
        return crate
    return f"{crate}::{sub_rel.replace('/', '::')}"


def _owning_crate_for(
    file: Path, members: tuple[_RustWorkspaceMember, ...],
) -> str | None:
    """Return the workspace crate name that owns `file`, or `None`."""
    if not members:
        return None
    for member in members:
        if _is_within(file, member.src_root):
            return member.crate_name
    return None


def _use_declaration_body(node: Any) -> Any | None:
    """Return the meaningful inner node of a `use_declaration`, or None.

    Skips the leading `pub` visibility modifier (if any) and the `use`
    keyword + trailing `;`. The body is one of: `scoped_identifier`,
    `identifier`, `scoped_use_list`, `use_wildcard`, or bare `crate`/
    `self`/`super` leaf tokens.
    """
    for child in node.children:
        if child.type in ("visibility_modifier", "use", ";"):
            continue
        return child
    return None


def _refs_from_use_body(
    body: Any,
    source_bytes: bytes,
    line: int,
    ctx: _RustResolverCtx,
) -> list[ImportRef]:
    """Dispatch a `use_declaration` body to the matching shape handler.

    Three shapes are produced by tree-sitter-rust: `scoped_use_list`,
    `use_wildcard`, and `scoped_identifier`/bare-leaf. A top-level
    `use_as_clause` (`use prefix::name as alias;`) is peeled once before
    re-dispatch.
    """
    if body.type == "use_as_clause":
        peeled = _peel_use_as_clause(body)
        if peeled is None:
            return []
        body = peeled
    if body.type == "scoped_use_list":
        return _refs_from_scoped_use_list(body, source_bytes, line, ctx)
    if body.type == "use_wildcard":
        return _refs_from_use_wildcard(body, source_bytes, line, ctx)
    if body.type in ("scoped_identifier", *_PATH_SEGMENT_NODE_TYPES):
        return _ref_from_scoped_path(body, source_bytes, line, ctx)
    return []


def _peel_use_as_clause(body: Any) -> Any | None:
    """Return the inner path of a `use_as_clause`, or None if missing."""
    for child in body.children:
        if child.type == "scoped_identifier" or child.type in _PATH_SEGMENT_NODE_TYPES:
            return child
    return None


def _refs_from_scoped_use_list(
    body: Any, source_bytes: bytes, line: int, ctx: _RustResolverCtx,
) -> list[ImportRef]:
    """Emit one ImportRef per leaf item in `use prefix::{...}`.

    Items may themselves be nested `scoped_use_list`s, e.g.
    `use crate::{a::{x, y}, b::Item}`. The walk recurses, prepending each
    level's prefix to the leaf's segments — so `crate::{a::{x, y}, b::I}`
    emits three refs against `a, a, b` respectively, not one against the
    bare `crate` prefix.
    """
    refs: list[ImportRef] = []
    _walk_scoped_use_list(body, [], source_bytes, line, ctx, refs)
    return refs


def _walk_scoped_use_list(
    node: Any,
    parent_prefix: list[str],
    source_bytes: bytes,
    line: int,
    ctx: _RustResolverCtx,
    out: list[ImportRef],
) -> None:
    """Recurse over a `scoped_use_list`, emitting one ref per leaf item."""
    own_prefix, use_list = _scoped_use_list_split(node, source_bytes)
    if use_list is None:
        return
    full_prefix = parent_prefix + own_prefix
    for nc in use_list.named_children:
        if nc.type == "scoped_use_list":
            _walk_scoped_use_list(
                nc, full_prefix, source_bytes, line, ctx, out,
            )
            continue
        if nc.type not in _USE_LIST_REF_NODE_TYPES:
            continue
        full = full_prefix + _use_list_item_segments(nc, source_bytes)
        out.append(ImportRef(
            target_module="::".join(full),
            target_package=_resolve_rust_use(full, ctx),
            line=line,
        ))


def _refs_from_use_wildcard(
    body: Any, source_bytes: bytes, line: int, ctx: _RustResolverCtx,
) -> list[ImportRef]:
    """Emit one ImportRef for `use prefix::*`."""
    prefix_segs = _wildcard_prefix(body, source_bytes)
    return [ImportRef(
        target_module="::".join(prefix_segs) + "::*",
        target_package=_resolve_rust_use(prefix_segs, ctx),
        line=line,
    )]


def _ref_from_scoped_path(
    body: Any, source_bytes: bytes, line: int, ctx: _RustResolverCtx,
) -> list[ImportRef]:
    """Emit one ImportRef for `use prefix::name` (full-path resolution)."""
    segs = _flatten_path(body, source_bytes)
    if not segs:
        return []
    return [ImportRef(
        target_module="::".join(segs),
        target_package=_resolve_rust_use(segs, ctx),
        line=line,
    )]


def _scoped_use_list_split(
    node: Any, source_bytes: bytes,
) -> tuple[list[str], Any | None]:
    """Decompose a `scoped_use_list` into (prefix segments, inner `use_list` node).

    Returns `[], None` when either component is absent — callers short-circuit.
    """
    prefix_segs: list[str] = []
    use_list_node: Any | None = None
    for child in node.children:
        if child.type == "scoped_identifier" or child.type in _PATH_SEGMENT_NODE_TYPES:
            prefix_segs = _flatten_path(child, source_bytes)
        elif child.type == "use_list":
            use_list_node = child
    return prefix_segs, use_list_node


def _use_list_item_segments(item: Any, source_bytes: bytes) -> list[str]:
    """Suffix segments contributed by one item inside a `use_list`.

    Nested `scoped_use_list` and bare `self` items contribute no segment —
    they resolve via the parent prefix alone.
    """
    if item.type == "use_as_clause":
        inner = _peel_use_as_clause(item)
        return _flatten_path(inner, source_bytes) if inner is not None else []
    if item.type == "use_wildcard":
        return _wildcard_prefix(item, source_bytes) + ["*"]
    if item.type == "scoped_identifier":
        return _flatten_path(item, source_bytes)
    if item.type == "identifier":
        return [_node_text(item, source_bytes)]
    return []


def _wildcard_prefix(node: Any, source_bytes: bytes) -> list[str]:
    """Extract the prefix path of a `use_wildcard` (`prefix::*`)."""
    for child in node.children:
        if child.type == "scoped_identifier" or child.type in _PATH_SEGMENT_NODE_TYPES:
            return _flatten_path(child, source_bytes)
    return []


def _flatten_path(node: Any, source_bytes: bytes) -> list[str]:
    """Flatten a Rust path node into its segment-text list.

    Handles nested `scoped_identifier` recursively (`crate::foo::bar` is a
    `scoped_identifier` whose first child is the inner `scoped_identifier`
    for `crate::foo`). Skips the `::` punctuation children.
    """
    if node.type == "scoped_identifier":
        out: list[str] = []
        for child in node.children:
            if child.type == "scoped_identifier":
                out.extend(_flatten_path(child, source_bytes))
            elif child.type in _PATH_SEGMENT_NODE_TYPES:
                out.append(_node_text(child, source_bytes))
        return out
    if node.type in _PATH_SEGMENT_NODE_TYPES:
        return [_node_text(node, source_bytes)]
    return []


def _resolve_rust_use(
    segments: list[str], ctx: _RustResolverCtx,
) -> str | None:
    """Resolve a `use` path to a project package name, else `None`.

    Three branches in priority order:

    1. Workspace `crate::...` (when `ctx.owning_crate` is set): replace the
       `crate` keyword with the file's owning crate name, join the result
       with `::`, and try longest-first against the qualified package set.
    2. Workspace `<sibling_crate>::...` (when `segments[0]` is in
       `ctx.workspace_crate_names`): join with `::` and try longest-first.
    3. Single-crate `crate::...`: strip the prefix, join with `/`, and try
       longest-first against the registered package set.

    Other shapes (stdlib `std::`, relative `self::`/`super::`, external
    crates not in `workspace_crate_names`) return `None`.
    """
    if not segments:
        return None
    if ctx.owning_crate is not None and segments[0] == _CRATE_PREFIX:
        qualified = [ctx.owning_crate, *segments[1:]]
        return _longest_first(qualified, "::", ctx.project_packages)
    if segments[0] in ctx.workspace_crate_names:
        return _longest_first(segments, "::", ctx.project_packages)
    if segments[0] == _CRATE_PREFIX:
        return _longest_first(segments[1:], "/", ctx.project_packages)
    return None


def _longest_first(
    parts: list[str], sep: str, project_packages: frozenset[str],
) -> str | None:
    """Try `sep`-joined prefixes longest-first against `project_packages`."""
    for n in range(len(parts), 0, -1):
        candidate = sep.join(parts[:n])
        if candidate in project_packages:
            return candidate
    return None


def _is_fully_pub(item_node: Any) -> bool:
    """True iff `item_node` has a `visibility_modifier` of exactly `pub`.

    Rejects `pub(crate)`, `pub(super)`, and `pub(in path::to::mod)`. The
    discriminator is the visibility modifier's child shape: bare `pub` has
    one child of type `"pub"`; restricted forms have additional `(`, target,
    `)` children.
    """
    for child in item_node.children:
        if child.type != "visibility_modifier":
            continue
        return child.child_count == 1 and child.children[0].type == "pub"
    return False


def _collect_public_item(
    node: Any, source_bytes: bytes, symbols: dict[str, int],
) -> None:
    """Add `node`'s fully-`pub` top-level identifier (if any) to `symbols`."""
    if node.type not in _PUBLIC_ITEM_NODE_TYPES:
        return
    if not _is_fully_pub(node):
        return
    name = _item_name(node, source_bytes)
    if name is None:
        return
    symbols.setdefault(name, _count_fn_params(node))


def _item_name(node: Any, source_bytes: bytes) -> str | None:
    """Return the declared identifier for a public item node, or None.

    Functions/consts/statics use `identifier`; structs/enums/traits/types
    use `type_identifier`. We scan children for either and return the first
    match — the grammar always places the name immediately after the
    keyword token.
    """
    for child in node.children:
        if child.type in ("identifier", "type_identifier"):
            return _node_text(child, source_bytes)
    return None


def _count_fn_params(fn_node: Any) -> int:
    """Parameter count for a function (0 for non-function items).

    Counts `parameter` named children only. `self_parameter` (the leading
    `&self` / `&mut self` / `self` on methods) is excluded — matches Go's
    receiver-not-counted policy.
    """
    if fn_node.type != "function_item":
        return 0
    for child in fn_node.children:
        if child.type == "parameters":
            return sum(
                1 for c in child.named_children
                if c.type == "parameter"
            )
    return 0
