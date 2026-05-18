"""TypeScript adapter: tree-sitter only.

A "package" here is a direct subdirectory of the *effective* audit root that
contains at least one TypeScript file (recursively). Files directly at the
effective root get `package=None` — included in n_files/total_loc/oversized
totals, excluded from the package-level graph.

The effective root may differ from the user-supplied `root`: if `root` itself
has no TypeScript files directly inside AND has a `src/` subdirectory with
TS content, the adapter auto-descends into `src/`. This makes
`coden architecture my-project/` work the same as
`coden architecture my-project/src/` for the standard `src/`-layout repo.

Supported extensions: `.ts`, `.tsx`, `.cts`, `.mts`. The `.tsx` extension uses
the `tsx` tree-sitter grammar (which understands JSX-in-types ambiguity);
the other three use the `typescript` grammar. Switching is per-file and
handled by `_grammar_for_file`.

Type-only imports — `import type { X } from "..."` and `import { type X } from "..."`
— count as real architectural imports. Tree-sitter exposes both forms as
ordinary `import_statement` nodes with a `source` field, so the shared
import walker picks them up without special-casing. The rationale: even
though `type` imports erase at compile time, they still bind two files
together in the architectural sense — changing the type's source forces a
recompile of every importer, and cycles among types are real cycles.

Public-facade extraction adds three TS-only export forms on top of the JS
inventory: `export interface`, `export type` (type alias), and `export enum`.
All three contribute their declared name with a param count of 0.

tsconfig `paths` + `baseUrl` resolution is performed against
`<effective-root>/tsconfig.json` when present — alias-style imports like
`import Foo from "@/components/Foo"` (with the canonical Next.js
`"@/*": ["./src/*"]` mapping) and bare-rooted imports under a `baseUrl` both
resolve correctly. JSONC comments in the tsconfig are tolerated (Next.js's
`create-next-app` template ships them). Limitations: per-workspace tsconfigs,
`extends` chains, and project `references` are not followed.

What v1 deliberately does NOT do:

- TypeScript project references (`tsconfig.json::references`).
- tsconfig `extends` chains.
- `.d.ts` ambient declaration files — they aren't enumerated by the
  source-file walker because `.d.ts` is not in `EXTENSIONS`.
- `namespace` declarations and `decorator` syntax — neither contributes to
  the package public-symbol facade.
- `in_function_imports` is always `()` — same rule as the JS adapter.
- Dynamic `import("x")` expressions — always untracked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._npm_package import NpmPackageAdapter, _identifier_text


_TS_EXTRA_EXPORT_INNER_TYPES: frozenset[str] = frozenset({
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
})
# Why: every TypeScript-only export form whose tree-sitter inner type the JS
# dispatcher doesn't recognise. All three expose a `name` field that's either
# `type_identifier` (interface, type alias) or `identifier` (enum) — handled
# uniformly by `_identifier_text`.


_TSX_EXTENSION: str = ".tsx"
# Why: tree-sitter-typescript ships two grammars. The `.tsx` grammar handles
# the JSX vs. type-assertion ambiguity (`<X>foo`); the `typescript` grammar
# rejects JSX. Plain TS files must use the strict grammar to avoid mis-parsing.


class TypeScriptAdapter(NpmPackageAdapter):
    """`LanguageAdapter` implementation for TypeScript."""

    LANGUAGE = "typescript"
    EXTENSIONS = frozenset({".ts", ".tsx", ".cts", ".mts"})
    INDEX_BASENAMES = ("index.ts", "index.tsx", "index.cts", "index.mts")
    LINE_COMMENT_PREFIXES = ("//",)

    def _grammar_for_file(self, file: Path) -> str:
        """`.tsx` → `tsx` grammar; everything else → `typescript` grammar."""
        return "tsx" if file.suffix.lower() == _TSX_EXTENSION else "typescript"

    def _extra_export_symbols(
        self,
        inner: Any,
        source_bytes: bytes,
        symbols: dict[str, int],
    ) -> None:
        """Fold `export interface`, `export type T = ...`, `export enum` into the facade."""
        if inner.type not in _TS_EXTRA_EXPORT_INNER_TYPES:
            return
        name = _identifier_text(inner.child_by_field_name("name"), source_bytes)
        if name:
            symbols.setdefault(name, 0)
