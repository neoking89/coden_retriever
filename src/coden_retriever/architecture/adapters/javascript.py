"""JavaScript adapter: tree-sitter only.

A "package" here is a direct subdirectory of the *effective* audit root that
contains at least one JavaScript file (recursively). Files directly at the
effective root get `package=None` — included in n_files/total_loc/oversized
totals, excluded from the package-level graph.

The effective root may differ from the user-supplied `root`: if `root` itself
has no JavaScript files directly inside AND has a `src/` subdirectory with
JS content, the adapter auto-descends into `src/`. This makes
`coden architecture my-project/` work the same as
`coden architecture my-project/src/` for the standard `src/`-layout repo.

For `require('..')` / `require('../')` (relative specifiers that resolve
to the effective root itself), the adapter reads the root's
`package.json::main` (default `"index.js"`) and follows ONE level of
barrel re-export — `module.exports = require("./X")` or
`export * from "./X"` / `export { ... } from "./X"`. If that target maps
to a registered top-level package, every `require('..')` in the audit
attributes to it. This catches the standard `test/` and `examples/` →
root → `lib/` pattern that every Node library exhibits.

What v1 deliberately does NOT do:

- TypeScript (`.ts`/`.tsx`/`.cts`/`.mts`). Separate adapter — see `typescript.py`.
- Dynamic `import("x")` expressions — always untracked.
- Multi-target barrels (a root entry that re-exports from more than one
  package). Only the FIRST matching pattern is followed.
- `package.json::exports` / `browser` conditional resolution. `jsconfig.json`
  aliases are ignored. `tsconfig.json::compilerOptions.paths` (and `baseUrl`)
  IS now picked up — projects that ship a tsconfig alongside their JS sources
  (e.g. for editor IntelliSense) get the same alias resolution as the
  TypeScript adapter.
- `require()` detection beyond simple `(const|let|var) X = require("y")`
  and bare-statement `require("y")`. Destructured requires, conditional
  requires, computed-name requires are NOT detected.
- Re-exports (`export { x } from "./other"`, `export * from "./other"`)
  do NOT contribute to the package's public facade (only to the root-
  barrel target resolution above).
- `in_function_imports` is always `()` — JS imports must be top-level, and
  the `require()`-inside-function cycle-workaround pattern is rare in
  idiomatic code.

The outer skeleton (file walk, layout cache, parser caching, LOC counting,
`analyze_file` / `package_public_facade` orchestration) lives in
`BaseTreeSitterAdapter`. Everything `package.json`-driven (workspaces,
import resolution, barrel tracing, public-symbol extraction) lives in
`NpmPackageAdapter`. This module is now just the JS-specific constants.
"""
from __future__ import annotations

from ._npm_package import NpmPackageAdapter


class JavaScriptAdapter(NpmPackageAdapter):
    """`LanguageAdapter` implementation for JavaScript."""

    LANGUAGE = "javascript"
    EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
    INDEX_BASENAMES = ("index.js", "index.mjs", "index.cjs", "index.jsx")
    LINE_COMMENT_PREFIXES = ("//",)
