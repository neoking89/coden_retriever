"""Thresholds for the architecture audit. Every value carries its `# Why:`."""
from __future__ import annotations

OVERSIZED_FILE_LOC = 500
# Why: floor of the coupling-driven Blob rule (Marinescu, ICSM 2004;
# Lanza & Marinescu, "OO Metrics in Practice", Springer 2006 — conjunctive
# AND-filters trade recall for precision). 500 LOC ≈ a 10-minute read;
# god-file cluster in the agent-refactor backtest all sat above this.

OVERSIZED_FILE_IMPORTS = 15
# Why: > 15 module-top imports = the file touches > 15 modules of state.
# Roughly aligns with Sahraoui/Godin/Miceli's CBO > 14 ceiling for the
# Chidamber-Kemerer (TSE 1994) Coupling Between Objects metric. Pairs
# with LOC > 500 (above) as the coupling-driven Blob rule.

OVERSIZED_FILE_LOC_HARD = 665
# Why: size-driven Blob rule, fires regardless of import count. The number
# is Nagappan & Ball's defect-density knee from "Use of Relative Code Churn
# Measures to Predict System Defect Density" (ICSE 2005) — Windows Server
# 2003 corpus (~96k files) showed defect density spiking above ~665 LOC.
# The coupling-AND rule above misses files like Express's lib/response.js
# (842 LOC / 13 imports) that are long-but-low-coupling god-files; this
# size-driven path catches them. DECOR (Moha et al., TSE 2010) explicitly
# endorses ORing multiple Blob variants — coupling-driven and size-driven
# live as separate conjunctive rules whose results union.

KITCHEN_SINK_LOC = 5000
# Why: ≥ 5k LOC in a single package crosses the human-comprehension ceiling
# (Ousterhout, "A Philosophy of Software Design" ch. 4). mcp/ at 15.5k is
# the canonical example on this repo.

KITCHEN_SINK_FILES = 30
# Why: > 30 files in one package signals lost cohesion; mcp/ has 64
# (kitchen-sink), feature packages that stayed cohesive cap at ~20.

KITCHEN_SINK_FANOUT = 6
# Why: depending on > 6 other packages = central-coordinator role.
# Combined with the LOC/files threshold this excludes legitimate entry
# points (cli has fan-out 8 but is small) and isolates true kitchen-sinks
# (mcp at 12, daemon at 10).

SHALLOW_DEPTH_RATIO = 200
# Why: < 200 body-LOC per public symbol = public surface ≈ body
# (Ousterhout's "shallow module" signature). formatters/ at 153 is the v1
# ground-truth example; deep modules like parsers/ score 824.

SHALLOW_MIN_BODY = 300
# Why: ignore tiny packages where the ratio is noise. 300 LOC ≈ 5 small
# files; below that the depth-ratio metric is meaningless because
# `interface_area` dominates `body_loc`.

TOP_FINDINGS_DEFAULT = 10
# Why: per-section row cap. Keeps the report skimmable; matches the
# `--top N` default in REFINED_SPEC.

LOC_DISPLAY_DIVISOR = 1000
# Why: human-readable "15.5k LOC" output uses thousands. Centralized so
# the JSON path can keep raw integers while text stays compact.

IN_FUNCTION_TOP_PACKAGES = 3
# Why: the in-function-imports INFO line shows the three worst packages
# explicitly and rolls the rest into an "elsewhere" bucket — matches the
# REFINED_SPEC sample ("62 mcp/ · 32 cli/ · 12 daemon/ · 33 elsewhere").
