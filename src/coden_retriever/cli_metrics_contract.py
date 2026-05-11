"""CLI Metrics Contract - Shared utilities for metric handlers.

This module provides utility functions that enforce the contract for CLI metric handlers.
Use these functions in ALL metric handlers to prevent common bugs.

CRITICAL REQUIREMENTS (THE CONTRACT):

1. **-n/--limit parameter** (for list-based metrics: hotspots, clones, echo):
   - MUST default to 20
   - MUST always apply limit using apply_defensive_limit()
   - MUST preserve full result count for statistics

2. **-r/--reverse parameter** (ALL metrics):
   - MUST pass to formatter: formatter.format_items(items, format, reverse)
   - MUST use print_metric_output() for correct ordering

3. **--tokens parameter (token_budget)** (ALL metrics):
   - MUST default to None for CLI mode
   - MUST default to 4000 for MCP mode
   - CLI users control output via -n, NOT token budget

4. **Output ordering** (ALL metrics):
   - Normal mode: stats -> results
   - Reverse mode: results -> stats
   - MUST use print_metric_output() to enforce this

Usage Example:

```python
from .cli_metrics_contract import apply_defensive_limit, print_metric_output

def handle_hotspots_command(args, root_path, config):
    # 1. Create params with token_limit=None for CLI
    params = HotspotsParams(
        source_dir=str(root_path),
        limit=args.limit,
        token_limit=args.tokens,  # None by default for CLI!
    )

    # 2. Execute via daemon or MCP
    result = execute_analysis(params)

    # 3. Apply defensive limit (CRITICAL for list-based metrics)
    all_results = result["hotspots"]
    limited_results = apply_defensive_limit(all_results, args.limit)

    # 4. Format output
    formatted = format_hotspots(limited_results, args.format, args.reverse)
    stats = format_stats(result["summary"]) if args.stats else None

    # 5. Print with correct ordering (CRITICAL for all metrics)
    print_metric_output(formatted, stats, args.reverse)
```
"""

import sys
from typing import TypeVar

T = TypeVar('T')


def apply_defensive_limit(results: list[T], limit: int | None) -> list[T]:
    """Apply limit to results defensively.

    Use this for ALL list-based metrics (hotspots, clones, echo comments).
    This ensures limit is applied even if the MCP/daemon function forgets.

    Args:
        results: Full list of results from MCP/daemon
        limit: Limit from args.limit (typically 20, or None for unlimited)

    Returns:
        Sliced list with at most `limit` items

    Example:
        ```python
        all_hotspots = result["hotspots"]  # 57 items
        hotspots = apply_defensive_limit(all_hotspots, args.limit)  # 20 items
        ```

    Why defensive:
        Even if MCP function already applied limit, applying it again is safe.
        If MCP forgot to apply limit, handler still enforces it.
        Prevents bug: "coden -n 20" showing 50+ results.
    """
    return results[:limit] if limit else results


def print_metric_output(formatted_output: str, stats_output: str | None, reverse: bool) -> None:
    """Print metric output with contract-compliant ordering.

    Use this for ALL metrics (hotspots, clones, propagation, echo comments).

    Args:
        formatted_output: Formatted results string (to stdout)
        stats_output: Formatted stats string or None (to stderr)
        reverse: Whether to reverse output order (from args.reverse)

    Ordering:
        Normal mode (reverse=False):
            1. Stats to stderr (if present)
            2. Results to stdout

        Reverse mode (reverse=True):
            1. Results to stdout
            2. Stats to stderr (if present)

    Example:
        ```python
        formatted = formatter.format_items(results, args.format, args.reverse)
        stats = formatter.format_stats(summary) if args.stats else None
        print_metric_output(formatted, stats, args.reverse)
        ```

    Why critical:
        - Stats must ALWAYS go to stderr (not stdout)
        - Results must ALWAYS go to stdout (not stderr)
        - Order changes based on reverse flag
        - Prevents bug: stats appearing in wrong stream or wrong order
    """
    if reverse:
        # Reverse mode: results first, then stats
        print(formatted_output)
        if stats_output:
            print(stats_output, file=sys.stderr)
    else:
        # Normal mode: stats first, then results
        if stats_output:
            print(stats_output, file=sys.stderr)
        print(formatted_output)


# Checklist for adding new metrics - keep this as a reference

CHECKLIST = """
Checklist for Adding New List-Based Metric (e.g., -X for cross-references):

Handler Implementation:
- [ ] Import utilities: from .cli_metrics_contract import apply_defensive_limit, print_metric_output
- [ ] Create params with token_limit=args.tokens (None for CLI)
- [ ] Try daemon first, fallback to MCP
- [ ] Extract results: all_results = result["items"]
- [ ] Apply defensive limit: limited = apply_defensive_limit(all_results, args.limit)
- [ ] Format with reverse: formatted = format_items(limited, args.format, args.reverse)
- [ ] Print with utility: print_metric_output(formatted, stats, args.reverse)

Argparse Configuration:
- [ ] -X flag added
- [ ] -n/--limit defaults to 20
- [ ] --tokens defaults to None
- [ ] -r/--reverse exists
- [ ] --stats exists

Testing:
- [ ] coden -X -n 10  -> shows exactly 10 results
- [ ] coden -X -n 50  -> shows max 50 results
- [ ] coden -X        -> shows default 20 results
- [ ] coden -X -r     -> reverses order correctly
- [ ] coden -X --stats -> shows "X shown (Y total)"
"""
