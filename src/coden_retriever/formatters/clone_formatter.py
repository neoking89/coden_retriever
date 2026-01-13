"""Clone detection CLI formatter using BaseCLIMetricFormatter.

Provides consistent formatting for clone detection results with:
- Colored output based on clone severity (EXACT > NEAR-CLONE > SEMANTIC)
- Clickable VS Code hyperlinks for function navigation
- JSON output support
- Summary statistics
- Support for combined, semantic, and syntactic modes
"""

import json
from typing import Any

from .cli_metrics import BaseCLIMetricFormatter, FALSE_POSITIVE_WARNING, SeverityTier


_CLONE_TABLE_WIDTH = 130
_CLONE_FUNC_COL_WIDTH = 32


def _extract_filename(file_path: str) -> str:
    """Extract filename from a path, handling both Unix and Windows separators."""
    return file_path.split("/")[-1].split("\\")[-1]


def format_clone_parameters_header(
    mode: str = "combined",
    similarity_threshold: float = 0.95,
    line_threshold: float = 0.70,
    func_threshold: float = 0.50,
    min_lines: int = 3,
    limit: int | None = 50,
    exclude_tests: bool = True,
) -> str:
    """Format parameter summary header for clone detection.

    Args:
        mode: Detection mode ("combined", "semantic", "syntactic")
        similarity_threshold: Semantic similarity threshold (0-1 scale)
        line_threshold: Line-by-line Jaccard threshold for syntactic mode
        func_threshold: Function match threshold for syntactic mode
        min_lines: Minimum function lines to consider
        limit: Result limit (None = show all)
        exclude_tests: Whether tests are excluded

    Returns:
        Formatted parameter header string
    """
    mode_labels = {
        "combined": "Combined (Semantic + Syntactic)",
        "semantic": "Semantic Only (Model2Vec Embeddings)",
        "syntactic": "Syntactic Only (Line-by-Line Jaccard)",
    }

    lines = []
    lines.append("=" * 80)
    lines.append("CLONE DETECTION PARAMETERS")
    lines.append("=" * 80)
    lines.append(f"Mode: {mode_labels.get(mode, mode)}")

    if mode in ("combined", "semantic"):
        lines.append(f"Semantic Threshold: >= {similarity_threshold * 100:.0f}%")
    if mode in ("combined", "syntactic"):
        lines.append(f"Line Threshold: >= {line_threshold * 100:.0f}% (Jaccard)")
        lines.append(f"Function Threshold: >= {func_threshold * 100:.0f}% (lines matching)")

    lines.append(f"Minimum Function Lines: {min_lines}")
    lines.append(f"Exclude Tests: {exclude_tests}")

    if limit is None:
        lines.append("[!] Result Limit: ALL (may be slow for large repos)")
    else:
        lines.append(f"[!] Result Limit: TOP {limit} -- more results may exist (use -n -1 for all)")

    lines.append(FALSE_POSITIVE_WARNING)
    lines.append("=" * 80)
    return "\n".join(lines)


class CloneFormatter(BaseCLIMetricFormatter):
    """Formatter for clone detection CLI output.

    Uses SeverityTier to color clones by priority:
    - EXACT / NEAR-CLONE: CRITICAL/HIGH severity - red (highest priority)
    - SEMANTIC-STRUCTURAL / STRUCTURAL: ELEVATED/MODERATE severity - orange
    - SEMANTIC / PARTIAL: MEDIUM/LOW severity - yellow-green
    """

    def get_tier(self, item: dict[str, Any]) -> SeverityTier:
        """Get color tier based on clone category and similarity.

        Higher similarity = more urgent to refactor = higher severity.
        """
        category = item.get("category", "")
        similarity = item.get("similarity", 0)

        # Category-based tiers (extended categories from combined mode)
        if category in ("EXACT", "NEAR-CLONE"):
            return SeverityTier.HIGH  # red
        if category in ("SEMANTIC-STRUCTURAL", "STRUCTURAL"):
            return SeverityTier.MODERATE  # orange
        if category in ("SEMANTIC", "PARTIAL"):
            return SeverityTier.LOW  # yellow-green

        # Fallback: similarity-based tiers
        if similarity >= 0.98:
            return SeverityTier.HIGH
        if similarity >= 0.90:
            return SeverityTier.MODERATE
        return SeverityTier.LOW

    def format_items(
        self,
        items: list[dict[str, Any]],
        output_format: str,
        reverse: bool,
    ) -> str:
        """Format clone pairs for CLI output.

        Args:
            items: List of clone pair dicts
            output_format: Output format ("tree", "json")
            reverse: If True, show highest similarity first

        Returns:
            Formatted string with ANSI colors and OSC 8 hyperlinks
        """
        if output_format == "json":
            return json.dumps(items, indent=2)

        if not items:
            return "No code clones found at the specified similarity threshold."

        display_clones = items if reverse else list(reversed(items))

        # Detect columns by checking ALL clones (not just first)
        # This ensures columns show up even if first clone has null values
        has_semantic = any(c.get("semantic_sim") is not None for c in items)
        has_syntactic = any(c.get("syntactic_pct") is not None for c in items)
        has_blocks = any(c.get("blocks") for c in items)

        lines = []
        col_width = _CLONE_FUNC_COL_WIDTH

        # Build header based on available data
        header_parts = [f"{'Rank':<4}", f"{'Score':<6}"]
        if has_semantic:
            header_parts.append(f"{'Sem':<6}")
        if has_syntactic:
            header_parts.append(f"{'Syn':<6}")
        if has_blocks:
            header_parts.append(f"{'Blocks':<8}")
        header_parts.extend([
            f"{'Category':<18}",
            f"{'Function 1':<{col_width}}",
            f"{'Function 2':<{col_width}}",
        ])
        header = " | ".join(header_parts)
        lines.append(header)
        lines.append("-" * _CLONE_TABLE_WIDTH)

        for i, clone in enumerate(display_clones, 1):
            rank = i if reverse else len(display_clones) - i + 1
            similarity = clone.get("similarity", 0)
            semantic_sim = clone.get("semantic_sim")
            syntactic_pct = clone.get("syntactic_pct")
            blocks = clone.get("blocks", [])
            max_block_size = clone.get("max_block_size", 0)
            category = clone.get("category", "UNKNOWN")
            e1 = clone.get("entity1", {})
            e2 = clone.get("entity2", {})

            tier = self.get_tier(clone)

            # Format score columns
            score_str = f"{similarity * 100:.1f}%"
            colored_score = self.colorize(f"{score_str:<6}", tier)

            row_parts = [f"{rank:<4}", colored_score]

            if has_semantic:
                if semantic_sim is not None:
                    sem_str = f"{semantic_sim * 100:.1f}%"
                else:
                    sem_str = "-"
                row_parts.append(f"{sem_str:<6}")

            if has_syntactic:
                if syntactic_pct is not None:
                    syn_str = f"{syntactic_pct * 100:.1f}%"
                else:
                    syn_str = "-"
                row_parts.append(f"{syn_str:<6}")

            if has_blocks:
                if blocks:
                    block_str = f"{len(blocks)}({max_block_size})"
                else:
                    block_str = "-"
                row_parts.append(f"{block_str:<8}")

            colored_category = self.colorize(f"{category:<18}", tier)
            row_parts.append(colored_category)

            # Create clickable hyperlinks for function names
            name1 = e1.get("name", "?")
            file1 = e1.get("file", "")
            line1 = e1.get("line", 0)
            func1_link = self.make_hyperlink(name1, file1, line1, tier)

            name2 = e2.get("name", "?")
            file2 = e2.get("file", "")
            line2 = e2.get("line", 0)
            func2_link = self.make_hyperlink(name2, file2, line2, tier)

            # Add file:line suffix for context
            file1_short = _extract_filename(file1)
            file2_short = _extract_filename(file2)
            func1_full = f"{func1_link} ({file1_short}:{line1})"
            func2_full = f"{func2_link} ({file2_short}:{line2})"

            row_parts.extend([f"{func1_full:<{col_width}}", f"{func2_full:<{col_width}}"])

            lines.append(" | ".join(row_parts))

        lines.append("-" * _CLONE_TABLE_WIDTH)
        return "\n".join(lines)

    def format_stats(self, summary: dict[str, Any]) -> str:
        """Format clone detection summary statistics.

        Args:
            summary: Summary statistics dict from detect_clones

        Returns:
            Formatted statistics string for stderr
        """
        total = summary.get("total_functions", 0)
        pairs_found = summary.get("clone_pairs_found", 0)
        mode = summary.get("mode", "semantic")

        # Category counts (extended categories)
        exact = summary.get("exact_duplicates", 0)
        near = summary.get("near_clones", 0)
        sem_struct = summary.get("semantic_structural", 0)
        structural = summary.get("structural", 0)
        semantic = summary.get("semantic_clones", summary.get("semantic", 0))
        partial = summary.get("partial", 0)

        # Thresholds
        sem_threshold = summary.get("semantic_threshold_used", summary.get("threshold_used", 0.95))
        line_threshold = summary.get("line_threshold_used")
        func_threshold = summary.get("func_threshold_used")
        results_returned = summary.get("results_returned", pairs_found)

        mode_labels = {
            "combined": "Combined",
            "semantic": "Semantic",
            "syntactic": "Syntactic",
        }

        lines = [
            "",
            "=" * 80,
            f"Clone Detection ({mode_labels.get(mode, mode)}) | {total:,} functions | {pairs_found:,} pairs found",
            "-" * 80,
        ]

        # Show relevant thresholds
        if mode in ("combined", "semantic"):
            lines.append(f"Semantic Threshold: {sem_threshold * 100:.0f}%")
        if mode in ("combined", "syntactic") and line_threshold is not None:
            lines.append(f"Line Threshold: {line_threshold * 100:.0f}%")
        if mode in ("combined", "syntactic") and func_threshold is not None:
            lines.append(f"Function Threshold: {func_threshold * 100:.0f}%")

        lines.append("-" * 80)
        lines.append("Categories:")

        # Show category breakdown based on mode
        if mode == "combined":
            lines.append(f"  EXACT (100%):           {exact:>4}")
            lines.append(f"  NEAR-CLONE (98-100%):   {near:>4}")
            lines.append(f"  SEMANTIC-STRUCTURAL:    {sem_struct:>4}")
            lines.append(f"  STRUCTURAL:             {structural:>4}")
            lines.append(f"  SEMANTIC:               {semantic:>4}")
            lines.append(f"  PARTIAL:                {partial:>4}")
        elif mode == "syntactic":
            lines.append(f"  EXACT (100%):           {exact:>4}")
            lines.append(f"  NEAR-CLONE (95-100%):   {near:>4}")
            lines.append(f"  STRUCTURAL:             {structural:>4}")
            lines.append(f"  PARTIAL:                {partial:>4}")
        else:  # semantic
            lines.append(f"  EXACT (100%):           {exact:>4}")
            lines.append(f"  NEAR-CLONE (98-100%):   {near:>4}")
            lines.append(f"  SEMANTIC (95-98%):      {semantic:>4}")

        if results_returned < pairs_found:
            lines.append("-" * 80)
            lines.append(f"Showing {results_returned} of {pairs_found} pairs (use -n for more)")

        if summary.get("token_budget_exceeded"):
            lines.append("-" * 80)
            lines.append("Note: Results truncated due to token budget")

        lines.append("=" * 80)
        return "\n".join(lines)
