"""Shared utilities for interactive UI components.

Provides common functionality used by tool_picker, directory_browser,
and other prompt_toolkit-based UI components.
"""


def calculate_viewport(
    selected_index: int,
    total_items: int,
    max_visible: int,
) -> tuple[int, int]:
    """Calculate the visible range for a scrollable list.

    Centers the selected item in the viewport when possible,
    clamping to list boundaries.

    Args:
        selected_index: Currently selected item index (0-based).
        total_items: Total number of items in the list.
        max_visible: Maximum number of items to display at once.

    Returns:
        Tuple of (start, end) indices defining the visible range.
        Items from start (inclusive) to end (exclusive) should be rendered.

    Example:
        >>> calculate_viewport(selected_index=10, total_items=30, max_visible=12)
        (4, 16)  # Shows items 4-15, with item 10 roughly centered
    """
    start = max(0, selected_index - max_visible // 2)
    end = min(total_items, start + max_visible)
    if end - start < max_visible:
        start = max(0, end - max_visible)
    return start, end


def format_scroll_indicator(selected_index: int, total_items: int) -> str:
    """Format a scroll position indicator string.

    Args:
        selected_index: Currently selected item index (0-based).
        total_items: Total number of items in the list.

    Returns:
        Formatted string like "  (5/20)" showing 1-based position.
    """
    return f"  ({selected_index + 1}/{total_items})"
