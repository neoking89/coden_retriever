"""CLI utility functions shared across handlers."""
import argparse
from typing import Literal

from rich_argparse import RawDescriptionRichHelpFormatter


def get_asyncio():
    """Lazy import asyncio to avoid 18ms startup cost."""
    import asyncio
    return asyncio


def parse_duration(duration_str: str) -> int:
    """Parse a duration string (e.g., '30m', '1h', '90s') to seconds."""
    if not duration_str:
        return 0

    duration_str = duration_str.strip().lower()

    # Seconds-per-minute and seconds-per-hour for duration conversion
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600

    if duration_str.endswith('s'):
        return int(duration_str[:-1])
    elif duration_str.endswith('m'):
        return int(duration_str[:-1]) * SECONDS_PER_MINUTE
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * SECONDS_PER_HOUR
    else:
        return int(duration_str)


def normalize_limit(limit: int | None) -> int | None:
    """Convert negative values to None (unlimited) for limit arguments.

    This allows users to explicitly request all results via -n -1.
    Any negative value is treated as unlimited for robustness.
    """
    if limit is not None and limit < 0:
        return None
    return limit


class DefaultValueHelpFormatter(RawDescriptionRichHelpFormatter):
    """Argparse help formatter that appends default values to each argument help."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help
        if not help_text:
            base_help = super()._get_help_string(action)
            return base_help if base_help is not None else ""

        if (
            "%(default)" not in help_text
            and action.default is not argparse.SUPPRESS
        ):
            default_value = action.default
            default_str = '""' if default_value == "" else str(default_value)
            help_text = f"{help_text} (default: {default_str})"

        return help_text


def get_clone_mode(args: argparse.Namespace) -> Literal["combined", "semantic", "syntactic"]:
    """Determine clone detection mode from CLI flags."""
    if getattr(args, "clone_semantic", False):
        return "semantic"
    if getattr(args, "clone_syntactic", False):
        return "syntactic"
    return "combined"
