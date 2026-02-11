"""Tramp data detection analysis.

Identifies parameter names appearing across many functions (tramp data pattern).
"""

from .detector import detect_tramp_data

__all__ = ["detect_tramp_data"]
