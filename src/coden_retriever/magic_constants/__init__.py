"""Magic constant detection — finds repeated literal values across a codebase."""
from .detector import detect_magic_constants

__all__ = ["detect_magic_constants"]
