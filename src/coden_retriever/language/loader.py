"""
Language loader module.

Handles loading Tree-sitter language grammars.
"""
import ctypes
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class LanguageLoader:
    """Loads Tree-sitter language grammars."""

    def __init__(self):
        self._lib_path: Optional[str] = None
        self._lib_handle: Optional[ctypes.CDLL] = None
        self._loaded_languages: Dict[str, Any] = {}
        self._failed_languages: Set[str] = set()
        self._initialize()

    def _initialize(self) -> None:
        """Find and load the tree-sitter-languages shared library."""
        self._lib_path = self._find_library()

        if not self._lib_path:
            logger.warning("Could not find tree-sitter-languages library")
            return

        try:
            self._lib_handle = ctypes.cdll.LoadLibrary(self._lib_path)
            logger.debug(f"Loaded tree-sitter library: {self._lib_path}")
        except OSError as e:
            logger.error(f"Failed to load tree-sitter library: {e}")
            self._lib_handle = None

    def _find_library(self) -> Optional[str]:
        """Locate the tree-sitter-languages shared library file."""
        try:
            import tree_sitter_languages
        except ImportError:
            return None

        base_path = Path(tree_sitter_languages.__file__).parent
        extensions = [".so", ".dylib", ".dll"]

        for ext in extensions:
            lib_file = base_path / f"languages{ext}"
            if lib_file.exists():
                return str(lib_file)

            for found in base_path.glob(f"languages*{ext}*"):
                if found.is_file():
                    return str(found)

        return None

    def load(self, lang_name: str) -> Optional[Any]:
        """Load a language grammar by name."""
        if lang_name in self._loaded_languages:
            return self._loaded_languages[lang_name]

        if lang_name in self._failed_languages:
            return None

        if not self._lib_handle:
            self._failed_languages.add(lang_name)
            return None

        # Lazy import to avoid circular import issues
        try:
            from tree_sitter import Language
        except ImportError:
            logger.error("tree-sitter not installed")
            self._failed_languages.add(lang_name)
            return None

        symbol_name = f"tree_sitter_{lang_name}"

        try:
            func = getattr(self._lib_handle, symbol_name)
            func.restype = ctypes.c_void_p
            func.argtypes = []

            lang_ptr = func()
            if not lang_ptr:
                raise RuntimeError(f"tree_sitter_{lang_name}() returned null")

            # Try different Language constructor signatures for version compatibility
            try:
                # tree-sitter 0.21+: Language(ptr, name) - preferred
                language = Language(lang_ptr, lang_name)  # type: ignore[call-arg]
            except TypeError:
                try:
                    # tree-sitter 0.25+: Language(library_path, name)
                    language = Language(self._lib_path, lang_name)  # type: ignore[call-arg]
                except TypeError:
                    # tree-sitter 0.20: Language(ptr) - works but no name
                    language = Language(lang_ptr)

            self._loaded_languages[lang_name] = language
            logger.debug(f"Loaded language: {lang_name}")
            return language

        except AttributeError:
            logger.debug(f"Language not available: {lang_name}")
            self._failed_languages.add(lang_name)
            return None
        except Exception as e:
            logger.debug(f"Failed to load language {lang_name}: {e}")
            self._failed_languages.add(lang_name)
            return None
