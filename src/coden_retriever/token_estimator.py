"""
Token estimation module.

Provides generic token estimation with multiple backends.
"""
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class TokenEstimator:
    """
    Generic token estimation with multiple backends.

    Supports:
    - Heuristic estimation (zero dependencies, ~90% accurate for code)
    - tiktoken (OpenAI models)
    - tokenizers (HuggingFace, generic BPE)
    """

    # Code typically tokenizes at ~3.2 chars/token, prose at ~4 chars/token
    CODE_CHARS_PER_TOKEN: float = 3.2
    PROSE_CHARS_PER_TOKEN: float = 4.0

    def __init__(self, backend: str = "auto"):
        """
        Initialize token estimator.

        Args:
            backend: One of "auto", "heuristic", "tiktoken", "tokenizers"
        """
        self._backend = backend
        # Any: encoder can be tiktoken.Encoding or tokenizers.Tokenizer
        self._encoder: Any = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the best available tokenization backend."""
        if self._backend == "heuristic":
            return

        if self._backend in ("auto", "tiktoken"):
            try:
                import tiktoken
                # cl100k_base is used by GPT-4, works reasonably for most LLMs
                self._encoder = tiktoken.get_encoding("cl100k_base")
                self._backend = "tiktoken"
                logger.debug("Using tiktoken backend for token estimation")
                return
            except ImportError:
                if self._backend == "tiktoken":
                    logger.warning("tiktoken not installed, falling back to heuristic")

        if self._backend in ("auto", "tokenizers"):
            try:
                from tokenizers import Tokenizer
                # Use a lightweight pre-trained tokenizer
                self._encoder = Tokenizer.from_pretrained("bert-base-uncased")
                self._backend = "tokenizers"
                logger.debug("Using tokenizers backend for token estimation")
                return
            except (ImportError, Exception):
                if self._backend == "tokenizers":
                    logger.warning("tokenizers not available, falling back to heuristic")

        self._backend = "heuristic"
        logger.debug("Using heuristic backend for token estimation")

    def count(self, text: str, is_code: bool = True) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for
            is_code: Whether text is code (affects heuristic estimation)

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        if self._backend == "tiktoken" and self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass

        if self._backend == "tokenizers" and self._encoder:
            try:
                return len(self._encoder.encode(text).ids)
            except Exception:
                pass

        # Heuristic fallback
        chars_per_token = self.CODE_CHARS_PER_TOKEN if is_code else self.PROSE_CHARS_PER_TOKEN
        return max(1, int(len(text) / chars_per_token))

    def estimate_chars_for_tokens(self, token_budget: int, is_code: bool = True) -> int:
        """Estimate how many characters fit in a token budget."""
        chars_per_token = self.CODE_CHARS_PER_TOKEN if is_code else self.PROSE_CHARS_PER_TOKEN
        return int(token_budget * chars_per_token)


# Global token estimator instance
_token_estimator: TokenEstimator | None = None


def get_token_estimator() -> TokenEstimator:
    """Get or create the global token estimator instance."""
    global _token_estimator
    if _token_estimator is None:
        _token_estimator = TokenEstimator(backend="auto")
    return _token_estimator


@lru_cache(maxsize=1024)
def _count_tokens_cached(text: str, is_code: bool) -> int:
    """Cached token count for repeated text (common in search results)."""
    return get_token_estimator().count(text, is_code)


def count_tokens(text: str, is_code: bool = True) -> int:
    """Count tokens in text using the global estimator with caching."""
    # Only cache for reasonable-sized strings (avoid memory bloat)
    if len(text) < 50000:
        return _count_tokens_cached(text, is_code)
    return get_token_estimator().count(text, is_code)
