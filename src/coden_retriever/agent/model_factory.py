"""Model factory for creating LLM model instances.

Handles provider-specific model creation logic following the Open/Closed Principle.
New providers can be added by extending the provider configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config_loader import GenerationSettings, get_config, load_config
from ..constants import (
    OLLAMA_DEFAULT_URL,
    LLAMACPP_DEFAULT_URL,
    OLLAMA_DEFAULT_API_KEY,
    LLAMACPP_DEFAULT_API_KEY,
)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    default_url: str
    default_api_key: str
    requires_api_key: bool = False


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "llamacpp": ProviderConfig(
        default_url=LLAMACPP_DEFAULT_URL,
        default_api_key=LLAMACPP_DEFAULT_API_KEY,
    ),
    "ollama": ProviderConfig(
        default_url=OLLAMA_DEFAULT_URL,
        default_api_key=OLLAMA_DEFAULT_API_KEY,
    ),
    "openai": ProviderConfig(
        default_url="",
        default_api_key="",
        requires_api_key=True,
    ),
}


def get_provider_url(provider: str) -> str:
    """Get provider URL from config or fallback to defaults."""
    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        return ""

    try:
        config = load_config()
        return config.model.provider_urls.get(provider, provider_config.default_url)
    except ImportError:
        return provider_config.default_url


class ModelFactory:
    """Factory for creating LLM model instances based on provider prefixes.

    Supported formats:
    - "llamacpp:model_name" - llama-cpp-server
    - "ollama:model_name" - Ollama server
    - "openai:model_name" - Official OpenAI API
    - "model_name" with base_url - Any OpenAI-compatible endpoint
    """

    def __init__(
        self,
        model_str: str,
        base_url: Optional[str] = None,
        generation: Optional[GenerationSettings] = None,
    ):
        self.model_str = model_str
        self.base_url = base_url
        self.generation = generation or GenerationSettings()

        self._cached_model: OpenAIChatModel | None = None
        self._cached_model_key: tuple[str, str | None, str | None] | None = None

    def _get_api_key_from_config(self) -> str | None:
        """Get api_key from config cache for immediate updates via /config set."""
        config = get_config()
        return config.model.generation.api_key if config else self.generation.api_key

    def get_model(self) -> OpenAIChatModel:
        """Get cached model instance, creating if necessary.

        Uses api_key from config cache for immediate updates via /config set.
        """
        api_key = self._get_api_key_from_config()
        current_key = (self.model_str, self.base_url, api_key)
        if self._cached_model is not None and self._cached_model_key == current_key:
            return self._cached_model

        self._cached_model = self._create_model()
        self._cached_model_key = current_key
        return self._cached_model

    def clear_cache(self) -> None:
        """Clear the cached model to force recreation on next get_model() call."""
        self._cached_model = None
        self._cached_model_key = None

    def _create_model(self) -> OpenAIChatModel:
        """Create a new model instance based on the model string prefix.

        API key resolution order:
        1. Config cache api_key (from /config set, for immediate updates)
        2. self.generation.api_key (from constructor parameter)
        3. OPENAI_API_KEY env var (for openai: prefix)
        4. Provider-specific defaults ("ollama", "not-needed")
        """
        # Get api_key from config cache for immediate updates
        api_key = self._get_api_key_from_config()

        for prefix, config in PROVIDER_CONFIGS.items():
            if self.model_str.startswith(f"{prefix}:"):
                return self._create_prefixed_model(prefix, config, api_key)

        return self._create_custom_model(api_key)

    def _create_prefixed_model(
        self, prefix: str, config: ProviderConfig, api_key: str | None
    ) -> OpenAIChatModel:
        """Create a model for a known provider prefix.

        Args:
            prefix: Provider prefix (e.g., "ollama", "openai").
            config: Provider configuration.
            api_key: API key from config cache (for immediate updates).
        """
        model_name = self.model_str.split(":", 1)[1]

        if config.requires_api_key:
            effective_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not effective_key:
                raise ValueError(
                    f"API key required for {prefix}: models. "
                    "Set OPENAI_API_KEY env var or use /config set api_key <key>"
                )
            return OpenAIChatModel(
                model_name,
                provider=OpenAIProvider(api_key=effective_key),
            )

        base_url = self.base_url or get_provider_url(prefix)
        effective_key = api_key or config.default_api_key
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=effective_key),
        )

    def _create_custom_model(self, api_key: str | None) -> OpenAIChatModel:
        """Create a model for custom OpenAI-compatible endpoints.

        Args:
            api_key: API key from config cache (for immediate updates).
        """
        # A leading `/` almost always means a persisted config was corrupted by
        # a slash-command typo (e.g. `/model /model foo`). Surface that clearly
        # instead of letting the user hit the generic base_url error.
        if self.model_str.startswith("/"):
            raise ValueError(
                f"Invalid model name '{self.model_str}': starts with '/'. "
                "Your persisted config likely stored a slash-command typo. "
                "Fix it via `/model <name>` (e.g. ollama:qwen2.5-coder:14b) "
                "or edit the 'model.default' field in your settings.json."
            )
        if not self.base_url:
            raise ValueError(
                f"base_url is required for custom model '{self.model_str}'. "
                "Use --base-url or prefix with llamacpp:/ollama:/openai:"
            )
        effective_key = api_key or "not-needed"
        return OpenAIChatModel(
            self.model_str,
            provider=OpenAIProvider(base_url=self.base_url, api_key=effective_key),
        )
