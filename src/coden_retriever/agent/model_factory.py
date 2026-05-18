"""Model factory for creating LLM model instances.

Resolves a model-string + optional base_url into a pydantic-ai
`OpenAIChatModel` backed by the right provider. Native `OllamaProvider`
handles the `ollama:` prefix; `OpenAIProvider` handles `llamacpp:`,
`openai:`, and bare model strings.
"""

import os
from typing import Optional

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ._constants import (
    LLAMACPP_DEFAULT_API_KEY,
    LLAMACPP_DEFAULT_URL,
    OLLAMA_DEFAULT_URL,
)


class ModelFactory:
    """Factory for creating LLM model instances based on provider prefixes.

    Supported formats:
    - "llamacpp:model_name" - llama-cpp-server (no native provider; OpenAI-compat)
    - "ollama:model_name"   - native OllamaProvider
    - "openai:model_name"   - Official OpenAI API
    - "model_name" + base_url - Any OpenAI-compatible endpoint
    """

    def __init__(
        self,
        model_str: str,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
    ):
        self.model_str = model_str
        self.base_url = base_url
        self.api_key = api_key

    def create_model(self) -> OpenAIChatModel:
        if self.model_str.startswith("ollama:"):
            return self._create_ollama()
        if self.model_str.startswith("llamacpp:"):
            return self._create_llamacpp()
        if self.model_str.startswith("openai:"):
            return self._create_openai()
        return self._create_custom_model()

    def _create_ollama(self) -> OpenAIChatModel:
        model_name = self.model_str.split(":", 1)[1]
        return OpenAIChatModel(
            model_name,
            provider=OllamaProvider(base_url=self.base_url or OLLAMA_DEFAULT_URL),
        )

    def _create_llamacpp(self) -> OpenAIChatModel:
        model_name = self.model_str.split(":", 1)[1]
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                base_url=self.base_url or LLAMACPP_DEFAULT_URL,
                api_key=self.api_key or LLAMACPP_DEFAULT_API_KEY,
            ),
        )

    def _create_openai(self) -> OpenAIChatModel:
        model_name = self.model_str.split(":", 1)[1]
        effective_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not effective_key:
            raise ValueError(
                "API key required for openai: models. "
                "Set OPENAI_API_KEY env var or pass api_key=<key>"
            )
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(api_key=effective_key),
        )

    def _create_custom_model(self) -> OpenAIChatModel:
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
        return OpenAIChatModel(
            self.model_str,
            provider=OpenAIProvider(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
            ),
        )
