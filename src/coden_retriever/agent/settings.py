"""Generation settings for the agent core.

Live-update contract: pass a `Callable[[], GenerationSettings]` (e.g.
`lambda: get_my_config().generation`) as `settings_provider` when callers
need runtime temperature/timeout/max_tokens changes to take effect mid-
session. `api_key` is read once at construction; if it changes, rebuild
the `CodingAgent`.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationSettings:
    """Model generation parameters passed to pydantic-ai's ModelSettings.

    Attributes:
        temperature: Controls randomness (0.0=deterministic, 1.0+=creative).
        max_tokens: Maximum response length (None=model default).
        timeout: Request timeout in seconds.
        api_key: API key override (read once at CodingAgent construction).
    """

    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    api_key: Optional[str] = None
