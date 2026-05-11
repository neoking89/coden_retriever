"""Semantic search configuration value object."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticConfig:
    """Whether semantic search is on, and which ONNX model directory to use.

    The two fields travel together: a model_path with enabled=False is dead
    weight, and most call sites already pair them. Carrying them as one value
    object keeps signatures honest and lets the receiving entity hold the pair
    as a single attribute.
    """

    enabled: bool = False
    model_path: str | None = None
