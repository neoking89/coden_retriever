"""Centralized threshold configuration for CLI analysis arguments."""
import argparse
from dataclasses import dataclass

from ..constants import (
    DEFAULT_CLONE_SEMANTIC_THRESHOLD,
    DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD,
    DEFAULT_ECHO_COMMENT_THRESHOLD,
    DEFAULT_HOTSPOT_RISK_THRESHOLD,
    DEFAULT_PROPAGATION_COST_THRESHOLD,
    SENSITIVE_VALUE_DEFAULT_THRESHOLD,
)


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for a CLI threshold argument."""

    name: str  # CLI argument name without -- (e.g., "clone-threshold")
    default: float
    analysis_flag: str  # Short flag (e.g., "-C")
    analysis_name: str  # Human-readable name (e.g., "Code Clones")
    short_help: str  # Brief help for search parser
    detailed_help: str  # Full help for flag parser (includes range, examples)
    example_value: float  # Value to use in examples
    validate_0_1: bool = True  # If True, validate range 0.0-1.0


THRESHOLD_CONFIGS: dict[str, ThresholdConfig] = {
    "risk": ThresholdConfig(
        name="risk-threshold",
        default=DEFAULT_HOTSPOT_RISK_THRESHOLD,
        analysis_flag="-H",
        analysis_name="Hotspots",
        short_help="(-H) Hotspot min risk score (raw score, typically 50-200+)",
        detailed_help="Hotspots (-H): min risk score for flagging. Raw score = coupling * log(complexity). Default: 50",
        example_value=DEFAULT_HOTSPOT_RISK_THRESHOLD,
        validate_0_1=False,
    ),
    "propagation": ThresholdConfig(
        name="propagation-threshold",
        default=DEFAULT_PROPAGATION_COST_THRESHOLD,
        analysis_flag="-P",
        analysis_name="Propagation Cost",
        short_help="(-P) Propagation cost threshold (0.0-1.0)",
        detailed_help="Propagation (-P): min internal coupling %% for flagging modules. Range: 0-1. Default: 0.25 (25%%)",
        example_value=DEFAULT_PROPAGATION_COST_THRESHOLD,
    ),
    "clone": ThresholdConfig(
        name="clone-threshold",
        default=DEFAULT_CLONE_SEMANTIC_THRESHOLD,
        analysis_flag="-C",
        analysis_name="Code Clones",
        short_help="(-C) Clone similarity threshold (0.0-1.0)",
        detailed_help="Clones (-C): min semantic similarity for flagging. Range: 0-1. Default: 0.95 (very similar)",
        example_value=0.90,
    ),
    "echo": ThresholdConfig(
        name="echo-threshold",
        default=DEFAULT_ECHO_COMMENT_THRESHOLD,
        analysis_flag="-E",
        analysis_name="Echo Comments",
        short_help="(-E) Echo comment similarity threshold (0.0-1.0)",
        detailed_help="Echo Comments (-E): semantic similarity threshold. Range: 0-1. Default: 0.80. Stricter (0.95) = near-identical only, Looser (0.75) = more detections",
        example_value=DEFAULT_ECHO_COMMENT_THRESHOLD,
    ),
    "dead_code": ThresholdConfig(
        name="dead-code-threshold",
        default=DEFAULT_DEAD_CODE_CONFIDENCE_THRESHOLD,
        analysis_flag="-D",
        analysis_name="Dead Code",
        short_help="(-D) Dead code confidence threshold (0.0-1.0)",
        detailed_help="Dead Code (-D): min confidence score for flagging. Range: 0-1. Default: 0.5 (medium confidence)",
        example_value=0.7,
    ),
    "sensitive_value": ThresholdConfig(
        name="sensitive-threshold",
        default=SENSITIVE_VALUE_DEFAULT_THRESHOLD,
        analysis_flag="-S",
        analysis_name="Sensitive Values",
        short_help="(-S) Sensitive value confidence threshold (0.0-1.0)",
        detailed_help="Sensitive Values (-S): min confidence for flagging. Range: 0-1. Default: 0.35. Lower = more recall, higher = more precision",
        example_value=0.50,
    ),
}


def _validate_threshold(value: str) -> float:
    """Validate threshold is between 0.0 and 1.0."""
    fval = float(value)
    if not (0.0 <= fval <= 1.0):
        raise argparse.ArgumentTypeError(f"threshold must be between 0.0 and 1.0, got {fval}")
    return fval


def _validate_positive_float(value: str) -> float:
    """Validate threshold is a positive number (no upper bound)."""
    fval = float(value)
    if fval < 0:
        raise argparse.ArgumentTypeError(f"threshold must be non-negative, got {fval}")
    return fval


def add_threshold_argument(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    config: ThresholdConfig,
    use_detailed_help: bool = False,
) -> None:
    """Add a threshold argument to a parser using centralized config."""
    help_text = config.detailed_help if use_detailed_help else config.short_help
    validator = _validate_threshold if config.validate_0_1 else _validate_positive_float
    parser.add_argument(
        f"--{config.name}",
        type=validator,
        default=config.default,
        metavar="FLOAT",
        help=help_text,
    )
