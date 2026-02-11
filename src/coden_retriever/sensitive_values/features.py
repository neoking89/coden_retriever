"""Feature extraction for sensitive value classification.

Extracts 18 character-distribution and heuristic features from string values.
Features are designed to distinguish secrets/credentials from normal code strings.
"""
from __future__ import annotations

import base64
import binascii
import math
import re
import string
from collections import Counter

from ..constants import (
    SENSITIVE_VALUE_BASE64_PADDING,
    SENSITIVE_VALUE_MIN_BASE64_LENGTH,
    SENSITIVE_VALUE_MIN_HEX_LENGTH,
)

# High-precision prefix patterns for known secret formats
KNOWN_PREFIXES: list[str] = [
    "AKIA", "ASIA", "ABIA",                            # AWS
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_",           # GitHub
    "github_pat_",
    "glpat-",                                           # GitLab
    "sk_live_", "pk_test_", "pk_live_", "sk_test_",    # Stripe
    "sk-proj-", "sk-ant-",                              # OpenAI / Anthropic
    "xox", "xoxb-", "xoxp-",                           # Slack
    "AIza",                                             # Google
    "SG.",                                              # SendGrid
    "bearer_", "oauth_",                                # Generic auth
    "eyJ",                                              # JWT
    "-----BEGIN",                                       # PEM keys
    "MII",                                              # DER-encoded keys
    "AAAA",                                             # SSH keys
    "0x",                                               # Ethereum addresses
    "bc1q", "bc1p",                                     # Bitcoin bech32
]

# File path keywords that suggest sensitive content
SENSITIVE_PATH_KEYWORDS: list[str] = [
    ".ssh", "id_rsa", "id_ed25519", "credentials", "shadow",
    "secrets.yml", "secrets.yaml", "secrets.json", "secrets.env",
    "wallet.dat", "secring", "private", "passwords",
    ".gnupg", ".aws", "boot.key", "tax_return",
]

# Feature vector labels (18 features)
FEATURE_NAMES: list[str] = [
    "shannon_entropy",
    "string_length",
    "ratio_uppercase",
    "ratio_lowercase",
    "ratio_digits",
    "ratio_special",
    "ratio_spaces",
    "unique_char_ratio",
    "is_valid_base64",
    "is_valid_hex",
    "has_known_prefix",
    "has_sensitive_path_kw",
    "is_connection_string",
    "has_cred_in_url",
    "max_same_char_run",
    "char_class_transitions",
    "no_spaces",
    "has_base64_padding",
]


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy in bits per character."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length) for c in counts.values()
    )


def ratio_of_class(text: str, char_set: str) -> float:
    """Proportion of characters belonging to a character class."""
    if not text:
        return 0.0
    return sum(1 for c in text if c in char_set) / len(text)


def is_valid_base64(text: str) -> bool:
    """Check if string is plausible base64 (valid chars + right padding)."""
    stripped = text.rstrip("=")
    if len(stripped) < SENSITIVE_VALUE_MIN_BASE64_LENGTH:
        return False
    # Standard base64 uses A-Za-z0-9+/ and URL-safe variant uses A-Za-z0-9-_
    b64_chars = set(string.ascii_letters + string.digits + "+/-_")
    if not all(c in b64_chars for c in stripped):
        return False
    try:
        decoded = base64.b64decode(text + SENSITIVE_VALUE_BASE64_PADDING, validate=False)
        return len(decoded) > 0
    except (ValueError, binascii.Error):
        return False


def is_valid_hex(text: str) -> bool:
    """Check if string is all hex characters and reasonable length."""
    if len(text) < SENSITIVE_VALUE_MIN_HEX_LENGTH:
        return False
    return all(c in string.hexdigits for c in text)


def has_known_prefix(text: str) -> bool:
    """Check if string starts with a known secret prefix."""
    return any(text.startswith(p) for p in KNOWN_PREFIXES)


def has_sensitive_path_keyword(text: str) -> bool:
    """Check if string contains sensitive path-related keywords."""
    lower = text.lower()
    return any(kw in lower for kw in SENSITIVE_PATH_KEYWORDS)


def looks_like_connection_string(text: str) -> bool:
    """Check for DB connection string patterns (scheme://user:pass@host)."""
    return bool(re.match(
        r"^(postgres|mysql|mongodb|redis|ftp|amqp|smtp)(\+\w+)?://", text
    ))


def max_same_char_run(text: str) -> int:
    """Longest consecutive run of the same character."""
    if not text:
        return 0
    max_run = 1
    current = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def char_class_transitions(text: str) -> int:
    """Count transitions between character classes (upper/lower/digit/other)."""
    def _char_class(c: str) -> int:
        if c.isupper():
            return 0
        if c.islower():
            return 1
        if c.isdigit():
            return 2
        return 3

    if len(text) < 2:  # Need at least 2 chars for 1 transition
        return 0
    return sum(
        1 for i in range(1, len(text))
        if _char_class(text[i]) != _char_class(text[i - 1])
    )


def extract_features(text: str) -> list[float]:
    """Extract the 18-element feature vector for a single string value."""
    length = len(text) if text else 1
    return [
        shannon_entropy(text),
        float(length),
        ratio_of_class(text, string.ascii_uppercase),
        ratio_of_class(text, string.ascii_lowercase),
        ratio_of_class(text, string.digits),
        ratio_of_class(text, string.punctuation),
        ratio_of_class(text, " "),
        len(set(text)) / length,                        # unique char ratio
        float(is_valid_base64(text)),
        float(is_valid_hex(text)),
        float(has_known_prefix(text)),
        float(has_sensitive_path_keyword(text)),
        float(looks_like_connection_string(text)),
        float("@" in text and "://" in text),           # cred-in-URL pattern
        float(max_same_char_run(text)),
        float(char_class_transitions(text)),
        float(" " not in text),                         # secrets rarely have spaces
        float(text.endswith("=") or text.endswith("==")),  # base64 padding
    ]
