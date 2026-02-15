"""Feature extraction for sensitive value classification.

Extracts 32 character-distribution and heuristic features from string values.
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
    SENSITIVE_VALUE_BASE64_READABLE_RATIO,
    SENSITIVE_VALUE_MIN_ALPHA_CHARS_FOR_CAPS,
    SENSITIVE_VALUE_MIN_BASE64_LENGTH,
    SENSITIVE_VALUE_MIN_HEX_LENGTH,
    SENSITIVE_VALUE_MIN_PASSWORD_SPECIAL_CHARS,
    SENSITIVE_VALUE_MIN_PLURAL_STRIP_LENGTH,
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
    "wallet.dat", "secring", "passwords", "private_key",
    ".gnupg", ".aws", "boot.key", "tax_return",
    "/private/", "\\private\\",  # path segment (avoids matching prose like "private functions")
]

# Prefixes indicating non-secret hash digests (content-addressable identifiers)
_HASH_DIGEST_PREFIXES: list[str] = [
    "sha256:", "sha1:", "md5:", "sha512:", "sha384:",
    "blake2b:", "blake2s:", "sha3:",
]

# UUID v1-v5 pattern: 8-4-4-4-12 hex digits
_UUID_PATTERN: re.Pattern[str] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Middle initial or abbreviated title pattern (e.g., "Dr. Sarah", "Robert M. Michelson")
_SINGLE_CHAR_PERIOD_PATTERN: re.Pattern[str] = re.compile(r"\b[A-Z][a-z]{0,2}\.\s")

# URL credential pattern (scheme://authority where authority contains username:password@host)
_CRED_IN_URL_PATTERN: re.Pattern[str] = re.compile(r"://([^/]+)")

# Chars strongly associated with leet-speak passwords (not formatting/CSS)
_PASSWORD_SPECIAL_CHARS: frozenset[str] = frozenset("!@#$%^&*")

# English derivational suffixes that appear in code labels but never in names.
# Structural pattern: "configuration", "processing", "available" vs "Rodriguez"
_NON_NAME_SUFFIXES: tuple[str, ...] = (
    "tion", "sion", "ment", "ness", "ful", "less", "ive", "ous",
    "ity", "able", "ible", "ance", "ence", "ize", "ise", "ing",
    "ated", "ling", "ory", "ure",
)

# Formatting characters that appear in code labels but never in person names
_FORMATTING_CHARS: frozenset[str] = frozenset(":;|()[]{}=<>@")

# Closed grammatical class: name particles across European/Arabic languages.
# These are function words that appear between given name and surname,
# NOT domain vocabulary. The set is linguistically fixed and does not grow.
_NAME_PARTICLES: frozenset[str] = frozenset({
    "de", "da", "di", "del", "der", "den", "des",  # Romance / Germanic
    "van", "von",                                    # Dutch / German
    "la", "le", "los", "las",                        # Articles as name parts
    "al", "el",                                      # Arabic / Spanish
    "do", "dos",                                     # Portuguese
})

# Feature vector labels (32 features)
FEATURE_NAMES: list[str] = [
    "shannon_entropy",
    "looks_like_identifier",
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
    "looks_like_uuid",
    "has_password_pattern",
    "word_count",
    "has_hash_digest_prefix",
    "base64_readable",
    "lowercase_hex_only",
    "has_non_name_suffix",
    "has_formatting_chars",
    "is_all_lowercase_words",
    "is_all_caps_phrase",
    "has_leading_whitespace",
    "has_lowercase_middle_word",
    "has_single_char_period",
    "has_hyphenated_capital_word",
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


def looks_like_identifier(text: str) -> bool:
    """Check if string matches CamelCase or snake_case code identifier patterns.

    CamelCase: "BM25Index", "HttpClientFactory", "OAuth2Provider"
    snake_case: "func_threshold", "max_retries", "output_format"
    These are safe code identifiers, not secrets.
    """
    if not text or " " in text:
        return False
    # CamelCase: alphanumeric only, starts uppercase, has lowercase chars
    if text.isalnum() and text[0].isupper() and any(c.islower() for c in text):
        return True
    # snake_case: lowercase + underscores + digits, must have underscore
    # Exclude strings where last segment is a long hex value (likely a secret key)
    if "_" in text and all(c.islower() or c == "_" or c.isdigit() for c in text):
        last_segment = text.rsplit("_", 1)[-1]
        hex_chars = set("0123456789abcdef")
        if len(last_segment) > SENSITIVE_VALUE_MIN_HEX_LENGTH and all(
            c in hex_chars for c in last_segment
        ):
            return False
        return True
    # SCREAMING_SNAKE_CASE: uppercase + underscores + digits (env var names)
    if "_" in text and all(c.isupper() or c == "_" or c.isdigit() for c in text):
        return True
    return False


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


def looks_like_uuid(text: str) -> bool:
    """Check if string matches UUID format (8-4-4-4-12 hex digits)."""
    return bool(_UUID_PATTERN.match(text))


def has_password_pattern(text: str) -> bool:
    """Check if string has letters + digits + 2 distinct password-special chars.

    Requires 2+ distinct chars from `!@#$%^&*` to avoid false positives
    on CSS colors (`bg:#hex`) and format strings (`{value*100}`).
    """
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    special_count = len({c for c in text if c in _PASSWORD_SPECIAL_CHARS})
    return has_letter and has_digit and special_count >= SENSITIVE_VALUE_MIN_PASSWORD_SPECIAL_CHARS


def word_count(text: str) -> int:
    """Count space-separated words. Natural language has many; secrets have few."""
    return len(text.split())


def has_hash_digest_prefix(text: str) -> bool:
    """Check if string starts with a hash algorithm prefix (sha256:, md5:, etc.)."""
    lower = text.lower()
    return any(lower.startswith(p) for p in _HASH_DIGEST_PREFIXES)


def base64_decodes_to_readable(text: str) -> bool:
    """Check if base64 decoding yields printable ASCII text.

    Safe base64 (encoded config/URLs) decodes to readable text.
    Secret base64 (encrypted keys/tokens) decodes to random bytes.
    """
    if not is_valid_base64(text):
        return False
    try:
        decoded = base64.b64decode(
            text + SENSITIVE_VALUE_BASE64_PADDING, validate=False,
        )
        decoded_str = decoded.decode("ascii")
        # Check that most characters are printable ASCII
        printable_ratio = sum(1 for c in decoded_str if c.isprintable()) / max(len(decoded_str), 1)
        return printable_ratio > SENSITIVE_VALUE_BASE64_READABLE_RATIO
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return False


def is_lowercase_hex_only(text: str) -> bool:
    """Check if string is exclusively lowercase hex chars (a-f, 0-9).

    Hash digests (MD5, SHA) are typically lowercase hex.
    Secret keys often include uppercase letters.
    """
    if len(text) < SENSITIVE_VALUE_MIN_HEX_LENGTH:
        return False
    hex_lower = set("0123456789abcdef")
    return all(c in hex_lower for c in text)


def has_non_name_suffix(text: str) -> bool:
    """Check if any word has an English derivational suffix uncommon in names.

    Words like "configuration", "processing", "available" end with suffixes
    (-tion, -ing, -able) that distinguish code labels from person names.
    Handles plural forms: "settings" -> "setting" -> -ing match.
    """
    words = text.lower().split()
    for w in words:
        forms = [w]
        if w.endswith("s") and len(w) >= SENSITIVE_VALUE_MIN_PLURAL_STRIP_LENGTH:
            forms.append(w[:-1])
        if any(f.endswith(s) for f in forms for s in _NON_NAME_SUFFIXES):
            return True
    return False


def has_formatting_chars(text: str) -> bool:
    """Check if string contains formatting characters absent from names.

    Code labels often contain : | ( ) [ ] etc. Person names never do.
    """
    return bool(set(text) & _FORMATTING_CHARS)


def is_all_lowercase_words(text: str) -> bool:
    """Check if all alphabetic characters are lowercase.

    Style strings ("bold cyan"), error messages ("no match") are lowercase.
    Person names always have uppercase letters.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    return len(alpha_chars) > 0 and all(c.islower() for c in alpha_chars)


def is_all_caps_phrase(text: str) -> bool:
    """Check if string is an ALL-CAPS phrase with spaces.

    Matches log headers like "DEBUG SESSION END", "FINAL RESPONSE".
    Person names are never all-uppercase, so this signals safe.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < SENSITIVE_VALUE_MIN_ALPHA_CHARS_FOR_CAPS:
        return False
    return " " in text and all(c.isupper() for c in alpha_chars)


def has_leading_whitespace(text: str) -> bool:
    """Check if string starts with whitespace (indented UI/menu text)."""
    return len(text) > 0 and text[0] in (" ", "\t")


def has_lowercase_middle_word(text: str) -> bool:
    """Check if string has a name particle between capitalized words.

    Name particles (van, de, da, von) are a closed grammatical class.
    They appear lowercase between capitalized given name and surname.
    Code phrases ("Stop the daemon") use articles/prepositions not in this set.
    """
    words = text.split()
    if len(words) < 3:
        return False
    for i in range(1, len(words)):
        w = words[i]
        if not w or not w[0] or w[0].isupper():
            continue
        if w.lower() not in _NAME_PARTICLES:
            continue
        has_prev_cap = any(w2[0].isupper() for w2 in words[:i] if w2)
        has_next_cap = any(w2[0].isupper() for w2 in words[i + 1:] if w2)
        if has_prev_cap and has_next_cap:
            return True
    return False


def has_single_char_period(text: str) -> bool:
    """Check if string contains an abbreviated title or middle initial.

    Matches middle initials ("Robert M. Michelson") and titles ("Dr. Sarah").
    Code labels never contain abbreviated titles or single-letter initials.
    """
    return bool(_SINGLE_CHAR_PERIOD_PATTERN.search(text))


def has_hyphenated_capital_word(text: str) -> bool:
    """Check if string has a hyphenated word with exactly two capitalized parts.

    Name patterns: Jean-Pierre, Anne-Marie, Karl-Heinz (always 2 parts).
    Excludes ALL-CAPS words (MID-SESSION), 3+ part headers (Content-Type),
    single-word strings, and strings with formatting chars (HTTP headers).
    """
    words = text.split()
    if len(words) < 2:
        return False
    # Formatting chars signal headers/labels ("Cache-Control: ..."), not names
    if set(text) & _FORMATTING_CHARS:
        return False
    for word in words:
        if "-" in word:
            parts = [p for p in word.split("-") if p]
            if len(parts) != 2 or not all(p and p[0].isupper() for p in parts):
                continue
            # Exclude ALL-CAPS hyphenated words (headers/constants, not names)
            if all(c.isupper() or not c.isalpha() for c in word):
                continue
            return True
    return False


def has_cred_in_url(text: str) -> bool:
    """Check if URL has credentials in authority part (user:pass@host).

    Matches: postgres://admin:pass@host, ftp://user:key@server
    Does NOT match: cdn.jsdelivr.net/npm/chart.js@4.4.1 (@ in path).
    """
    match = _CRED_IN_URL_PATTERN.search(text)
    if not match:
        return False
    return "@" in match.group(1)


def extract_features(text: str) -> list[float]:
    """Extract the 32-element feature vector for a single string value."""
    length = len(text) if text else 1
    return [
        shannon_entropy(text),
        float(looks_like_identifier(text)),              # CamelCase/snake_case = safe
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
        float(has_cred_in_url(text)),                     # cred-in-URL pattern
        float(max_same_char_run(text)),
        float(char_class_transitions(text)),
        float(" " not in text),                         # secrets rarely have spaces
        float(text.endswith("=") or text.endswith("==")),  # base64 padding
        float(looks_like_uuid(text)),                   # UUIDs are safe, not secrets
        float(has_password_pattern(text)),               # leet-speak passwords
        float(word_count(text)),                         # natural language = safe
        float(has_hash_digest_prefix(text)),             # sha256:... = safe digest
        float(base64_decodes_to_readable(text)),         # readable decoded = safe
        float(is_lowercase_hex_only(text)),              # lowercase hex = hash digest
        float(has_non_name_suffix(text)),                 # -tion/-ing/-able = label
        float(has_formatting_chars(text)),                # :;|()[] = not a name
        float(is_all_lowercase_words(text)),              # all lowercase = not a name
        float(is_all_caps_phrase(text)),                  # ALL CAPS headers = safe
        float(has_leading_whitespace(text)),              # indented text = safe
        float(has_lowercase_middle_word(text)),            # name particles = PII
        float(has_single_char_period(text)),               # middle initials = PII
        float(has_hyphenated_capital_word(text)),           # hyphenated names = PII
    ]
