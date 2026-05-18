r"""LaTeX math-mode → Unicode substitution for terminal rendering.

LLMs frequently emit math notation like ``$\rightarrow$``, ``$\frac{a}{b}$``
or ``$\mathbb{R}$`` even when prompted for plain text. Rich's Markdown
renderer shows the raw command; we substitute terminal-safe Unicode
(Latin-1, Dingbats, Math Operators) so the panel reads naturally.

Coverage tiers reflect observed frequency across GPT-4, Claude, Gemma,
LLaMA. Exotic commands are intentionally absent — adding them costs
regex breadth and almost never fires on real output.
"""

from __future__ import annotations

import re


class _LatexTables:
    """All command → Unicode lookup tables in one namespace."""

    # Math commands that map 1:1 to a single Unicode codepoint (or short
    # sequence). Tiered by observed frequency in LLM output.
    SYMBOLS: dict[str, str] = {
        # ─── Frequently emitted ─────────────────────────────────────────
        # Arrows — chevron / triangle style for a modern terminal look
        r"\rightarrow": "❯",            # ❯  Dingbats
        r"\to":         "❯",            # ❯  shorthand for \rightarrow
        r"\leftarrow":  "❮",            # ❮  Dingbats
        r"\Rightarrow": "»",            # »  Latin-1
        r"\Leftarrow":  "«",            # «  Latin-1
        # Arithmetic
        r"\times": "×",
        r"\div":   "÷",
        # Comparisons
        r"\neq":    "≠",
        r"\leq":    "≤",
        r"\geq":    "≥",
        r"\approx": "≈",
        # Constants / big operators
        r"\infty": "∞",
        r"\sum":   "∑",
        r"\prod":  "∏",
        # Greek letters (ML / stats contexts)
        r"\alpha":   "α",
        r"\beta":    "β",
        r"\gamma":   "γ",
        r"\delta":   "δ",
        r"\epsilon": "ε",
        r"\lambda":  "λ",
        r"\mu":      "μ",
        r"\pi":      "π",
        r"\sigma":   "σ",
        r"\theta":   "θ",
        r"\phi":     "φ",
        r"\omega":   "ω",
        # Set membership
        r"\in":       "∈",
        r"\notin":    "∉",
        r"\subset":   "⊂",
        r"\subseteq": "⊆",
        # Punctuation
        r"\cdot":  "·",
        r"\ldots": "…",
        r"\dots":  "…",                 # alias for \ldots
        # Brackets (inner products, tuples)
        r"\langle": "⟨",
        r"\rangle": "⟩",

        # ─── Occasionally emitted ───────────────────────────────────────
        r"\pm":             "±",
        r"\leftrightarrow": "❮❯",
        r"\Leftrightarrow": "«»",
        r"\uparrow":        "▲",
        r"\downarrow":      "▼",
        r"\cup":            "∪",
        r"\cap":            "∩",
        r"\emptyset":       "∅",
        r"\supset":         "⊃",
        r"\forall":         "∀",
        r"\exists":         "∃",
        r"\nabla":          "∇",
        r"\partial":        "∂",
        r"\cdots":          "⋯",
        r"\star":           "★",
        r"\bullet":         "•",
        r"\checkmark":      "✔",
        r"\dagger":         "†",
    }

    # ``$\xrightarrow{Label}$`` and ``$\xleftarrow{Label}$`` — produced by
    # Gemma and some GPT-4 outputs. Exotic variants (``\xLeftarrow``, …)
    # are absent on purpose.
    LABELED_ARROWS: dict[str, tuple[str, str]] = {
        r"\xrightarrow": ("─[", "]❯"),  # ─[Label]❯
        r"\xleftarrow":  ("❮[", "]─"),  # ❮[Label]─
    }

    # Number sets that have dedicated double-struck Unicode glyphs.
    MATHBB: dict[str, str] = {
        "R": "ℝ",
        "N": "ℕ",
        "Z": "ℤ",
        "Q": "ℚ",
        "C": "ℂ",
    }

    # Single Unicode codepoint used for the square-root prefix.
    SQRT_PREFIX: str = "√"


# ── Regex assembly ───────────────────────────────────────────────────────


def _delimited(inner: str) -> str:
    """Wrap a LaTeX-command body with optional ``$..$`` or ``$$..$$``."""
    return r"\$\$?\s*" + inner + r"\s*\$\$?"


def _alternation(commands: list[str]) -> str:
    # Longest-first so ``\rightarrow`` matches before ``\to``.
    return "|".join(re.escape(cmd) for cmd in sorted(commands, key=len, reverse=True))


class _LatexPatterns:
    """All compiled regex patterns used by :func:`replace_latex_symbols`."""

    INLINE = re.compile(
        _delimited("(" + _alternation(list(_LatexTables.SYMBOLS)) + ")")
    )
    LABELED = re.compile(
        _delimited(
            "(" + _alternation(list(_LatexTables.LABELED_ARROWS)) + r")\s*\{([^}]*)\}"
        )
    )
    FRAC = re.compile(_delimited(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}"))
    SQRT = re.compile(_delimited(r"\\sqrt\s*\{([^}]*)\}"))
    MATHBB = re.compile(_delimited(r"\\mathbb\s*\{([^}]*)\}"))


# ── Substitution callbacks ───────────────────────────────────────────────


def _sub_simple(match: re.Match[str]) -> str:
    return _LatexTables.SYMBOLS[match.group(1)]


def _sub_labeled(match: re.Match[str]) -> str:
    prefix, suffix = _LatexTables.LABELED_ARROWS[match.group(1)]
    return f"{prefix}{match.group(2).strip()}{suffix}"


def _sub_frac(match: re.Match[str]) -> str:
    return f"{match.group(1).strip()}/{match.group(2).strip()}"


def _sub_sqrt(match: re.Match[str]) -> str:
    return f"{_LatexTables.SQRT_PREFIX}({match.group(1).strip()})"


def _sub_mathbb(match: re.Match[str]) -> str:
    letter = match.group(1).strip()
    return _LatexTables.MATHBB.get(letter, letter)


# ── Public API ───────────────────────────────────────────────────────────


def replace_latex_symbols(text: str) -> str:
    r"""Replace common LaTeX math-mode commands with Unicode equivalents.

    Handles simple symbols (``$\rightarrow$`` → ❯), labeled arrows
    (``$\xrightarrow{Parser}$`` → ─[Parser]❯), fractions, square roots,
    and double-struck number sets (``$\mathbb{R}$`` → ℝ). Unknown
    commands are passed through verbatim.
    """
    # Fast path — every supported pattern requires a '$' delimiter.
    if "$" not in text:
        return text

    # Parameterised commands first; otherwise the trailing simple-symbol
    # pass consumes just the leading ``$\xrightarrow$`` fragment.
    text = _LatexPatterns.LABELED.sub(_sub_labeled, text)
    text = _LatexPatterns.FRAC.sub(_sub_frac, text)
    text = _LatexPatterns.SQRT.sub(_sub_sqrt, text)
    text = _LatexPatterns.MATHBB.sub(_sub_mathbb, text)
    return _LatexPatterns.INLINE.sub(_sub_simple, text)
