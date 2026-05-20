"""Single source of truth for unsupported-language warnings.

Both the CLI handler (via `run_audit`) and the MCP wrapper consume these
helpers, so the message text + supported-languages list stay in sync wherever
they surface.
"""
from __future__ import annotations

_INTERNAL_LANGUAGES = frozenset({"stub"})
# Why: `stub` is the polyglot-seam smoke-test adapter; advertising it would
# point users at an empty-output dead end.

_WARNING_PREFIX = "architecture: "
_WARNING_SENTINEL = "is not yet supported"
# Why: a "architecture: " prefix alone also matches CLI handler errors like
# "architecture: path does not exist". Pairing it with the sentinel phrase
# uniquely identifies messages produced by `unsupported_language_message`.


def supported_languages() -> tuple[str, ...]:
    """Names of adapters intended for end-user use, in registry order."""
    from .runner import _ADAPTER_FACTORIES
    return tuple(name for name in _ADAPTER_FACTORIES if name not in _INTERNAL_LANGUAGES)


def unsupported_language_message(detected: str) -> str:
    """Single-line warning text used by both CLI stderr and MCP wrapper."""
    supported = ", ".join(supported_languages())
    return (
        f"{_WARNING_PREFIX}'{detected}' {_WARNING_SENTINEL} "
        f"(supported: {supported}). The LanguageAdapter seam is polyglot-ready — "
        f"adding '{detected}' needs one adapter file under "
        f"src/coden_retriever/architecture/adapters/."
    )


def is_unsupported_language_message(msg: str) -> bool:
    """True iff `msg` was produced by `unsupported_language_message`."""
    return msg.startswith(_WARNING_PREFIX) and _WARNING_SENTINEL in msg


def multi_module_warning_text(walked: int, detected: int) -> str:
    """Warning shown when the adapter walked fewer source files than exist under `root`.

    Fires for both shapes of v1 multi-module explicit-fail:
      - zero scan (e.g. Maven `<modules>` parent → walked=0, detected>0)
      - partial scan (e.g. Cargo workspace where the audit catches only the
        root crate's incidental files → walked << detected)

    The text names the symptom (X of Y), gives the likely cause (multi-module
    / workspace layout), and the user-actionable fix (point at one module).
    """
    return (
        f"multi-module layout detected — adapter walked {walked} of {detected} "
        f"source files; point the audit at a single module / crate for full coverage"
    )


def parse_unsupported_language(msg: str) -> str:
    """Extract the language name from a message produced by `unsupported_language_message`.

    Format the producer guarantees: ``architecture: '<lang>' is not yet supported ...``.
    The producer only ever feeds this with auto-detected names drawn from
    `LANGUAGE_MAP` (no embedded quotes), so the split is safe. Callers receiving an
    arbitrary user-supplied `lang` should use that value directly instead of
    re-deriving it from the message.
    """
    return msg.split("'", 2)[1]
