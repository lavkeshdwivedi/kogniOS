from __future__ import annotations

import re

from .base import GuardrailError, InputGuardrail, OutputGuardrail


def block_keywords(*keywords: str, case_sensitive: bool = False) -> InputGuardrail:
    """Reject input containing any of the given keywords."""
    flags = 0 if case_sensitive else re.IGNORECASE
    patterns = [re.compile(re.escape(kw), flags) for kw in keywords]

    def _guard(text: str) -> str:
        for pattern in patterns:
            if pattern.search(text):
                raise GuardrailError(
                    f"Input rejected: contains blocked keyword '{pattern.pattern}'"
                )
        return text

    return _guard


def max_length(limit: int, truncate: bool = False) -> InputGuardrail:
    """Reject (or truncate) input exceeding `limit` characters."""

    def _guard(text: str) -> str:
        if len(text) > limit:
            if truncate:
                return text[:limit]
            raise GuardrailError(f"Input rejected: length {len(text)} exceeds limit {limit}")
        return text

    return _guard


def pii_scrubber() -> OutputGuardrail:
    """Scrub common PII patterns from output (email addresses, phone numbers, SSNs)."""
    _email = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    _phone = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    _ssn = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def _guard(text: str) -> str:
        text = _email.sub("[EMAIL]", text)
        text = _phone.sub("[PHONE]", text)
        text = _ssn.sub("[SSN]", text)
        return text

    return _guard


def profanity_filter(word_list: list[str] | None = None) -> InputGuardrail:
    """Reject input containing profanity. Pass a custom word_list or use a minimal default."""
    words = word_list or ["badword1", "badword2"]  # minimal; real use supplies own list
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
        re.IGNORECASE,
    )

    def _guard(text: str) -> str:
        if pattern.search(text):
            raise GuardrailError("Input rejected: contains profanity")
        return text

    return _guard
