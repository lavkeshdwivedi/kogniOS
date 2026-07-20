"""Voice-quality guardrails for editorial content generation.

Opt-in only. Default kognios usage is unaffected — none of these checks
run unless you explicitly pass the flags.

Usage::

    from kognios.guardrails.voice import voice_guardrail

    # opt-in to all checks
    guard = voice_guardrail(banned_phrases=True, excerpt_patterns=True)
    clean_text = guard(llm_output)          # raises GuardrailError on match

    # check individual fields
    from kognios.guardrails.voice import find_banned_phrases, check_excerpt
    hits = find_banned_phrases(paragraph)   # returns list of found phrases
    bad  = check_excerpt(excerpt_text)      # returns matched pattern or None
"""

from __future__ import annotations

import re

from .base import GuardrailError, OutputGuardrail


# ── Banned phrases ────────────────────────────────────────────────────────────
# AI-cliché words and constructions that read as generated rather than written.
# Add to this list when a new pattern slips through into published content.

BANNED_PHRASES: list[str] = [
    "in an era where",
    "landscape",
    "paradigm shift",
    "delve into",
    "furthermore",
    "it's worth noting",
    "game-changer",
    "unlock",
    "unveil",
    "revolutionize",
    "leverage",
    "synergy",
    "seamless",
    "seamlessly",
    "robust",
    "cutting-edge",
    "best-in-class",
    "world-class",
    "unravel",
    "unprecedented",
    "groundbreaking",
    "transformative",
    "it's important to note",
    "needless to say",
    "at the end of the day",
    "taught me the importance of",
    "powerful reminder",
    "sensory overload",
    "blends tradition and technology",
    "outside of our comfort zone",
    "outside our comfort zones",
    "fresh eyes",
    "as i reflect",
    "in the end",
    "as i look back",
    "in this article",
    "journey of discovery",
    "the importance of focus",
    "invigorated and refreshed",
    "newfound appreciation",
    "seek out diverse perspectives",
    "meaningful and lasting impact",
    "here's the thing",
    "the reality is",
    "the truth is",
    "truth is",
    "let me tell you",
    "make no mistake",
    "the short answer is",
    "spoiler alert",
    "here's the kicker",
    "but here's the kicker",
    "it turns out",
    "the question isn't",
    "the real question is",
    "the elephant in the room",
    "moving the needle",
    "double down",
    "peel back the layers",
    "when push comes to shove",
    "at its core",
    "fundamentally",
    "essentially",
    "ultimately",
    "arguably",
    "dive deep",
    "deep dive",
    "think of it as",
    "think about it",
    "sit with that for a second",
    "let that sink in",
    # Travel-blog hype
    "amazing",
    "vibrant",
    "stunning",
    "breathtaking",
    "magical",
    "unforgettable",
    "once-in-a-lifetime",
    "hidden gem",
    "off the beaten path",
    "must-see",
    "left a lasting impression",
    "stayed with me",
    "changed how i see the world",
    "reminded me of what really matters",
    "made me appreciate",
    "put things in perspective",
    "as the sun set",
    "as i sat there",
    "looking back now",
    "in that moment",
    # Repeated-observation clichés
    "i've seen teams",
    "i've seen engineers",
    "i've seen people",
    "i've seen companies",
    "i've seen this",
    "i've seen it",
    "it's not just about",
    "it is not just about",
    "the bottom line is",
    "the key is to",
    "the key is that",
    "the key to",
    "and this leads to",
    "and this is where",
    "this is where it gets",
    "this is what",
    "in other words",
    "what this means is",
    "what that means is",
    "to be clear",
    "to be fair",
    "to be honest",
    "the good news is",
    "the bad news is",
    "here's what i mean",
    "here is what i mean",
    "you might be wondering",
    "you may be wondering",
    "what you need to understand",
    "what you need to know",
    # Structural AI clichés
    "in conclusion",
    "in summary",
    "to summarize",
    "to sum up",
    "in closing",
    "as we've seen",
    "as we have seen",
    "as discussed",
    "one of the biggest challenges",
    "one of the main challenges",
    "one of the primary challenges",
    "one of the primary",
    "another challenge is",
    "another challenge",
    "one of the key",
    "one of the most important",
    "it's worth mentioning",
    "it is worth mentioning",
    "it's important to",
    "it is important to",
    "needless to mention",
    "goes without saying",
    "this is especially true",
    "this is particularly true",
    "which is why it's essential",
    "which is why it is essential",
    "which is why it's important",
    "which is why it is important",
    "overall,",
    "overall this",
    "overall it",
    "not just a testament",
    "not just about",
    "not only about",
    "but also a reminder",
    "but also a",
    "the benefits of",
    "the benefits are clear",
    "the implications of",
    "the concept of",
    "the importance of",
    "the discipline behind",
    "the principle behind",
    "the same principles",
    "the same approach",
    "by applying",
    "by establishing",
    "by taking a proactive",
    "requires a deep understanding",
    "requires a cultural",
    "considering the state of",
    "highlighting how",
    # "X has an unexpected parallel in Y" — announces a comparison instead of showing it
    "unexpected parallel",
    "has a parallel in",
    "has parallels in",
    "has parallels with",
    "similar parallel",
    "this milestone has",
    "and this milestone",
    # Passive vague endings
    "can lead to a lack",
    "leads to a lack",
    "may lead to a lack",
]


# ── Excerpt-specific patterns ─────────────────────────────────────────────────
# Patterns that are specifically bad in excerpt/description fields — either
# as openers or anywhere in the text.

_EXCERPT_BAD_OPENERS: re.Pattern[str] = re.compile(
    r"^(I(?:'ve|'ve|ve)\s+seen\b|In today[''']s|In an era|The\s+\w+\s+landscape)",
    re.IGNORECASE,
)

_EXCERPT_BAD_ANYWHERE: re.Pattern[str] = re.compile(
    r"(unexpected parallel|has a parallel in|has parallels (?:in|with)|"
    r"this milestone has|and this milestone|"
    r"can lead to a lack|leads to a lack|"
    r"where .{5,60} can lead to)",
    re.IGNORECASE,
)


# ── Public helpers ────────────────────────────────────────────────────────────


def find_banned_phrases(text: str) -> list[str]:
    """Return which BANNED_PHRASES appear in *text* (case-insensitive)."""
    if not text:
        return []
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def check_excerpt(text: str) -> str | None:
    """Return the first bad pattern found in an excerpt, or None if clean.

    Checks both opener patterns and mid-excerpt AI constructions.
    """
    if not text:
        return None
    m = _EXCERPT_BAD_OPENERS.match(text)
    if m:
        return m.group(0)
    m = _EXCERPT_BAD_ANYWHERE.search(text)
    if m:
        return m.group(0)
    return None


# ── Guardrail factory ─────────────────────────────────────────────────────────


def voice_guardrail(
    *,
    banned_phrases: bool = False,
    excerpt_patterns: bool = False,
) -> OutputGuardrail:
    """Return an OutputGuardrail that enforces editorial voice quality.

    Both checks are **off by default** so standard kognios usage is
    unaffected. Opt in explicitly::

        guard = voice_guardrail(banned_phrases=True, excerpt_patterns=True)

    Parameters
    ----------
    banned_phrases:
        When True, raises GuardrailError if the text contains any phrase
        from BANNED_PHRASES. Useful as a post-generation check on body
        paragraphs or excerpts before writing to disk.
    excerpt_patterns:
        When True, also checks for excerpt-specific AI constructions
        (bad openers, "unexpected parallel" constructions, passive vague
        endings). Intended for excerpt/description fields specifically.

    Raises
    ------
    GuardrailError
        With the matched phrase/pattern in the message.
    """

    def _guard(text: str) -> str:
        if banned_phrases:
            hits = find_banned_phrases(text)
            if hits:
                raise GuardrailError(f"Voice guardrail: banned phrase found: {hits[0]!r}")
        if excerpt_patterns:
            bad = check_excerpt(text)
            if bad:
                raise GuardrailError(f"Voice guardrail: excerpt AI pattern found: {bad!r}")
        return text

    return _guard
