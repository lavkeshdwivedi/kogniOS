"""Guards the names downstream repos import (lavkesh, geo-pulse).

If one of these breaks, a consumer's scheduled job fails at import time, so it
should fail here first. Keep this list in step with what consumers use.
"""

import importlib

import pytest

CONSUMER_IMPORTS = [
    ("kognios", "ModelChain"),
    ("kognios", "free_tier_chain"),
    ("kognios.guardrails.voice", "BANNED_PHRASES"),
    ("kognios.guardrails.voice", "check_excerpt"),
    ("kognios.guardrails.voice", "find_banned_phrases"),
    ("kognios.models.chain", "free_tier_chain"),
    ("kognios", "collect_keys"),
    ("kognios.models.chain", "collect_keys"),
    ("kognios.models.chain", "_collect_keys"),
    ("kognios.models.groq", "GroqModel"),
    ("kognios.models.gemini", "GeminiModel"),
    ("kognios.models.anthropic", "AnthropicModel"),
]


@pytest.mark.parametrize(("module", "name"), CONSUMER_IMPORTS)
def test_consumer_import(module, name):
    assert hasattr(importlib.import_module(module), name)
