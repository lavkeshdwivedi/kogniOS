"""Tests for guardrails — all sync, no API calls."""

from __future__ import annotations

import pytest

from kognios.guardrails import (
    GuardrailError,
    block_keywords,
    max_length,
    pii_scrubber,
    profanity_filter,
)


def test_block_keywords_blocks():
    guard = block_keywords("forbidden", "banned")
    with pytest.raises(GuardrailError):
        guard("this is a forbidden request")


def test_block_keywords_passes():
    guard = block_keywords("forbidden")
    assert guard("this is fine") == "this is fine"


def test_block_keywords_case_insensitive():
    guard = block_keywords("Forbidden")
    with pytest.raises(GuardrailError):
        guard("FORBIDDEN word here")


def test_max_length_raises():
    guard = max_length(10)
    with pytest.raises(GuardrailError):
        guard("this is too long for the limit")


def test_max_length_passes():
    guard = max_length(100)
    assert guard("short") == "short"


def test_max_length_truncate():
    guard = max_length(5, truncate=True)
    assert guard("hello world") == "hello"


def test_pii_scrubber_email():
    guard = pii_scrubber()
    result = guard("contact me at user@example.com please")
    assert "user@example.com" not in result
    assert "[EMAIL]" in result


def test_pii_scrubber_phone():
    guard = pii_scrubber()
    result = guard("call me at 555-867-5309")
    assert "555-867-5309" not in result
    assert "[PHONE]" in result


def test_pii_scrubber_ssn():
    guard = pii_scrubber()
    result = guard("ssn is 123-45-6789")
    assert "123-45-6789" not in result
    assert "[SSN]" in result


def test_profanity_filter_blocks():
    guard = profanity_filter(["badword1"])
    with pytest.raises(GuardrailError):
        guard("this has badword1 in it")


def test_agent_input_guardrail_blocks(monkeypatch):
    from unittest.mock import MagicMock
    from kognios import Agent

    agent = Agent(
        model=MagicMock(),
        input_guardrails=[block_keywords("forbidden")],
    )
    with pytest.raises(GuardrailError):
        agent.run("forbidden request")


def test_agent_output_guardrail_scrubs():
    from unittest.mock import MagicMock
    from kognios import Agent
    from kognios.models.base import ModelResponse

    fake_model = MagicMock()
    fake_model.complete = MagicMock(
        return_value=ModelResponse(content="email is user@example.com", usage={})
    )
    agent = Agent(model=fake_model, output_guardrails=[pii_scrubber()])
    result = agent.run("what is the email?")
    assert "user@example.com" not in result
    assert "[EMAIL]" in result
