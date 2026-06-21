"""Tests for multi-step planning, agent-to-agent messaging, and code interpreter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kognios import Agent
from kognios.models.base import ModelResponse
from kognios.tools.builtins import code_interpreter


# ── Code interpreter ──────────────────────────────────────────────────────────


def test_code_interpreter_basic():
    result = code_interpreter("print('hello world')")
    assert "hello world" in result


def test_code_interpreter_math():
    result = code_interpreter("x = 2 + 2\nprint(x)")
    assert "4" in result


def test_code_interpreter_timeout():
    result = code_interpreter("import time; time.sleep(30)", timeout=1)
    assert "timed out" in result.lower() or "timeout" in result.lower()


def test_code_interpreter_syntax_error():
    result = code_interpreter("def bad syntax!!!")
    assert "Error" in result or "SyntaxError" in result


# ── Plan and run ──────────────────────────────────────────────────────────────


def test_plan_and_run_calls_model_multiple_times():
    fake_model = MagicMock()
    # First call: the plan
    plan_response = ModelResponse(content="1. Step one\n2. Step two", usage={})
    # Subsequent calls: step results + final
    step_response = ModelResponse(content="step done", usage={})
    final_response = ModelResponse(content="final answer", usage={})
    fake_model.complete = MagicMock(
        side_effect=[plan_response, step_response, step_response, final_response]
    )

    agent = Agent(model=fake_model)
    result = agent.plan_and_run("Do something complex")
    assert result == "final answer"
    assert fake_model.complete.call_count >= 3


@pytest.mark.asyncio
async def test_aplan_and_run():
    fake_model = MagicMock()
    plan_response = ModelResponse(content="1. Do this\n2. Do that", usage={})
    step_response = ModelResponse(content="done", usage={})
    final_response = ModelResponse(content="async final", usage={})
    fake_model.acomplete = AsyncMock(
        side_effect=[plan_response, step_response, step_response, final_response]
    )

    agent = Agent(model=fake_model)
    result = await agent.aplan_and_run("Async goal")
    assert result == "async final"


# ── Agent-to-agent ────────────────────────────────────────────────────────────


def test_agent_send():
    fake_model = MagicMock()
    fake_model.complete = MagicMock(
        return_value=ModelResponse(content="response from target", usage={})
    )
    sender = Agent(model=MagicMock())
    target = Agent(model=fake_model)

    result = sender.send(target, "hello target")
    assert result == "response from target"


@pytest.mark.asyncio
async def test_agent_asend():
    fake_model = MagicMock()
    fake_model.acomplete = AsyncMock(return_value=ModelResponse(content="async response", usage={}))
    sender = Agent(model=MagicMock())
    target = Agent(model=fake_model)

    result = await sender.asend(target, "hello async")
    assert result == "async response"
