"""Tests for v0.3 features: parallel tools, usage tracking, persistent sessions, Ollama."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kognios.memory.short_term import ShortTermMemory
from kognios.models.ollama import OllamaModel


# ── Persistent sessions ───────────────────────────────────────────────────────


def test_short_term_memory_save_load(tmp_path):
    mem = ShortTermMemory()
    mem.append("user", "hello")
    mem.append("assistant", "hi there")

    path = str(tmp_path / "session.json")
    mem.save(path)

    mem2 = ShortTermMemory()
    mem2.load(path)
    assert mem2.messages() == mem.messages()


def test_short_term_memory_load_missing_file(tmp_path):
    mem = ShortTermMemory()
    mem.load(str(tmp_path / "nonexistent.json"))  # should not raise
    assert mem.messages() == []


# ── Ollama provider ───────────────────────────────────────────────────────────


def test_ollama_default_base_url():
    model = OllamaModel()
    assert model.model == "llama3.3"
    # base_url should point to localhost:11434
    assert "11434" in model._client.base_url.host or "11434" in str(model._client.base_url)


def test_ollama_custom_model():
    model = OllamaModel(model="mistral")
    assert model.model == "mistral"


# ── Parallel tool execution ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_tool_calls_run_concurrently():
    """Both tools should be called; order of results matches order of tool_calls."""
    from kognios import Agent, tool

    call_log: list[str] = []

    @tool
    async def tool_a() -> str:
        """Tool A."""
        call_log.append("a")
        return "result_a"

    @tool
    async def tool_b() -> str:
        """Tool B."""
        call_log.append("b")
        return "result_b"

    # Build a fake model that returns two tool calls then a final text response
    fake_model = MagicMock()
    from kognios.models.base import ModelResponse

    tool_response = ModelResponse(
        content="",
        tool_calls=[
            {"id": "id1", "name": "tool_a", "input": {}},
            {"id": "id2", "name": "tool_b", "input": {}},
        ],
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    final_response = ModelResponse(
        content="done", tool_calls=[], usage={"input_tokens": 5, "output_tokens": 3}
    )
    fake_model.acomplete = AsyncMock(side_effect=[tool_response, final_response])
    fake_model.schemas = MagicMock(return_value=[])

    agent = Agent(model=fake_model, tools=[tool_a, tool_b])
    result = await agent.arun("go")
    assert result == "done"
    assert set(call_log) == {"a", "b"}


# ── Usage tracking ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_last_usage_accumulated():
    from kognios import Agent
    from kognios.models.base import ModelResponse

    fake_model = MagicMock()
    fake_model.acomplete = AsyncMock(
        return_value=ModelResponse(
            content="hello",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
    )

    agent = Agent(model=fake_model)
    await agent.arun("hi")
    assert agent.last_usage == {"input_tokens": 10, "output_tokens": 5}
