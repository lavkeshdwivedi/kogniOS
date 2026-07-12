import pytest
from unittest.mock import MagicMock
from kognios.agent import Agent
from kognios.models.base import ModelResponse
from kognios.tools.registry import tool
from kognios.memory.short_term import ShortTermMemory


def make_model(content="answer", tool_calls=None):
    model = MagicMock()
    model.complete.return_value = ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
    )
    return model


def test_basic_run():
    model = make_model("Paris is the capital of France.")
    agent = Agent(model=model)
    result = agent.run("What is the capital of France?")
    assert result == "Paris is the capital of France."
    model.complete.assert_called_once()


def test_system_prompt_uses_instructions():
    model = make_model("ok")
    agent = Agent(model=model, instructions="Be concise.")
    agent.run("hello")
    _, kwargs = model.complete.call_args
    assert "Be concise." in kwargs.get("system", "")


def test_tool_call_loop():
    @tool
    def add(x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    # First call returns tool_call, second call returns final answer.
    model = MagicMock()
    model.complete.side_effect = [
        ModelResponse(
            content="", tool_calls=[{"id": "1", "name": "add", "input": {"x": 2, "y": 3}}]
        ),
        ModelResponse(content="The answer is 5.", tool_calls=[]),
    ]
    agent = Agent(model=model, tools=[add])
    result = agent.run("What is 2 + 3?")
    assert result == "The answer is 5."
    assert model.complete.call_count == 2


def test_max_iterations_raises():
    model = MagicMock()
    model.complete.return_value = ModelResponse(
        content="", tool_calls=[{"id": "1", "name": "loop", "input": {}}]
    )

    @tool
    def loop() -> str:
        """Loops forever."""
        return "still going"

    agent = Agent(model=model, tools=[loop], max_iterations=3)
    with pytest.raises(RuntimeError, match="did not produce a response"):
        agent.run("go")


def test_memory_persists_across_turns():
    model = make_model("remembered")
    mem = ShortTermMemory()
    agent = Agent(model=model, memory=mem)
    agent.run("first message")
    agent.run("second message")
    msgs = mem.messages()
    assert len(msgs) == 4  # 2 turns × (user + assistant)


def test_knowledge_injected_into_system():
    kb = MagicMock()
    kb.search.return_value = ["Relevant chunk about Python."]
    model = make_model("answer")
    agent = Agent(model=model, knowledge=kb)
    agent.run("tell me about Python")
    kb.search.assert_called_once_with("tell me about Python")
    _, kwargs = model.complete.call_args
    assert "Relevant chunk about Python." in kwargs.get("system", "")


def test_max_tool_errors_raises():
    @tool
    def broken() -> str:
        """Always fails."""
        return "Error: something went wrong"

    model = MagicMock()
    model.complete.return_value = ModelResponse(
        content="", tool_calls=[{"id": "1", "name": "broken", "input": {}}]
    )
    agent = Agent(model=model, tools=[broken], max_tool_errors=2)
    with pytest.raises(RuntimeError, match="failed 2 times"):
        agent.run("do something")


def test_max_tool_errors_default_allows_some_errors():
    call_count = {"n": 0}

    @tool
    def flaky() -> str:
        """Fails twice then succeeds."""
        call_count["n"] += 1
        if call_count["n"] < 3:
            return "Error: not ready"
        return "ok"

    model = MagicMock()
    model.complete.side_effect = [
        ModelResponse(content="", tool_calls=[{"id": "1", "name": "flaky", "input": {}}]),
        ModelResponse(content="", tool_calls=[{"id": "2", "name": "flaky", "input": {}}]),
        ModelResponse(content="", tool_calls=[{"id": "3", "name": "flaky", "input": {}}]),
        ModelResponse(content="done", tool_calls=[]),
    ]
    agent = Agent(model=model, tools=[flaky], max_tool_errors=3)
    result = agent.run("go")
    assert result == "done"


def test_max_context_tokens_trims_messages():
    agent = Agent(model=MagicMock(), max_context_tokens=50)
    # 10 messages × 100 chars = 1000 chars ≈ 250 tokens; limit is 50 tokens
    messages = [{"role": "user", "content": "x" * 100}] * 10
    trimmed = agent._trim_messages(list(messages))
    assert len(trimmed) < len(messages)


def test_max_context_tokens_none_keeps_all():
    agent = Agent(model=MagicMock(), max_context_tokens=None)
    messages = [{"role": "user", "content": "x" * 500}] * 5
    trimmed = agent._trim_messages(list(messages))
    assert len(trimmed) == 5


def test_max_context_tokens_preserves_last_message():
    agent = Agent(model=MagicMock(), max_context_tokens=1)
    messages = [{"role": "user", "content": "a" * 1000}, {"role": "user", "content": "last"}]
    trimmed = agent._trim_messages(list(messages))
    assert trimmed[-1]["content"] == "last"


def test_parallel_sync_tool_calls_both_executed():
    call_log = []

    @tool
    def tool_a() -> str:
        """Tool A."""
        call_log.append("a")
        return "result_a"

    @tool
    def tool_b() -> str:
        """Tool B."""
        call_log.append("b")
        return "result_b"

    model = MagicMock()
    model.complete.side_effect = [
        ModelResponse(
            content="",
            tool_calls=[
                {"id": "1", "name": "tool_a", "input": {}},
                {"id": "2", "name": "tool_b", "input": {}},
            ],
        ),
        ModelResponse(content="done", tool_calls=[]),
    ]
    agent = Agent(model=model, tools=[tool_a, tool_b])
    result = agent.run("run both")
    assert result == "done"
    assert set(call_log) == {"a", "b"}
