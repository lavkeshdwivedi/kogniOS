import pytest
from unittest.mock import MagicMock
from kognios.agent import Agent
from kognios.models.base import ModelChunk, ModelResponse


def make_stream_model(chunks: list[ModelChunk]):
    model = MagicMock()
    model.stream.return_value = iter(chunks)
    return model


def test_agent_stream_yields_text():
    model = make_stream_model(
        [
            ModelChunk(text="Hello"),
            ModelChunk(text=" world"),
            ModelChunk(final=True),
        ]
    )
    agent = Agent(model=model)
    result = "".join(agent.stream("hi"))
    assert result == "Hello world"


def test_agent_stream_tool_then_answer():
    from kognios.tools.registry import tool

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hi, {name}!"

    model = MagicMock()
    model.stream.side_effect = [
        # First call: tool call
        iter(
            [
                ModelChunk(
                    tool_calls=[{"id": "1", "name": "greet", "input": {"name": "Alice"}}],
                    final=True,
                )
            ]
        ),
        # Second call: final answer
        iter([ModelChunk(text="I greeted Alice."), ModelChunk(final=True)]),
    ]
    agent = Agent(model=model, tools=[greet])
    result = "".join(agent.stream("greet Alice"))
    assert result == "I greeted Alice."
    assert model.stream.call_count == 2


@pytest.mark.asyncio
async def test_agent_arun():
    async def async_complete(*args, **kwargs):
        return ModelResponse(content="async answer")

    model = MagicMock()
    model.acomplete = async_complete
    agent = Agent(model=model)
    result = await agent.arun("question")
    assert result == "async answer"
