"""
Integration test examples — skipped by default, require real API keys.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

from __future__ import annotations

import os
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_anthropic_basic_run():
    """Real API call to Anthropic — requires ANTHROPIC_API_KEY."""
    from kognios import Agent, AnthropicModel

    agent = Agent(model=AnthropicModel(model="claude-haiku-4-5-20251001"))
    result = agent.run("Reply with exactly: hello")
    assert "hello" in result.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_openai_basic_run():
    """Real API call to OpenAI — requires OPENAI_API_KEY."""
    from kognios import Agent, OpenAIModel

    agent = Agent(model=OpenAIModel(model="gpt-4o-mini"))
    result = agent.run("Reply with exactly: hello")
    assert "hello" in result.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_numpy_vector_knowledge_with_real_embeddings():
    """Real embedding API call — requires OPENAI_API_KEY."""
    from kognios.knowledge.numpy_vector import NumpyVectorKnowledge

    kb = NumpyVectorKnowledge()  # uses default OpenAI embed_fn
    kb.load("The capital of France is Paris.")
    results = kb.search("What is the capital of France?", top_k=1)
    assert len(results) == 1
    assert "Paris" in results[0]
