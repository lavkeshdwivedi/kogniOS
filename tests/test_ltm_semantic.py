"""Tests for LongTermMemory semantic search — uses stub embeddings, no real API calls."""

from __future__ import annotations

import pytest


def _stub_embed(texts: list[str]) -> list[list[float]]:
    """One-hot by first character ordinal, dim=128."""
    dim = 128
    result = []
    for text in texts:
        vec = [0.0] * dim
        if text:
            vec[ord(text[0]) % dim] = 1.0
        result.append(vec)
    return result


def test_semantic_recall_returns_relevant_facts():
    pytest.importorskip("numpy")
    from kognios.memory.long_term import LongTermMemory

    ltm = LongTermMemory(db_path=":memory:", embed_fn=_stub_embed)
    ltm.store("greeting", "hello world")
    ltm.store("farewell", "goodbye everyone")
    ltm.store("question", "what is the weather?")

    results = ltm.semantic_recall("hello", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "greeting"


def test_semantic_recall_without_embed_fn_raises():
    from kognios.memory.long_term import LongTermMemory

    ltm = LongTermMemory(db_path=":memory:")
    ltm.store("key", "value")
    with pytest.raises(RuntimeError, match="embed_fn"):
        ltm.semantic_recall("query")


def test_semantic_recall_empty_db():
    pytest.importorskip("numpy")
    from kognios.memory.long_term import LongTermMemory

    ltm = LongTermMemory(db_path=":memory:", embed_fn=_stub_embed)
    results = ltm.semantic_recall("query", top_k=5)
    assert results == []


def test_store_without_embed_fn_does_not_crash():
    from kognios.memory.long_term import LongTermMemory

    ltm = LongTermMemory(db_path=":memory:")
    ltm.store("key", "value")  # should not raise
    assert ltm.recall("key") == "value"


def test_all_facts_still_works_with_embed_fn():
    pytest.importorskip("numpy")
    from kognios.memory.long_term import LongTermMemory

    ltm = LongTermMemory(db_path=":memory:", embed_fn=_stub_embed)
    ltm.store("a", "apple")
    ltm.store("b", "banana")
    facts = ltm.all_facts()
    assert facts == {"a": "apple", "b": "banana"}
