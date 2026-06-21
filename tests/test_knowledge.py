import pytest
from kognios.knowledge.sqlite_fts import SQLiteKnowledge


@pytest.fixture
def kb():
    return SQLiteKnowledge(db_path=":memory:")


def test_load_and_search(kb):
    kb.load("The quick brown fox jumps over the lazy dog.")
    results = kb.search("fox")
    assert len(results) >= 1
    assert "fox" in results[0]


def test_search_returns_top_k(kb):
    for i in range(10):
        kb.load(f"Document {i}: Python is a programming language used for topic_{i}.")
    results = kb.search("Python programming", top_k=3)
    assert len(results) <= 3


def test_load_multiline_text(kb):
    text = "AgentOS is a framework.\n" * 100
    kb.load(text)
    results = kb.search("AgentOS framework")
    assert len(results) >= 1


def test_search_no_results(kb):
    kb.load("The sky is blue.")
    results = kb.search("quantum entanglement")
    assert results == []


def test_fts5_available(kb):
    # If FTS5 was unavailable, __init__ would have raised. Getting here means it works.
    assert kb is not None
