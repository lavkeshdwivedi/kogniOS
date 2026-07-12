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


# --- HybridKnowledge tests ---


def _stub_embed(texts: list[str]) -> list[list[float]]:
    """Simple bag-of-chars embedding (dim=64) — no API key required."""
    dim = 64
    result = []
    for text in texts:
        vec = [0.0] * dim
        for ch in text:
            vec[ord(ch) % dim] += 1.0
        total = sum(vec) or 1.0
        result.append([v / total for v in vec])
    return result


@pytest.fixture
def hybrid_kb():
    pytest.importorskip("numpy")
    from kognios.knowledge.hybrid import HybridKnowledge
    from kognios.knowledge.numpy_vector import NumpyVectorKnowledge

    fts = SQLiteKnowledge(db_path=":memory:")
    vec = NumpyVectorKnowledge(embed_fn=_stub_embed)
    return HybridKnowledge(fts, vec)


def test_hybrid_load_and_search(hybrid_kb):
    hybrid_kb.load("The quick brown fox jumps over the lazy dog.")
    results = hybrid_kb.search("fox", top_k=3)
    assert len(results) >= 1
    assert any("fox" in r for r in results)


def test_hybrid_search_empty(hybrid_kb):
    results = hybrid_kb.search("python", top_k=5)
    assert results == []


def test_hybrid_rrf_deduplicates(hybrid_kb):
    hybrid_kb.load("Python is a programming language.")
    hybrid_kb.load("Python runs everywhere.")
    results = hybrid_kb.search("Python", top_k=5)
    # No duplicate chunks in output
    assert len(results) == len(set(results))


def test_hybrid_top_k_respected(hybrid_kb):
    for i in range(10):
        hybrid_kb.load(f"Fact {i}: kognios is a Python agent framework with feature_{i}.")
    results = hybrid_kb.search("kognios agent", top_k=3)
    assert len(results) <= 3
