"""Tests for NumpyVectorKnowledge — uses a stub embed_fn, no real API calls."""

from __future__ import annotations

import pytest


def _stub_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings: one-hot by first character ordinal."""
    dim = 128
    result = []
    for text in texts:
        vec = [0.0] * dim
        if text:
            vec[ord(text[0]) % dim] = 1.0
        result.append(vec)
    return result


def test_numpy_vector_search_returns_chunks():
    pytest.importorskip("numpy")
    from kognios.knowledge.numpy_vector import NumpyVectorKnowledge

    kb = NumpyVectorKnowledge(embed_fn=_stub_embed)
    kb.load("alpha beta gamma")
    results = kb.search("alpha", top_k=1)
    assert len(results) == 1


def test_numpy_vector_empty_search():
    pytest.importorskip("numpy")
    from kognios.knowledge.numpy_vector import NumpyVectorKnowledge

    kb = NumpyVectorKnowledge(embed_fn=_stub_embed)
    results = kb.search("anything", top_k=5)
    assert results == []


def test_numpy_vector_multiple_loads():
    pytest.importorskip("numpy")
    from kognios.knowledge.numpy_vector import NumpyVectorKnowledge

    kb = NumpyVectorKnowledge(embed_fn=_stub_embed)
    kb.load("first document content here")
    kb.load("second document content here")
    results = kb.search("document", top_k=5)
    assert len(results) >= 1
