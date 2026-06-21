from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import KnowledgeBase
from .loaders import load_text as _load_text
from .sqlite_fts import _chunk


class NumpyVectorKnowledge(KnowledgeBase):
    """Semantic search via numpy cosine similarity + configurable embedding function.

    Requires: pip install 'kognios[vector]'
    """

    def __init__(self, embed_fn: Callable[[list[str]], list[list[float]]] | None = None):
        """
        Args:
            embed_fn: callable that takes a list of strings and returns a list of embedding
                      vectors. Defaults to OpenAI text-embedding-3-small.
        """
        try:
            import numpy as np

            self._np = np
        except ImportError:
            raise ImportError("numpy is required: pip install 'kognios[vector]'")

        self._embed_fn = embed_fn or _default_openai_embed
        self._chunks: list[str] = []
        self._embeddings: list[list[float]] | None = None  # lazy, built on first search

    def load(self, source: str | Path, source_type: str = "text") -> int:
        text = _load_text(source, source_type)
        new_chunks = _chunk(text)
        self._chunks.extend(new_chunks)
        self._embeddings = None  # invalidate cache
        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        if not self._chunks:
            return []
        np = self._np
        # Build embedding matrix lazily
        if self._embeddings is None:
            self._embeddings = self._embed_fn(self._chunks)
        matrix = np.array(self._embeddings, dtype=float)
        # Normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix = matrix / norms
        # Embed query
        q_vec = np.array(self._embed_fn([query])[0], dtype=float)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        # Cosine similarity
        scores = matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self._chunks[i] for i in top_idx]


def _default_openai_embed(texts: list[str]) -> list[list[float]]:
    try:
        import openai
    except ImportError:
        raise ImportError("openai is required for default embeddings: pip install openai")
    client = openai.OpenAI()
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]
