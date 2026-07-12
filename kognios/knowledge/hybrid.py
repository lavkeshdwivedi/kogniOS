from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .base import KnowledgeBase


class HybridKnowledge(KnowledgeBase):
    """Combines FTS5 and vector search via Reciprocal Rank Fusion (RRF).

    Documents must be loaded via this class — it loads into both backends simultaneously.
    No extra dependencies beyond ``kognios[vector]`` for the vector backend.

    Example::

        from kognios import SQLiteKnowledge, NumpyVectorKnowledge, HybridKnowledge
        fts = SQLiteKnowledge()
        vec = NumpyVectorKnowledge()
        kb = HybridKnowledge(fts, vec)
        kb.load("my_doc.pdf")
        results = kb.search("agent frameworks", top_k=5)
    """

    def __init__(
        self,
        fts_kb: KnowledgeBase,
        vector_kb: KnowledgeBase,
        k: int = 60,
    ):
        self._fts = fts_kb
        self._vec = vector_kb
        self._k = k

    def load(self, source: str | Path, source_type: str = "text") -> int:
        count = self._fts.load(source, source_type)
        self._vec.load(source, source_type)
        return count

    def search(self, query: str, top_k: int = 5) -> list[str]:
        candidate_n = top_k * 4
        fts_results = self._fts.search(query, top_k=candidate_n)
        vec_results = self._vec.search(query, top_k=candidate_n)

        scores: dict[str, float] = defaultdict(float)
        for rank, chunk in enumerate(fts_results):
            scores[chunk] += 1.0 / (self._k + rank + 1)
        for rank, chunk in enumerate(vec_results):
            scores[chunk] += 1.0 / (self._k + rank + 1)

        ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
        return ranked[:top_k]
