from __future__ import annotations

import json
from collections.abc import Callable

from ..storage.sqlite import get_connection


class LongTermMemory:
    def __init__(
        self,
        db_path: str = "kognios.db",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        self._conn = get_connection(db_path)
        self.embed_fn = embed_fn  # if set, embeddings are stored and semantic_recall works

    def store(self, key: str, value: str) -> None:
        embedding_blob: bytes | None = None
        if self.embed_fn is not None:
            vec = self.embed_fn([value])[0]
            embedding_blob = json.dumps(vec).encode()
        self._conn.execute(
            "INSERT INTO long_term_memory(key, value, updated_at, embedding)"
            " VALUES (?, ?, datetime('now'), ?)"
            " ON CONFLICT(key) DO UPDATE SET"
            " value=excluded.value, updated_at=excluded.updated_at, embedding=excluded.embedding",
            (key, value, embedding_blob),
        )
        self._conn.commit()

    def recall(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM long_term_memory WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def all_facts(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM long_term_memory").fetchall()
        return {k: v for k, v in rows}

    def semantic_recall(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """Return (key, value) pairs most semantically similar to query.

        Requires embed_fn to have been set and facts to have been stored with it.
        Falls back to returning all_facts() items if no embeddings are stored.
        """
        if self.embed_fn is None:
            raise RuntimeError("semantic_recall requires embed_fn to be set on LongTermMemory")

        rows = self._conn.execute(
            "SELECT key, value, embedding FROM long_term_memory WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            # No embeddings stored yet — fall back to keyword LIKE search on the key column
            first_token = query.split()[0] if query.split() else ""
            like_rows = self._conn.execute(
                "SELECT key, value FROM long_term_memory WHERE key LIKE ? LIMIT ?",
                (f"%{first_token}%", top_k),
            ).fetchall()
            return [(k, v) for k, v in like_rows]

        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy is required for semantic_recall: pip install 'kognios[vector]'"
            )

        keys = [r[0] for r in rows]
        values = [r[1] for r in rows]
        embeddings = [json.loads(r[2]) for r in rows]

        matrix = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix = matrix / norms

        q_vec = np.array(self.embed_fn([query])[0], dtype=float)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(keys[i], values[i]) for i in top_idx]

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM long_term_memory WHERE key = ?", (key,))
        self._conn.commit()

    def remember(self, key: str, value: str) -> None:
        """Alias for store()."""
        self.store(key, value)

    def forget(self, key: str) -> None:
        """Alias for delete()."""
        self.delete(key)
