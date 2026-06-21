from __future__ import annotations

from pathlib import Path

from .base import KnowledgeBase
from .loaders import load_text as _load_text_source
from ..storage.sqlite import get_connection

_CHUNK_TOKENS = 512
_OVERLAP_TOKENS = 64
_CHARS_PER_TOKEN = 4


def _chunk(text: str) -> list[str]:
    chunk_size = _CHUNK_TOKENS * _CHARS_PER_TOKEN
    step = (_CHUNK_TOKENS - _OVERLAP_TOKENS) * _CHARS_PER_TOKEN
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


class SQLiteKnowledge(KnowledgeBase):
    def __init__(self, db_path: str = "kognios.db"):
        self._conn = get_connection(db_path)
        self._check_fts5()

    def _check_fts5(self) -> None:
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        except Exception:
            raise RuntimeError(
                "SQLite FTS5 is not available in this Python build. "
                "Install a Python distribution compiled with FTS5 support."
            )

    def load(self, source: str | Path, source_type: str = "text") -> int:
        text = self._fetch(source, source_type)
        chunks = _chunk(text)
        src_label = str(source)[:200]
        self._conn.executemany(
            "INSERT INTO knowledge_chunks(chunk, source) VALUES (?, ?)",
            [(c, src_label) for c in chunks],
        )
        self._conn.commit()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        safe_query = query.replace('"', '""')
        rows = self._conn.execute(
            "SELECT chunk FROM knowledge_chunks"
            " WHERE knowledge_chunks MATCH ?"
            " ORDER BY rank LIMIT ?",
            (safe_query, top_k),
        ).fetchall()
        return [r[0] for r in rows]

    def _fetch(self, source: str | Path, source_type: str) -> str:
        return _load_text_source(source, source_type)
