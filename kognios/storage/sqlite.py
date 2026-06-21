from __future__ import annotations

import sqlite3

_connections: dict[str, sqlite3.Connection] = {}


def get_connection(db_path: str = "kognios.db") -> sqlite3.Connection:
    # Never cache :memory: connections — each caller gets its own isolated DB.
    if db_path == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("PRAGMA foreign_keys=ON")
        run_migrations(conn)
        return conn
    if db_path not in _connections:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        run_migrations(conn)
        _connections[db_path] = conn
    return _connections[db_path]


def run_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            key       TEXT PRIMARY KEY,
            value     TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks
            USING fts5(chunk, source UNINDEXED);
    """)
    # Add embedding column if it doesn't exist yet
    cols = [r[1] for r in conn.execute("PRAGMA table_info(long_term_memory)").fetchall()]
    if "embedding" not in cols:
        conn.execute("ALTER TABLE long_term_memory ADD COLUMN embedding BLOB")
    conn.commit()
