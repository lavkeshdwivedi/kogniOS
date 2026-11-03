"""An action ledger: remember what a tool already did so a retry or a resumed run never repeats it.

Tools with side effects (send a message, post, charge) are unsafe to run twice. Wrap them with
``ledger.once`` and the second call with the same arguments returns the first result instead
of acting again. The ledger also keeps named watermarks, such as "last item processed", so a
scheduled run can pick up where the previous one stopped.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable

from .storage.sqlite import get_connection

# stored while an action is in flight; a key still holding it after a crash means the outcome is unknown
PENDING = "\x00pending"
UNKNOWN_OUTCOME = (
    "Error: an earlier attempt of this action never reported back, so it may already have run. "
    "Check the outside system before repeating it."
)


class ActionLedger:
    def __init__(self, db_path: str = "kognios.db"):
        self._conn = get_connection(db_path)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS action_ledger (
                key        TEXT PRIMARY KEY,
                result     TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS ledger_watermark (
                name  TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def key(action: str, args: dict) -> str:
        blob = json.dumps(args, sort_keys=True, default=str)
        return f"{action}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"

    def seen(self, key: str) -> str | None:
        row = self._conn.execute("SELECT result FROM action_ledger WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def record(self, key: str, result: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO action_ledger (key, result) VALUES (?, ?)", (key, result)
        )
        self._conn.commit()

    def begin(self, key: str) -> None:
        """Mark an action as started. If the process dies before ``record`` or ``forget``,
        ``seen`` returns ``PENDING`` so the action is not blindly repeated."""
        self.record(key, PENDING)

    def forget(self, key: str) -> None:
        """Clear a key so a failed action can be retried."""
        self._conn.execute("DELETE FROM action_ledger WHERE key = ?", (key,))
        self._conn.commit()

    def once(self, fn: Callable) -> Callable:
        """Run ``fn`` at most once per distinct set of arguments. A call that raised or returned
        an ``Error:`` string is cleared and can be retried. A call that never reported back
        (the process died mid-action) returns ``UNKNOWN_OUTCOME`` instead of running again."""

        @functools.wraps(fn)
        def wrapper(**kwargs):
            key = self.key(fn.__name__, kwargs)
            previous = self.seen(key)
            if previous == PENDING:
                return UNKNOWN_OUTCOME
            if previous is not None:
                return previous
            self.begin(key)
            try:
                result = str(fn(**kwargs))
            except BaseException:
                self.forget(key)
                raise
            if result.startswith("Error:"):
                self.forget(key)
            else:
                self.record(key, result)
            return result

        if hasattr(fn, "_tool_schema"):
            wrapper._tool_schema = fn._tool_schema
        return wrapper

    def watermark(self, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM ledger_watermark WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def advance(self, name: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ledger_watermark (name, value) VALUES (?, ?)", (name, value)
        )
        self._conn.commit()
