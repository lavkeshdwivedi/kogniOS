"""An action ledger: remember what a tool already did so a retry or a resumed run never repeats it.

Tools with side effects (send a message, post, charge) are unsafe to run twice. Wrap them with
``ledger.once`` and the second call with the same arguments returns the first result instead
of acting again. The ledger also keeps named watermarks, such as "last item processed", so a
scheduled run can pick up where the previous one stopped.

An action is claimed atomically before it runs, so two threads or two parallel tool calls with
the same arguments cannot both act: the second waits for the first and gets its result.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import sqlite3
import threading
from collections.abc import Callable

# stored while an action is in flight; a key still holding it with no live claimant in this
# process means the process died (or the action timed out) and the outcome is unknown
PENDING = "\x00pending"
UNKNOWN_OUTCOME = (
    "Error: an earlier attempt of this action never reported back, so it may already have run. "
    "Check the outside system before repeating it."
)

RUN, DONE, UNKNOWN = "run", "done", "unknown"


class ActionLedger:
    def __init__(self, db_path: str = "kognios.db"):
        # a private connection behind a lock: the shared one is not safe for parallel tool calls
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        with self._lock:
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
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM action_ledger WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def record(self, key: str, result: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO action_ledger (key, result) VALUES (?, ?)", (key, result)
            )
            self._conn.commit()

    def acquire(self, key: str) -> tuple[str, str | None]:
        """Claim an action. Returns ``("run", None)`` when the caller should do it now,
        ``("done", result)`` when it already ran, or ``("unknown", None)`` when an earlier
        attempt never reported back. If another thread is running the same action right now,
        this waits for it and then returns its result. Blocking, so async callers should run
        it in a worker thread."""
        while True:
            with self._lock:
                prior = self.seen(key)
                if prior is None:
                    self.record(key, PENDING)
                    self._inflight[key] = threading.Event()
                    return RUN, None
                if prior != PENDING:
                    return DONE, prior
                event = self._inflight.get(key)
                if event is None:
                    return UNKNOWN, None
            event.wait()

    def _release(self, key: str, result: str | None) -> None:
        with self._lock:
            if result is None:
                self._conn.execute("DELETE FROM action_ledger WHERE key = ?", (key,))
                self._conn.commit()
            elif result != PENDING:
                self.record(key, result)
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def finish(self, key: str, result: str) -> None:
        """The action ran: store its result and wake anyone waiting on it."""
        self._release(key, result)

    def release(self, key: str) -> None:
        """The action failed in a way that is safe to retry: clear the claim."""
        self._release(key, None)

    def hold(self, key: str) -> None:
        """The outcome is unknown (timeout, interrupt): keep the in-flight marker so the
        action is not blindly repeated, and wake waiters, who will be told it is unknown."""
        self._release(key, PENDING)

    def once(self, fn: Callable) -> Callable:
        """Run ``fn`` at most once per distinct set of arguments, even across threads.

        A call that raised or returned an ``Error:`` string is cleared and can be retried. A
        call interrupted mid-action (``KeyboardInterrupt``, process death) or that never
        reported back returns ``UNKNOWN_OUTCOME`` next time instead of running again.
        Positional and keyword spellings of the same call share one key."""
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = self.key(fn.__name__, dict(bound.arguments))
            status, prior = self.acquire(key)
            if status == DONE:
                return prior
            if status == UNKNOWN:
                return UNKNOWN_OUTCOME
            try:
                result = str(fn(*bound.args, **bound.kwargs))
            except Exception:
                self.release(key)
                raise
            except BaseException:
                self.hold(key)
                raise
            if result.startswith("Error:"):
                self.release(key)
            else:
                self.finish(key, result)
            return result

        if hasattr(fn, "_tool_schema"):
            wrapper._tool_schema = fn._tool_schema
        return wrapper

    def watermark(self, name: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM ledger_watermark WHERE name = ?", (name,)
            ).fetchone()
        return row[0] if row else None

    def advance(self, name: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_watermark (name, value) VALUES (?, ?)",
                (name, value),
            )
            self._conn.commit()
