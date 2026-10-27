"""Durable run state for agents: save after every step, pause for approval, resume later.

A checkpoint is the full message history of a run plus where the run stopped. Because it
is saved after every step, a crashed or paused run can pick up from its last step instead
of starting over, and every earlier step stays readable for debugging (time travel).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from .storage.sqlite import get_connection

RUNNING = "running"
INTERRUPTED = "interrupted"
DONE = "done"


@dataclass
class Checkpoint:
    run_id: str
    step: int
    status: str
    message: str
    messages: list[dict]
    # tool calls from the last model turn that have not run yet (set while interrupted)
    pending_calls: list[dict] = field(default_factory=list)
    # ids of the pending calls that need a reviewer's decision before they can run
    awaiting: list[str] = field(default_factory=list)
    tool_error_counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    iterations_used: int = 0
    result: str | None = None
    # "run" is an agent loop. "plan", "route" and "stage" are records written by
    # plan_and_run and Team so a multi-agent flow can resume without redoing finished work
    kind: str = "run"
    plan: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ApprovalRequired(Exception):
    """Raised when a run reaches a tool call that needs a reviewer's decision.

    The run is already saved. Call ``agent.resume(run_id, decisions)`` with a decision for
    each id in ``awaiting``: ``True`` to approve, ``False`` to reject, or a dict such as
    ``{"approve": True, "input": {...}}`` to approve with edited arguments.
    """

    def __init__(self, run_id: str, pending: list[dict]):
        self.run_id = run_id
        self.pending = pending
        names = ", ".join(tc["name"] for tc in pending)
        super().__init__(f"Run '{run_id}' is waiting for approval of: {names}")


class CheckpointStore(Protocol):
    def save(self, cp: Checkpoint) -> None: ...

    def load(self, run_id: str) -> Checkpoint | None: ...

    def history(self, run_id: str) -> list[Checkpoint]: ...


class MemoryCheckpointStore:
    """In-process store, good for tests. Nothing survives a restart."""

    def __init__(self) -> None:
        self._runs: dict[str, list[Checkpoint]] = {}

    def save(self, cp: Checkpoint) -> None:
        # deep copy through JSON so later mutation of the live run cannot rewrite history
        self._runs.setdefault(cp.run_id, []).append(
            Checkpoint(**json.loads(json.dumps(asdict(cp))))
        )

    def load(self, run_id: str) -> Checkpoint | None:
        steps = self._runs.get(run_id)
        return steps[-1] if steps else None

    def history(self, run_id: str) -> list[Checkpoint]:
        return list(self._runs.get(run_id, []))


class SQLiteCheckpointStore:
    """Durable store on the same SQLite file the memory and knowledge layers use."""

    def __init__(self, db_path: str = "kognios.db") -> None:
        self._conn = get_connection(db_path)

    def save(self, cp: Checkpoint) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_checkpoints (run_id, seq, status, state, created_at) "
            "VALUES (?, COALESCE((SELECT MAX(seq) FROM agent_checkpoints WHERE run_id = ?), 0) + 1, ?, ?, ?)",
            (cp.run_id, cp.run_id, cp.status, json.dumps(asdict(cp)), cp.created_at),
        )
        self._conn.commit()

    def load(self, run_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT state FROM agent_checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return Checkpoint(**json.loads(row[0])) if row else None

    def history(self, run_id: str) -> list[Checkpoint]:
        rows = self._conn.execute(
            "SELECT state FROM agent_checkpoints WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [Checkpoint(**json.loads(r[0])) for r in rows]


def save_record(
    store: CheckpointStore,
    run_id: str,
    kind: str,
    message: str,
    result: str | None = None,
    plan: str | None = None,
) -> None:
    """Persist a finished bookkeeping record (a plan, a routing decision, a plan_and_run result)."""
    store.save(
        Checkpoint(
            run_id=run_id,
            step=0,
            status=DONE,
            message=message,
            messages=[],
            result=result,
            kind=kind,
            plan=plan,
        )
    )


def load_record(store: CheckpointStore, run_id: str) -> Checkpoint | None:
    """Latest saved record for run_id, or None."""
    return store.load(run_id)
