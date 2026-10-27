import threading

import pytest

from kognios import Agent, Team, tool
from kognios.checkpoint import ApprovalRequired, MemoryCheckpointStore, SQLiteCheckpointStore
from kognios.models.base import ModelResponse
from tests.test_async_checkpoint import ScriptedModel, call

PLAN = "1. gather\n2. write"


def text(s):
    return ModelResponse(content=s)


def make_agent(script, store, executed=None, approval=None):
    executed = executed if executed is not None else []

    @tool
    def danger(x: str) -> str:
        """Does something that needs review."""
        executed.append(x)
        return f"did {x}"

    model = ScriptedModel(script)
    agent = Agent(
        model,
        tools=[danger],
        checkpointer=store,
        require_approval=approval,
        max_iterations=5,
    )
    return agent, model, executed


# ── plan_and_run ────────────────────────────────────────────────────────────


def test_plan_run_resumes_after_crash_without_redoing_finished_steps():
    store = MemoryCheckpointStore()
    agent, model, _ = make_agent([text(PLAN), text("gathered"), RuntimeError("boom")], store)
    with pytest.raises(RuntimeError):
        agent.plan_and_run("report", run_id="p1")
    assert model.calls == 3

    model.script = [text("written"), text("final answer")]
    model.calls, model.seen_messages = 0, []
    assert agent.plan_and_run("report", run_id="p1") == "final answer"
    assert model.calls == 2  # no new plan, step 1 not redone
    assert "Step 1 result: gathered" in model.seen_messages[0][0]["content"]


def test_finished_plan_run_replays_without_model_calls():
    store = MemoryCheckpointStore()
    agent, model, _ = make_agent([text(PLAN), text("a"), text("b"), text("done")], store)
    assert agent.plan_and_run("report", run_id="p2") == "done"
    model.script, model.calls = [], 0
    assert agent.plan_and_run("report", run_id="p2") == "done"
    assert model.calls == 0


def test_plan_run_without_run_id_is_unchanged_and_unsaved():
    store = MemoryCheckpointStore()
    agent, model, _ = make_agent([text(PLAN), text("a"), text("b"), text("done")], store)
    assert agent.plan_and_run("report") == "done"
    assert store.load("anything") is None and store._runs == {}


def test_plan_run_id_needs_a_checkpointer():
    agent, _, _ = make_agent([], None)
    with pytest.raises(ValueError):
        agent.plan_and_run("report", run_id="p3")


def test_plan_run_pauses_for_approval_and_resumes_with_decisions():
    store = MemoryCheckpointStore()
    script = [text(PLAN), text("gathered"), call("c1", "danger", x="ship"), text("written")]
    agent, model, executed = make_agent(script, store, approval={"danger"})
    with pytest.raises(ApprovalRequired) as e:
        agent.plan_and_run("report", run_id="p4")
    assert e.value.run_id == "p4/step-2" and executed == []

    model.script = [text("written"), text("final")]  # step 2 continues after its tool ran
    out = agent.plan_and_run("report", run_id="p4", decisions={"c1": True})
    assert out == "final" and executed == ["ship"]


def test_plan_run_survives_a_process_restart_on_sqlite(tmp_path):
    db = str(tmp_path / "k.db")
    first, model, _ = make_agent(
        [text(PLAN), text("a"), RuntimeError("boom")], SQLiteCheckpointStore(db)
    )
    with pytest.raises(RuntimeError):
        first.plan_and_run("report", run_id="p5")
    second, model2, _ = make_agent([text("b"), text("end")], SQLiteCheckpointStore(db))
    assert second.plan_and_run("report", run_id="p5") == "end"
    assert model2.calls == 2


async def test_aplan_run_resumes_after_crash():
    store = MemoryCheckpointStore()
    agent, model, _ = make_agent([text(PLAN), text("gathered"), RuntimeError("boom")], store)
    with pytest.raises(RuntimeError):
        await agent.aplan_and_run("report", run_id="a1")
    model.script, model.calls = [text("written"), text("final")], 0
    assert await agent.aplan_and_run("report", run_id="a1") == "final"
    assert model.calls == 2


async def test_aplan_run_approval_roundtrip():
    store = MemoryCheckpointStore()
    script = [text(PLAN), text("g"), call("c9", "danger", x="go"), text("w")]
    agent, model, executed = make_agent(script, store, approval={"danger"})
    with pytest.raises(ApprovalRequired):
        await agent.aplan_and_run("report", run_id="a2")
    model.script = [text("written"), text("final")]  # step 2 continues after its tool ran
    assert await agent.aplan_and_run("report", run_id="a2", decisions={"c9": True}) == "final"
    assert executed == ["go"]


# ── Team ────────────────────────────────────────────────────────────────────


def make_team(store, router_script, a_script, b_script, approval_b=None):
    router = ScriptedModel(router_script)
    a, am, _ = make_agent(a_script, store)
    b, bm, executed = make_agent(b_script, store, approval=approval_b)
    a.description, b.description = "does a", "does b"
    return Team(router, {"a": a, "b": b}, checkpointer=store), router, am, bm, executed


ROUTE_B = text('{"agent": "b", "message": "task for b"}')


def test_team_run_does_not_reroute_after_crash():
    store = MemoryCheckpointStore()
    team, router, _, bm, _ = make_team(store, [ROUTE_B], [], [RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        team.run("hello", run_id="t1")
    bm.script = [text("b done")]
    assert team.run("hello", run_id="t1") == "b done"
    assert router.calls == 1


def test_team_run_approval_resume():
    store = MemoryCheckpointStore()
    team, router, _, bm, executed = make_team(
        store, [ROUTE_B], [], [call("k1", "danger", x="q")], approval_b={"danger"}
    )
    with pytest.raises(ApprovalRequired):
        team.run("hello", run_id="t2")
    bm.script = [text("ok")]
    assert team.resume("t2", {"k1": True}) == "ok"
    assert executed == ["q"] and router.calls == 1


def test_team_resume_unknown_run_raises():
    team, *_ = make_team(MemoryCheckpointStore(), [], [], [])
    with pytest.raises(KeyError):
        team.resume("nope", {})


def test_team_run_id_needs_team_checkpointer():
    a, _, _ = make_agent([], MemoryCheckpointStore())
    team = Team(ScriptedModel([]), {"a": a})
    with pytest.raises(ValueError):
        team.run("hi", run_id="x")


def test_team_pipeline_skips_finished_stages_on_resume():
    store = MemoryCheckpointStore()
    team, _, am, bm, _ = make_team(store, [], [text("from a")], [RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        team.pipeline("start", run_id="pl")
    bm.script = [text("from b")]
    assert team.pipeline("start", run_id="pl") == "from b"
    assert am.calls == 1  # stage a finished before the crash


def test_team_broadcast_reruns_only_the_crashed_agent():
    store = MemoryCheckpointStore()
    team, _, am, bm, _ = make_team(store, [], [text("A")], [RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        team.broadcast("go", run_id="bc")
    bm.script = [text("B")]
    assert team.broadcast("go", run_id="bc") == {"a": "A", "b": "B"}
    assert am.calls == 1


async def test_team_arun_and_apipeline_resume():
    store = MemoryCheckpointStore()
    team, router, am, bm, _ = make_team(store, [ROUTE_B], [text("from a")], [RuntimeError("x")])
    with pytest.raises(RuntimeError):
        await team.arun("hello", run_id="ta")
    bm.script = [text("ok")]
    assert await team.arun("hello", run_id="ta") == "ok" and router.calls == 1
    bm.script = [RuntimeError("y")]
    with pytest.raises(RuntimeError):
        await team.apipeline("s", run_id="tp")
    bm.script = [text("end")]
    assert await team.apipeline("s", run_id="tp") == "end" and am.calls == 1


# ── async saves stay off the event loop ─────────────────────────────────────


class ThreadSpyStore(MemoryCheckpointStore):
    def __init__(self):
        super().__init__()
        self.threads = set()

    def save(self, cp):
        self.threads.add(threading.get_ident())
        super().save(cp)


async def test_async_checkpoint_saves_run_off_the_event_loop():
    store = ThreadSpyStore()
    agent, _, _ = make_agent([text("hi")], store)
    await agent.arun("hello", run_id="spy")
    assert threading.get_ident() not in store.threads and store.threads


def test_sync_checkpoint_saves_stay_on_the_caller_thread():
    store = ThreadSpyStore()
    agent, _, _ = make_agent([text("hi")], store)
    agent.run("hello", run_id="spy2")
    assert store.threads == {threading.get_ident()}
