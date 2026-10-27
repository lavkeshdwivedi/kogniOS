import pytest

from kognios import Agent, tool
from kognios.checkpoint import (
    ApprovalRequired,
    MemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from kognios.models.base import BaseModel, ModelResponse


class ScriptedModel(BaseModel):
    """Replays fixed responses. An Exception in the script is raised, to simulate a crash."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen_messages = []

    def complete(self, messages, tools=None, system="", **kw):
        self.calls += 1
        self.seen_messages.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(id_, name, **args):
    return ModelResponse(content="", tool_calls=[{"id": id_, "name": name, "input": args}])


def make_tools():
    executed = []

    @tool
    def add(x: int, y: int) -> int:
        """Add numbers."""
        executed.append(("add", x, y))
        return x + y

    @tool
    def delete_item(item_id: str) -> str:
        """Delete an item."""
        executed.append(("delete_item", item_id))
        return f"deleted {item_id}"

    return add, delete_item, executed


def test_every_step_is_saved_and_history_is_readable():
    add, _, _ = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("1", "add", x=2, y=3), ModelResponse(content="5")])
    agent = Agent(model=model, tools=[add], checkpointer=store)

    assert agent.run("2+3?", run_id="r1") == "5"

    steps = store.history("r1")
    assert [s.status for s in steps] == ["running", "running", "done"]
    assert [s.step for s in steps] == [1, 2, 3]
    # earlier steps keep their own message history (time travel)
    assert len(steps[0].messages) < len(steps[1].messages)
    assert steps[-1].result == "5"


def test_finished_run_returns_saved_result_without_calling_the_model():
    store = MemoryCheckpointStore()
    model = ScriptedModel([ModelResponse(content="done")])
    agent = Agent(model=model, checkpointer=store)

    assert agent.run("hi", run_id="r1") == "done"
    assert agent.run("hi again", run_id="r1") == "done"
    assert model.calls == 1


def test_crashed_run_resumes_from_last_step_and_never_repeats_a_tool():
    add, _, executed = make_tools()
    store = MemoryCheckpointStore()
    crashing = ScriptedModel([call("1", "add", x=1, y=2), RuntimeError("provider down")])
    first = Agent(model=crashing, tools=[add], checkpointer=store)

    with pytest.raises(RuntimeError, match="provider down"):
        first.run("1+2?", run_id="r1")
    assert executed == [("add", 1, 2)]

    # a fresh process: new agent, same store, same run id
    healthy = ScriptedModel([ModelResponse(content="3")])
    second = Agent(model=healthy, tools=[add], checkpointer=store)
    assert second.run("1+2?", run_id="r1") == "3"

    assert executed == [("add", 1, 2)]  # the tool did not run a second time
    assert healthy.calls == 1  # only the missing step was paid for


def test_approval_gate_pauses_before_the_tool_runs():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("c1", "delete_item", item_id="s-9"), ModelResponse(content="ok")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )

    with pytest.raises(ApprovalRequired) as info:
        agent.run("remove s-9", run_id="r1")

    assert info.value.run_id == "r1"
    assert [tc["name"] for tc in info.value.pending] == ["delete_item"]
    assert executed == []
    saved = store.load("r1")
    assert saved.status == "interrupted"
    assert saved.awaiting == ["c1"]


def test_running_an_interrupted_run_again_asks_again_and_still_runs_nothing():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("c1", "delete_item", item_id="s-9")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        agent.run("remove s-9", run_id="r1")
    with pytest.raises(ApprovalRequired):
        agent.run("remove s-9", run_id="r1")

    assert executed == []
    assert model.calls == 1


def test_approve_runs_the_tool_once_and_finishes():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel(
        [call("c1", "delete_item", item_id="s-9"), ModelResponse(content="removed")]
    )
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        agent.run("remove s-9", run_id="r1")

    assert agent.resume("r1", {"c1": True}) == "removed"
    assert executed == [("delete_item", "s-9")]
    assert store.load("r1").status == "done"


def test_reject_skips_the_tool_and_tells_the_model():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel(
        [call("c1", "delete_item", item_id="s-9"), ModelResponse(content="understood")]
    )
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        agent.run("remove s-9", run_id="r1")

    result = agent.resume("r1", {"c1": {"approve": False, "reason": "still in use"}})

    assert result == "understood"
    assert executed == []
    tool_result = model.seen_messages[-1][-1]["content"][0]["content"]
    assert "rejected by the reviewer" in tool_result
    assert "still in use" in tool_result


def test_approve_with_edited_arguments_runs_the_edit_and_the_model_sees_it():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("c1", "delete_item", item_id="s-9"), ModelResponse(content="done")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        agent.run("remove s-9", run_id="r1")

    agent.resume("r1", {"c1": {"approve": True, "input": {"item_id": "s-10"}}})

    assert executed == [("delete_item", "s-10")]
    assistant_turn = model.seen_messages[-1][-2]["content"]
    assert assistant_turn[0]["input"] == {"item_id": "s-10"}


def test_only_gated_tools_pause_and_the_rest_wait_with_them():
    add, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    two_calls = ModelResponse(
        content="",
        tool_calls=[
            {"id": "a", "name": "add", "input": {"x": 1, "y": 1}},
            {"id": "d", "name": "delete_item", "input": {"item_id": "s-1"}},
        ],
    )
    model = ScriptedModel([two_calls, ModelResponse(content="all done")])
    agent = Agent(
        model=model,
        tools=[add, delete_item],
        checkpointer=store,
        require_approval={"delete_item"},
    )
    with pytest.raises(ApprovalRequired) as info:
        agent.run("do both", run_id="r1")

    assert [tc["id"] for tc in info.value.pending] == ["d"]
    assert executed == []  # the safe call is held too, so a resume cannot double-run it

    assert agent.resume("r1", {"d": True}) == "all done"
    assert sorted(executed) == [("add", 1, 1), ("delete_item", "s-1")]


def test_sqlite_store_survives_a_new_connection(tmp_path):
    db = str(tmp_path / "runs.db")
    add, _, _ = make_tools()
    model = ScriptedModel([call("1", "add", x=1, y=1), ModelResponse(content="2")])
    Agent(model=model, tools=[add], checkpointer=SQLiteCheckpointStore(db)).run(
        "1+1?", run_id="durable"
    )

    reopened = SQLiteCheckpointStore(db)
    steps = reopened.history("durable")
    assert [s.status for s in steps] == ["running", "running", "done"]
    assert reopened.load("durable").result == "2"
    assert reopened.load("missing") is None


def test_misuse_fails_loudly():
    model = ScriptedModel([])
    with pytest.raises(ValueError, match="needs a checkpointer"):
        Agent(model=model, require_approval={"delete_item"})
    with pytest.raises(ValueError, match="checkpointer"):
        Agent(model=model).run("hi", run_id="r1")
    gated = Agent(model=model, checkpointer=MemoryCheckpointStore(), require_approval={"x"})
    with pytest.raises(ValueError, match="needs a run_id"):
        gated.run("hi")
    with pytest.raises(KeyError):
        gated.resume("never-started")


def test_runs_without_a_run_id_behave_exactly_as_before():
    add, _, executed = make_tools()
    model = ScriptedModel([call("1", "add", x=4, y=5), ModelResponse(content="9")])
    assert Agent(model=model, tools=[add]).run("4+5?") == "9"
    assert executed == [("add", 4, 5)]
