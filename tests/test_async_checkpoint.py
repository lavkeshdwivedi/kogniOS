import pytest

from kognios import Agent, tool
from kognios.checkpoint import ApprovalRequired, MemoryCheckpointStore, SQLiteCheckpointStore
from kognios.models.base import BaseModel, ModelChunk, ModelResponse


class ScriptedModel(BaseModel):
    """Replays fixed turns. An Exception in the script is raised, to simulate a crash.

    A turn is a ModelResponse for acomplete, or a list of ModelChunk for astream.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen_messages = []

    def complete(self, messages, tools=None, system="", **kw):
        return self._next(messages)

    async def acomplete(self, messages, tools=None, system=""):
        return self._next(messages)

    async def astream(self, messages, tools=None, system=""):
        for chunk in self._next(messages):
            yield chunk

    def _next(self, messages):
        self.calls += 1
        self.seen_messages.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(id_, name, **args):
    return ModelResponse(content="", tool_calls=[{"id": id_, "name": name, "input": args}])


def calls(*specs):
    return ModelResponse(
        content="",
        tool_calls=[{"id": i, "name": n, "input": a} for i, n, a in specs],
    )


def stream_call(id_, name, **args):
    return [ModelChunk(final=True, tool_calls=[{"id": id_, "name": name, "input": args}])]


def stream_text(*parts):
    return [ModelChunk(text=p) for p in parts] + [ModelChunk(final=True)]


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


async def collect(gen):
    return [event async for event in gen]


# arun


@pytest.mark.asyncio
async def test_arun_saves_every_step():
    add, _, _ = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("1", "add", x=2, y=3), ModelResponse(content="5")])
    agent = Agent(model=model, tools=[add], checkpointer=store)

    assert await agent.arun("2+3?", run_id="r1") == "5"

    steps = store.history("r1")
    assert [s.status for s in steps] == ["running", "running", "done"]
    assert len(steps[0].messages) < len(steps[1].messages)
    assert steps[-1].result == "5"


@pytest.mark.asyncio
async def test_arun_without_run_id_saves_nothing():
    store = MemoryCheckpointStore()
    agent = Agent(model=ScriptedModel([ModelResponse(content="hi")]), checkpointer=store)
    assert await agent.arun("hello") == "hi"
    assert store._runs == {}


@pytest.mark.asyncio
async def test_arun_finished_run_returns_saved_result_without_the_model():
    store = MemoryCheckpointStore()
    model = ScriptedModel([ModelResponse(content="done")])
    agent = Agent(model=model, checkpointer=store)
    assert await agent.arun("hi", run_id="r1") == "done"
    assert await agent.arun("hi again", run_id="r1") == "done"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_arun_crash_resumes_without_repeating_a_tool():
    add, _, executed = make_tools()
    store = MemoryCheckpointStore()
    crashing = ScriptedModel([call("1", "add", x=1, y=2), RuntimeError("provider down")])
    agent = Agent(model=crashing, tools=[add], checkpointer=store)
    with pytest.raises(RuntimeError, match="provider down"):
        await agent.arun("1+2?", run_id="r1")
    assert executed == [("add", 1, 2)]

    healthy = ScriptedModel([ModelResponse(content="3")])
    agent2 = Agent(model=healthy, tools=[add], checkpointer=store)
    assert await agent2.arun("1+2?", run_id="r1") == "3"
    assert executed == [("add", 1, 2)]
    assert healthy.calls == 1


@pytest.mark.asyncio
async def test_arun_works_with_sqlite_store(tmp_path):
    add, _, _ = make_tools()
    store = SQLiteCheckpointStore(str(tmp_path / "k.db"))
    model = ScriptedModel([call("1", "add", x=1, y=1), ModelResponse(content="2")])
    agent = Agent(model=model, tools=[add], checkpointer=store)
    assert await agent.arun("1+1?", run_id="r1") == "2"
    assert store.load("r1").status == "done"
    assert len(store.history("r1")) == 3


# approval gate


@pytest.mark.asyncio
async def test_arun_pauses_before_a_gated_tool_and_holds_the_whole_turn():
    add, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel(
        [calls(("1", "add", {"x": 1, "y": 1}), ("2", "delete_item", {"item_id": "a"}))]
    )
    agent = Agent(
        model=model, tools=[add, delete_item], checkpointer=store, require_approval={"delete_item"}
    )

    with pytest.raises(ApprovalRequired) as e:
        await agent.arun("clean up", run_id="r1")

    assert [tc["name"] for tc in e.value.pending] == ["delete_item"]
    assert executed == []  # even the ungated add waits, so resume cannot repeat it
    cp = store.load("r1")
    assert cp.status == "interrupted" and cp.awaiting == ["2"]


@pytest.mark.asyncio
async def test_aresume_approves_and_finishes():
    add, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel(
        [calls(("1", "add", {"x": 1, "y": 1}), ("2", "delete_item", {"item_id": "a"}))]
    )
    agent = Agent(
        model=model, tools=[add, delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        await agent.arun("clean up", run_id="r1")

    model.script.append(ModelResponse(content="all done"))
    assert await agent.aresume("r1", {"2": True}) == "all done"
    assert sorted(executed) == [("add", 1, 1), ("delete_item", "a")]
    assert store.load("r1").status == "done"


@pytest.mark.asyncio
async def test_aresume_rejection_tells_the_model_why():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("1", "delete_item", item_id="a")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        await agent.arun("go", run_id="r1")

    model.script.append(ModelResponse(content="ok, not deleting"))
    out = await agent.aresume("r1", {"1": {"approve": False, "reason": "too risky"}})
    assert out == "ok, not deleting"
    assert executed == []
    last_turn = model.seen_messages[-1][-1]
    assert "rejected" in str(last_turn) and "too risky" in str(last_turn)


@pytest.mark.asyncio
async def test_aresume_with_edited_arguments_runs_the_edit_and_records_it():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("1", "delete_item", item_id="a")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        await agent.arun("go", run_id="r1")

    model.script.append(ModelResponse(content="done"))
    await agent.aresume("r1", {"1": {"approve": True, "input": {"item_id": "b"}}})
    assert executed == [("delete_item", "b")]
    assistant_turn = model.seen_messages[-1][-2]
    assert "'b'" in str(assistant_turn) or '"b"' in str(assistant_turn)


@pytest.mark.asyncio
async def test_aresume_unknown_run_raises():
    agent = Agent(model=ScriptedModel([]), checkpointer=MemoryCheckpointStore())
    with pytest.raises(KeyError):
        await agent.aresume("nope", {})


@pytest.mark.asyncio
async def test_arun_gate_without_run_id_is_an_error():
    _, delete_item, _ = make_tools()
    agent = Agent(
        model=ScriptedModel([]),
        tools=[delete_item],
        checkpointer=MemoryCheckpointStore(),
        require_approval={"delete_item"},
    )
    with pytest.raises(ValueError, match="run_id"):
        await agent.arun("go")


@pytest.mark.asyncio
async def test_sync_run_can_be_resumed_async_and_back():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([call("1", "delete_item", item_id="a")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    with pytest.raises(ApprovalRequired):
        agent.run("go", run_id="r1")
    model.script.append(ModelResponse(content="done"))
    assert await agent.aresume("r1", {"1": True}) == "done"
    assert executed == [("delete_item", "a")]
    assert agent.run("go", run_id="r1") == "done"


# astream_events


@pytest.mark.asyncio
async def test_astream_events_text_tool_and_done_and_saves_steps():
    add, _, _ = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([stream_call("1", "add", x=2, y=3), stream_text("The ", "sum is 5")])
    agent = Agent(model=model, tools=[add], checkpointer=store)

    events = await collect(agent.astream_events("2+3?", run_id="r1"))

    kinds = [e["type"] for e in events]
    assert kinds == ["tool_call", "tool_result", "text", "text", "done"]
    assert events[0]["name"] == "add" and events[1]["result"] == "5"
    assert events[-1]["result"] == "The sum is 5"
    assert [s.status for s in store.history("r1")] == ["running", "running", "done"]


@pytest.mark.asyncio
async def test_astream_events_emits_approval_required_then_stops():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([stream_call("1", "delete_item", item_id="a")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )

    events = await collect(agent.astream_events("go", run_id="r1"))

    assert [e["type"] for e in events] == ["approval_required"]
    assert events[0]["run_id"] == "r1"
    assert events[0]["pending"][0]["name"] == "delete_item"
    assert executed == []
    assert store.load("r1").status == "interrupted"


@pytest.mark.asyncio
async def test_astream_events_resumes_with_decisions():
    _, delete_item, executed = make_tools()
    store = MemoryCheckpointStore()
    model = ScriptedModel([stream_call("1", "delete_item", item_id="a")])
    agent = Agent(
        model=model, tools=[delete_item], checkpointer=store, require_approval={"delete_item"}
    )
    await collect(agent.astream_events("go", run_id="r1"))

    model.script.append(stream_text("deleted it"))
    events = await collect(agent.astream_events("go", run_id="r1", decisions={"1": True}))

    assert [e["type"] for e in events] == ["tool_call", "tool_result", "text", "done"]
    assert executed == [("delete_item", "a")]
    assert events[-1]["result"] == "deleted it"


@pytest.mark.asyncio
async def test_astream_events_finished_run_replays_only_done():
    store = MemoryCheckpointStore()
    model = ScriptedModel([stream_text("hi")])
    agent = Agent(model=model, checkpointer=store)
    await collect(agent.astream_events("hello", run_id="r1"))
    events = await collect(agent.astream_events("hello", run_id="r1"))
    assert events == [{"type": "done", "run_id": "r1", "result": "hi"}]
    assert model.calls == 1


@pytest.mark.asyncio
async def test_astream_events_crash_resumes_without_repeating_a_tool():
    add, _, executed = make_tools()
    store = MemoryCheckpointStore()
    crashing = ScriptedModel([stream_call("1", "add", x=1, y=2), RuntimeError("provider down")])
    agent = Agent(model=crashing, tools=[add], checkpointer=store)
    with pytest.raises(RuntimeError, match="provider down"):
        await collect(agent.astream_events("1+2?", run_id="r1"))
    assert executed == [("add", 1, 2)]

    healthy = ScriptedModel([stream_text("3")])
    agent2 = Agent(model=healthy, tools=[add], checkpointer=store)
    events = await collect(agent2.astream_events("1+2?", run_id="r1"))
    assert events[-1] == {"type": "done", "run_id": "r1", "result": "3"}
    assert executed == [("add", 1, 2)]


@pytest.mark.asyncio
async def test_astream_events_without_run_id_still_streams():
    model = ScriptedModel([stream_text("a", "b")])
    agent = Agent(model=model)
    events = await collect(agent.astream_events("hello"))
    assert [e["type"] for e in events] == ["text", "text", "done"]
    assert events[-1]["result"] == "ab"


@pytest.mark.asyncio
async def test_plain_astream_still_yields_text_chunks():
    model = ScriptedModel([stream_text("a", "b")])
    agent = Agent(model=model)
    assert [c async for c in agent.astream("hello")] == ["a", "b"]


# registry.acall regression: a sync tool used to run twice when a timeout was set


@pytest.mark.asyncio
async def test_acall_runs_a_sync_tool_exactly_once():
    add, _, executed = make_tools()
    from kognios.tools.registry import ToolRegistry

    registry = ToolRegistry([add])
    assert await registry.acall("add", {"x": 1, "y": 2}, timeout=5) == "3"
    assert await registry.acall("add", {"x": 1, "y": 2}, timeout=None) == "3"
    assert executed == [("add", 1, 2), ("add", 1, 2)]


@pytest.mark.asyncio
async def test_acall_still_awaits_async_tools_and_times_out():
    import asyncio

    from kognios.tools.registry import ToolRegistry

    @tool
    async def fast(x: int) -> int:
        """Return x."""
        return x

    @tool
    async def slow() -> str:
        """Never finishes in time."""
        await asyncio.sleep(2)
        return "late"

    registry = ToolRegistry([fast, slow])
    assert await registry.acall("fast", {"x": 7}, timeout=5) == "7"
    assert "timed out" in await registry.acall("slow", {}, timeout=0.05)
