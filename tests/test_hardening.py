from unittest.mock import MagicMock

import pytest

from kognios import ActionLedger, Agent, GuardrailError, MemoryCheckpointStore, pii_scrubber, tool
from kognios.ledger import PENDING
from kognios.models.base import ModelChunk, ModelResponse
from kognios.tools.builtins import code_interpreter, file_tools, read_file


def _model(*responses):
    m = MagicMock()
    m.complete.side_effect = list(responses)
    return m


def _call(name, **inp):
    return ModelResponse(content="", tool_calls=[{"id": "t1", "name": name, "input": inp}])


def _sender():
    sent = []

    @tool
    def send(to: str) -> str:
        """Send."""
        sent.append(to)
        return f"sent {to}"

    return send, sent


def _once_agent(send, ledger):
    return Agent(
        model=_model(_call("send", to="bob"), ModelResponse(content="done")),
        tools=[send],
        ledger=ledger,
        once_tools={"send"},
        checkpointer=MemoryCheckpointStore(),
    )


# ── ledger wired into the agent ──────────────────────────────────────────────


def test_once_tools_needs_ledger_and_run_id():
    send, _ = _sender()
    with pytest.raises(ValueError):
        Agent(model=_model(), tools=[send], once_tools={"send"})
    with pytest.raises(ValueError):
        Agent(model=_model(), tools=[send], ledger=ActionLedger(":memory:"), once_tools={"send"})
    agent = _once_agent(send, ActionLedger(":memory:"))
    with pytest.raises(ValueError, match="needs a run_id"):
        agent.run("hi")


def test_once_tool_not_repeated_when_a_run_replays_its_turn():
    send, sent = _sender()
    ledger = ActionLedger(":memory:")
    _once_agent(send, ledger).run("go", run_id="r1")
    # a crash before the checkpoint replays the turn: new call id, same arguments
    _once_agent(send, ledger).run("go", run_id="r1")
    assert sent == ["bob"]
    # a different run is a different scope, so the same message may go out again
    _once_agent(send, ledger).run("go", run_id="r2")
    assert sent == ["bob", "bob"]


def test_in_flight_marker_blocks_blind_retry(tmp_path):
    send, sent = _sender()
    db = str(tmp_path / "ledger.db")
    tc = {"id": "t", "name": "send", "input": {"to": "bob"}}

    def agent_on(ledger):
        return Agent(
            model=_model(), tools=[send], ledger=ledger, once_tools={"send"},
            checkpointer=MemoryCheckpointStore(),
        )

    dying = agent_on(ActionLedger(db))
    key = dying._once_key(tc, "r1")
    assert dying.ledger.acquire(key)[0] == "run"  # claimed, then the process "dies"
    out = agent_on(ActionLedger(db))._call_tool_sync(tc, "r1")  # a fresh process
    assert out.startswith("Error:") and "may already have run" in out and sent == []


def test_agent_marks_the_action_in_flight_while_it_runs():
    ledger = ActionLedger(":memory:")
    seen = []

    @tool
    def peek(x: str) -> str:
        """Peek."""
        seen.append(ledger.seen(agent._once_key(tc, "r")))
        return "ok"

    agent = Agent(
        model=_model(), tools=[peek], ledger=ledger, once_tools={"peek"},
        checkpointer=MemoryCheckpointStore(),
    )
    tc = {"id": "t", "name": "peek", "input": {"x": "a"}}
    agent._call_tool_sync(tc, "r")
    assert seen == [PENDING] and ledger.seen(agent._once_key(tc, "r")) == "ok"


def test_ledger_once_runs_once_across_threads():
    import threading
    import time

    ledger = ActionLedger(":memory:")
    runs = []

    def act(to: str) -> str:
        runs.append(to)
        time.sleep(0.05)
        return f"sent {to}"

    safe = ledger.once(act)
    gate = threading.Barrier(8)
    out = []

    def worker():
        gate.wait()
        out.append(safe(to="y"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert runs == ["y"] and out == ["sent y"] * 8


def test_positional_and_keyword_calls_share_a_key():
    ledger = ActionLedger(":memory:")
    runs = []

    def act(to: str, body: str = "hi") -> str:
        runs.append(to)
        return "ok"

    safe = ledger.once(act)
    safe("a")
    safe(to="a")
    safe("a", "hi")
    assert runs == ["a"]


def test_interrupt_leaves_the_action_unknown_not_retryable():
    ledger = ActionLedger(":memory:")

    def act() -> str:
        raise KeyboardInterrupt

    safe = ledger.once(act)
    with pytest.raises(KeyboardInterrupt):
        safe()
    assert "may already have run" in safe()


def test_same_turn_duplicates_run_once_and_do_not_trip_the_breaker():
    send, sent = _sender()
    calls = [{"id": f"t{i}", "name": "send", "input": {"to": "bob"}} for i in range(6)]
    agent = Agent(
        model=_model(ModelResponse(content="", tool_calls=calls), ModelResponse(content="done")),
        tools=[send], ledger=ActionLedger(":memory:"), once_tools={"send"},
        checkpointer=MemoryCheckpointStore(),
    )
    assert agent.run("go", run_id="r") == "done"
    assert sent == ["bob"]


def test_timeout_keeps_the_key_pending_so_a_retry_cannot_double_the_action():
    import time

    ledger = ActionLedger(":memory:")
    ran = []

    @tool
    def slow(x: str) -> str:
        """Slow."""
        time.sleep(0.3)
        ran.append(x)
        return "ok"

    agent = Agent(
        model=_model(), tools=[slow], ledger=ledger, once_tools={"slow"},
        checkpointer=MemoryCheckpointStore(), tool_timeout=0.05,
    )
    tc = {"id": "t", "name": "slow", "input": {"x": "a"}}
    assert "timed out" in agent._call_tool_sync(tc, "r")
    assert "may already have run" in agent._call_tool_sync(tc, "r")
    time.sleep(0.4)
    assert ran == ["a"]


def test_async_once_tool_runs_once_for_parallel_identical_calls():
    import asyncio

    send, sent = _sender()
    calls = [{"id": f"t{i}", "name": "send", "input": {"to": "bob"}} for i in range(4)]
    m = MagicMock()

    async def acomplete(*a, **k):
        return responses.pop(0)

    responses = [ModelResponse(content="", tool_calls=calls), ModelResponse(content="done")]
    m.acomplete.side_effect = acomplete
    agent = Agent(
        model=m, tools=[send], ledger=ActionLedger(":memory:"), once_tools={"send"},
        checkpointer=MemoryCheckpointStore(),
    )
    assert asyncio.run(agent.arun("go", run_id="r")) == "done"
    assert sent == ["bob"]


def test_astream_and_astream_events_hold_text_for_output_guardrails():
    import asyncio

    def chunks():
        return [ModelChunk(text="mail a@"), ModelChunk(text="b.com now"), ModelChunk(final=True)]

    async def gen(*a, **k):
        for c in chunks():
            yield c

    m = MagicMock()
    m.astream.side_effect = gen
    agent = Agent(
        model=m, output_guardrails=[pii_scrubber()], checkpointer=MemoryCheckpointStore()
    )

    async def collect():
        plain = "".join([t async for t in agent.astream("q")])
        events = [e async for e in agent.astream_events("q", run_id="r")]
        return plain, events

    plain, events = asyncio.run(collect())
    assert "a@b.com" not in plain and "now" in plain
    texts = [e["text"] for e in events if e["type"] == "text"]
    assert len(texts) == 1 and "a@b.com" not in texts[0]
    assert "a@b.com" not in events[-1]["result"]


def test_failed_once_tool_can_retry():
    ledger = ActionLedger(":memory:")
    n = {"i": 0}

    @tool
    def flaky(x: str) -> str:
        """Flaky."""
        n["i"] += 1
        return "Error: down" if n["i"] == 1 else "ok"

    agent = Agent(
        model=_model(), tools=[flaky], ledger=ledger, once_tools={"flaky"},
        checkpointer=MemoryCheckpointStore(),
    )
    tc = {"id": "t", "name": "flaky", "input": {"x": "a"}}
    assert agent._call_tool_sync(tc, "r") == "Error: down"
    assert agent._call_tool_sync(tc, "r") == "ok"
    assert agent._call_tool_sync(tc, "r") == "ok" and n["i"] == 2
    assert ledger.seen(agent._once_key(tc, "r")) != PENDING


# ── tool guardrails ──────────────────────────────────────────────────────────


def test_tool_guardrail_scrubs_and_blocks():
    @tool
    def leak() -> str:
        """Leak."""
        return "mail me at a@b.com"

    scrub = Agent(model=_model(), tools=[leak], tool_guardrails=[pii_scrubber()])
    assert "a@b.com" not in scrub._call_tool_sync({"name": "leak", "input": {}})

    def deny(text):
        raise GuardrailError("injection")

    block = Agent(model=_model(), tools=[leak], tool_guardrails=[deny])
    out = block._call_tool_sync({"name": "leak", "input": {}})
    assert out.startswith("Error: tool result blocked") and "injection" in out


# ── guardrail ordering and output paths ──────────────────────────────────────


def test_input_guardrail_runs_before_retrieval():
    kb = MagicMock()
    kb.search.return_value = []
    agent = Agent(
        model=_model(ModelResponse(content="ok")),
        knowledge=kb,
        input_guardrails=[lambda t: t.replace("SECRET", "[x]")],
    )
    agent.run("find SECRET now")
    kb.search.assert_called_once_with("find [x] now")


def test_output_guardrail_applies_to_structured_output():
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        note: str

    m = _model(ModelResponse(content="x"))
    m.structured_complete.return_value = {"note": "mail a@b.com"}
    agent = Agent(model=m, output_guardrails=[pii_scrubber()])
    assert "a@b.com" not in agent.run("q", output_type=Out).note


def test_stream_holds_text_for_output_guardrails():
    m = MagicMock()
    m.stream.return_value = iter(
        [ModelChunk(text="mail a@"), ModelChunk(text="b.com now"), ModelChunk(final=True)]
    )
    agent = Agent(model=m, output_guardrails=[pii_scrubber()])
    out = "".join(agent.stream("q"))
    assert "a@b.com" not in out and "now" in out


# ── trimming ─────────────────────────────────────────────────────────────────


def test_trim_never_leaves_an_orphan_tool_result():
    use = {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "t"}]}
    res = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x" * 400}],
    }
    agent = Agent(model=_model(), max_context_tokens=150)
    msgs = [{"role": "user", "content": "q" * 400}, use, res, {"role": "assistant", "content": "a" * 400}]
    assert not agent._is_tool_result(agent._trim_messages(msgs)[0])
    # a turn in progress ends on a tool result: its tool call stays with it
    tail = [{"role": "user", "content": "q" * 4000}, use, res]
    assert Agent(model=_model(), max_context_tokens=10)._trim_messages(tail) == [use, res]


# ── built-in tools ───────────────────────────────────────────────────────────


def test_read_file_is_capped(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("y" * 100)
    out = read_file(str(f), max_chars=10)
    assert out.startswith("y" * 10) and "truncated" in out


def test_file_tools_confined_to_root(tmp_path):
    rd, wr = file_tools(tmp_path / "work")
    assert wr(path="a/b.txt", content="hi").startswith("Written")
    assert rd(path="a/b.txt") == "hi"
    assert "outside the allowed folder" in wr(path="../escape.txt", content="x")
    assert "outside the allowed folder" in rd(path=str(tmp_path / "other.txt"))
    assert not (tmp_path / "escape.txt").exists()


def test_code_interpreter_does_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("KOGNIOS_TEST_SECRET", "hunter2")
    out = code_interpreter("import os; print(os.environ.get('KOGNIOS_TEST_SECRET'))")
    assert out == "None"


def test_read_file_clamps_a_negative_or_huge_max_chars(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("y" * 50)
    assert "longer than 0 chars" in read_file(str(f), max_chars=-5)
    assert read_file(str(f), max_chars=10**12) == "y" * 50


def test_code_interpreter_can_find_the_home_folder():
    assert code_interpreter("from pathlib import Path; print(Path.home().is_absolute())") == "True"
