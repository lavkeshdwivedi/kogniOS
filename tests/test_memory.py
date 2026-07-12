from kognios.memory.short_term import ShortTermMemory
from kognios.memory.long_term import LongTermMemory


def test_short_term_append_and_messages():
    mem = ShortTermMemory(max_turns=3)
    mem.append("user", "hello")
    mem.append("assistant", "hi there")
    msgs = mem.messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "hi there"}


def test_short_term_sliding_window():
    mem = ShortTermMemory(max_turns=2)
    for i in range(5):
        mem.append("user", f"msg {i}")
        mem.append("assistant", f"reply {i}")
    msgs = mem.messages()
    # max_turns=2 → keeps last 4 messages
    assert len(msgs) == 4
    assert msgs[0]["content"] == "msg 3"


def test_short_term_clear():
    mem = ShortTermMemory()
    mem.append("user", "test")
    mem.clear()
    assert mem.messages() == []


def test_long_term_store_recall():
    mem = LongTermMemory(db_path=":memory:")
    mem.store("name", "Alice")
    assert mem.recall("name") == "Alice"


def test_long_term_update():
    mem = LongTermMemory(db_path=":memory:")
    mem.store("name", "Alice")
    mem.store("name", "Bob")
    assert mem.recall("name") == "Bob"


def test_long_term_recall_missing():
    mem = LongTermMemory(db_path=":memory:")
    assert mem.recall("no_such_key") is None


def test_long_term_all_facts():
    mem = LongTermMemory(db_path=":memory:")
    mem.store("a", "1")
    mem.store("b", "2")
    facts = mem.all_facts()
    assert facts == {"a": "1", "b": "2"}


def test_long_term_delete():
    mem = LongTermMemory(db_path=":memory:")
    mem.store("key", "value")
    mem.delete("key")
    assert mem.recall("key") is None


def test_long_term_remember_forget_aliases():
    mem = LongTermMemory(db_path=":memory:")
    mem.remember("city", "London")
    assert mem.recall("city") == "London"
    mem.forget("city")
    assert mem.recall("city") is None


# --- ShortTermMemory compaction tests ---


class _StubModel:
    """Minimal model stub that returns a fixed summary."""

    def complete(self, messages, system="", **kw):
        from kognios.models.base import ModelResponse

        return ModelResponse(content="Summary of dropped turns.", tool_calls=[], usage={})


def test_short_term_compaction_triggers():
    model = _StubModel()
    mem = ShortTermMemory(max_turns=2, compaction_model=model)
    # Fill past the limit (max_turns=2 → limit=4 messages)
    for i in range(3):
        mem.append("user", f"msg {i}")
        mem.append("assistant", f"reply {i}")
    msgs = mem.messages()
    # Should have a system summary + 4 recent messages
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert any("summary" in m["content"].lower() for m in msgs if m["role"] == "system")


def test_short_term_no_compaction_without_model():
    mem = ShortTermMemory(max_turns=2)
    for i in range(3):
        mem.append("user", f"msg {i}")
        mem.append("assistant", f"reply {i}")
    msgs = mem.messages()
    # No system message — plain sliding window
    assert all(m["role"] != "system" for m in msgs)
    assert len(msgs) == 4
