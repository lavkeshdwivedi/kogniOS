import json
from unittest.mock import MagicMock

from kognios import ActionLedger, Agent, load_skills, review, review_and_revise, tool
from kognios.models.base import ModelResponse


def _model(*replies):
    m = MagicMock()
    m.complete.side_effect = [ModelResponse(content=r) for r in replies]
    return m


def test_skills_index_and_lazy_body(tmp_path):
    (tmp_path / "a.md").write_text("---\nname: triage\ndescription: sort mail\n---\nStep one.\n")
    (tmp_path / "bad.md").write_text("no frontmatter")
    skills = load_skills(tmp_path)
    assert len(skills) == 1 and "triage: sort mail" in skills.index()
    assert "Step one." not in skills.index()
    assert skills.get("triage") == "Step one."
    assert skills.get("nope").startswith("Error:")


def test_skills_tool_registers_on_agent(tmp_path):
    (tmp_path / "a.md").write_text("---\nname: triage\ndescription: d\n---\nBody\n")
    agent = Agent(model=_model("ok"), tools=[load_skills(tmp_path).tool()])
    assert agent.registry.call("use_skill", {"name": "triage"}) == "Body"


def test_ledger_runs_side_effect_once():
    ledger = ActionLedger(":memory:")
    calls = []

    @tool
    def send(to: str) -> str:
        """Send a message."""
        calls.append(to)
        return f"sent {to}"

    safe = ledger.once(send)
    assert safe(to="a") == "sent a" and safe(to="a") == "sent a"
    assert safe(to="b") == "sent b"
    assert calls == ["a", "b"]


def test_ledger_does_not_record_errors():
    ledger = ActionLedger(":memory:")
    n = {"i": 0}

    def flaky() -> str:
        n["i"] += 1
        return "Error: down" if n["i"] == 1 else "ok"

    safe = ledger.once(flaky)
    assert safe() == "Error: down" and safe() == "ok" and safe() == "ok"
    assert n["i"] == 2


def test_watermark_roundtrip():
    ledger = ActionLedger(":memory:")
    assert ledger.watermark("inbox") is None
    ledger.advance("inbox", "42")
    assert ledger.watermark("inbox") == "42"


def _verdicts(*pairs):
    return json.dumps({"results": [{"check": c, "verdict": v, "reason": "r"} for c, v in pairs]})


def test_review_missing_check_is_unverified():
    report = review(_model(_verdicts(("a", "PASS"))), "draft", ["a", "b"])
    assert not report.passed
    assert [r.check for r in report.unverified] == ["b"]


def test_review_garbage_reply_never_passes():
    assert not review(_model("not json"), "draft", ["a"]).passed


def test_review_and_revise_loops_on_failure_only():
    m = _model(_verdicts(("a", "FAIL")), _verdicts(("a", "PASS")))
    draft, report = review_and_revise(m, "v1", ["a"], lambda d, r: "v2")
    assert draft == "v2" and report.passed


def test_review_and_revise_skips_unverified():
    m = _model(_verdicts(("a", "CAN'T VERIFY")))
    revise = MagicMock()
    draft, report = review_and_revise(m, "v1", ["a"], revise)
    revise.assert_not_called()
    assert draft == "v1" and report.unverified


def test_tool_output_cap():
    @tool
    def big() -> str:
        """Return a lot."""
        return "x" * 100

    capped = Agent(model=_model("ok"), tools=[big], max_tool_output_chars=10)
    out = capped._call_tool_sync({"name": "big", "input": {}})
    assert out.startswith("x" * 10) and "truncated 90 of 100" in out
    assert Agent(model=_model("ok"), tools=[big])._call_tool_sync({"name": "big"}) == "x" * 100
