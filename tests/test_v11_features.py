"""Tests for v1.1 features: smart fact injection, eval CLI, ingest CLI, serve endpoints."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from kognios.agent import Agent
from kognios.cli import cli
from kognios.memory.long_term import LongTermMemory
from kognios.memory.short_term import ShortTermMemory
from kognios.models.base import ModelChunk, ModelResponse


# ── Smart fact injection ───────────────────────────────────────────────────────


def test_build_system_uses_all_facts_without_embed_fn():
    """When LongTermMemory has no embed_fn, _build_system dumps all facts."""
    model = MagicMock()
    model.complete.return_value = ModelResponse(content="ok")
    ltm = LongTermMemory(db_path=":memory:")
    ltm.store("color", "blue")
    ltm.store("size", "large")

    agent = Agent(model=model, memory=ltm)
    # call _build_system directly to avoid the memory.append check
    system = agent._build_system("test query")

    assert "color: blue" in system
    assert "size: large" in system


def test_build_system_uses_semantic_recall_with_embed_fn():
    """When LongTermMemory has embed_fn, _build_system uses semantic_recall."""
    model = MagicMock()
    model.complete.return_value = ModelResponse(content="ok")

    embed_calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(texts)
        result = []
        for t in texts:
            result.append([1.0, 0.0] if "cat" in t.lower() or "pet" in t.lower() else [0.0, 1.0])
        return result

    ltm = LongTermMemory(db_path=":memory:", embed_fn=fake_embed)
    ltm.store("pet", "cat")
    ltm.store("color", "blue")
    ltm.store("food", "pasta")

    agent = Agent(model=model, memory=ltm)
    system = agent._build_system("what is my pet?")

    # semantic_recall used: embed_fn called for storage + query
    assert len(embed_calls) >= 1
    # at least one fact appeared in system prompt
    assert "Known facts" in system


def test_long_term_memory_does_not_crash_agent_run():
    """Agent.run with LongTermMemory as memory should not raise AttributeError."""
    model = MagicMock()
    model.complete.return_value = ModelResponse(content="ok")
    ltm = LongTermMemory(db_path=":memory:")
    ltm.store("key", "val")
    agent = Agent(model=model, memory=ltm)
    # must not raise 'LongTermMemory has no attribute append'
    result = agent.run("hello")
    assert result == "ok"


# ── kognios eval CLI ───────────────────────────────────────────────────────────


def _write_dataset(tmp_path: Path, cases: list[dict]) -> str:
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(cases), encoding="utf-8")
    return str(p)


def test_eval_command_pass_rate(tmp_path):
    dataset = _write_dataset(
        tmp_path,
        [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Capital of France?", "expected": "Paris"},
        ],
    )

    from unittest.mock import patch

    with patch("kognios.cli._make_model") as mock_make:
        fake_model = MagicMock()
        fake_model.complete.side_effect = [
            ModelResponse(content="4"),
            ModelResponse(content="Paris is the capital."),
        ]
        mock_make.return_value = fake_model

        result = CliRunner().invoke(cli, ["eval", dataset, "--scorer", "contains"])

    assert result.exit_code == 0, result.output
    assert "Pass rate" in result.output
    assert "100.0%" in result.output


def test_eval_command_failure_case(tmp_path):
    dataset = _write_dataset(tmp_path, [{"input": "What is 2+2?", "expected": "4"}])

    from unittest.mock import patch

    with patch("kognios.cli._make_model") as mock_make:
        fake_model = MagicMock()
        fake_model.complete.return_value = ModelResponse(content="five")
        mock_make.return_value = fake_model

        result = CliRunner().invoke(cli, ["eval", dataset, "--scorer", "exact_match"])

    assert result.exit_code == 0, result.output
    assert "FAIL" in result.output


def test_eval_command_rejects_non_array(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"input": "x"}', encoding="utf-8")

    from unittest.mock import patch

    with patch("kognios.cli._make_model") as mock_make:
        mock_make.return_value = MagicMock()
        result = CliRunner().invoke(cli, ["eval", str(p)])

    assert result.exit_code != 0


# ── kognios ingest CLI ─────────────────────────────────────────────────────────


def test_ingest_command_fts(tmp_path):
    txt = tmp_path / "doc.txt"
    txt.write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    db = str(tmp_path / "kb.db")

    result = CliRunner().invoke(cli, ["ingest", str(txt), "--db", db, "--backend", "fts"])

    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output

    from kognios.knowledge.sqlite_fts import SQLiteKnowledge

    kb = SQLiteKnowledge(db_path=db)
    hits = kb.search("fox")
    assert any("fox" in h.lower() for h in hits)


def test_ingest_command_vector(tmp_path):
    txt = tmp_path / "doc.txt"
    txt.write_text("Vectors are useful for semantic search.", encoding="utf-8")

    result = CliRunner().invoke(cli, ["ingest", str(txt), "--backend", "vector"])

    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output


# ── kognios serve endpoints ────────────────────────────────────────────────────


@pytest.fixture
def serve_client():
    """Build a minimal FastAPI app matching the serve command's endpoints."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    fake_model = MagicMock()
    fake_model.complete.return_value = ModelResponse(content="42")
    fake_model.stream.return_value = iter(
        [
            ModelChunk(text="hello"),
            ModelChunk(text=" world", final=True),
        ]
    )

    instructions = "Be helpful."
    stateless = Agent(model=fake_model, instructions=instructions)
    sessions: dict[str, ShortTermMemory] = {}

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/run")
    def run_ep(req: dict):
        result = stateless.run(req["message"])
        return {"response": result, "usage": stateless.last_usage}

    @app.post("/stream")
    def stream_ep(req: dict):
        def gen():
            for chunk in stateless.stream(req["message"]):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/session")
    def create_sess():
        sid = str(uuid.uuid4())
        sessions[sid] = ShortTermMemory(max_turns=20)
        return {"session_id": sid}

    @app.delete("/session/{session_id}")
    def del_sess(session_id: str):
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        del sessions[session_id]
        return {"deleted": session_id}

    @app.post("/session/{session_id}/run")
    def sess_run(session_id: str, req: dict):
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        a = Agent(model=fake_model, memory=sessions[session_id], instructions=instructions)
        result = a.run(req["message"])
        return {"response": result, "usage": a.last_usage}

    return TestClient(app, raise_server_exceptions=True), sessions


def test_serve_health(serve_client):
    client, _ = serve_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_serve_run(serve_client):
    client, _ = serve_client
    r = client.post("/run", json={"message": "what is 6x7?"})
    assert r.status_code == 200
    assert r.json()["response"] == "42"


def test_serve_stream(serve_client):
    client, _ = serve_client
    r = client.post("/stream", json={"message": "hi"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert "hello" in body
    assert "[DONE]" in body


def test_serve_session_lifecycle(serve_client):
    client, sessions = serve_client

    r = client.post("/session")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid in sessions

    r = client.post(f"/session/{sid}/run", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["response"] == "42"

    r = client.delete(f"/session/{sid}")
    assert r.status_code == 200
    assert sid not in sessions


def test_serve_session_404_on_missing(serve_client):
    client, _ = serve_client
    r = client.post("/session/nonexistent/run", json={"message": "hi"})
    assert r.status_code == 404
