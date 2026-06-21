"""Tests for the evaluation harness — all mocked, no real API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kognios import Agent
from kognios.eval import AgentEvaluator, EvalCase, EvalReport
from kognios.eval.scorers import contains, exact_match, regex_match
from kognios.models.base import ModelResponse


# ── Scorers ───────────────────────────────────────────────────────────────────


def test_contains_pass():
    assert contains("The answer is 4", "4") == 1.0


def test_contains_fail():
    assert contains("The answer is 5", "4") == 0.0


def test_exact_match_pass():
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive():
    assert exact_match("paris", "PARIS") == 1.0


def test_exact_match_fail():
    assert exact_match("Paris, France", "Paris") == 0.0


def test_regex_match():
    scorer = regex_match(r"\d{4}")
    assert scorer("The year was 1991", "") == 1.0
    assert scorer("No year here", "") == 0.0


# ── EvalReport ────────────────────────────────────────────────────────────────


def test_eval_report_pass_rate():
    from kognios.eval.harness import EvalResult

    results = [
        EvalResult(case=EvalCase("a", "a"), output="a", passed=True, score=1.0, duration_s=0.1),
        EvalResult(case=EvalCase("b", "b"), output="x", passed=False, score=0.0, duration_s=0.1),
    ]
    report = EvalReport(results)
    assert report.pass_rate == 0.5
    assert report.mean_score == 0.5


# ── AgentEvaluator ────────────────────────────────────────────────────────────


def test_evaluator_run_sync():
    fake_model = MagicMock()
    fake_model.complete = MagicMock(
        side_effect=[
            ModelResponse(content="4", usage={}),
            ModelResponse(content="Paris", usage={}),
        ]
    )
    agent = Agent(model=fake_model)
    evaluator = AgentEvaluator(agent, scorer=exact_match)
    report = evaluator.run(
        [
            EvalCase(input="2+2?", expected="4"),
            EvalCase(input="Capital of France?", expected="Paris"),
        ]
    )
    assert report.pass_rate == 1.0
    assert len(report.results) == 2


def test_evaluator_run_failure_case():
    fake_model = MagicMock()
    fake_model.complete = MagicMock(return_value=ModelResponse(content="wrong answer", usage={}))
    agent = Agent(model=fake_model)
    evaluator = AgentEvaluator(agent, scorer=exact_match)
    report = evaluator.run([EvalCase(input="q", expected="right answer")])
    assert report.pass_rate == 0.0
    assert report.results[0].passed is False


def test_evaluator_captures_exceptions():
    fake_model = MagicMock()
    fake_model.complete = MagicMock(side_effect=RuntimeError("API down"))
    agent = Agent(model=fake_model)
    evaluator = AgentEvaluator(agent, scorer=exact_match)
    report = evaluator.run([EvalCase(input="q", expected="a")])
    assert report.results[0].error != ""
    assert report.results[0].passed is False


@pytest.mark.asyncio
async def test_evaluator_arun():
    fake_model = MagicMock()
    fake_model.acomplete = AsyncMock(
        return_value=ModelResponse(content="yes it contains", usage={})
    )
    agent = Agent(model=fake_model)
    evaluator = AgentEvaluator(agent, scorer=contains)
    report = await evaluator.arun([EvalCase(input="test", expected="contains")])
    assert report.pass_rate == 1.0
