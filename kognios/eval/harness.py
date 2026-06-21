from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class EvalCase:
    """A single evaluation test case."""

    input: str
    expected: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single test case."""

    case: EvalCase
    output: str
    passed: bool
    score: float  # 0.0 – 1.0
    duration_s: float
    error: str = ""


@dataclass
class EvalReport:
    """Aggregated results for all test cases."""

    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def mean_duration_s(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.duration_s for r in self.results) / len(self.results)

    def summary(self) -> str:
        return (
            f"Cases: {len(self.results)} | "
            f"Pass rate: {self.pass_rate:.1%} | "
            f"Mean score: {self.mean_score:.3f} | "
            f"Mean latency: {self.mean_duration_s:.2f}s"
        )

    def print_report(self) -> None:
        print(self.summary())
        for i, r in enumerate(self.results, 1):
            status = "PASS" if r.passed else "FAIL"
            err = f" | error: {r.error}" if r.error else ""
            print(f"  [{status}] case {i}: score={r.score:.3f} | {r.duration_s:.2f}s{err}")
            print(f"         input:    {r.case.input[:80]}")
            print(f"         expected: {r.case.expected[:80]}")
            print(f"         got:      {r.output[:80]}")


class AgentEvaluator:
    """Run an Agent against a dataset and score each response.

    Example::

        from kognios.eval import AgentEvaluator, EvalCase, exact_match

        evaluator = AgentEvaluator(agent, scorer=exact_match)
        report = evaluator.run([
            EvalCase(input="What is 2+2?", expected="4"),
            EvalCase(input="Capital of France?", expected="Paris"),
        ])
        report.print_report()
    """

    def __init__(
        self,
        agent: "Agent",
        scorer: Callable[[str, str], float] | None = None,
        pass_threshold: float = 0.5,
    ):
        self.agent = agent
        self.scorer = scorer or contains
        self.pass_threshold = pass_threshold

    def run(self, cases: list[EvalCase]) -> EvalReport:
        """Run all cases synchronously."""
        results = []
        for case in cases:
            result = self._run_case(case)
            results.append(result)
        return EvalReport(results)

    async def arun(self, cases: list[EvalCase]) -> EvalReport:
        """Run all cases asynchronously (parallel)."""
        import asyncio

        tasks = [self._arun_case(case) for case in cases]
        results = await asyncio.gather(*tasks)
        return EvalReport(list(results))

    def _run_case(self, case: EvalCase) -> EvalResult:
        t0 = time.perf_counter()
        output = ""
        error = ""
        try:
            output = self.agent.run(case.input)
        except Exception as exc:
            error = str(exc)
        duration = time.perf_counter() - t0
        score = self.scorer(output, case.expected) if not error else 0.0
        return EvalResult(
            case=case,
            output=output,
            passed=score >= self.pass_threshold,
            score=score,
            duration_s=duration,
            error=error,
        )

    async def _arun_case(self, case: EvalCase) -> EvalResult:
        t0 = time.perf_counter()
        output = ""
        error = ""
        try:
            output = await self.agent.arun(case.input)
        except Exception as exc:
            error = str(exc)
        duration = time.perf_counter() - t0
        score = self.scorer(output, case.expected) if not error else 0.0
        return EvalResult(
            case=case,
            output=output,
            passed=score >= self.pass_threshold,
            score=score,
            duration_s=duration,
            error=error,
        )


# default scorer used when none is provided
def contains(output: str, expected: str) -> float:
    return 1.0 if expected.lower() in output.lower() else 0.0
