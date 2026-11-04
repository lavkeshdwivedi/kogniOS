"""A skeptical reviewer: a second model call that checks a draft before it goes out.

The reviewer sees only the final content and a checklist, never the conversation that produced
it, so it cannot inherit the author's assumptions. Each check ends PASS, FAIL or CAN'T VERIFY,
each with a reason. ``review_and_revise`` feeds failures back to the author for another pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .models.base import BaseModel

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "CAN'T VERIFY"

_SYSTEM = (
    "You are a skeptical reviewer. Assume the draft is wrong until shown otherwise. For each "
    "check, answer PASS only with evidence from the draft itself, FAIL when it is broken, and "
    "CAN'T VERIFY when the draft alone cannot settle it. Reply with JSON only: "
    '{"results": [{"check": "...", "verdict": "PASS|FAIL|CAN\'T VERIFY", "reason": "..."}]}'
)


@dataclass
class CheckResult:
    check: str
    verdict: str
    reason: str = ""


@dataclass
class ReviewReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.verdict == PASS for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.verdict == FAIL]

    @property
    def unverified(self) -> list[CheckResult]:
        return [r for r in self.results if r.verdict == UNVERIFIED]


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def review(model: BaseModel, content: str, checks: list[str]) -> ReviewReport:
    """Review ``content`` against ``checks`` in a fresh context. A check the reviewer skips
    or garbles counts as CAN'T VERIFY, never as a pass."""
    prompt = "Checks:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(checks, 1))
    prompt += f"\n\nDraft:\n{content}"
    reply = model.complete([{"role": "user", "content": prompt}], system=_SYSTEM).content
    by_check = {
        str(r.get("check", "")).strip(): r
        for r in _extract_json(reply).get("results", [])
        if isinstance(r, dict)
    }
    results = []
    for check in checks:
        raw = by_check.get(check.strip())
        verdict = str(raw.get("verdict", "")).upper() if raw else ""
        if verdict not in (PASS, FAIL):
            verdict = UNVERIFIED
        results.append(CheckResult(check, verdict, str(raw.get("reason", "")) if raw else ""))
    return ReviewReport(results)


def review_and_revise(
    model: BaseModel,
    draft: str,
    checks: list[str],
    revise: Callable[[str, ReviewReport], str],
    max_rounds: int = 2,
) -> tuple[str, ReviewReport]:
    """Review, and while checks FAIL hand the report to ``revise`` for a new draft. Returns the
    last draft and its report. CAN'T VERIFY results do not trigger a revision, because no edit
    can settle them: they are left for a human."""
    report = review(model, draft, checks)
    for _ in range(max_rounds):
        if not report.failures:
            break
        draft = revise(draft, report)
        report = review(model, draft, checks)
    return draft, report
