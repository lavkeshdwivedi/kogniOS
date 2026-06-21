from __future__ import annotations

import re


def exact_match(output: str, expected: str) -> float:
    """1.0 if output equals expected (case-insensitive, stripped), else 0.0."""
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def contains(output: str, expected: str) -> float:
    """1.0 if expected string appears anywhere in output (case-insensitive)."""
    return 1.0 if expected.lower() in output.lower() else 0.0


def regex_match(pattern: str):
    """Factory: returns a scorer that checks output against a regex pattern."""
    compiled = re.compile(pattern, re.IGNORECASE)

    def _scorer(output: str, expected: str) -> float:
        return 1.0 if compiled.search(output) else 0.0

    _scorer.__name__ = f"regex_match({pattern!r})"
    return _scorer


def llm_judge(judge_model, rubric: str = ""):
    """Factory: returns a scorer that asks an LLM to score the output 0-10.

    judge_model: any kogniOS BaseModel with .complete()
    rubric: optional additional criteria for the judge
    """

    def _scorer(output: str, expected: str) -> float:
        prompt = (
            f"You are an impartial judge. Score the following AI response on a scale of 0 to 10.\n\n"
            f"Expected answer: {expected}\n"
            f"Actual response: {output}\n"
            f"{rubric}\n"
            f"Reply with ONLY a single integer from 0 to 10."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = judge_model.complete(messages)
            score_str = response.content.strip()
            score = int(re.search(r"\d+", score_str).group())
            return min(max(score / 10.0, 0.0), 1.0)
        except Exception:
            return 0.0

    return _scorer
