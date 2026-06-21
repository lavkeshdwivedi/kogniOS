"""
Evaluation harness example.

Run a set of test cases against an agent and score the results.
No API key needed — uses mock responses for this demo.
"""

from unittest.mock import MagicMock
from kognios import Agent
from kognios.eval import AgentEvaluator, EvalCase, contains, exact_match
from kognios.models.base import ModelResponse


def main() -> None:
    # Use a mock model so no API key is needed
    fake_model = MagicMock()
    fake_model.complete = MagicMock(
        side_effect=[
            ModelResponse(content="The answer is 4", usage={}),
            ModelResponse(content="Paris is the capital of France", usage={}),
            ModelResponse(content="Python was created in 1991", usage={}),
        ]
    )

    agent = Agent(model=fake_model, instructions="Be concise.")

    cases = [
        EvalCase(input="What is 2+2?", expected="4"),
        EvalCase(input="Capital of France?", expected="Paris"),
        EvalCase(input="When was Python created?", expected="1991"),
    ]

    evaluator = AgentEvaluator(agent, scorer=contains)
    report = evaluator.run(cases)
    report.print_report()
    print(f"\nFinal: {report.pass_rate:.0%} pass rate")


if __name__ == "__main__":
    main()
