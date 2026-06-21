"""
Multi-step planning example.

The agent breaks a complex goal into steps, executes each one,
and returns a consolidated answer.
"""

from kognios import Agent, AnthropicModel
from kognios.tools.builtins import web_search, calculator


def main() -> None:
    agent = Agent(
        model=AnthropicModel(),
        tools=[web_search, calculator],
        instructions="You are a thorough research assistant.",
    )

    result = agent.plan_and_run(
        "What is the population of Tokyo, and how does it compare to New York City? "
        "Calculate the ratio."
    )
    print(result)


if __name__ == "__main__":
    main()
