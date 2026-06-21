"""Agent with short-term and long-term memory across turns."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kognios import Agent, AnthropicModel, ShortTermMemory, LongTermMemory

model = AnthropicModel(model="claude-haiku-4-5-20251001")
short_mem = ShortTermMemory(max_turns=10)
long_mem = LongTermMemory(db_path=":memory:")

long_mem.store("user_name", "Lavkesh")
long_mem.store("preferred_language", "Python")

agent = Agent(
    model=model,
    memory=short_mem,
    instructions="You are a personalized assistant. Use known facts about the user.",
)
# Inject long-term memory into the agent so _build_system picks it up.
agent.memory = type("_Combined", (), {
    "messages": short_mem.messages,
    "append": short_mem.append,
    "all_facts": long_mem.all_facts,
})()

print("Turn 1:")
r1 = agent.run("My favourite framework is Django.")
print(r1)

print("\nTurn 2:")
r2 = agent.run("What did I just tell you?")
print(r2)
