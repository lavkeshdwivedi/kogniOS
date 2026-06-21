"""Minimal agent — no tools, no memory, just a direct LLM call."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kognios import Agent, AnthropicModel

model = AnthropicModel(model="claude-haiku-4-5-20251001")
agent = Agent(model=model, instructions="You are a concise assistant.")

answer = agent.run("What is the capital of France?")
print(answer)
