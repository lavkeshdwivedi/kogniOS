"""Agent that answers from a loaded knowledge base."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kognios import Agent, AnthropicModel, SQLiteKnowledge

DOCUMENT = """
AgentOS is a lightweight Python agent framework built from scratch.
It has four core primitives: Agent + Tools, Memory, Knowledge/RAG, and Team routing.
The @tool decorator auto-generates JSON schemas from Python type hints.
Short-term memory stores conversation history in RAM with a sliding window.
Long-term memory persists key/value facts in SQLite.
Knowledge is indexed in SQLite FTS5 and retrieved via BM25 ranking.
Team routing dispatches messages to specialized sub-agents via a router LLM.
"""

kb = SQLiteKnowledge(db_path=":memory:")
kb.load(DOCUMENT)

model = AnthropicModel(model="claude-haiku-4-5-20251001")
agent = Agent(
    model=model,
    knowledge=kb,
    instructions="Answer only from the provided context.",
)

answer = agent.run("How does AgentOS handle long-term memory?")
print(answer)
