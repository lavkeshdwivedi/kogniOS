# Kogni·OS

> A from-scratch Python agent framework. No heavy deps, no vector DB.  
> SQLite ships with Python. That's all you need.

[![PyPI](https://img.shields.io/pypi/v/kognios.svg)](https://pypi.org/project/kognios/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/lavkeshdwivedi/kogniOS/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lavkeshdwivedi/kogniOS/blob/main/CONTRIBUTING.md)
[![Tests](https://github.com/lavkeshdwivedi/kogniOS/actions/workflows/ci.yml/badge.svg)](https://github.com/lavkeshdwivedi/kogniOS/actions)

---

## Why Kogni·OS?

Most agent frameworks are wrappers around wrappers. Kogni·OS is written from scratch so you can **read and understand every line** and extend it without fighting abstractions.

- SQLite FTS5 for RAG (no Chroma, no FAISS, no vector DB)
- ReAct loop in ~800 lines of core code
- Zero heavy dependencies
- Built-in streaming, async, and structured output
- 9 providers: Anthropic, OpenAI, Groq, Gemini, Mistral, Cohere, Ollama, Bedrock, xAI

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                   Team                       │
│  router_model ──▶ pick agent ──▶ agent.run() │
└───────────────────────┬─────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │           Agent             │
          │  ┌──────────────────────┐  │
          │  │   ReAct Loop         │  │
          │  │  think → tool → obs  │  │
          │  └──────────┬───────────┘  │
          │             │               │
          │  ┌──────────▼───────────┐  │
          │  │    BaseModel ABC      │  │
          │  │  Anthropic │ OpenAI  │  │
          │  │  Groq      │ Gemini  │  │
          │  └──────────────────────┘  │
          │                            │
          │  ┌──────────────────────┐  │
          │  │   ToolRegistry       │  │  ◀── @tool decorator
          │  │   @tool fns + schema │  │
          │  └──────────────────────┘  │
          │                            │
          │  ┌──────────────────────┐  │
          │  │   Memory             │  │
          │  │  ShortTerm (RAM)     │  │
          │  │  LongTerm  (SQLite)  │  │
          │  └──────────────────────┘  │
          │                            │
          │  ┌──────────────────────┐  │
          │  │   KnowledgeBase      │  │
          │  │  SQLite FTS5 / BM25  │  │  ◀── load(file|url|text)
          │  └──────────────────────┘  │
          └────────────────────────────┘
```

---

## Install

```bash
pip install git+https://github.com/lavkeshdwivedi/kogniOS
```

Or from source (recommended for contributors):

```bash
git clone https://github.com/lavkeshdwivedi/kogniOS
cd kogniOS
pip install -e ".[dev]"
```

Requires **Python ≥ 3.11** and at least one provider key:

| Provider | Env var |
|---|---|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| xAI (Grok) | `XAI_API_KEY` |
| Ollama | no key (run locally) |

---

## 30-second quickstart

```python
from kognios import Agent, AnthropicModel, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"22°C and sunny in {city}."

agent = Agent(
    model=AnthropicModel(),
    tools=[get_weather],
    instructions="Be concise.",
)
print(agent.run("What's the weather in Paris?"))
# → "It's 22°C and sunny in Paris."
```

---

## Core primitives

### 1 · Agent + Tools (ReAct loop)

`@tool` reads your type hints to generate an OpenAI-compatible JSON schema automatically.
No decorating, no YAML, no config files.

```python
from kognios import Agent, AnthropicModel, tool

@tool
def add(x: int, y: int) -> int:
    """Add two integers."""
    return x + y

@tool
def multiply(x: int, y: int) -> int:
    """Multiply two integers."""
    return x * y

agent = Agent(model=AnthropicModel(), tools=[add, multiply])
print(agent.run("What is (3 + 4) × 6?"))
```

The ReAct loop runs **think → call tool → observe → repeat** until the model stops
calling tools, with a configurable `max_iterations` guard (default 10).

Tool calls time out after 30 seconds by default. Adjust per agent:

```python
agent = Agent(model=..., tools=[...], tool_timeout=10.0)  # stricter
agent = Agent(model=..., tools=[...], tool_timeout=None)  # no limit
```

---

### 2 · Memory

```python
from kognios import Agent, AnthropicModel, ShortTermMemory, LongTermMemory

short = ShortTermMemory(max_turns=20)   # sliding window, in RAM
long  = LongTermMemory(db_path="my.db") # SQLite key/value

long.store("user_name", "Alice")
long.store("location",  "Berlin")

agent = Agent(model=AnthropicModel(), memory=short)
# Known facts from long-term memory are auto-injected into the system prompt.
```

- **ShortTermMemory**: conversation history with sliding-window truncation
- **LongTermMemory**: persistent SQLite key/value; survives restarts

**Semantic long-term memory**: retrieve relevant facts by meaning, not key:

```python
from kognios import Agent, AnthropicModel, LongTermMemory

def my_embed(texts):
    import openai
    client = openai.OpenAI()
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in r.data]

long = LongTermMemory(db_path="my.db", embed_fn=my_embed)
long.store("user_goal", "Build a Python agent framework")
long.store("user_name", "Alice")

# retrieve the most relevant facts for a query
relevant = long.semantic_recall("What is the user building?", top_k=2)
# → [("user_goal", "Build a Python agent framework"), ...]
```

Requires `pip install 'kognios[vector]'` (numpy).

---

### 3 · Knowledge / RAG

```python
from kognios import Agent, AnthropicModel, SQLiteKnowledge

kb = SQLiteKnowledge(db_path="docs.db")
kb.load("docs/manual.txt")                           # file path
kb.load("https://example.com/spec", source_type="url")  # URL
kb.load("Inline text goes here…")                   # raw string
kb.load("report.pdf")                               # PDF (pip install 'kognios[pdf]')
kb.load("page.html")                                # HTML (tags stripped)
kb.load("data.csv")                                 # CSV (flattened to text)
kb.load("config.json")                              # JSON (flattened)
kb.load("report.docx")                              # Word doc (pip install 'kognios[docx]')
kb.load("https://github.com/owner/repo", source_type="github")  # GitHub repo

agent = Agent(model=AnthropicModel(), knowledge=kb)
print(agent.run("Summarise the key points."))
```

Chunks are indexed with **SQLite FTS5** and retrieved via **BM25 ranking**, no external vector database required.

---

### 4 · Team routing

```python
from kognios import Agent, Team, AnthropicModel

model = AnthropicModel()
agents = {
    "finance": Agent(model=model, name="finance",
                     description="Stocks, markets, financial data."),
    "weather": Agent(model=model, name="weather",
                     description="Weather forecasts and climate."),
}
team = Team(router_model=model, agents=agents)
print(team.run("What is Apple's stock price?"))
# → Router picks "finance", delegates the message, returns the answer.
```

---

## Multi-step planning

Ask an agent to plan and execute a complex goal in steps:

```python
agent = Agent(model=AnthropicModel(), tools=[web_search, calculator])
result = agent.plan_and_run("Research the top 3 Python web frameworks and compare their performance.")
print(result)
```

The agent first generates a numbered plan, then executes each step with accumulated context, and finally consolidates the results.

## Agent-to-agent messaging

Agents can talk directly to each other without a Team router:

```python
researcher = Agent(model=AnthropicModel(), name="researcher",
                   instructions="Research and summarize topics.")
writer = Agent(model=AnthropicModel(), name="writer",
               instructions="Write clear, engaging prose.")

# Researcher gathers info, writer polishes it
summary = researcher.run("Summarize the key features of SQLite.")
article = writer.send(researcher, f"Rewrite this as a blog intro: {summary}")
print(article)
```

---

## Streaming

```python
for chunk in agent.stream("Write me a haiku about SQLite."):
    print(chunk, end="", flush=True)
print()
```

---

## Async

```python
import asyncio
from kognios import Agent, AnthropicModel

async def main():
    agent = Agent(model=AnthropicModel())
    result = await agent.arun("What is the capital of Japan?")
    print(result)

asyncio.run(main())
```

Async streaming:

```python
async for chunk in agent.astream("Tell me a joke."):
    print(chunk, end="", flush=True)
```

---

## Structured output

```python
from pydantic import BaseModel
from kognios import Agent, AnthropicModel

class CityInfo(BaseModel):
    city: str
    country: str
    population_millions: float

agent = Agent(model=AnthropicModel())
info = agent.run("Tell me about Tokyo.", output_type=CityInfo)
print(info.city, info.population_millions)  # Tokyo  13.96
```

Requires `pip install 'kognios[structured]'`

---

## Providers

```python
from kognios.models.anthropic import AnthropicModel
from kognios.models.openai    import OpenAIModel
from kognios.models.groq      import GroqModel
from kognios.models.gemini    import GeminiModel
from kognios.models.ollama    import OllamaModel
from kognios.models.mistral   import MistralModel
from kognios.models.cohere    import CohereModel
from kognios.models.bedrock   import BedrockModel
from kognios.models.xai       import XAIModel
from kognios.models.together  import TogetherModel

model = AnthropicModel(model="claude-sonnet-4-6")
model = OpenAIModel(model="gpt-4o")
model = GroqModel(model="openai/gpt-oss-120b")                          # GROQ_API_KEY
model = GeminiModel(model="gemini-2.5-flash")                           # GEMINI_API_KEY
model = OllamaModel(model="llama3.3")                                   # needs Ollama running locally
model = MistralModel(model="mistral-large-latest")                      # MISTRAL_API_KEY
model = CohereModel(model="command-a-plus-05-2026")                     # COHERE_API_KEY
model = BedrockModel(model="anthropic.claude-3-7-sonnet-20250219-v1:0") # AWS_* env vars
model = XAIModel(model="grok-4")                                        # XAI_API_KEY
model = TogetherModel(model="deepseek-ai/DeepSeek-V3")                  # TOGETHER_API_KEY
```

All providers share the same `BaseModel` interface; one line to swap.

---

## ModelChain — multi-provider failover

Chain providers so a 429 on Groq auto-falls over to Gemini, Together, xAI, or Anthropic:

```python
from kognios import Agent, free_tier_chain

# Reads GROQ_API_KEY[_N], GEMINI_API_KEY[_N], TOGETHER_API_KEY[_N],
# XAI_API_KEY[_N], ANTHROPIC_API_KEY[_N] from environment.
# One model instance per (key, model) pair — independent cooldown buckets.
chain = free_tier_chain(preferred="groq")
agent = Agent(model=chain, tools=[...])
```

Or build an explicit chain:

```python
from kognios import ModelChain, GroqModel, GeminiModel, AnthropicModel

chain = ModelChain([
    GroqModel(model="openai/gpt-oss-120b"),
    GeminiModel(model="gemini-2.5-flash"),
    AnthropicModel(),
])
```

Features: 429-aware exponential backoff, per-model cooldown, dead-model tracking,
`<think>` block stripping, overall deadline, full streaming and async support.

---

## MCP (Model Context Protocol)

Connect any MCP server to your agent in two lines:

```python
from kognios import Agent, AnthropicModel, MCPClient

async def main():
    async with MCPClient.stdio("uvx", ["mcp-server-filesystem", "."]) as mcp:
        agent = Agent(model=AnthropicModel(), tools=mcp.tools())
        result = await agent.arun("List Python files in this directory.")
        print(result)
```

Sync path:

```python
with MCPClient.stdio("uvx", ["mcp-server-filesystem", "."]).sync() as mcp:
    agent = Agent(model=AnthropicModel(), tools=mcp.tools())
    print(agent.run("How many Python files are there?"))
```

Install the optional dep: `pip install 'kognios[mcp]'`

---

## Built-in tools

Import any of these into your agent:

```python
from kognios.tools.builtins import (
    calculator,        # safe AST-based math evaluator
    python_eval,       # sandboxed exec with stdout capture
    read_file,         # read any local file
    write_file,        # write file + create parent dirs
    http_get,          # fetch URL (urllib, no requests needed)
    web_search,        # DuckDuckGo search
    code_interpreter,  # isolated subprocess exec with timeout
    browse,            # JS-rendered page fetching (pip install 'kognios[browser]')
)
```

---

## CLI

```bash
# Interactive chat (streams by default)
kognios chat

# Choose provider and model
kognios chat -p openai -m gpt-4o

# One-shot Q&A over a local document
kognios ask README.md "What does this project do?"
```

---

## HTTP server

Serve any agent over HTTP with a single command:

```bash
pip install 'kognios[serve]'
kognios serve --provider anthropic --port 8000
```

Then call it:

```bash
curl -X POST http://localhost:8000/run \
     -H 'Content-Type: application/json' \
     -d '{"message": "What is the capital of France?"}'
# → {"response": "Paris.", "usage": {"input_tokens": 12, "output_tokens": 3}}
```

---

## Evaluation

Run your agent against a test dataset and measure quality:

```python
from kognios import Agent, AnthropicModel
from kognios.eval import AgentEvaluator, EvalCase, contains, exact_match

agent = Agent(model=AnthropicModel())

cases = [
    EvalCase(input="What is 2+2?", expected="4"),
    EvalCase(input="Capital of France?", expected="Paris"),
]

evaluator = AgentEvaluator(agent, scorer=contains)
report = evaluator.run(cases)
report.print_report()
# Cases: 2 | Pass rate: 100.0% | Mean score: 1.000 | Mean latency: 0.82s
```

Built-in scorers: `exact_match`, `contains`, `regex_match(pattern)`, `llm_judge(model)`.

Async parallel evaluation: `await evaluator.arun(cases)`.

---

## Guardrails

Validate or transform agent input/output with guardrails:

```python
from kognios import Agent, AnthropicModel, block_keywords, max_length, pii_scrubber, GuardrailError

agent = Agent(
    model=AnthropicModel(),
    input_guardrails=[
        block_keywords("jailbreak", "ignore previous instructions"),
        max_length(1000),
    ],
    output_guardrails=[pii_scrubber()],
)

try:
    result = agent.run("Tell me how to jailbreak this system")
except GuardrailError as e:
    print(f"Blocked: {e}")
```

Built-in guardrails: `block_keywords(*kws)`, `max_length(n)`, `pii_scrubber()`, `profanity_filter()`.

## Plugin system

Third-party packages can extend Kogni·OS with new providers, tools, and memory backends by declaring Python entry points:

```toml
# In your package's pyproject.toml:
[project.entry-points."kognios.providers"]
my_provider = "my_package:MyModel"

[project.entry-points."kognios.tools"]
my_tool = "my_package:my_tool_fn"
```

Then discover and load them at runtime:

```python
from kognios.plugins import load_plugins, list_plugins

providers = load_plugins("kognios.providers")
print(list_plugins("kognios.tools"))
```

---

## Tracing

Record every LLM call and tool execution as a span:

```python
from kognios import Agent, AnthropicModel
from kognios.tracing import Tracer

tracer = Tracer()
agent = Agent(model=AnthropicModel(), tracer=tracer)
agent.run("What is 2+2?")

tracer.print_spans()
# [TRACE] llm.complete 823.4ms model=claude-sonnet-4-6
```

---

## Examples

```bash
python examples/basic_agent.py
python examples/agent_with_tools.py
python examples/agent_with_memory.py
python examples/agent_with_knowledge.py
python examples/multi_agent_team.py
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

154 unit tests, all mocked, no API key required.

---

## Project layout

```
kognios/
├── agent.py              # Agent class + ReAct loop
├── team.py               # Team router
├── cli.py                # Click CLI
├── plugins.py            # Entry-point plugin discovery
├── tracing.py            # Tracer + span sinks
├── models/
│   ├── base.py           # BaseModel ABC, ModelResponse, ModelChunk
│   ├── chain.py          # ModelChain + free_tier_chain factory
│   ├── anthropic.py
│   ├── openai.py
│   ├── groq.py
│   ├── gemini.py
│   ├── ollama.py
│   ├── mistral.py
│   ├── cohere.py
│   ├── bedrock.py
│   ├── xai.py
│   └── together.py
├── tools/
│   ├── registry.py       # @tool decorator + ToolRegistry (with timeout)
│   └── builtins/         # calculator, python_eval, read_file, write_file,
│                         # http_get, web_search, code_interpreter, browse
├── memory/
│   ├── short_term.py     # sliding-window RAM buffer
│   └── long_term.py      # SQLite key/value + semantic recall
├── knowledge/
│   ├── sqlite_fts.py     # FTS5 / BM25 knowledge base
│   ├── numpy_vector.py   # cosine-similarity vector knowledge
│   └── loaders.py        # file/URL/PDF/CSV/JSON/DOCX/GitHub loaders
├── eval/                 # AgentEvaluator, EvalCase, EvalReport, scorers
├── guardrails/           # block_keywords, max_length, pii_scrubber
├── mcp/                  # MCPClient (stdio + SSE)
└── storage/
    └── sqlite.py         # shared connection + migrations
```

---

## Contributing

See [CONTRIBUTING.md](https://github.com/lavkeshdwivedi/kogniOS/blob/main/CONTRIBUTING.md) for setup instructions, coding conventions, and the PR process.

Good areas to contribute:
- New model providers
- Long-term memory backends (Redis, DynamoDB)
- Async-native `kognios eval --concurrent` flag
- OpenTelemetry sink for the Tracer

---

## Roadmap

See [ROADMAP.md](https://github.com/lavkeshdwivedi/kogniOS/blob/main/ROADMAP.md) for the full plan.

---

## License

MIT. See [LICENSE](https://github.com/lavkeshdwivedi/kogniOS/blob/main/LICENSE).

Built by [Lavkesh Dwivedi](https://github.com/lavkeshdwivedi).
