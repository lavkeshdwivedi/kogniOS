# Roadmap

This document tracks what's done, what's in progress, and what's planned.
Ideas and PRs are welcome for any item marked **Planned**.

---

## v0.2 (current release)

- [x] Agent + Tools (ReAct loop, `@tool` decorator, JSON schema auto-gen)
- [x] ShortTermMemory (sliding-window RAM buffer)
- [x] LongTermMemory (SQLite key/value, auto-injected into system prompt)
- [x] Knowledge / RAG (SQLite FTS5, BM25 ranking, 512-token chunks)
- [x] Team routing (router LLM picks named sub-agent)
- [x] Streaming (`agent.stream()`, `Iterator[str]`)
- [x] Async (`agent.arun()`, `agent.astream()`)
- [x] Structured output (Pydantic models via forced tool_use / JSON schema injection)
- [x] Providers: Anthropic, OpenAI, Groq, Gemini
- [x] Built-in tools: calculator, python_eval, read_file, write_file, http_get, web_search
- [x] CLI: `kognios chat`, `kognios ask`
- [x] PDF ingestion (optional `pip install 'kognios[pdf]'`)
- [x] MCP client (`MCPClient`): connect agents to any MCP server via stdio or SSE
- [x] 115 unit tests, all mocked

---

## v0.3 (near-term)

- [x] **Parallel tool execution**: run independent tool calls concurrently with `asyncio.gather`
- [x] **Token counting**: expose `usage` dict on `agent.last_usage` for cost tracking
- [x] **Persistent agent sessions**: save/restore short-term memory to disk (`memory.save(path)` / `memory.load(path)`)
- [x] **Ollama** (local models) provider
- [ ] **More providers**
  - [x] Cohere (Command R+)
  - [x] Mistral
  - [x] AWS Bedrock
- [x] **Vector knowledge backend**: `NumpyVectorKnowledge` with numpy cosine similarity and pluggable embedding function
- [x] **`kognios serve`** CLI command: minimal FastAPI HTTP wrapper

---

## v0.4 (medium-term)

- [x] **Multi-step planning**: agent can create and execute a step-by-step plan before answering
- [x] **Long-term memory retrieval**: semantic search over facts via `LongTermMemory.semantic_recall(query)`
- [x] **Knowledge loaders**: HTML, CSV/JSON, DOCX/ODT, GitHub repo
- [x] **Browser tool**: Playwright-based `browse(url)` with urllib fallback
- [x] **Code interpreter tool**: isolated subprocess with a timeout
- [x] **Agent-to-agent messaging**: direct peer calls without a Team router (`agent.send(other_agent, msg)`)

---

## v1.0 (longer-term)

- [x] **Stable public API** with semantic versioning guarantees from v1.0.0
- [x] **Plugin system**: `load_plugins("kognios.providers")` / `list_plugins()` via Python entry points
- [x] **Tracing / observability**: `Tracer` records LLM calls and tool executions as spans; `print_spans()` for debugging
- [x] **Evaluation harness**: `AgentEvaluator` runs a test set through an agent; scorers: `exact_match`, `contains`, `regex_match`, `llm_judge`
- [x] **Guardrails**: input/output validation hooks: `block_keywords`, `max_length`, `pii_scrubber`, `profanity_filter`

---

## v1.1 (production serving )+ CLI completeness

- [x] **`kognios serve` streaming**: `POST /stream` SSE endpoint for real-time token delivery
- [x] **`kognios serve` session management**: `POST /session`, `POST /session/{id}/run`, `POST /session/{id}/stream`, `DELETE /session/{id}` for stateful HTTP conversations
- [x] **`kognios serve` provider completeness**: all 8 providers available (added mistral, cohere, bedrock)
- [x] **`kognios serve` health endpoint**: `GET /health`
- [x] **`kognios serve --kb`**: attach a persistent knowledge base to the served agent
- [x] **`kognios eval` CLI command**: run a JSON eval dataset against any agent from the command line
- [x] **`kognios ingest` CLI command**: ingest documents (text, PDF, HTML, CSV, JSON, DOCX, URL) into a persistent knowledge base; reports chunk count on completion
- [x] **`KnowledgeBase.load()` returns chunk count**: both FTS and vector backends return `int`
- [x] **Smart fact injection**: `_build_system` uses `semantic_recall(query, top_k=5)` instead of `all_facts()` when `LongTermMemory.embed_fn` is set
- [x] **`LongTermMemory` safe as agent memory**: `Agent.run/arun` guards `memory.append` with `hasattr`, so `LongTermMemory` no longer crashes when passed as the `memory` argument

---

## Out of scope (won't do)

- LangChain / LlamaIndex compatibility shims : the whole point is to stay lean
- GUI / visual workflow builder
- Proprietary cloud hosting : Kogni·OS is a library, not a service

---

Want to pick something up? Comment on the relevant
[issue](https://github.com/lavkeshdwivedi/kogniOS/issues) or open a new one
and tag it `roadmap`.
