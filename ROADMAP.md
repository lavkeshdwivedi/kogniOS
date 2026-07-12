# Roadmap

Tracks what shipped and what's planned. PRs welcome on any **Planned** item.

---

## Released

### v1.2.0 (current)
- [x] `ModelChain.astream()` — async streaming across provider chain
- [x] `ModelChain.stream()` deadline enforcement (matches `complete()` robustness)
- [x] `Agent.tool_timeout` — 30 s default; hung tools cancelled and return an error string
- [x] xAI / Grok in `free_tier_chain()` (`grok-4`, `grok-3`, `grok-3-mini`)
- [x] `_collect_keys` scans all N env var slots (no longer stops at first gap)
- [x] `AnthropicModel` default corrected to `claude-sonnet-4-6`
- [x] `GroqModel` default updated to `openai/gpt-oss-120b`; retired models removed from pool

### v1.1
- [x] `kognios serve` streaming (SSE) + session management
- [x] `kognios eval` + `kognios ingest` CLI commands
- [x] Smart fact injection via `LongTermMemory.semantic_recall`
- [x] Prometheus Tracer sink
- [x] `NumpyVectorKnowledge` + `LongTermMemory.semantic_recall`

### v1.0
- [x] Stable public API + semver guarantees
- [x] Plugin system (Python entry points)
- [x] Evaluation harness (`AgentEvaluator`, 4 built-in scorers)
- [x] Guardrails (`block_keywords`, `max_length`, `pii_scrubber`, `profanity_filter`)
- [x] Tracer / observability with span sinks

### v0.3
- [x] Providers: Ollama, Mistral, Cohere, AWS Bedrock, xAI, Together
- [x] `ModelChain` + `free_tier_chain()` multi-provider failover with 429 backoff
- [x] Parallel async tool execution (`asyncio.gather`)
- [x] `agent.last_usage` token counting
- [x] `ShortTermMemory.save/load` persistence
- [x] `Agent.plan_and_run()` / `aplan_and_run()` multi-step planning
- [x] `Agent.send()` / `asend()` agent-to-agent messaging
- [x] Knowledge loaders: HTML, CSV, JSON, DOCX, GitHub repo
- [x] `browse` (Playwright + urllib fallback) and `code_interpreter` builtins
- [x] `MCPClient` (stdio + SSE)

### v0.2 (initial release)
- [x] Agent + ReAct loop + `@tool` decorator
- [x] `ShortTermMemory`, `LongTermMemory` (SQLite)
- [x] `SQLiteKnowledge` (FTS5 / BM25)
- [x] Team routing
- [x] Streaming + async
- [x] Structured output (Pydantic)
- [x] Providers: Anthropic, OpenAI, Groq, Gemini

---

## Planned

### v1.3 — Reliability
- [ ] **Context window management**: `max_context_tokens` on `Agent`; auto-trim oldest turns
  before each LLM call so long sessions never crash on context overflow
- [ ] **Native JSON mode**: `OpenAIModel.structured_complete()` uses `response_format=json_object`
  instead of prompt injection (Groq and Gemini inherit it via `OpenAIModel`)
- [ ] **Tool retry guard**: `max_tool_errors: int = 3` stops infinite retry when a tool
  keeps returning `"Error: ..."` — currently the LLM can loop forever on broken tools

### v1.4 — RAG & Memory
- [ ] **Hybrid search**: FTS5 + vector combined via Reciprocal Rank Fusion — no new dependency
- [ ] **Memory compaction**: optional LLM-based summarisation of dropped turns instead of
  silent deletion (`compaction_model` param on `ShortTermMemory`)
- [ ] **Semantic recall fallback fix**: `LongTermMemory.semantic_recall()` falls back to
  key-substring LIKE search instead of returning arbitrary first-N rows

### v1.5 — Multi-agent
- [ ] **Parallel tool calls in sync `agent.run()`**: `ThreadPoolExecutor` when >1 tool call
  arrives (async already does this via `asyncio.gather`)
- [ ] **`Team.pipeline(msg)`**: linear chain — output of agent N is input to agent N+1
- [ ] **`Team.broadcast(msg)`**: fan-out to all agents, return `dict[name, result]`
- [ ] **Prompt caching (Anthropic)**: `cache_control: ephemeral` on system messages when
  `cache_system=True` (~90% cost reduction on repeated identical system prompts)

### v2.0 — Production
- [ ] **Agent checkpointing**: `save_agent(agent, path)` / `load_agent(path, model)` —
  serialise messages + memory to JSON for resume-after-crash or human-in-the-loop
- [ ] **OpenTelemetry sink**: `OTelTracer` class in `kognios/tracing.py`, optional dep

---

## Out of scope

- LangChain / LlamaIndex compatibility shims — the whole point is to stay lean
- GUI / visual workflow builder
- Proprietary cloud hosting — kognios is a library, not a service

---

PRs welcome. Open an [issue](https://github.com/lavkeshdwivedi/kogniOS/issues) and tag it `roadmap`.
