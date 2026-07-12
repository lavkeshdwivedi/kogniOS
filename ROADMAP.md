# Roadmap

Tracks what shipped and what's planned. PRs welcome on any **Planned** item.

---

## Released

### v1.4.0 (current)
- [x] `HybridKnowledge(fts_kb, vector_kb, k=60)` — FTS5 + vector via Reciprocal Rank Fusion
- [x] `kognios ingest --backend hybrid` — single-command hybrid ingestion
- [x] `ShortTermMemory.compaction_model` — LLM-based summarisation of dropped turns
- [x] `ShortTermMemory.acompact()` — async compaction variant
- [x] `LongTermMemory.remember()` / `.forget()` — ergonomic aliases for `store()` / `delete()`
- [x] `LongTermMemory.semantic_recall()` fallback: LIKE search on key instead of arbitrary first-N rows
- [x] `AzureOpenAIModel` — Azure OpenAI provider (`openai.AzureOpenAI`), optional `api_key`
- [x] `VertexAIModel` — Google Vertex AI (Gemini) via `google-cloud-aiplatform`; `[vertexai]` extra

### v1.3.0
- [x] `Agent.max_tool_errors` — circuit breaker; raises after N consecutive `"Error: ..."` results from a tool
- [x] `Agent.max_context_tokens` — auto-trim oldest turns before each LLM call; prevents context overflow
- [x] Parallel sync tool calls in `Agent.run()` via `ThreadPoolExecutor` (matches `arun()`)
- [x] `Team.pipeline()` / `apipeline()` — sequential agent chain
- [x] `Team.broadcast()` / `abroadcast()` — parallel fan-out to all agents
- [x] `OpenAIModel.structured_complete()` uses native `response_format=json_object` (Groq inherits it)

### v1.2.0
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

### v1.5 — Performance & Developer Experience

#### Prompt Caching & Cost Reduction
- [ ] **Anthropic prompt caching (`cache_system=True`)**: when `AnthropicModel` is
  constructed with `cache_system=True`, wrap the system prompt block with
  `cache_control: {"type": "ephemeral"}` — ~90% cost reduction and ~85% latency
  reduction on repeated identical system prompts
- [ ] **Cache hit reporting in `agent.last_usage`**: surface `cache_read_input_tokens`
  and `cache_creation_input_tokens` from the Anthropic response
- [ ] **Tool result caching (`cache_tool_results=True` on `Agent`)**: deduplicate
  identical `(tool_name, frozen_args)` calls within a single `run()` / `arun()` turn —
  avoids redundant I/O for deterministic tools like `read_file` or `http_get`

#### Reliability
- [ ] **Tool retry (`tool_max_retries` on `Agent`)**: when a tool returns `"Error: ..."`
  retry up to N times with exponential backoff before incrementing the `max_tool_errors`
  counter — reduces false positives from transient network errors
- [ ] **Typed streaming events (`StreamEvent`)**: discriminated union
  `StreamEvent(kind: Literal["text","tool_start","tool_end","done"], ...)` via a new
  `agent.stream_events(message)` method; `agent.stream()` remains unchanged

#### Structured Output
- [ ] **Native structured output for Gemini**: `GeminiModel.structured_complete()` via
  the `response_schema` parameter (Gemini 2.x JSON Schema support)
- [ ] **Native structured output for Mistral**: `MistralModel.structured_complete()` via
  `response_format={"type": "json_object"}`
- [ ] **`ModelChain.structured_complete()`**: delegates to chain members in turn until
  one returns valid JSON; currently missing

#### CLI Enhancements
- [ ] **`--save-session` / `--load-session` flags on `kognios chat`**: persist
  `ShortTermMemory` to JSON between sessions using existing `.save()` / `.load()` methods
- [ ] **`/tools` slash command in chat REPL**: lists registered tool names; `/tool <name>`
  calls a tool directly for debugging
- [ ] **`--tools` flag on `kognios chat`**: comma-separated built-in tool names
  (`calculator,web_search,read_file`) attached at startup without writing code
- [ ] **`--max-context-tokens` flag on `kognios chat`**: expose `Agent.max_context_tokens`
  from the CLI

#### Web Search
- [ ] **Configurable search backend**: `KOGNIOS_SEARCH_BACKEND=ddg|serper|brave`; Serper
  and Brave unlock higher rate limits and richer snippets via `SERPER_API_KEY` /
  `BRAVE_SEARCH_API_KEY`
- [ ] **`web_search(query, max_chars=500)`**: snippet length control so agents can
  request briefer or fuller snippets depending on context budget

### v2.0 — Production Hardening
- [ ] **Agent checkpointing**: `save_agent(agent, path)` / `load_agent(path, model)` —
  serialise messages + `ShortTermMemory` to JSON for resume-after-crash or
  human-in-the-loop workflows; `LongTermMemory` and `NumpyVectorKnowledge` already
  persist to disk and are excluded from the snapshot
- [ ] **OpenTelemetry sink (`OtelTracer`)**: new class in `kognios/tracing.py` that
  translates `Span` objects to OTLP spans via `opentelemetry-sdk`; optional `[otel]`
  extra; configure endpoint via `OTEL_EXPORTER_OTLP_ENDPOINT`
- [ ] **`kognios serve` authentication middleware**: `--api-key` flag generates a
  bearer-token check on all endpoints; also accepts `KOGNIOS_SERVE_API_KEY` env var
- [ ] **`POST /team` endpoint on `kognios serve`**: accepts `{message, agents: [...]}`
  and invokes `Team.run()` server-side — useful for demos without writing Python
- [ ] **`AsyncLongTermMemory`**: `aiosqlite`-backed variant so `Agent.arun()` never
  blocks the event loop on fact storage or retrieval; optional `[async-db]` extra

---

## Out of scope

- LangChain / LlamaIndex compatibility shims — the whole point is to stay lean
- GUI / visual workflow builder
- Proprietary cloud hosting — kognios is a library, not a service
- Bundled vector database servers (Pinecone, Weaviate, Qdrant) — use the plugin system;
  `NumpyVectorKnowledge` is the reference implementation for local vector search

---

PRs welcome. Open an [issue](https://github.com/lavkeshdwivedi/kogniOS/issues) and tag it `roadmap`.
