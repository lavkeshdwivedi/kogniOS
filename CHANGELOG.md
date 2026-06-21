# Changelog

All notable changes to Kogni·OS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [1.1.4] - 2026-07-08

### Fixed
- README badge and inline links now use absolute GitHub URLs so they resolve correctly on PyPI (LICENSE, CONTRIBUTING.md, ROADMAP.md).

---

## [1.1.3] - 2026-07-08

### Changed
- Default models updated: Anthropic `claude-sonnet-5`, Bedrock `anthropic.claude-sonnet-5`.

---

## [1.1.2] - 2026-07-08

### Added
- **xAI provider** (`XAIModel`) via the OpenAI-compatible `https://api.x.ai/v1` endpoint; set `XAI_API_KEY`.

### Fixed
- Provider models (Mistral, Cohere, Groq, Gemini) no longer require `OPENAI_API_KEY` when their own key is absent; each now falls back to a provider-named placeholder so the client initialises cleanly.
- `numpy`, `fastapi`, and `uvicorn` added to dev extras so CI runs the full test suite without optional extras installed separately.

### Changed
- Default models updated: Groq `llama-4-scout`, Gemini `gemini-2.5-flash`, Cohere `command-a-plus-05-2026`, xAI `grok-4`, Bedrock `anthropic.claude-3-7-sonnet-20250219-v1:0`, Ollama `llama3.3`.

---

## [1.1.1] - 2026-07-07

### Changed
- Updated README: correct test count (154), refreshed Contributing section.

---

## [1.1.0] - 2026-07-07

### Added
- **`kognios serve` streaming**: `POST /stream` SSE endpoint; each token delivered as `data: {"text": "..."}`, terminated with `data: [DONE]`.
- **`kognios serve` session management**: `POST /session` creates a named session backed by `ShortTermMemory`; `POST /session/{id}/run` and `POST /session/{id}/stream` are stateful across turns; `DELETE /session/{id}` cleans up.
- **`kognios serve --kb <db>`**: attach a `SQLiteKnowledge` knowledge base (created by `kognios ingest`) to the served agent; all endpoints (stateless and session) use it for RAG.
- **`kognios serve` health endpoint**: `GET /health` returns `{"status": "ok", "provider": ..., "model": ...}`.
- **All 8 providers in `kognios serve`**: mistral, cohere, and bedrock now available via `--provider`.
- **`kognios eval <dataset.json>`**: CLI command to run a JSON eval dataset against any provider/model; supports `--scorer contains|exact_match|regex_match` and `--threshold`.
- **`kognios ingest <file>`**: CLI command to ingest documents (plain text, PDF, HTML, CSV, JSON, DOCX, URL) into a persistent SQLite knowledge base (`--backend fts|vector`). Reports chunk count on completion.
- **Smart fact injection**: `Agent._build_system` calls `LongTermMemory.semantic_recall(query, top_k=5)` instead of `all_facts()` when `embed_fn` is set, so only query-relevant facts enter the system prompt.
- **`LongTermMemory` safe as agent memory**: `Agent.run` and `arun` now guard `memory.append` with `hasattr`, so `LongTermMemory` can be passed as the `memory` argument without raising `AttributeError`.
- **`KnowledgeBase.load()` returns chunk count**: `SQLiteKnowledge.load()` and `NumpyVectorKnowledge.load()` now return `int` (number of chunks stored).

---

## [Unreleased - pre-1.1 additions]

### Added
- **Prometheus Tracer sink**: `prometheus_sink()` factory in `kognios/tracing.py`.
  Records span duration as `kognios_span_duration_seconds` (Histogram) and errors as
  `kognios_span_errors_total` (Counter), both labelled by `span.name`.
  Requires `prometheus-client>=0.17`; install via `pip install 'kognios[prometheus]'`.
  New `prometheus` optional extra added to `pyproject.toml`; also included in `[all]`.
- **Redis memory plugin example**: `examples/redis_memory_plugin/` with a
  `RedisLongTermMemory` class duck-typing the `LongTermMemory` interface
  (`remember`, `recall`, `facts`), Redis hash storage, optional TTL, lazy import,
  entry-point registration under `kognios.memory`, and 15 unit tests requiring no
  real Redis server.
- **PyPI build readiness**: `python -m build` produces `dist/kognios-1.0.0.whl`
  and `dist/kognios-1.0.0.tar.gz`; both pass `python -m twine check dist/*`.
  Ready to upload with `python -m twine upload dist/*`.

### Changed
- Default `pytest` run now excludes integration tests automatically via
  `addopts = "-m 'not integration'"` in `pyproject.toml`. Run integration
  tests explicitly with `pytest -m integration`.

---

## [1.0.0] - 2026-07-06

### Added
- **Plugin system**: third-party providers/tools/memory discoverable via Python entry points (`kognios.providers`, `kognios.tools`, `kognios.memory`)
- **`__version__`** attribute (importlib.metadata)
- **Stable public API** with semantic versioning guarantees from this release onward

### v1.0 features (all shipped in this release)
- `AgentEvaluator`: run test datasets through an agent; scorers: `exact_match`, `contains`, `regex_match`, `llm_judge`
- Guardrails: `block_keywords`, `max_length`, `pii_scrubber`, `profanity_filter`; `input_guardrails`/`output_guardrails` on `Agent`
- `Tracer`: span-based observability for LLM calls and tool executions; sink callbacks and OpenTelemetry stub

---

## [0.3.0] - 2026-07-06

### Added
- **Providers**: Ollama (local), Mistral, Cohere Command R+, AWS Bedrock
- **Parallel tool execution**: `asyncio.gather` across tool calls in `arun`/`astream`
- **Token counting**: `agent.last_usage` dict accumulated across the full run
- **Persistent sessions**: `ShortTermMemory.save(path)` / `.load(path)`
- **`NumpyVectorKnowledge`**: cosine similarity semantic search with pluggable `embed_fn`
- **`LongTermMemory.semantic_recall(query)`**: semantic search over stored facts
- **Knowledge loaders**: HTML, CSV, JSON, DOCX, GitHub repo crawler
- **Multi-step planning**: `Agent.plan_and_run()` / `aplan_and_run()`
- **Agent-to-agent messaging**: `Agent.send()` / `asend()`
- **`code_interpreter` builtin**: subprocess-isolated exec with timeout
- **`browse` builtin**: Playwright (headless Chromium) with urllib fallback
- **MCP client**: `MCPClient` connects agents to any MCP server via stdio or SSE
- **`kognios serve`**: FastAPI `POST /run` endpoint
- 115 tests (up from 38)

---

## [0.2.0] - 2026-06-01 (initial public release)

### Added
- `Agent` with ReAct loop, `@tool` decorator, JSON schema auto-gen
- `ShortTermMemory` (sliding-window RAM buffer)
- `LongTermMemory` (SQLite key/value, auto-injected into system prompt)
- `SQLiteKnowledge`: FTS5 BM25 full-text search
- `Team` routing: router LLM picks named sub-agent
- Streaming (`agent.stream()`) and async (`agent.arun()`, `agent.astream()`)
- Structured output via Pydantic models
- Providers: Anthropic, OpenAI, Groq, Gemini
- Built-in tools: `calculator`, `python_eval`, `read_file`, `write_file`, `http_get`, `web_search`
- CLI: `kognios chat`, `kognios ask`
- PDF ingestion (`kognios[pdf]`)
- 38 unit tests
