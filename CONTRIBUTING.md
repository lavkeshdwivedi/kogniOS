# Contributing to Kogni·OS

Thank you for considering a contribution! This document covers everything
you need to go from idea → merged PR.

---

## Table of contents

1. [Quick setup](#quick-setup)
2. [Project structure](#project-structure)
3. [Running tests](#running-tests)
4. [Coding conventions](#coding-conventions)
5. [Adding a new model provider](#adding-a-new-model-provider)
6. [Adding a new built-in tool](#adding-a-new-built-in-tool)
7. [Submitting a PR](#submitting-a-pr)
8. [Reporting bugs](#reporting-bugs)

---

## Quick setup

```bash
# Windows (recommended local path)
cd C:\Claude\Projects
git clone https://github.com/lavkeshdwivedi/kogniOS
cd kogniOS

# macOS / Linux
git clone https://github.com/lavkeshdwivedi/kogniOS
cd kogniOS

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite (no API key needed — all mocked)
pytest tests/ -v
```

Python **3.11 or newer** is required.

---

## Project structure

```
kognios/          ← the importable package
  agent.py        ← Agent class + ReAct loop
  team.py         ← Team router
  cli.py          ← Click CLI (kognios chat / kognios ask)
  models/         ← provider adapters
  tools/          ← @tool decorator + built-in tools
  memory/         ← short-term (RAM) + long-term (SQLite)
  knowledge/      ← FTS5 RAG pipeline
  storage/        ← shared SQLite connection + migrations
examples/         ← runnable demos (require an API key)
tests/            ← unit tests (mocked, no key needed)
```

---

## Running tests

```bash
pytest tests/ -v                # all tests
pytest tests/test_agent.py -v  # single file
pytest -k "memory" -v          # filter by name
```

Tests live in `tests/` and use `unittest.mock`. No external services are
called — every LLM response is a `MagicMock`.

To lint:

```bash
ruff check kognios tests
ruff format --check kognios tests
```

---

## Coding conventions

- **Python 3.11+** syntax — use `X | Y` unions, `list[T]`, `dict[K, V]`.
- **`from __future__ import annotations`** at the top of every module.
- **No comments** unless the *why* is genuinely non-obvious. Good names are enough.
- **No extra abstractions** — three similar lines beat a premature helper.
- **No new dependencies** without a strong reason. SQLite + stdlib is the baseline.
- Line length: **100** (enforced by `ruff`).
- Format with `ruff format` before opening a PR.

---

## Adding a new model provider

1. Create `kognios/models/<name>.py`.
2. Subclass `BaseModel` from `kognios.models.base`.
3. Implement `complete()` — required abstract method.
4. Optionally override `stream()`, `acomplete()`, `astream()`, `structured_complete()`.
5. Export from `kognios/__init__.py`.
6. Add a row to the provider table in `README.md`.

Minimal example (OpenAI-compatible endpoint):

```python
# kognios/models/cohere.py
from __future__ import annotations
import os
from .openai import OpenAIModel

class CohereModel(OpenAIModel):
    def __init__(self, model: str = "command-r-plus", **kwargs):
        super().__init__(
            model=model,
            api_key=os.environ.get("COHERE_API_KEY", ""),
            base_url="https://api.cohere.com/compatibility/v1",
            **kwargs,
        )
```

If the provider uses a non-OpenAI wire format, subclass `BaseModel` directly
and look at `kognios/models/anthropic.py` as a reference.

Add a test in `tests/test_agent.py` or a new `tests/test_<name>_model.py` that
mocks the underlying SDK call and asserts `ModelResponse` fields.

---

## Adding a new built-in tool

1. Create `kognios/tools/builtins/<name>.py`.
2. Decorate your function with `@tool` from `kognios.tools.registry`.
3. Add a type-hinted docstring — the first line becomes the tool description.
4. Export from `kognios/tools/builtins/__init__.py`.
5. Add tests to `tests/test_builtins.py`.

```python
# kognios/tools/builtins/uuid_gen.py
from __future__ import annotations
import uuid
from ..registry import tool

@tool
def generate_uuid() -> str:
    """Generate a random UUID v4."""
    return str(uuid.uuid4())
```

---

## Submitting a PR

1. **Fork** the repo and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, write tests, run `pytest` and `ruff check`.
3. Commit with a clear message:
   ```
   feat: add CohereModel provider
   fix: handle empty tool_calls in OpenAI streaming
   docs: add structured output example to README
   ```
4. Push and open a pull request against `main`.
5. Fill in the PR template — describe *what* changed and *why*.

PRs that pass CI and include tests will be reviewed promptly.

---

## Reporting bugs

Open a [GitHub Issue](https://github.com/lavkeshdwivedi/kogniOS/issues/new/choose)
using the **Bug report** template. Include:

- Python version (`python --version`)
- Kogni·OS version (`pip show kognios`)
- Provider and model you were using
- Minimal reproduction snippet
- Full traceback

---

## Code of conduct

Be kind and constructive. Disagreements about code are fine; personal attacks
are not. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Publishing to PyPI

Build and publish (maintainers only):

```bash
# Install build tools
pip install build twine

# Build wheel + sdist
python -m build

# Upload to PyPI (requires PyPI API token)
twine upload dist/*

# Or upload to TestPyPI first:
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ kognios
```

Users can install directly from PyPI:
```bash
pip install kognios
pip install 'kognios[all]'   # all optional deps
```

---

## Running integration tests

Unit tests (default, no API keys needed):
```bash
pytest tests/ -m "not integration"
```

Integration tests (require real API keys):
```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
pytest tests/ -m integration -v
```
