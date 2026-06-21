---
name: Bug report
about: Something isn't working as expected
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug

A clear, concise description of what went wrong.

## Minimal reproduction

```python
# paste the smallest snippet that reproduces the issue
from kognios import Agent, AnthropicModel

agent = Agent(model=AnthropicModel())
agent.run("...")
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the **full traceback** if applicable.

```
Traceback (most recent call last):
  ...
```

## Environment

| Item | Version |
|---|---|
| Python | `python --version` |
| kognios | `pip show kognios` |
| Provider | Anthropic / OpenAI / Groq / Gemini |
| Model | e.g. claude-sonnet-4-6 |
| OS | Windows / macOS / Linux |

## Additional context

Anything else that might help — screenshots, related issues, etc.
