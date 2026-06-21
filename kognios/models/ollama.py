from __future__ import annotations

import os

from .openai import OpenAIModel


class OllamaModel(OpenAIModel):
    """Local Ollama model via its OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama3.3",
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key="ollama",
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            **kwargs,
        )
