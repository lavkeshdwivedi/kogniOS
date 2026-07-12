from __future__ import annotations

import os

from .openai import OpenAIModel


class GroqModel(OpenAIModel):
    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GROQ_API_KEY", "groq"),
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )
