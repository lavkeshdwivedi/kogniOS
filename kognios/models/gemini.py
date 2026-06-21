from __future__ import annotations

import os

from .openai import OpenAIModel


class GeminiModel(OpenAIModel):
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GEMINI_API_KEY", "gemini"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            **kwargs,
        )
