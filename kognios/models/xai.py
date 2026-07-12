from __future__ import annotations

import os

from .openai import OpenAIModel


class XAIModel(OpenAIModel):
    """xAI Grok via its OpenAI-compatible API."""

    def __init__(self, model: str = "grok-4", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("XAI_API_KEY", "xai"),
            base_url="https://api.x.ai/v1",
            **kwargs,
        )
