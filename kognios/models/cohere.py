from __future__ import annotations

import os

from .openai import OpenAIModel


class CohereModel(OpenAIModel):
    """Cohere Command R+ via its OpenAI-compatible v2 API."""

    def __init__(self, model: str = "command-a-plus-05-2026", **kwargs):
        super().__init__(
            model=model,
            api_key=os.environ.get("COHERE_API_KEY", "cohere"),
            base_url="https://api.cohere.com/compatibility/v1",
            **kwargs,
        )
