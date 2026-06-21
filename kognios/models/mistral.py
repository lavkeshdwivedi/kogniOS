from __future__ import annotations

import os

from .openai import OpenAIModel


class MistralModel(OpenAIModel):
    """Mistral AI via its OpenAI-compatible API."""

    def __init__(self, model: str = "mistral-large-latest", **kwargs):
        super().__init__(
            model=model,
            api_key=os.environ.get("MISTRAL_API_KEY", "mistral"),
            base_url="https://api.mistral.ai/v1",
            **kwargs,
        )
