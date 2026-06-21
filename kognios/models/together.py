from __future__ import annotations

import os

from .openai import OpenAIModel


class TogetherModel(OpenAIModel):
    def __init__(
        self, model: str = "deepseek-ai/DeepSeek-V3", api_key: str | None = None, **kwargs
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
            **kwargs,
        )
