from __future__ import annotations

import os
from collections.abc import Iterator

from .base import BaseModel, ModelChunk, ModelResponse


class BedrockModel(BaseModel):
    """Anthropic Claude models via AWS Bedrock.

    Requires: pip install 'kognios[bedrock]'
    Auth: set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
          (or use an IAM role / AWS profile).
    """

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-5",
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_region: str | None = None,
        **kwargs,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install 'kognios[bedrock]'")

        self.model = model
        self.kwargs = kwargs
        self._client = anthropic.AnthropicBedrock(
            aws_access_key=aws_access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_key=aws_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_region=aws_region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ModelResponse:
        params = self._base_params(messages, tools, system)
        response = self._client.messages.create(**params)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Iterator[ModelChunk]:
        params = self._base_params(messages, tools, system)
        with self._client.messages.stream(**params) as s:
            for text in s.text_stream:
                yield ModelChunk(text=text)
            final = s.get_final_message()
            tool_calls = _extract_tool_calls(final.content)
            yield ModelChunk(tool_calls=tool_calls, final=True)

    def _base_params(self, messages, tools, system) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.kwargs.get("max_tokens", 4096),
            "messages": messages,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [_to_anthropic_tool(t) for t in tools]
        return params

    def _parse_response(self, response) -> ModelResponse:
        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


def _extract_tool_calls(content_blocks) -> list[dict]:
    return [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in content_blocks
        if b.type == "tool_use"
    ]


def _to_anthropic_tool(schema: dict) -> dict:
    return {
        "name": schema["function"]["name"],
        "description": schema["function"].get("description", ""),
        "input_schema": schema["function"].get("parameters", {"type": "object", "properties": {}}),
    }
