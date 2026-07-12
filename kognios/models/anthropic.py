from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import anthropic

from .base import BaseModel, ModelChunk, ModelResponse


class AnthropicModel(BaseModel):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._async_client = (
            anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        )

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        **call_kwargs,
    ) -> ModelResponse:
        params = self._base_params(messages, tools, system)
        if call_kwargs:
            params.update(call_kwargs)
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

    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ModelResponse:
        params = self._base_params(messages, tools, system)
        response = await self._async_client.messages.create(**params)
        return self._parse_response(response)

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> AsyncIterator[ModelChunk]:
        params = self._base_params(messages, tools, system)
        async with self._async_client.messages.stream(**params) as s:
            async for text in s.text_stream:
                yield ModelChunk(text=text)
            final = await s.get_final_message()
            tool_calls = _extract_tool_calls(final.content)
            yield ModelChunk(tool_calls=tool_calls, final=True)

    def structured_complete(
        self,
        messages: list[dict],
        schema: dict,
        system: str = "",
    ) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.kwargs.get("max_tokens", 4096),
            "messages": messages,
            "tools": [
                {
                    "name": "_output",
                    "description": "Return structured output.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "_output"},
        }
        if system:
            params["system"] = system
        response = self._client.messages.create(**params)
        for block in response.content:
            if block.type == "tool_use" and block.name == "_output":
                return block.input
        raise RuntimeError("Structured output tool was not called by the model")

    def _base_params(self, messages, tools, system) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.kwargs.get("max_tokens", 4096),
            "messages": messages,
            **{k: v for k, v in self.kwargs.items() if k != "max_tokens"},
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
