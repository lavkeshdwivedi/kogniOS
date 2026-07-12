from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator

import openai

from .base import BaseModel, ModelChunk, ModelResponse


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.kwargs = kwargs
        client_kwargs: dict = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)

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
        response = self._client.chat.completions.create(**params)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Iterator[ModelChunk]:
        params = self._base_params(messages, tools, system)
        params["stream"] = True
        accumulated_tools: dict[int, dict] = {}

        for chunk in self._client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                yield ModelChunk(text=delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tools:
                        accumulated_tools[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        accumulated_tools[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        accumulated_tools[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        accumulated_tools[idx]["arguments"] += tc.function.arguments

        tool_calls = [
            {"id": v["id"], "name": v["name"], "input": json.loads(v["arguments"] or "{}")}
            for v in accumulated_tools.values()
        ]
        yield ModelChunk(tool_calls=tool_calls, final=True)

    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ModelResponse:
        params = self._base_params(messages, tools, system)
        response = await self._async_client.chat.completions.create(**params)
        return self._parse_response(response)

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> AsyncIterator[ModelChunk]:
        params = self._base_params(messages, tools, system)
        params["stream"] = True
        accumulated_tools: dict[int, dict] = {}

        async for chunk in await self._async_client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                yield ModelChunk(text=delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tools:
                        accumulated_tools[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        accumulated_tools[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        accumulated_tools[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        accumulated_tools[idx]["arguments"] += tc.function.arguments

        tool_calls = [
            {"id": v["id"], "name": v["name"], "input": json.loads(v["arguments"] or "{}")}
            for v in accumulated_tools.values()
        ]
        yield ModelChunk(tool_calls=tool_calls, final=True)

    def _base_params(self, messages, tools, system) -> dict:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        params: dict = {"model": self.model, "messages": msgs, **self.kwargs}
        if tools:
            params["tools"] = tools
        return params

    def structured_complete(
        self,
        messages: list[dict],
        schema: dict | None = None,
        system: str = "",
    ) -> dict:
        params = self._base_params(messages, None, system)
        params["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**params)
        return json.loads(response.choices[0].message.content or "{}")

    def _parse_response(self, response) -> ModelResponse:
        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments or "{}"),
                    }
                )
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return ModelResponse(content=msg.content or "", tool_calls=tool_calls, usage=usage)
