from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class ModelChunk:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    final: bool = False


class BaseModel(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        **call_kwargs,
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Iterator[ModelChunk]:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ModelResponse:
        return await asyncio.to_thread(self.complete, messages, tools=tools, system=system)

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> AsyncIterator[ModelChunk]:
        raise NotImplementedError(f"{type(self).__name__} does not support async streaming")
        # make this a valid async generator (unreachable but required by type checker)
        yield  # type: ignore[misc]

    def structured_complete(
        self,
        messages: list[dict],
        schema: dict,
        system: str = "",
    ) -> dict:
        """Default: inject schema into system prompt and parse JSON. Override for better reliability."""
        schema_str = json.dumps(schema, indent=2)
        augmented = (system + "\n\n" if system else "") + (
            f"Respond with a JSON object exactly matching this schema:\n{schema_str}\n"
            "Return only valid JSON. No markdown fences, no prose."
        )
        response = self.complete(messages, system=augmented)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
