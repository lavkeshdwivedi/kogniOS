from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.base import BaseModel as LLMModel


class ShortTermMemory:
    def __init__(
        self,
        max_turns: int = 20,
        compaction_model: "LLMModel | None" = None,
    ):
        self.max_turns = max_turns
        self.compaction_model = compaction_model
        self._messages: list[dict] = []

    def append(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            if self.compaction_model is not None:
                dropped = self._messages[:-limit]
                summary = self._compact_sync(dropped)
                self._messages = [
                    {"role": "system", "content": f"Prior conversation summary: {summary}"}
                ] + self._messages[-limit:]
            else:
                self._messages = self._messages[-limit:]

    def _compact_sync(self, dropped: list[dict]) -> str:
        turns_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in dropped)
        response = self.compaction_model.complete(  # type: ignore[union-attr]
            messages=[
                {
                    "role": "user",
                    "content": f"Summarise this conversation in 2-3 sentences:\n\n{turns_text}",
                }
            ],
            system="You are a concise summariser. Preserve key facts and decisions.",
        )
        return response.content

    async def acompact(self, dropped: list[dict]) -> str:
        """Async variant of _compact_sync for callers on an async event loop."""
        turns_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in dropped)
        response = await self.compaction_model.acomplete(  # type: ignore[union-attr]
            messages=[
                {
                    "role": "user",
                    "content": f"Summarise this conversation in 2-3 sentences:\n\n{turns_text}",
                }
            ],
            system="You are a concise summariser. Preserve key facts and decisions.",
        )
        return response.content

    def messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def save(self, path: str) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._messages, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        p = pathlib.Path(path)
        if p.exists():
            self._messages = json.loads(p.read_text(encoding="utf-8"))
