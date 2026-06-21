from __future__ import annotations

import json
import pathlib


class ShortTermMemory:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._messages: list[dict] = []

    def append(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        # Keep at most max_turns pairs (user + assistant = 2 messages per turn).
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

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
