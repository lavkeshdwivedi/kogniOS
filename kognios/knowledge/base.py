from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class KnowledgeBase(ABC):
    @abstractmethod
    def load(self, source: str | Path, source_type: str = "text") -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[str]: ...
