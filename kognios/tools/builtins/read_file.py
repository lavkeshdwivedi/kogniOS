from __future__ import annotations

from pathlib import Path

from ..registry import tool


@tool
def read_file(path: str) -> str:
    """Read the contents of a file and return it as text."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as exc:
        return f"Error: {exc}"
