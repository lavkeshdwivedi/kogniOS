from __future__ import annotations

from pathlib import Path

from ..registry import tool


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file, creating parent directories as needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error: {exc}"
