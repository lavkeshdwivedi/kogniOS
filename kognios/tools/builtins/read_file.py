from __future__ import annotations

from pathlib import Path

from ..registry import tool

DEFAULT_MAX_CHARS = 20_000


def read_text_capped(path: Path, max_chars: int) -> str:
    # read one character past the cap so truncation is detectable without loading a huge file
    with path.open(encoding="utf-8", errors="replace") as f:
        text = f.read(max_chars + 1)
    if len(text) > max_chars:
        return f"{text[:max_chars]}\n[truncated: file is longer than {max_chars} chars]"
    return text


@tool
def read_file(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Read a file and return it as text, cut off at max chars (default 20000)."""
    try:
        return read_text_capped(Path(path), int(max_chars))
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as exc:
        return f"Error: {exc}"
