from __future__ import annotations

from pathlib import Path

from ..registry import tool
from .read_file import DEFAULT_MAX_CHARS, read_text_capped


def file_tools(root: str | Path) -> tuple:
    """``read_file`` and ``write_file`` confined to one directory.

    Any path that resolves outside ``root`` (absolute paths, ``..``, or a symlink that points
    out) is refused. The plain ``read_file`` and ``write_file`` built-ins accept any path, so
    use these when the model should only touch a working folder.
    """
    base = Path(root).resolve()

    def _inside(path: str) -> Path:
        target = (base / path).resolve()
        if target != base and base not in target.parents:
            raise PermissionError(f"path is outside the allowed folder: {path}")
        return target

    @tool
    def read_file(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """Read a file in the working folder, cut off at max chars (default 20000)."""
        try:
            return read_text_capped(_inside(path), int(max_chars))
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as exc:
            return f"Error: {exc}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write text to a file in the working folder, creating parent directories."""
        try:
            target = _inside(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Written {len(content)} characters to {path}"
        except Exception as exc:
            return f"Error: {exc}"

    return read_file, write_file
