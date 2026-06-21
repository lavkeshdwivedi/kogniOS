from __future__ import annotations

import io
import contextlib

from ..registry import tool


@tool
def python_eval(code: str) -> str:
    """Execute Python code and return stdout + the final expression value."""
    stdout_buf = io.StringIO()
    local_ns: dict = {}
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<agent>", "exec"), {}, local_ns)
    except Exception as exc:
        return f"Error: {exc}"
    output = stdout_buf.getvalue()
    return output if output else "(no output)"
