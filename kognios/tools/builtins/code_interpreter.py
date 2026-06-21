from __future__ import annotations

import subprocess
import sys
import textwrap

from ..registry import tool


@tool
def code_interpreter(code: str, timeout: int = 10) -> str:
    """Execute Python code in an isolated subprocess and return stdout+stderr.

    Safer than python_eval: runs in a separate process that is killed after timeout seconds.
    """
    # Wrap code so we capture both print output and the last expression
    wrapper = textwrap.dedent(f"""
import sys, io, traceback
_code = {repr(code)}
_buf = io.StringIO()
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = _buf
sys.stderr = _buf
try:
    _compiled = compile(_code, "<agent>", "exec")
    _ns = {{}}
    exec(_compiled, _ns)
except Exception:
    traceback.print_exc()
finally:
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr
print(_buf.getvalue(), end="")
""")
    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: code execution timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"
