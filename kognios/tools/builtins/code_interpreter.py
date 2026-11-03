from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

from ..registry import tool


# the child gets only what Python needs to start, so API keys in the parent's environment stay out
_ENV_ALLOW = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL")


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.upper() in _ENV_ALLOW}


@tool
def code_interpreter(code: str, timeout: int = 10) -> str:
    """Execute Python code in an isolated subprocess and return stdout+stderr.

    Safer than python_eval: a separate process with no inherited secrets in its environment,
    run in an empty temporary folder and killed after timeout seconds. It is not a sandbox: the
    code can still reach the network and any file the user account can read.
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
        with tempfile.TemporaryDirectory() as scratch:
            result = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_clean_env(),
                cwd=scratch,
            )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: code execution timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"
