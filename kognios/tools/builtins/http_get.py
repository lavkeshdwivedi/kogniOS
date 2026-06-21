from __future__ import annotations

import urllib.request

from ..registry import tool


@tool
def http_get(url: str) -> str:
    """Fetch the content of a URL and return the response body as text."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kognios/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")[:8000]
    except Exception as exc:
        return f"Error: {exc}"
