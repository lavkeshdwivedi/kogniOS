from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..registry import tool


@tool
def web_search(query: str) -> str:
    """Search the web and return a brief summary of results."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://ddg-api.fly.dev/search?q={encoded}&limit=3"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        results = data if isinstance(data, list) else data.get("results", [])
        lines = []
        for r in results[:3]:
            title = r.get("title", "")
            snippet = r.get("body", r.get("snippet", ""))
            link = r.get("href", r.get("url", ""))
            lines.append(f"- {title}: {snippet} ({link})")
        return "\n".join(lines) if lines else "No results found."
    except Exception as exc:
        return f"Search failed: {exc}"
