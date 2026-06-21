from __future__ import annotations

import re
import urllib.request

from ..registry import tool


def _strip_html(html: str) -> str:
    html = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_urllib(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "kognios/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return _strip_html(raw)


@tool
def browse(url: str) -> str:
    """Fetch a web page and return its visible text content.

    Uses Playwright (headless Chromium) if installed for JS-rendered pages,
    otherwise falls back to a simple HTTP fetch.

    Install Playwright support: pip install 'kognios[browser]'
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            text = page.inner_text("body")
            browser.close()
            return text[:8000]  # cap output
    except ImportError:
        # Playwright not installed — fall back to urllib
        pass
    except Exception:
        # Playwright failed (no browser installed, etc.) — fall back
        pass
    return _fetch_urllib(url)[:8000]
