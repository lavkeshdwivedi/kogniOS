"""Source loaders for knowledge bases. All return plain text."""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from pathlib import Path


def load_text(source: str | Path, source_type: str = "auto") -> str:
    """Load text from a source. source_type: auto|text|url|pdf|html|csv|json|docx|github."""
    src = str(source)

    if source_type == "url":
        return _fetch_url(src)
    if source_type == "pdf":
        return _load_pdf(src)
    if source_type == "html":
        return _load_html(src)
    if source_type == "csv":
        return _load_csv(src)
    if source_type == "json":
        return _load_json(src)
    if source_type == "docx":
        return _load_docx(src)
    if source_type == "github":
        return _load_github_repo(src)

    # auto-detect
    if src.startswith("http://") or src.startswith("https://"):
        if "github.com" in src and not src.endswith((".md", ".txt", ".py")):
            return _load_github_repo(src)
        raw = _fetch_url(src)
        # detect HTML
        if "<html" in raw[:500].lower() or "<!doctype" in raw[:200].lower():
            return _strip_html(raw)
        return raw

    # file path
    p = Path(src)
    if p.suffix == ".pdf":
        return _load_pdf(src)
    if p.suffix in (".html", ".htm"):
        return _strip_html(p.read_text(encoding="utf-8", errors="replace"))
    if p.suffix == ".csv":
        return _load_csv(src)
    if p.suffix == ".json":
        return _load_json(src)
    if p.suffix in (".docx",):
        return _load_docx(src)

    # raw string or plain file
    if "\n" not in src and len(src) <= 255:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return src


def _fetch_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    # Remove script/style blocks
    html = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_html(src: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        raw = _fetch_url(src)
    else:
        raw = Path(src).read_text(encoding="utf-8", errors="replace")
    return _strip_html(raw)


def _load_pdf(src: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pip install 'kognios[pdf]'")
    reader = PdfReader(src)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_csv(src: str) -> str:
    """Flatten CSV to tab-separated rows of text."""
    p = Path(src)
    content = p.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(content))
    rows = ["\t".join(row) for row in reader]
    return "\n".join(rows)


def _load_json(src: str) -> str:
    """Flatten JSON to readable text."""
    p = Path(src)
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    return _flatten_json(data)


def _flatten_json(obj, prefix: str = "") -> str:
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            lines.append(_flatten_json(v, key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            lines.append(_flatten_json(item, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}" if prefix else str(obj))
    return "\n".join(lines)


def _load_docx(src: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("pip install 'kognios[docx]'")
    doc = Document(src)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_github_repo(url_or_owner_repo: str) -> str:
    """Crawl Python/Markdown/text files from a GitHub repo via the API."""
    # Accept either full URL or "owner/repo"
    src = url_or_owner_repo
    if src.startswith("https://github.com/"):
        src = src[len("https://github.com/") :]
    src = src.rstrip("/")
    parts = src.split("/")
    owner, repo = parts[0], parts[1] if len(parts) > 1 else parts[0]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    req = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "kognios"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        tree = json.loads(resp.read())

    EXTENSIONS = {".py", ".md", ".txt", ".rst", ".yaml", ".yml", ".toml"}
    blobs = [
        item
        for item in tree.get("tree", [])
        if item["type"] == "blob" and Path(item["path"]).suffix in EXTENSIONS
    ][:50]  # cap at 50 files

    texts = [f"=== {owner}/{repo} ==="]
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
    for item in blobs:
        try:
            raw = _fetch_url(raw_base + item["path"])
            texts.append(f"\n--- {item['path']} ---\n{raw[:4000]}")
        except Exception:
            pass
    return "\n".join(texts)
