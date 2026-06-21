"""Tests for knowledge loaders — no network, no external deps."""

from __future__ import annotations

import json

from kognios.knowledge.loaders import load_text, _strip_html, _flatten_json


def test_strip_html_removes_tags():
    html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
    result = _strip_html(html)
    assert "Hello" in result
    assert "World" in result
    assert "<" not in result


def test_strip_html_removes_script():
    html = "<html><script>alert('x')</script><p>Keep this</p></html>"
    result = _strip_html(html)
    assert "Keep this" in result
    assert "alert" not in result


def test_load_text_raw_string():
    result = load_text("hello world this is a test")
    assert result == "hello world this is a test"


def test_load_text_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("file content here")
    result = load_text(str(f))
    assert result == "file content here"


def test_load_text_html_file(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>Hello HTML</p></body></html>")
    result = load_text(str(f))
    assert "Hello HTML" in result
    assert "<" not in result


def test_load_text_csv_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25")
    result = load_text(str(f))
    assert "Alice" in result
    assert "Bob" in result


def test_load_text_json_file(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"name": "Alice", "scores": [1, 2, 3]}))
    result = load_text(str(f))
    assert "Alice" in result
    assert "1" in result


def test_flatten_json_dict():
    result = _flatten_json({"a": 1, "b": "hello"})
    assert "a: 1" in result
    assert "b: hello" in result


def test_flatten_json_nested():
    result = _flatten_json({"user": {"name": "Alice", "age": 30}})
    assert "Alice" in result


def test_sqlite_knowledge_loads_html(tmp_path):
    from kognios.knowledge.sqlite_fts import SQLiteKnowledge

    kb = SQLiteKnowledge(db_path=":memory:")
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>semantic search engine</p></body></html>")
    kb.load(str(f))
    results = kb.search("semantic search")
    assert len(results) >= 1
    assert "semantic search engine" in results[0]
