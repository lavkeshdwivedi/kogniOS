"""Unit tests for RedisLongTermMemory.

No real Redis server required — redis.Redis is mocked throughout.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Helpers — build a fake redis module so we don't need redis installed
# ---------------------------------------------------------------------------

def _make_fake_redis_module():
    """Return a minimal fake ``redis`` module with a configurable Redis client."""
    fake_redis_mod = types.ModuleType("redis")

    class FakeRedis:
        """In-memory Redis stub that covers the hash commands we use."""

        def __init__(self, *args, **kwargs):
            self._store: dict[str, dict] = {}  # key -> {field: value}
            self._ttls: dict[str, int] = {}

        # Hash commands
        def hset(self, name: str, key: str, value: str) -> int:
            bucket = self._store.setdefault(name, {})
            existed = key in bucket
            bucket[key] = value
            return 0 if existed else 1

        def hget(self, name: str, key: str) -> str | None:
            return self._store.get(name, {}).get(key)

        def hgetall(self, name: str) -> dict:
            return dict(self._store.get(name, {}))

        def hdel(self, name: str, *keys) -> int:
            bucket = self._store.get(name, {})
            removed = 0
            for k in keys:
                if k in bucket:
                    del bucket[k]
                    removed += 1
            return removed

        def delete(self, *names) -> int:
            removed = 0
            for name in names:
                if name in self._store:
                    del self._store[name]
                    removed += 1
            return removed

        def expire(self, name: str, seconds: int) -> int:
            self._ttls[name] = seconds
            return 1

    fake_redis_mod.Redis = FakeRedis
    return fake_redis_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_redis(monkeypatch):
    """Inject fake redis module for every test in this file."""
    fake = _make_fake_redis_module()
    monkeypatch.setitem(sys.modules, "redis", fake)
    yield fake


# We must import AFTER the monkeypatch is active.  Use a lazy import in each
# test via importlib, or simply import the module here after the fixture
# machinery runs (pytest applies autouse fixtures before the module is used).

# ---------------------------------------------------------------------------
# Import under test  (delayed so monkeypatch is in effect)
# ---------------------------------------------------------------------------

import importlib, pathlib

def _load_module():
    spec = importlib.util.spec_from_file_location(
        "redis_memory",
        pathlib.Path(__file__).parent.parent / "redis_memory.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRedisLongTermMemory:

    def _get_class(self):
        mod = _load_module()
        return mod.RedisLongTermMemory

    def test_remember_and_recall(self):
        cls = self._get_class()
        mem = cls()
        mem.remember("name", "Alice")
        assert mem.recall("name") == "Alice"

    def test_recall_missing_key_returns_none(self):
        cls = self._get_class()
        mem = cls()
        assert mem.recall("nonexistent") is None

    def test_remember_overwrites_existing(self):
        cls = self._get_class()
        mem = cls()
        mem.remember("color", "blue")
        mem.remember("color", "red")
        assert mem.recall("color") == "red"

    def test_facts_returns_all_as_strings(self):
        cls = self._get_class()
        mem = cls()
        mem.remember("a", "1")
        mem.remember("b", "2")
        facts = mem.facts()
        assert "a=1" in facts
        assert "b=2" in facts
        assert len(facts) == 2

    def test_facts_empty_when_nothing_stored(self):
        cls = self._get_class()
        mem = cls()
        assert mem.facts() == []

    def test_delete_removes_single_key(self):
        cls = self._get_class()
        mem = cls()
        mem.remember("x", "10")
        mem.remember("y", "20")
        mem.delete("x")
        assert mem.recall("x") is None
        assert mem.recall("y") == "20"

    def test_clear_removes_all_facts(self):
        cls = self._get_class()
        mem = cls()
        mem.remember("p", "1")
        mem.remember("q", "2")
        mem.clear()
        assert mem.facts() == []

    def test_ttl_applied_on_write(self, patch_redis):
        """expire() should be called on the hash key when ttl is set."""
        cls = self._get_class()
        mem = cls(ttl=300)
        mem.remember("k", "v")
        hash_key = "kognios:memory:facts"
        assert hash_key in mem._redis._ttls
        assert mem._redis._ttls[hash_key] == 300

    def test_no_ttl_by_default(self, patch_redis):
        """expire() should NOT be called when ttl=None."""
        cls = self._get_class()
        mem = cls()  # ttl=None
        mem.remember("k", "v")
        assert not mem._redis._ttls  # no TTLs recorded

    def test_custom_prefix(self):
        cls = self._get_class()
        mem = cls(prefix="bot:mem")
        mem.remember("lang", "Python")
        assert mem.recall("lang") == "Python"
        # Facts are stored under the custom prefix hash key
        assert "lang=Python" in mem.facts()

    def test_redis_client_injection(self, patch_redis):
        """Passing redis_client= should skip the import path."""
        cls = self._get_class()
        fake_client = patch_redis.Redis()
        mem = cls(redis_client=fake_client)
        mem.remember("injected", "yes")
        assert mem.recall("injected") == "yes"

    def test_import_error_when_redis_missing(self, monkeypatch):
        """Without redis installed, constructing without redis_client raises ImportError."""
        monkeypatch.setitem(sys.modules, "redis", None)
        cls = self._get_class()
        with pytest.raises(ImportError, match="redis"):
            cls()  # no redis_client provided, so it tries to import redis

    def test_as_system_prompt_fragment_empty(self):
        cls = self._get_class()
        mem = cls()
        assert mem.as_system_prompt_fragment() == ""

    def test_as_system_prompt_fragment_with_facts(self):
        cls = self._get_class()
        mem = cls()
        mem.remember("name", "Alice")
        fragment = mem.as_system_prompt_fragment()
        assert "Long-term memory" in fragment
        assert "name=Alice" in fragment

    def test_module_is_importable(self):
        """Basic smoke-test: the module loads without errors."""
        mod = _load_module()
        assert hasattr(mod, "RedisLongTermMemory")
