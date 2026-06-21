"""Redis-backed long-term memory plugin for kogniOS.

Implements the same interface as ``kognios.memory.LongTermMemory`` but stores
facts in a Redis list so they survive restarts and are shared across processes.

Install::

    pip install kognios redis>=5.0
    # or via the example pyproject.toml:
    pip install -e .

Usage::

    from redis_memory import RedisLongTermMemory

    mem = RedisLongTermMemory(prefix="mybot", ttl=3600)
    mem.remember("user_name", "Alice")
    print(mem.recall("user_name"))  # "Alice"
    print(mem.facts())              # ["user_name=Alice"]
"""

from __future__ import annotations


class RedisLongTermMemory:
    """Long-term memory backed by Redis.

    Parameters
    ----------
    host:
        Redis server hostname (default ``"localhost"``).
    port:
        Redis server port (default ``6379``).
    db:
        Redis database index (default ``0``).
    prefix:
        Key prefix used for all Redis keys (default ``"kognios:memory"``).
    ttl:
        Optional TTL in seconds applied to each Redis key on write.
        ``None`` (the default) means keys never expire.
    redis_client:
        Optional pre-constructed ``redis.Redis`` instance.  When supplied,
        ``host``/``port``/``db`` are ignored.  Useful for testing.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "kognios:memory",
        ttl: int | None = None,
        redis_client=None,
    ) -> None:
        self._prefix = prefix
        self._ttl = ttl

        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "redis package is required for RedisLongTermMemory: "
                    "pip install 'redis>=5.0'"
                ) from exc
            self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, name: str) -> str:
        return f"{self._prefix}:{name}"

    # ------------------------------------------------------------------
    # Public interface  (mirrors kognios.memory.LongTermMemory)
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        """Store *value* under *key*.

        Existing entries with the same key are overwritten (delete + set).
        """
        rkey = self._key(key)
        # Use a Redis hash to map key -> value under the prefix namespace.
        hash_key = f"{self._prefix}:facts"
        self._redis.hset(hash_key, key, value)
        if self._ttl is not None:
            self._redis.expire(hash_key, self._ttl)

    def recall(self, key: str) -> str | None:
        """Return the value stored for *key*, or ``None`` if not found."""
        hash_key = f"{self._prefix}:facts"
        value = self._redis.hget(hash_key, key)
        return value  # already str | None because decode_responses=True

    def facts(self) -> list[str]:
        """Return all stored facts as ``"key=value"`` strings."""
        hash_key = f"{self._prefix}:facts"
        data: dict = self._redis.hgetall(hash_key)
        return [f"{k}={v}" for k, v in data.items()]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def delete(self, key: str) -> None:
        """Remove a single key from memory."""
        hash_key = f"{self._prefix}:facts"
        self._redis.hdel(hash_key, key)

    def clear(self) -> None:
        """Remove all facts stored under this prefix."""
        hash_key = f"{self._prefix}:facts"
        self._redis.delete(hash_key)

    def as_system_prompt_fragment(self) -> str:
        """Format all facts as a block suitable for injection into a system prompt."""
        items = self.facts()
        if not items:
            return ""
        lines = "\n".join(f"  - {f}" for f in items)
        return f"Long-term memory:\n{lines}"
