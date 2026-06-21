from __future__ import annotations

import os
import random
import re
import time
from collections.abc import Iterator

from .base import BaseModel, ModelChunk, ModelResponse


# ── Free-tier model pools for free_tier_chain() ───────────────────────────────
# Order within each pool: best quality / most quota first, fastest fallback last.

_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

_TOGETHER_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
]

_ANTHROPIC_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

# ── Think-block stripping ─────────────────────────────────────────────────────
# Reasoning models (Qwen3, DeepSeek-R1, some GPT-OSS variants) emit
# <think>...</think> preambles. Strip them so callers see clean output.

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# ── Multi-key collector ───────────────────────────────────────────────────────


def _collect_keys(base_env: str, max_keys: int = 10) -> list[str]:
    """Return all keys for a provider from PROVIDER_KEY, PROVIDER_KEY_2, ...

    Stops at the first missing slot. De-duplicates so the same key pasted
    twice (with or without whitespace) only appears once.
    """
    out: list[str] = []
    seen: set[str] = set()
    primary = (os.environ.get(base_env) or "").strip()
    if primary and primary not in seen:
        out.append(primary)
        seen.add(primary)
    for i in range(2, max_keys + 1):
        val = (os.environ.get(f"{base_env}_{i}") or "").strip()
        if not val:
            break
        if val not in seen:
            out.append(val)
            seen.add(val)
    return out


# ── ModelChain ────────────────────────────────────────────────────────────────


class ModelChain(BaseModel):
    """Try models in order with 429-aware backoff and per-model cooldown.

    Drop-in replacement for any single model. Enhancements over a bare
    fallback loop:

    * Dead-model tracking: 401, 403, 404, and decommissioned-400 responses
      permanently skip that instance for the lifetime of this chain object,
      so subsequent calls jump straight past it.
    * Rate pacing: min_interval seconds between outbound calls keeps the
      process under free-tier per-minute ceilings.
    * Think-block stripping: removes <think>...</think> from reasoning-model
      output when strip_think=True (default).
    * Overall deadline: overall_timeout seconds cap across the whole call so
      one slow provider cannot stall the run forever.
    * 429 backoff: exponential back-off for max_retries attempts, then marks
      that model cooling for the retry-after window and moves on.

    Raises RuntimeError when every model has been exhausted.

    Usage::

        from kognios.models.chain import ModelChain, free_tier_chain

        # Simple explicit chain
        chain = ModelChain([GroqModel(), GeminiModel(), AnthropicModel()])

        # Factory that reads keys from env and builds the full pool
        chain = free_tier_chain(preferred="gemini")
        response = chain.complete(messages, system=system_prompt)
    """

    def __init__(
        self,
        models: list[BaseModel],
        min_interval: float = 2.0,
        default_cooldown: float = 65.0,
        max_retries: int = 2,
        max_backoff: float = 30.0,
        overall_timeout: float = 180.0,
        strip_think: bool = True,
    ):
        self.models = models
        self.min_interval = min_interval
        self.default_cooldown = default_cooldown
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.overall_timeout = overall_timeout
        self.strip_think = strip_think
        self._cooldown_until: dict[int, float] = {}
        self._dead: set[int] = set()
        self._last_call: float = 0.0

    def _pace(self) -> None:
        gap = time.time() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.time()

    def _parse_retry_after(self, exc: Exception) -> float:
        try:
            headers = exc.response.headers  # type: ignore[attr-defined]
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                return max(1.0, float(ra))
        except Exception:
            pass
        return self.default_cooldown

    def _is_rate_limit(self, exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 429

    def _is_dead_error(self, exc: Exception) -> bool:
        sc = getattr(exc, "status_code", None)
        if sc in (401, 403, 404):
            return True
        if sc == 400:
            msg = str(exc).lower()
            return "decommission" in msg or "not found" in msg
        return False

    def _clean(self, text: str) -> str:
        return _strip_think(text) if self.strip_think else text

    def _deadline(self) -> float:
        return (time.time() + self.overall_timeout) if self.overall_timeout else float("inf")

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        **call_kwargs,
    ) -> ModelResponse:
        deadline = self._deadline()
        last_err: Exception | None = None
        for model in self.models:
            mid = id(model)
            if mid in self._dead:
                continue
            if time.time() > deadline:
                break
            wait = self._cooldown_until.get(mid, 0.0) - time.time()
            if wait > 0:
                if time.time() + wait > deadline:
                    continue
                time.sleep(wait)

            for attempt in range(self.max_retries + 1):
                if time.time() > deadline:
                    break
                self._pace()
                try:
                    resp = model.complete(messages, tools=tools, system=system, **call_kwargs)
                    resp.content = self._clean(resp.content)
                    return resp
                except Exception as exc:
                    last_err = exc
                    if self._is_dead_error(exc):
                        self._dead.add(mid)
                        break
                    if self._is_rate_limit(exc):
                        if attempt < self.max_retries:
                            delay = min(
                                1.2 * (2**attempt) + random.uniform(0, 0.4),
                                self.max_backoff,
                            )
                            time.sleep(delay)
                        else:
                            self._cooldown_until[mid] = time.time() + self._parse_retry_after(exc)
                            break
                    else:
                        break

        raise RuntimeError(f"All models in chain exhausted. Last error: {last_err}")

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Iterator[ModelChunk]:
        last_err: Exception | None = None
        for model in self.models:
            mid = id(model)
            if mid in self._dead:
                continue
            wait = self._cooldown_until.get(mid, 0.0) - time.time()
            if wait > 0:
                time.sleep(wait)
            self._pace()
            try:
                yield from model.stream(messages, tools=tools, system=system)
                return
            except Exception as exc:
                last_err = exc
                if self._is_dead_error(exc):
                    self._dead.add(mid)
                elif self._is_rate_limit(exc):
                    self._cooldown_until[mid] = time.time() + self._parse_retry_after(exc)

        raise RuntimeError(f"All models in chain exhausted. Last error: {last_err}")

    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ModelResponse:
        import asyncio

        deadline = self._deadline()
        last_err: Exception | None = None
        for model in self.models:
            mid = id(model)
            if mid in self._dead:
                continue
            if time.time() > deadline:
                break
            wait = self._cooldown_until.get(mid, 0.0) - time.time()
            if wait > 0:
                if time.time() + wait > deadline:
                    continue
                await asyncio.sleep(wait)
            try:
                resp = await model.acomplete(messages, tools=tools, system=system)
                resp.content = self._clean(resp.content)
                return resp
            except Exception as exc:
                last_err = exc
                if self._is_dead_error(exc):
                    self._dead.add(mid)
                elif self._is_rate_limit(exc):
                    self._cooldown_until[mid] = time.time() + self._parse_retry_after(exc)

        raise RuntimeError(f"All models in chain exhausted. Last error: {last_err}")


# ── Factory ───────────────────────────────────────────────────────────────────


def free_tier_chain(
    preferred: str | None = None,
    min_interval: float = 2.0,
    overall_timeout: float = 180.0,
    groq_models: list[str] | None = None,
    gemini_models: list[str] | None = None,
    together_models: list[str] | None = None,
    anthropic_models: list[str] | None = None,
    **model_kwargs,
) -> ModelChain:
    """Build a ModelChain with free-tier providers first, paid providers last.

    Reads GROQ_API_KEY, GROQ_API_KEY_2, ..., GROQ_API_KEY_N from env and
    similarly for GEMINI, TOGETHER, ANTHROPIC. Creates one model instance
    per (key, model_name) pair so each has its own cooldown bucket: a 429
    on key1/model-A does not block key2/model-A.

    Default provider order: Groq -> Gemini -> Together -> Anthropic.
    Pass preferred="gemini" to promote that provider to the front.

    Extra keyword arguments (e.g. max_tokens=400, temperature=0.25) are
    forwarded to every model constructor.
    """
    from .groq import GroqModel
    from .gemini import GeminiModel
    from .together import TogetherModel
    from .anthropic import AnthropicModel

    groq_pool = groq_models or _GROQ_MODELS
    gemini_pool = gemini_models or _GEMINI_MODELS
    together_pool = together_models or _TOGETHER_MODELS
    anthropic_pool = anthropic_models or _ANTHROPIC_MODELS

    provider_order = ["groq", "gemini", "together", "anthropic"]
    if preferred and preferred in provider_order:
        provider_order = [preferred] + [p for p in provider_order if p != preferred]

    models: list[BaseModel] = []
    for provider in provider_order:
        if provider == "groq":
            for key in _collect_keys("GROQ_API_KEY"):
                for name in groq_pool:
                    models.append(GroqModel(model=name, api_key=key, **model_kwargs))
        elif provider == "gemini":
            for key in _collect_keys("GEMINI_API_KEY"):
                for name in gemini_pool:
                    models.append(GeminiModel(model=name, api_key=key, **model_kwargs))
        elif provider == "together":
            for key in _collect_keys("TOGETHER_API_KEY"):
                for name in together_pool:
                    models.append(TogetherModel(model=name, api_key=key, **model_kwargs))
        elif provider == "anthropic":
            for key in _collect_keys("ANTHROPIC_API_KEY"):
                for name in anthropic_pool:
                    models.append(AnthropicModel(model=name, api_key=key, **model_kwargs))

    return ModelChain(
        models,
        min_interval=min_interval,
        overall_timeout=overall_timeout,
    )
