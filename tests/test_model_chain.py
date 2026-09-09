"""Tests for ModelChain — 429 backoff, model fallback, cooldown tracking."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from kognios.models.base import ModelResponse
from kognios.models.chain import ModelChain


class _RateLimitError(Exception):
    def __init__(self, retry_after: float | None = None):
        super().__init__("429 Too Many Requests")
        self.status_code = 429
        headers: dict = {}
        if retry_after is not None:
            headers["retry-after"] = str(retry_after)
        self.response = MagicMock()
        self.response.headers = headers


def _mock_model(content="ok", side_effect=None):
    m = MagicMock()
    if side_effect is not None:
        m.complete.side_effect = side_effect
    else:
        m.complete.return_value = ModelResponse(content=content)
    return m


def test_first_model_succeeds():
    m1 = _mock_model("hello")
    m2 = _mock_model("fallback")
    chain = ModelChain([m1, m2], min_interval=0)
    resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "hello"
    m2.complete.assert_not_called()


def test_falls_back_on_non_429_error():
    m1 = _mock_model(side_effect=RuntimeError("connection refused"))
    m2 = _mock_model("from fallback")
    chain = ModelChain([m1, m2], min_interval=0)
    resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "from fallback"


def test_falls_back_after_429_exhausted():
    m1 = _mock_model(side_effect=_RateLimitError(retry_after=1))
    m2 = _mock_model("fallback ok")
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    with patch("time.sleep"):
        resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "fallback ok"


def test_raises_when_all_exhausted():
    m1 = _mock_model(side_effect=_RateLimitError())
    m2 = _mock_model(side_effect=RuntimeError("also failed"))
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    with patch("time.sleep"), pytest.raises(RuntimeError, match="exhausted"):
        chain.complete([{"role": "user", "content": "hi"}])


def test_cooldown_tracked_per_model():
    m1 = _mock_model(side_effect=_RateLimitError(retry_after=60))
    m2 = _mock_model("ok")
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    with patch("time.sleep"):
        chain.complete([{"role": "user", "content": "hi"}])
    assert chain._cooldown_until[id(m1)] > time.time()
    assert id(m2) not in chain._cooldown_until


def test_retries_429_before_moving_on():
    success = ModelResponse(content="retry worked")
    m1 = _mock_model(side_effect=[_RateLimitError(), _RateLimitError(), success])
    chain = ModelChain([m1], min_interval=0, max_retries=2)
    with patch("time.sleep"):
        resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "retry worked"
    assert m1.complete.call_count == 3


def test_falls_back_on_empty_content():
    m1 = _mock_model(content="")
    m2 = _mock_model("from fallback")
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "from fallback"


def test_falls_back_on_reasoning_only_response():
    m1 = _mock_model(content="<think>burned the whole budget reasoning</think>")
    m2 = _mock_model("from fallback")
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "from fallback"


def test_raises_when_all_models_return_empty():
    m1 = _mock_model(content="")
    m2 = _mock_model(content="   ")
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    with pytest.raises(RuntimeError, match="exhausted"):
        chain.complete([{"role": "user", "content": "hi"}])


def test_empty_content_with_tool_calls_is_not_rejected():
    resp_with_tools = ModelResponse(content="", tool_calls=[{"name": "lookup"}])
    m1 = _mock_model()
    m1.complete.return_value = resp_with_tools
    m2 = _mock_model("should not be reached")
    chain = ModelChain([m1, m2], min_interval=0, max_retries=0)
    resp = chain.complete([{"role": "user", "content": "hi"}])
    assert resp is resp_with_tools
    m2.complete.assert_not_called()


def test_exported_from_kognios():
    from kognios import ModelChain as MC

    assert MC is ModelChain
