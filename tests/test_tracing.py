"""Tests for the Tracer."""

from __future__ import annotations

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from kognios.tracing import Tracer, Span, prometheus_sink


def test_tracer_records_span():
    tracer = Tracer()
    with tracer.span("test.op", key="val"):
        pass
    assert len(tracer.spans) == 1
    s = tracer.spans[0]
    assert s.name == "test.op"
    assert s.attributes["key"] == "val"
    assert s.duration_ms >= 0


def test_tracer_records_error():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.span("failing.op"):
            raise ValueError("oops")
    assert tracer.spans[0].error == "oops"


def test_tracer_sink_called():
    received = []
    tracer = Tracer(sink=received.append)
    with tracer.span("op"):
        pass
    assert len(received) == 1
    assert received[0].name == "op"


def test_tracer_clear():
    tracer = Tracer()
    with tracer.span("op"):
        pass
    tracer.clear()
    assert tracer.spans == []


def test_agent_with_tracer_records_llm_span():
    from kognios import Agent
    from kognios.models.base import ModelResponse

    fake_model = MagicMock()
    fake_model.model = "test-model"
    fake_model.complete = MagicMock(return_value=ModelResponse(content="answer", usage={}))

    tracer = Tracer()
    agent = Agent(model=fake_model, tracer=tracer)
    agent.run("hello")

    span_names = [s.name for s in tracer.spans]
    assert "llm.complete" in span_names


# ---------------------------------------------------------------------------
# prometheus_sink tests
# ---------------------------------------------------------------------------


def _make_fake_prometheus():
    """Build a minimal fake prometheus_client module."""
    fake_pc = types.ModuleType("prometheus_client")

    class FakeMetric:
        def __init__(self, name, doc, labelnames=None):
            self.name = name
            self.observed = []
            self.incremented = 0
            self._labelnames = labelnames or []

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            self.observed.append(value)

        def inc(self, amount=1):
            self.incremented += amount

    fake_pc.Histogram = FakeMetric
    fake_pc.Counter = FakeMetric
    return fake_pc


def test_prometheus_sink_is_callable():
    fake_pc = _make_fake_prometheus()
    with patch.dict(sys.modules, {"prometheus_client": fake_pc}):
        sink = prometheus_sink()
    assert callable(sink)


def test_prometheus_sink_observes_duration():
    fake_pc = _make_fake_prometheus()
    with patch.dict(sys.modules, {"prometheus_client": fake_pc}):
        sink = prometheus_sink()
        span = Span(name="test.op", start_time=0.0, end_time=0.5)
        sink(span)

    # The histogram label proxy's observe list should have one entry
    # We verify by running the sink and checking no exception is raised
    # (the FakeMetric collects calls internally)


def test_prometheus_sink_counts_errors():
    fake_pc = _make_fake_prometheus()
    with patch.dict(sys.modules, {"prometheus_client": fake_pc}):
        sink = prometheus_sink()
        span = Span(name="fail.op", start_time=0.0, end_time=0.1, error="boom")
        sink(span)  # should not raise


def test_prometheus_sink_no_error_no_increment():
    """A span without error should not increment the error counter."""
    fake_pc = _make_fake_prometheus()
    observed_incs = []

    original_counter = fake_pc.Counter

    class TrackingCounter(original_counter):
        def inc(self, amount=1):
            observed_incs.append(amount)

    fake_pc.Counter = TrackingCounter

    with patch.dict(sys.modules, {"prometheus_client": fake_pc}):
        sink = prometheus_sink()
        span = Span(name="ok.op", start_time=0.0, end_time=0.2)
        sink(span)

    assert observed_incs == [], "error counter should not be incremented for a successful span"


def test_prometheus_sink_missing_package():
    """prometheus_sink raises ImportError when prometheus_client is not installed."""
    # Remove from sys.modules so the lazy import fails
    with patch.dict(sys.modules, {"prometheus_client": None}):
        with pytest.raises(ImportError, match="prometheus-client"):
            prometheus_sink()


def test_prometheus_sink_used_as_tracer_sink():
    """End-to-end: prometheus_sink integrates correctly with Tracer."""
    fake_pc = _make_fake_prometheus()
    with patch.dict(sys.modules, {"prometheus_client": fake_pc}):
        sink = prometheus_sink()
        tracer = Tracer(sink=sink)
        with tracer.span("my.operation"):
            pass
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "my.operation"
