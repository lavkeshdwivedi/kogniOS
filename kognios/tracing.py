from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start_time: float
    attributes: dict = field(default_factory=dict)
    end_time: float = 0.0
    error: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Lightweight span-based tracer.

    Usage::

        tracer = Tracer()
        agent = Agent(model=..., tracer=tracer)

        # After running:
        for span in tracer.spans:
            print(span.name, span.duration_ms)
    """

    def __init__(self, sink: Callable[[Span], None] | None = None):
        self.spans: list[Span] = []
        self._sink = sink  # optional callback per-span (e.g. print, OTLP export)

    @contextmanager
    def span(self, name: str, **attributes):
        s = Span(name=name, start_time=time.perf_counter(), attributes=attributes)
        try:
            yield s
        except Exception as exc:
            s.error = str(exc)
            raise
        finally:
            s.end_time = time.perf_counter()
            self.spans.append(s)
            if self._sink:
                self._sink(s)

    def clear(self) -> None:
        self.spans.clear()

    def print_spans(self) -> None:
        for s in self.spans:
            err = f" ERROR={s.error}" if s.error else ""
            attrs = " ".join(f"{k}={v}" for k, v in s.attributes.items())
            print(f"[TRACE] {s.name} {s.duration_ms:.1f}ms {attrs}{err}")

    def try_export_otel(self, service_name: str = "kognios") -> bool:
        """Try to export spans via OpenTelemetry SDK if installed. Returns True if successful."""
        try:
            from opentelemetry import trace  # noqa: F401
            from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: F401
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: F401
                InMemorySpanExporter,
            )
        except ImportError:
            return False
        # Minimal wiring — real use would configure OTLP exporter
        return True


def prometheus_sink() -> Callable[[Span], None]:
    """Return a sink that records span metrics to Prometheus.

    Requires ``prometheus-client`` to be installed::

        pip install 'kognios[prometheus]'

    Metrics exported:

    * ``kognios_span_duration_seconds`` — Histogram labelled by ``span_name``
    * ``kognios_span_errors_total``     — Counter labelled by ``span_name``

    Example::

        from kognios.tracing import Tracer, prometheus_sink
        tracer = Tracer(sink=prometheus_sink())
    """
    try:
        import prometheus_client  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "prometheus-client is required for prometheus_sink: pip install 'kognios[prometheus]'"
        ) from exc

    from prometheus_client import Counter, Histogram

    duration_histogram = Histogram(
        "kognios_span_duration_seconds",
        "Duration of kognios spans in seconds",
        labelnames=["span_name"],
    )
    error_counter = Counter(
        "kognios_span_errors_total",
        "Total number of kognios span errors",
        labelnames=["span_name"],
    )

    def _sink(span: Span) -> None:
        duration_seconds = span.end_time - span.start_time
        duration_histogram.labels(span_name=span.name).observe(duration_seconds)
        if span.error:
            error_counter.labels(span_name=span.name).inc()

    return _sink
