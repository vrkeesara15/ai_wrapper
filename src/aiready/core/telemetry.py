"""Tracing, metrics and structured logging.

* ``span("gateway.route", model=...)`` opens a span. When OpenTelemetry is installed and
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, spans go to the client's collector in OTLP (one open format for
  traces, metrics and logs). Spans are also kept in a local ring buffer and can be shipped to the AI-Ready
  trace store, so a request can be replayed from the trace viewer even without an OTel backend.
* Prometheus metrics carry ``app``, ``feature``, ``team`` and ``model`` labels so cost and latency can be
  reported per feature and charged back by team.
* Personal data is masked by the configured redactor before any span attribute is stored.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Iterator

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from pydantic import BaseModel, Field

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("aiready_trace_id", default=None)
_span_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("aiready_span_stack", default=())

TRACE_HEADER = "X-Trace-Id"


def new_trace_id() -> str:
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(trace_id: str | None) -> contextvars.Token:
    return _trace_id.set(trace_id)


class SpanRecord(BaseModel):
    trace_id: str
    span_id: str
    parent_id: str | None = None
    name: str
    service: str = ""
    start: float
    end: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000


class _Recorder:
    def __init__(self, capacity: int = 5000):
        self.buffer: deque[SpanRecord] = deque(maxlen=capacity)
        self.exporters: list[Callable[[SpanRecord], None]] = []
        self.redactor: Callable[[Any], Any] | None = None
        self.service = os.environ.get("AIREADY_SERVICE", "aiready")
        self._lock = threading.Lock()

    def emit(self, rec: SpanRecord) -> None:
        with self._lock:
            self.buffer.append(rec)
        for exp in list(self.exporters):
            try:
                exp(rec)
            except Exception:  # telemetry must never break a request
                logging.getLogger(__name__).debug("span exporter failed", exc_info=True)

    def trace(self, trace_id: str) -> list[SpanRecord]:
        with self._lock:
            return [s for s in self.buffer if s.trace_id == trace_id]


recorder = _Recorder()
_otel_tracer: Any = None


class Span:
    def __init__(self, rec: SpanRecord, otel_span: Any = None):
        self.rec, self._otel = rec, otel_span

    def set(self, **attrs: Any) -> "Span":
        if recorder.redactor:
            attrs = recorder.redactor(attrs)
        self.rec.attributes.update(attrs)
        if self._otel is not None:
            for k, v in attrs.items():
                self._otel.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))
        return self

    def event(self, name: str, **attrs: Any) -> None:
        if recorder.redactor:
            attrs = recorder.redactor(attrs)
        self.rec.events.append({"name": name, "ts": time.time(), **attrs})
        if self._otel is not None:
            self._otel.add_event(name, {k: str(v) for k, v in attrs.items()})

    def error(self, exc: BaseException | str) -> None:
        self.rec.status = "error"
        self.rec.attributes["error"] = str(exc)


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    trace_id = _trace_id.get()
    token = None
    if trace_id is None:
        trace_id = new_trace_id()
        token = _trace_id.set(trace_id)
    stack = _span_stack.get()
    span_id = uuid.uuid4().hex[:16]
    rec = SpanRecord(trace_id=trace_id, span_id=span_id, parent_id=stack[-1] if stack else None, name=name,
                     service=recorder.service, start=time.time())
    stack_token = _span_stack.set(stack + (span_id,))
    otel_cm = _otel_tracer.start_as_current_span(name) if _otel_tracer is not None else contextlib.nullcontext()
    with otel_cm as otel_span:
        s = Span(rec, otel_span)
        if attrs:
            s.set(**attrs)
        try:
            yield s
        except BaseException as exc:
            s.error(exc)
            raise
        finally:
            rec.end = time.time()
            _span_stack.reset(stack_token)
            if token is not None:
                _trace_id.reset(token)
            recorder.emit(rec)


# --------------------------------------------------------------------------- metrics


class Metrics:
    """Prometheus metrics shared by every service (one registry per process)."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        lbl = ["service", "app", "feature", "team", "model"]
        self.requests = Counter("aiready_requests_total", "Requests handled", lbl + ["status"], registry=r)
        self.latency = Histogram("aiready_request_latency_seconds", "End-to-end latency", lbl, registry=r,
                                 buckets=(0.05, 0.1, 0.25, 0.5, 0.8, 1, 2, 4, 8, 16, 32, 64))
        self.ttft = Histogram("aiready_time_to_first_token_seconds", "Time to first token", lbl, registry=r,
                              buckets=(0.1, 0.25, 0.5, 0.8, 1, 2, 4, 8))
        self.tokens = Counter("aiready_tokens_total", "Tokens processed", lbl + ["direction"], registry=r)
        self.cost = Counter("aiready_cost_total", "Cost in configured currency", lbl, registry=r)
        self.cache = Counter("aiready_cache_total", "Cache lookups", ["service", "cache", "result"], registry=r)
        self.guardrail = Counter("aiready_guardrail_total", "Guardrail outcomes", ["service", "check", "outcome"],
                                 registry=r)
        self.policy = Counter("aiready_policy_decisions_total", "Policy decisions", ["service", "action", "decision"],
                              registry=r)
        self.stage = Histogram("aiready_stage_latency_seconds", "Latency per pipeline stage", ["service", "stage"],
                               registry=r, buckets=(0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.3, 0.5, 1, 2, 5))
        self.quality = Gauge("aiready_quality_score", "Rolling quality score", ["service", "app", "feature", "scorer"],
                             registry=r)
        self.queue_depth = Gauge("aiready_queue_depth", "Queue depth", ["service", "queue"], registry=r)
        self.pipeline = Counter("aiready_pipeline_items_total", "Ingestion items", ["source", "stage", "outcome"],
                                registry=r)
        self.fallbacks = Counter("aiready_fallbacks_total", "Fallback events", ["service", "kind"], registry=r)

    @contextlib.contextmanager
    def time_stage(self, service: str, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stage.labels(service, stage).observe(time.perf_counter() - t0)


metrics = Metrics()


# --------------------------------------------------------------------------- setup


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage(), "service": recorder.service, "trace_id": _trace_id.get()}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_telemetry(service: str, *, json_logs: bool | None = None, level: str | None = None) -> None:
    global _otel_tracer
    recorder.service = service
    os.environ.setdefault("AIREADY_SERVICE", service)
    handler = logging.StreamHandler(sys.stdout)
    if json_logs if json_logs is not None else os.environ.get("AIREADY_JSON_LOGS", "1") == "1":
        handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level or os.environ.get("AIREADY_LOG_LEVEL", "INFO"))

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create({"service.name": service}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            _otel_tracer = trace.get_tracer("aiready")
        except ImportError:
            logging.getLogger(__name__).warning("OTEL endpoint set but opentelemetry not installed (pip install aiready[otel])")

    collector = os.environ.get("AIREADY_TRACE_COLLECTOR_URL")
    if collector:
        recorder.exporters.append(_http_exporter(collector))


def _http_exporter(url: str) -> Callable[[SpanRecord], None]:
    """Batches spans to the AI-Ready observability service in a background thread."""
    import queue

    import httpx

    q: "queue.Queue[SpanRecord]" = queue.Queue(maxsize=10000)

    def worker() -> None:
        client = httpx.Client(timeout=5.0)
        while True:
            batch = [q.get()]
            try:
                while len(batch) < 200:
                    batch.append(q.get(timeout=0.5))
            except queue.Empty:
                pass
            try:
                client.post(url.rstrip("/") + "/v1/spans", json=[b.model_dump() for b in batch])
            except Exception:
                pass

    threading.Thread(target=worker, name="span-exporter", daemon=True).start()

    def export(rec: SpanRecord) -> None:
        try:
            q.put_nowait(rec)
        except queue.Full:
            pass

    return export
