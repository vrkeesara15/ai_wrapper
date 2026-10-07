"""Trace store and trace viewer.

Every service emits ``core.telemetry.SpanRecord`` s (one per stage). They arrive here in batches
(``POST /v1/spans`` from the span exporter, or in-process via ``install_local_exporter``), are masked, and
stored so that a single request can be shown as a tree, searched by feature and time, costed, scored and
replayed.

Span attribute conventions read by this package (all optional; emitters add what they know):

=====================  =====================================================================
``app`` ``feature``    which application and feature the request belongs to (cost/quality grouping)
``team``               who pays (chargeback); else derived from ``observability.teams``
``user`` / ``sub``     signed-in user (never a service account)
``tenant``             tenant of the user
``model``              concrete model that served the call
``prompt_tokens``      tokens in (alias ``tokens_in``, or ``usage.prompt_tokens``)
``completion_tokens``  tokens out (alias ``tokens_out``, or ``usage.completion_tokens``)
``cost``               cost in the configured currency (else computed from ``observability.pricing``)
``cache_hit``          bool, the answer came from the cache
``ttft_ms``            time to first token
``request``            the gateway ChatRequest as JSON - makes the request replayable
``question`` ``answer`` ``passages`` ``citations`` ``found``   answer spans, used by the live scorers
``lag_seconds``        pipeline lag reported by ingestion (alerts)
=====================  =====================================================================
"""

from __future__ import annotations

import fnmatch
import json
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import BaseModel, Field
from sqlalchemy import Column, Float, Index, MetaData, String, Table, Text, create_engine, delete, select
from sqlalchemy.engine import Engine

from aiready.core.plugins import register
from aiready.core.telemetry import SpanRecord

# --------------------------------------------------------------------------- attribute helpers


def attr(span: SpanRecord, *names: str, default: Any = None) -> Any:
    """First present attribute among ``names``; dotted names look inside nested dicts (``usage.prompt_tokens``)."""
    for n in names:
        node: Any = span.attributes
        ok = True
        for part in n.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and node is not None:
            return node
    return default


def span_user(span: SpanRecord) -> str | None:
    u = attr(span, "user", "sub", "principal.sub")
    return str(u) if u is not None else None


def match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


class TraceSummary(BaseModel):
    trace_id: str
    root: str
    start: float
    duration_ms: float
    spans: int
    status: str
    app: str | None = None
    feature: str | None = None
    user: str | None = None
    model: str | None = None


class TraceNode(BaseModel):
    span: SpanRecord
    duration_ms: float
    children: list[TraceNode] = Field(default_factory=list)


class TraceTree(BaseModel):
    trace_id: str
    summary: TraceSummary
    roots: list[TraceNode]
    replays: list[str] = Field(default_factory=list)  # trace ids of replays of this trace


def summarize(trace_id: str, spans: list[SpanRecord]) -> TraceSummary:
    spans = sorted(spans, key=lambda s: s.start)
    ids = {s.span_id for s in spans}
    roots = [s for s in spans if not s.parent_id or s.parent_id not in ids] or spans[:1]
    start = min(s.start for s in spans)
    end = max((s.end or s.start) for s in spans)

    def first(*names: str) -> Any:
        for s in spans:
            v = attr(s, *names)
            if v is not None:
                return str(v)
        return None

    return TraceSummary(
        trace_id=trace_id, root=roots[0].name, start=start, duration_ms=round((end - start) * 1000, 3),
        spans=len(spans), status="error" if any(s.status == "error" for s in spans) else "ok",
        app=first("app"), feature=first("feature"), user=next((u for u in map(span_user, spans) if u), None),
        model=first("model"),
    )


def build_tree(trace_id: str, spans: list[SpanRecord]) -> TraceTree:
    """Full request tree: spans nested by parent id, children in start order. Orphans become roots so a
    partially exported trace is still viewable."""
    nodes = {s.span_id: TraceNode(span=s, duration_ms=round(s.duration_ms, 3)) for s in spans}
    roots: list[TraceNode] = []
    for s in sorted(spans, key=lambda x: x.start):
        node = nodes[s.span_id]
        parent = nodes.get(s.parent_id) if s.parent_id else None
        (parent.children if parent else roots).append(node)
    return TraceTree(trace_id=trace_id, summary=summarize(trace_id, spans), roots=roots)


# --------------------------------------------------------------------------- stores


class TraceStore(Protocol):
    def add(self, spans: list[SpanRecord]) -> int: ...
    def trace(self, trace_id: str) -> list[SpanRecord]: ...
    def find_traces(self, *, app: str | None = None, feature: str | None = None, user: str | None = None,
                    tenant: str | None = None, since: float | None = None, until: float | None = None,
                    limit: int = 50) -> list[str]: ...
    def spans(self, *, since: float | None = None, until: float | None = None,
              name_patterns: list[str] | None = None) -> list[SpanRecord]: ...
    def purge(self, older_than: float) -> int: ...


def _index_fields(s: SpanRecord) -> dict[str, Any]:
    return {
        "tenant": _s(attr(s, "tenant")), "app": _s(attr(s, "app")), "feature": _s(attr(s, "feature")),
        "team": _s(attr(s, "team")), "model": _s(attr(s, "model")), "user_sub": _s(span_user(s)),
    }


def _s(v: Any) -> str | None:
    return None if v is None else str(v)[:256]


@register("trace_store", "memory")
class MemoryTraceStore:
    """Process-local store for tests and single-process demos."""

    def __init__(self) -> None:
        self._spans: dict[tuple[str, str], SpanRecord] = {}
        self._lock = threading.RLock()

    def add(self, spans: list[SpanRecord]) -> int:
        with self._lock:
            for s in spans:
                self._spans[(s.trace_id, s.span_id)] = s
        return len(spans)

    def trace(self, trace_id: str) -> list[SpanRecord]:
        with self._lock:
            return sorted((s for (t, _), s in self._spans.items() if t == trace_id), key=lambda s: s.start)

    def find_traces(self, *, app: str | None = None, feature: str | None = None, user: str | None = None,
                    tenant: str | None = None, since: float | None = None, until: float | None = None,
                    limit: int = 50) -> list[str]:
        want = {"app": app, "feature": feature, "user_sub": user, "tenant": tenant}
        hits: dict[str, float] = {}
        with self._lock:
            for s in self._spans.values():
                if since is not None and s.start < since or until is not None and s.start > until:
                    continue
                idx = _index_fields(s)
                if all(v is None or idx[k] == v for k, v in want.items()):
                    hits[s.trace_id] = max(hits.get(s.trace_id, 0.0), s.start)
        return [t for t, _ in sorted(hits.items(), key=lambda x: -x[1])[:limit]]

    def spans(self, *, since: float | None = None, until: float | None = None,
              name_patterns: list[str] | None = None) -> list[SpanRecord]:
        with self._lock:
            out = [s for s in self._spans.values()
                   if (since is None or s.start >= since) and (until is None or s.start <= until)
                   and (not name_patterns or match_any(s.name, name_patterns))]
        return sorted(out, key=lambda s: s.start)

    def purge(self, older_than: float) -> int:
        with self._lock:
            old = [k for k, s in self._spans.items() if s.start < older_than]
            for k in old:
                del self._spans[k]
        return len(old)


@register("trace_store", "sql")
class SqlTraceStore:
    """SQL trace store (SQLite for the pilot, PostgreSQL in production) with the columns the viewer and the
    reports filter on promoted out of the JSON body and indexed."""

    def __init__(self, url: str | None = None, engine: Engine | None = None, table: str = "aiready_spans"):
        if engine is None:
            if url is None:
                from aiready.core.config import get_config

                url = get_config().get("state.url", "sqlite:///data/state.db")
            kwargs: dict[str, Any] = {"future": True}
            if url.startswith("sqlite"):
                kwargs["connect_args"] = {"check_same_thread": False}
                if ":memory:" in url:
                    from sqlalchemy.pool import StaticPool

                    kwargs["poolclass"] = StaticPool
                elif url.startswith("sqlite:///"):
                    from pathlib import Path

                    Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(url, **kwargs)
        self.engine = engine
        self.meta = MetaData()
        self.t = Table(
            table, self.meta,
            Column("trace_id", String(64), primary_key=True),
            Column("span_id", String(64), primary_key=True),
            Column("parent_id", String(64)),
            Column("name", String(256), nullable=False),
            Column("service", String(128)),
            Column("start", Float, nullable=False),
            Column("end", Float),
            Column("status", String(32)),
            Column("tenant", String(256)),
            Column("app", String(256)),
            Column("feature", String(256)),
            Column("team", String(256)),
            Column("model", String(256)),
            Column("user_sub", String(256)),
            Column("body", Text, nullable=False),
            Index(f"ix_{table}_start", "start"),
            Index(f"ix_{table}_feature", "feature", "start"),
            Index(f"ix_{table}_user", "user_sub", "start"),
        )
        self.meta.create_all(self.engine)
        self._lock = threading.RLock()

    def add(self, spans: list[SpanRecord]) -> int:
        if not spans:
            return 0
        t = self.t
        rows = [{"trace_id": s.trace_id, "span_id": s.span_id, "parent_id": s.parent_id, "name": s.name,
                 "service": s.service, "start": s.start, "end": s.end, "status": s.status, **_index_fields(s),
                 "body": s.model_dump_json()} for s in spans]
        with self._lock, self.engine.begin() as c:
            for s in spans:  # idempotent re-delivery: the exporter may resend a batch
                c.execute(delete(t).where(t.c.trace_id == s.trace_id, t.c.span_id == s.span_id))
            c.execute(t.insert(), rows)
        return len(rows)

    def trace(self, trace_id: str) -> list[SpanRecord]:
        t = self.t
        with self.engine.connect() as c:
            rows = c.execute(select(t.c.body).where(t.c.trace_id == trace_id).order_by(t.c.start)).all()
        return [SpanRecord.model_validate_json(r[0]) for r in rows]

    def find_traces(self, *, app: str | None = None, feature: str | None = None, user: str | None = None,
                    tenant: str | None = None, since: float | None = None, until: float | None = None,
                    limit: int = 50) -> list[str]:
        t = self.t
        q = select(t.c.trace_id, t.c.start)
        for col, val in ((t.c.app, app), (t.c.feature, feature), (t.c.user_sub, user), (t.c.tenant, tenant)):
            if val is not None:
                q = q.where(col == val)
        if since is not None:
            q = q.where(t.c.start >= since)
        if until is not None:
            q = q.where(t.c.start <= until)
        q = q.order_by(t.c.start.desc()).limit(limit * 20)
        seen: list[str] = []
        with self.engine.connect() as c:
            for tid, _ in c.execute(q):
                if tid not in seen:
                    seen.append(tid)
                    if len(seen) >= limit:
                        break
        return seen

    def spans(self, *, since: float | None = None, until: float | None = None,
              name_patterns: list[str] | None = None) -> list[SpanRecord]:
        t = self.t
        q = select(t.c.name, t.c.body)
        if since is not None:
            q = q.where(t.c.start >= since)
        if until is not None:
            q = q.where(t.c.start <= until)
        q = q.order_by(t.c.start)
        with self.engine.connect() as c:
            rows = c.execute(q).all()
        return [SpanRecord.model_validate_json(b) for n, b in rows if not name_patterns or match_any(n, name_patterns)]

    def purge(self, older_than: float) -> int:
        t = self.t
        with self._lock, self.engine.begin() as c:
            return c.execute(delete(t).where(t.c.start < older_than)).rowcount or 0


# --------------------------------------------------------------------------- grouping


def group_by_trace(spans: Iterable[SpanRecord]) -> dict[str, list[SpanRecord]]:
    out: dict[str, list[SpanRecord]] = defaultdict(list)
    for s in spans:
        out[s.trace_id].append(s)
    return out


def recorded_request(spans: list[SpanRecord]) -> tuple[SpanRecord, dict[str, Any]] | None:
    """The gateway request recorded in a trace (first span with a ``request`` attribute holding messages)."""
    for s in sorted(spans, key=lambda x: x.start):
        req = attr(s, "request")
        if isinstance(req, str):
            try:
                req = json.loads(req)
            except ValueError:
                continue
        if isinstance(req, dict) and req.get("messages"):
            return s, req
    return None


def now() -> float:
    return time.time()
