"""Usage database for chargeback and usage reports.

One row per gateway request (tokens, cost, latency, model, route, team, feature, user, outcome). Reports group
by any of those dimensions over a date range. The table lives in the platform state database (SQLite for the
pilot, PostgreSQL in production) unless ``gateway.usage.url`` points elsewhere.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

DIMENSIONS = ("tenant", "app", "team", "feature", "user", "model", "route", "operation", "status")


class UsageRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    tenant: str = "default"
    app: str = "default"
    team: str = "default"
    feature: str = "default"
    user: str = ""
    model: str = ""
    route: str = ""
    operation: str = "chat"
    status: str = "ok"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    trace_id: str | None = None


class UsageStore:
    def __init__(self, url: str | None = None, engine: Engine | None = None):
        if engine is None:
            if url is None:
                from aiready.core.stores import get_docstore

                engine = get_docstore().engine
            else:
                kwargs: dict[str, Any] = {"future": True}
                if url.startswith("sqlite"):
                    kwargs["connect_args"] = {"check_same_thread": False}
                engine = create_engine(url, **kwargs)
        self.engine = engine
        self.meta = MetaData()
        self.table = Table(
            "aiready_gateway_usage", self.meta,
            Column("id", String(64), primary_key=True),
            Column("ts", Float, nullable=False, index=True),
            *(Column(d, String(256), nullable=False, default="", index=d in ("team", "user", "model")) for d in DIMENSIONS),
            Column("prompt_tokens", Integer, nullable=False, default=0),
            Column("completion_tokens", Integer, nullable=False, default=0),
            Column("cost", Float, nullable=False, default=0.0),
            Column("latency_ms", Float, nullable=False, default=0.0),
            Column("cached", Boolean, nullable=False, default=False),
            Column("trace_id", String(64)),
        )
        self.meta.create_all(self.engine)
        self._lock = threading.Lock()

    def add(self, rec: UsageRecord) -> None:
        with self._lock, self.engine.begin() as c:
            c.execute(self.table.insert().values(**rec.model_dump()))

    def report(self, *, group_by: list[str] | None = None, start: float | None = None, end: float | None = None,
               filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Sum tokens, cost and requests per group over ``[start, end)``."""
        t = self.table
        dims = [d for d in (group_by or []) if d in DIMENSIONS]
        bad = [d for d in (group_by or []) if d not in DIMENSIONS] + [k for k in (filters or {}) if k not in DIMENSIONS]
        if bad:
            from aiready.core.errors import ValidationFailed

            raise ValidationFailed(f"unknown usage dimension(s) {bad}; use {list(DIMENSIONS)}")
        cols = [t.c[d] for d in dims]
        q = select(*cols, func.count().label("requests"),
                   func.sum(t.c.prompt_tokens).label("prompt_tokens"),
                   func.sum(t.c.completion_tokens).label("completion_tokens"),
                   func.sum(t.c.cost).label("cost"), func.avg(t.c.latency_ms).label("avg_latency_ms"),
                   func.sum(func.cast(t.c.cached, Integer)).label("cache_hits"))
        conds = []
        if start is not None:
            conds.append(t.c.ts >= start)
        if end is not None:
            conds.append(t.c.ts < end)
        for k, v in (filters or {}).items():
            conds.append(t.c[k] == v)
        if conds:
            q = q.where(and_(*conds))
        if cols:
            q = q.group_by(*cols).order_by(*cols)
        with self.engine.connect() as c:
            rows = c.execute(q).mappings().all()
        out = []
        for r in rows:
            row = {d: r[d] for d in dims}
            if not r["requests"]:
                continue
            pt, ct = int(r["prompt_tokens"] or 0), int(r["completion_tokens"] or 0)
            row.update(requests=int(r["requests"]), prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
                       cost=round(float(r["cost"] or 0), 6), avg_latency_ms=round(float(r["avg_latency_ms"] or 0), 1),
                       cache_hits=int(r["cache_hits"] or 0))
            out.append(row)
        return out
