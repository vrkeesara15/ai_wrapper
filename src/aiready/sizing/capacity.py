"""Capacity worksheet per pool, exactly as on the "Capacity and sizing method" slide.

    01 Measure demand   peak users at once x requests per user per minute x tokens per request (in and out)
                        / 60 = demand, tokens per second
    02 Measure supply   tokens per second one GPU node sustains for the chosen model at the target latency,
                        taken from a load test on the client's hardware, never from a brochure
    03 Divide, then add nodes for load = demand / supply; add one node so a failure is not an outage, then
                        headroom for growth
    04 Repeat per pool  chat models, small models, embedding and rerank; ingestion is sized by backlog and
                        the nightly window

Rounding: nodes for load are rounded up (a fraction of a node is a node); headroom is a percentage of the
load nodes, also rounded up: ``nodes to order = ceil(demand / supply) + 1 + ceil(ceil(demand / supply) x
headroom%)``. "No node count is quoted before the load test": without a measured supply the result is
``pending_load_test`` and carries no node count.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from aiready.core.errors import ValidationFailed

BROCHURE_WORDS = ("brochure", "datasheet", "data sheet", "vendor", "marketing", "spec sheet", "estimate")


class PoolWorksheet(BaseModel):
    """One pool. Bracketed values on the slide are ``None`` until filled in during the assessment."""

    pool: str
    model: str | None = None
    peak_users: float | None = None  # users active at the same time at peak
    requests_per_user_per_min: float | None = None
    tokens_per_request: float | None = None  # in + out; or give tokens_in and tokens_out
    tokens_in: float | None = None
    tokens_out: float | None = None
    supply_tokens_per_sec_per_node: float | None = None  # measured
    supply_source: str | None = None  # where the supply figure comes from (load test id / date / hardware)
    target_latency: str | None = None  # the latency the supply was measured at, e.g. "p95 TTFT 800 ms"
    redundancy_nodes: int = 1  # "add one node so a failure is not an outage"
    headroom_pct: float = 20.0  # growth headroom on top of the load nodes

    @model_validator(mode="after")
    def _check(self) -> PoolWorksheet:
        if self.tokens_per_request is None and (self.tokens_in is not None or self.tokens_out is not None):
            self.tokens_per_request = (self.tokens_in or 0) + (self.tokens_out or 0)
        if self.supply_tokens_per_sec_per_node is not None:
            if self.supply_tokens_per_sec_per_node <= 0:
                raise ValueError("measured supply must be positive")
            src = (self.supply_source or "").lower()
            if any(w in src for w in BROCHURE_WORDS):
                raise ValueError("supply must come from a load test on the client's hardware, never from a brochure")
        for name in ("peak_users", "requests_per_user_per_min", "tokens_per_request", "headroom_pct"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must not be negative")
        return self

    def missing(self) -> list[str]:
        return [n for n in ("peak_users", "requests_per_user_per_min", "tokens_per_request") if getattr(self, n) is None]


class PoolSizing(BaseModel):
    pool: str
    model: str | None = None
    status: Literal["sized", "pending_load_test", "incomplete"]
    missing: list[str] = Field(default_factory=list)
    demand_tokens_per_sec: float | None = None
    supply_tokens_per_sec_per_node: float | None = None
    supply_source: str | None = None
    nodes_for_load: float | None = None  # demand / supply, unrounded
    load_nodes: int | None = None  # rounded up
    redundancy_nodes: int = 1
    headroom_nodes: int | None = None
    nodes_to_order: int | None = None
    utilisation_at_peak: float | None = None  # demand / (supply x nodes_to_order)


def demand_tokens_per_sec(peak_users: float, requests_per_user_per_min: float, tokens_per_request: float) -> float:
    """Step 01: the product / 60."""
    return peak_users * requests_per_user_per_min * tokens_per_request / 60.0


def nodes_to_order(demand: float, supply: float, *, redundancy: int = 1, headroom_pct: float = 20.0) -> tuple[float, int, int, int]:
    """Step 03: (nodes for load, load nodes rounded up, headroom nodes, nodes to order)."""
    if supply <= 0:
        raise ValidationFailed("supply per node must be positive")
    raw = demand / supply
    load = math.ceil(raw - 1e-9) if raw > 0 else 0
    head = math.ceil(load * headroom_pct / 100.0 - 1e-9) if load else 0
    return raw, load, head, load + redundancy + head


def size_pool(ws: PoolWorksheet) -> PoolSizing:
    missing = ws.missing()
    if missing:
        return PoolSizing(pool=ws.pool, model=ws.model, status="incomplete", missing=missing,
                          redundancy_nodes=ws.redundancy_nodes)
    demand = demand_tokens_per_sec(ws.peak_users or 0, ws.requests_per_user_per_min or 0, ws.tokens_per_request or 0)
    if ws.supply_tokens_per_sec_per_node is None:
        return PoolSizing(pool=ws.pool, model=ws.model, status="pending_load_test", demand_tokens_per_sec=round(demand, 3),
                          redundancy_nodes=ws.redundancy_nodes, missing=["supply_tokens_per_sec_per_node"])
    raw, load, head, total = nodes_to_order(demand, ws.supply_tokens_per_sec_per_node, redundancy=ws.redundancy_nodes,
                                            headroom_pct=ws.headroom_pct)
    return PoolSizing(
        pool=ws.pool, model=ws.model, status="sized", demand_tokens_per_sec=round(demand, 3),
        supply_tokens_per_sec_per_node=ws.supply_tokens_per_sec_per_node, supply_source=ws.supply_source,
        nodes_for_load=round(raw, 4), load_nodes=load, redundancy_nodes=ws.redundancy_nodes, headroom_nodes=head,
        nodes_to_order=total,
        utilisation_at_peak=round(demand / (ws.supply_tokens_per_sec_per_node * total), 4) if total else 0.0,
    )


def default_pools() -> list[PoolWorksheet]:
    """The pools the slide repeats the worksheet for, with every bracketed value still open."""
    return [PoolWorksheet(pool="chat"), PoolWorksheet(pool="small"), PoolWorksheet(pool="embedding"),
            PoolWorksheet(pool="rerank")]


# --------------------------------------------------------------------------- ingestion


class IngestionWorksheet(BaseModel):
    """Ingestion is sized by the backlog (first full load) and the nightly window (daily changes)."""

    backlog_documents: float
    avg_chunks_per_document: float
    tokens_per_chunk: float
    daily_changed_documents: float
    nightly_window_hours: float = 6.0
    backfill_days: float = 7.0  # how many nights the first load may take
    embed_supply_tokens_per_sec_per_node: float | None = None  # measured on the embedding pool
    parse_docs_per_sec_per_worker: float | None = None  # measured parser throughput
    supply_source: str | None = None
    headroom_pct: float = 20.0

    @model_validator(mode="after")
    def _check(self) -> IngestionWorksheet:
        if self.nightly_window_hours <= 0 or self.backfill_days <= 0:
            raise ValueError("nightly_window_hours and backfill_days must be positive")
        if any(w in (self.supply_source or "").lower() for w in BROCHURE_WORDS):
            raise ValueError("supply must come from a load test on the client's hardware, never from a brochure")
        return self


class IngestionSizing(BaseModel):
    backlog_tokens: float
    backfill_tokens_per_sec: float
    nightly_tokens: float
    nightly_tokens_per_sec: float
    embed_nodes_backfill: int | None = None
    embed_nodes_nightly: int | None = None
    parse_workers_backfill: int | None = None
    parse_workers_nightly: int | None = None
    status: Literal["sized", "pending_load_test"]


def size_ingestion(ws: IngestionWorksheet) -> IngestionSizing:
    window_s = ws.nightly_window_hours * 3600
    backlog_tokens = ws.backlog_documents * ws.avg_chunks_per_document * ws.tokens_per_chunk
    nightly_tokens = ws.daily_changed_documents * ws.avg_chunks_per_document * ws.tokens_per_chunk
    backfill_tps = backlog_tokens / (window_s * ws.backfill_days)
    nightly_tps = nightly_tokens / window_s

    def nodes(demand: float, supply: float | None) -> int | None:
        if supply is None:
            return None
        load = math.ceil(demand / supply - 1e-9) if demand > 0 else 0
        return max(1, load + math.ceil(load * ws.headroom_pct / 100 - 1e-9)) if load else 0

    docs_backfill = ws.backlog_documents / (window_s * ws.backfill_days)
    docs_nightly = ws.daily_changed_documents / window_s
    measured = ws.embed_supply_tokens_per_sec_per_node is not None and ws.parse_docs_per_sec_per_worker is not None
    return IngestionSizing(
        backlog_tokens=backlog_tokens, backfill_tokens_per_sec=round(backfill_tps, 3), nightly_tokens=nightly_tokens,
        nightly_tokens_per_sec=round(nightly_tps, 3),
        embed_nodes_backfill=nodes(backfill_tps, ws.embed_supply_tokens_per_sec_per_node),
        embed_nodes_nightly=nodes(nightly_tps, ws.embed_supply_tokens_per_sec_per_node),
        parse_workers_backfill=nodes(docs_backfill, ws.parse_docs_per_sec_per_worker),
        parse_workers_nightly=nodes(docs_nightly, ws.parse_docs_per_sec_per_worker),
        status="sized" if measured else "pending_load_test",
    )
