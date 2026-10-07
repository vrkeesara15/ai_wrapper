"""Cost reports, chargeback and latency against the budget.

Cost is derived from the same spans as the traces, so a cost figure can always be traced back to the
requests that make it up. A span is a *usage event* when it carries token counts or a cost. Cost recorded by
the gateway wins; otherwise it is computed from ``observability.pricing`` (self-hosted models can be priced at
an internal rate so teams see a chargeback figure for them too).
"""

from __future__ import annotations

import csv
import io
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import BaseModel

from aiready.core.telemetry import SpanRecord
from aiready.observability.config import ModelPrice
from aiready.observability.traces import attr, match_any, span_user

DIMENSIONS = ("app", "feature", "team", "model", "user", "day")


class UsageRecord(BaseModel):
    ts: float
    trace_id: str
    span: str
    app: str = "-"
    feature: str = "-"
    team: str = "unassigned"
    model: str = "-"
    user: str = "-"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    error: bool = False
    cache_hit: bool | None = None

    @property
    def day(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%Y-%m-%d")

    def dim(self, name: str) -> str:
        return self.day if name == "day" else str(getattr(self, name))


class UsageRow(BaseModel):
    key: dict[str, str]
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency_p50_ms: float
    latency_p95_ms: float
    errors: int
    error_rate: float
    cache_hit_rate: float | None = None


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank on sorted values with linear interpolation (matches numpy's default 'linear')."""
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    k = (len(v) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(v[lo] + (v[hi] - v[lo]) * (k - lo))


def resolve_team(app: str, feature: str, teams: dict[str, str]) -> str:
    return teams.get(f"{app}/{feature}") or teams.get(app) or teams.get(f"*/{feature}") or "unassigned"


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int, pricing: dict[str, ModelPrice]) -> float:
    price = pricing.get(model) or pricing.get("*")
    if price is None:
        return 0.0
    return price.per_request + prompt_tokens / 1000 * price.input_per_1k + completion_tokens / 1000 * price.output_per_1k


def usage_records(spans: Iterable[SpanRecord], *, pricing: dict[str, ModelPrice] | None = None,
                  teams: dict[str, str] | None = None) -> list[UsageRecord]:
    """Turn usage spans into records. Missing app/feature/user/team are inherited from other spans of the
    same trace (child spans of the gateway rarely repeat them)."""
    pricing, teams = pricing or {}, teams or {}
    spans = list(spans)
    trace_ctx: dict[str, dict[str, str]] = defaultdict(dict)
    for s in spans:
        ctx = trace_ctx[s.trace_id]
        for key, names in (("app", ("app",)), ("feature", ("feature",)), ("team", ("team",))):
            v = attr(s, *names)
            if v is not None and key not in ctx:
                ctx[key] = str(v)
        u = span_user(s)
        if u and "user" not in ctx:
            ctx["user"] = u
    out: list[UsageRecord] = []
    for s in spans:
        pt = attr(s, "prompt_tokens", "tokens_in", "usage.prompt_tokens")
        ct = attr(s, "completion_tokens", "tokens_out", "usage.completion_tokens")
        cost = attr(s, "cost")
        if pt is None and ct is None and cost is None:
            continue
        ctx = trace_ctx[s.trace_id]
        app = str(attr(s, "app", default=ctx.get("app", "-")))
        feature = str(attr(s, "feature", default=ctx.get("feature", "-")))
        model = str(attr(s, "model", default="-"))
        team = str(attr(s, "team", default=ctx.get("team") or resolve_team(app, feature, teams)))
        pt_i, ct_i = int(pt or 0), int(ct or 0)
        hit = attr(s, "cache_hit")
        out.append(UsageRecord(
            ts=s.start, trace_id=s.trace_id, span=s.name, app=app, feature=feature, team=team, model=model,
            user=span_user(s) or ctx.get("user", "-"), prompt_tokens=pt_i, completion_tokens=ct_i,
            cost=float(cost) if cost is not None else compute_cost(model, pt_i, ct_i, pricing),
            latency_ms=s.duration_ms, error=s.status == "error", cache_hit=bool(hit) if hit is not None else None,
        ))
    return out


def aggregate(records: list[UsageRecord], by: list[str]) -> list[UsageRow]:
    """Group usage records by any of app, feature, team, model, user, day."""
    bad = [b for b in by if b not in DIMENSIONS]
    if bad:
        from aiready.core.errors import ValidationFailed

        raise ValidationFailed(f"unknown dimension(s) {bad}; use {list(DIMENSIONS)}")
    groups: dict[tuple[str, ...], list[UsageRecord]] = defaultdict(list)
    for r in records:
        groups[tuple(r.dim(b) for b in by)].append(r)
    rows = []
    for key, rs in groups.items():
        lat = [r.latency_ms for r in rs]
        hits = [r.cache_hit for r in rs if r.cache_hit is not None]
        errors = sum(r.error for r in rs)
        pt, ct = sum(r.prompt_tokens for r in rs), sum(r.completion_tokens for r in rs)
        rows.append(UsageRow(
            key=dict(zip(by, key)), requests=len(rs), prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
            cost=round(sum(r.cost for r in rs), 6), latency_p50_ms=round(percentile(lat, 50), 3),
            latency_p95_ms=round(percentile(lat, 95), 3), errors=errors, error_rate=round(errors / len(rs), 4),
            cache_hit_rate=round(sum(hits) / len(hits), 4) if hits else None,
        ))
    return sorted(rows, key=lambda r: -r.cost)


# --------------------------------------------------------------------------- chargeback


class ChargebackLine(BaseModel):
    team: str
    requests: int
    total_tokens: int
    cost: float
    share: float  # of the period's total cost
    budget: float | None = None
    budget_used: float | None = None
    by_feature: dict[str, float]
    by_model: dict[str, float]


class ChargebackReport(BaseModel):
    period: str
    since: float
    until: float
    currency: str
    total_cost: float
    lines: list[ChargebackLine]

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["period", "team", "requests", "total_tokens", "cost", "currency", "share", "budget", "budget_used",
                    "top_feature"])
        for ln in self.lines:
            top = max(ln.by_feature.items(), key=lambda x: x[1])[0] if ln.by_feature else ""
            w.writerow([self.period, ln.team, ln.requests, ln.total_tokens, f"{ln.cost:.4f}", self.currency,
                        f"{ln.share:.4f}", "" if ln.budget is None else f"{ln.budget:.2f}",
                        "" if ln.budget_used is None else f"{ln.budget_used:.4f}", top])
        return buf.getvalue()


def month_bounds(period: str | None = None) -> tuple[str, float, float]:
    """``YYYY-MM`` -> (period, first second, last second) in UTC; default the current month."""
    if not period:
        period = datetime.now(timezone.utc).strftime("%Y-%m")
    year, month = (int(x) for x in period.split("-"))
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
    return period, start.timestamp(), end.timestamp() - 1e-6


def chargeback(records: list[UsageRecord], *, period: str, since: float, until: float, currency: str = "EUR",
               budgets: dict[str, float] | None = None) -> ChargebackReport:
    budgets = budgets or {}
    recs = [r for r in records if since <= r.ts <= until]
    total = sum(r.cost for r in recs)
    by_team: dict[str, list[UsageRecord]] = defaultdict(list)
    for r in recs:
        by_team[r.team].append(r)
    lines = []
    for team, rs in by_team.items():
        cost = sum(r.cost for r in rs)
        feat: dict[str, float] = defaultdict(float)
        mod: dict[str, float] = defaultdict(float)
        for r in rs:
            feat[r.feature] += r.cost
            mod[r.model] += r.cost
        budget = budgets.get(f"team:{team}")
        lines.append(ChargebackLine(
            team=team, requests=len(rs), total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in rs),
            cost=round(cost, 6), share=round(cost / total, 6) if total else 0.0, budget=budget,
            budget_used=round(cost / budget, 6) if budget else None,
            by_feature={k: round(v, 6) for k, v in feat.items()}, by_model={k: round(v, 6) for k, v in mod.items()},
        ))
    lines.sort(key=lambda ln: -ln.cost)
    return ChargebackReport(period=period, since=since, until=until, currency=currency, total_cost=round(total, 6),
                            lines=lines)


# --------------------------------------------------------------------------- latency vs budget


class StageLatency(BaseModel):
    stage: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    budget_ms: float | None
    within_budget: bool | None  # p95 <= budget; None when no samples
    over_budget_share: float  # share of samples over the budget


def stage_of(span: SpanRecord, stage_map: dict[str, list[str]]) -> str | None:
    explicit = attr(span, "stage")
    if explicit and str(explicit) in stage_map:
        return str(explicit)
    for stage, patterns in stage_map.items():
        if match_any(span.name, patterns):
            return stage
    return None


def latency_report(spans: Iterable[SpanRecord], budget_ms: dict[str, float],
                   stage_map: dict[str, list[str]]) -> list[StageLatency]:
    """p50/p95/p99 per budget stage. ``first_token`` and ``model_start`` come from ``ttft_ms`` /
    ``model_start_ms`` attributes because they are points inside a span, not spans of their own."""
    samples: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        st = stage_of(s, stage_map)
        if st:
            samples[st].append(s.duration_ms)
        ttft = attr(s, "ttft_ms")
        if ttft is not None:
            samples["first_token"].append(float(ttft))
        ms = attr(s, "model_start_ms")
        if ms is not None:
            samples["model_start"].append(float(ms))
    out = []
    for stage in list(budget_ms) + [s for s in samples if s not in budget_ms]:
        vals = samples.get(stage, [])
        budget = budget_ms.get(stage)
        p95 = percentile(vals, 95)
        out.append(StageLatency(
            stage=stage, count=len(vals), p50_ms=round(percentile(vals, 50), 3), p95_ms=round(p95, 3),
            p99_ms=round(percentile(vals, 99), 3), budget_ms=budget,
            within_budget=(p95 <= budget) if vals and budget is not None else None,
            over_budget_share=round(sum(v > budget for v in vals) / len(vals), 4) if vals and budget else 0.0,
        ))
    return out


def cache_hit_rate(records: list[UsageRecord], spans: Iterable[SpanRecord] = ()) -> float | None:
    vals = [r.cache_hit for r in records if r.cache_hit is not None]
    seen = {(r.trace_id, r.span) for r in records}
    for s in spans:
        hit = attr(s, "cache_hit")
        if hit is not None and (s.trace_id, s.name) not in seen:
            vals.append(bool(hit))
    return round(sum(vals) / len(vals), 4) if vals else None


def window(minutes: float) -> tuple[float, float]:
    t = time.time()
    return t - minutes * 60, t


def as_dicts(rows: list[BaseModel]) -> list[dict[str, Any]]:
    return [r.model_dump() for r in rows]
