"""Configuration of the observability service (top-level YAML section ``observability``).

Defaults run locally with no infrastructure: spans in the shared SQL state database, regex masking, a 10%
live-scoring sample, the latency budget from the "Latency budget and response modes" slide.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

#: Design targets from the latency budget slide (milliseconds). Confirmed by load testing, never assumed.
DEFAULT_LATENCY_BUDGET_MS: dict[str, float] = {
    "screen": 50, "embed": 30, "search": 70, "rerank": 150, "model_start": 500, "first_token": 800, "check": 500,
}

#: Span-name globs mapped to budget stages. First match wins. Clients add their own span names here.
DEFAULT_STAGE_MAP: dict[str, list[str]] = {
    "screen": ["guardrails.screen*", "*.screen_input", "*.screen"],
    "embed": ["embedding.*", "*.embed", "*.embed_query"],
    "search": ["search.search", "search.hybrid", "*.search", "*.hybrid_search"],
    "rerank": ["rerank.*", "*.rerank"],
    "check": ["*.verify", "*.grounding*", "guardrails.validate*", "*.check"],
}


class StoreConfig(BaseModel):
    """Trace store plugin spec. ``type: sql`` (sqlite/postgres via SQLAlchemy) or ``memory``;
    ``class: pkg.mod:Class`` for anything else (ClickHouse, Elasticsearch...)."""

    type: str | None = "sql"
    url: str | None = None  # default: the shared ``state.url`` database
    retention_days: int = 90
    extra: dict[str, Any] = Field(default_factory=dict)

    def spec(self) -> dict[str, Any]:
        d: dict[str, Any] = {**self.extra}
        if self.type:
            d["type"] = self.type
        if self.url:
            d["url"] = self.url
        return d


class MaskingConfig(BaseModel):
    use_guardrails: bool = True
    extra_patterns: dict[str, str] = Field(default_factory=dict)  # ENTITY -> regex
    disabled_entities: list[str] = Field(default_factory=list)


class IngestConfig(BaseModel):
    # The span exporter in core.telemetry posts without credentials. Keep this false and give the exporter a
    # service token (or put the endpoint behind a network policy) - see config/observability.yaml.
    allow_anonymous: bool = False
    max_batch: int = 5000
    # Also store the spans of the process this service runs in (single-process pilot: every package in one
    # process, so the in-memory span recorder already holds the whole request).
    capture_local: bool = False
    flush_interval_seconds: float = 2.0


class ModelPrice(BaseModel):
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    per_request: float = 0.0


class ScoringConfig(BaseModel):
    enabled: bool = True
    sample_rate: float = 0.1  # share of production answers scored (deterministic on trace id)
    low_score_threshold: float = 0.6  # overall score below this goes to the review queue
    scorers: list[dict[str, Any]] = Field(default_factory=lambda: [
        {"type": "grounding"}, {"type": "citation_validity"}, {"type": "refusal"}, {"type": "pii_leak"},
    ])
    llm_judge: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "route": "judge"})
    answer_span_names: list[str] = Field(default_factory=lambda: ["rag.answer", "*.answer", "agents.run", "assistant.*"])


class ReviewConfig(BaseModel):
    causes: list[str] = Field(default_factory=lambda: ["retrieval", "prompt", "model", "data", "policy"])
    negative_rating_threshold: int = 2  # ratings on a 1..5 scale at or below this are negative


class AlertRuleConfig(BaseModel):
    name: str
    kind: str  # quality | latency | spend | error_rate | pipeline_lag
    threshold: float | None = None  # defaults: quality 0.7, latency = budget, error_rate 0.05, lag 3600s
    window_minutes: int = 60
    period: str | None = None  # "month" = month to date (default for spend), else the sliding window
    scope: dict[str, str] = Field(default_factory=dict)  # app / feature / team / model / stage / scorer
    severity: str = "warning"
    notifiers: list[str] = Field(default_factory=list)  # names from ``notifiers``; empty = all
    min_samples: int = 1


class AlertsConfig(BaseModel):
    enabled: bool = True
    interval_seconds: int = 60
    rules: list[AlertRuleConfig] = Field(default_factory=list)
    notifiers: dict[str, dict[str, Any]] = Field(default_factory=lambda: {"log": {"type": "log"}})


class AccessConfig(BaseModel):
    """Who may see what. Users always see their own traces; these roles see everything in their tenant."""

    viewer_roles: list[str] = Field(default_factory=lambda: ["admin", "ops", "auditor"])
    replay_roles: list[str] = Field(default_factory=lambda: ["admin"])
    reviewer_roles: list[str] = Field(default_factory=lambda: ["admin", "reviewer", "ai-engineer"])
    report_roles: list[str] = Field(default_factory=lambda: ["admin", "ops", "finance"])


class ObservabilityConfig(BaseModel):
    store: StoreConfig = Field(default_factory=StoreConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    latency_budget_ms: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_LATENCY_BUDGET_MS))
    stage_map: dict[str, list[str]] = Field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_STAGE_MAP.items()})
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)  # model -> price, used when spans carry no cost
    currency: str = "EUR"
    teams: dict[str, str] = Field(default_factory=dict)  # "app" or "app/feature" -> team, when spans carry none
    budgets: dict[str, float] = Field(default_factory=dict)  # "team:<t>" / "feature:<f>" / "app:<a>" -> monthly spend
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    access: AccessConfig = Field(default_factory=AccessConfig)


def load_config() -> ObservabilityConfig:
    from aiready.core.config import get_config

    return get_config().section("observability", ObservabilityConfig)
