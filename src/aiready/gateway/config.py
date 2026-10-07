"""Gateway configuration (top-level section ``gateway:``).

Model list, routing rules, budgets, rate limits and keys are configuration: the application never changes when
they do. Named maps (``routes``, ``budgets``, ``rate_limits``) are dicts rather than lists so overlays and the
admin API can change one entry without restating the others. Defaults run with the mock backend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Scope = Literal["user", "team", "app", "feature", "tenant"]
ResponseMode = Literal["check_then_send", "stream_then_verify"]


class AuthConfig(BaseModel):
    # Require a service identity (X-Service-Token or a gateway API key) on top of the user token.
    require_client_identity: bool = False
    api_key_header: str = "X-API-Key"
    # Operations a service may call without a signed-in user (pipelines, ingestion, batch jobs).
    service_only_operations: list[str] = Field(default_factory=lambda: ["embed", "rerank", "batch"])
    # Claim holding the caller's team (for budgets/chargeback); falls back to the API key's team, then the first group.
    team_claim: str = "team"


class RateLimitRule(BaseModel):
    """Sliding-window request limit for one scope. ``match`` restricts it to one value ('*' = every value)."""

    scope: Scope = "user"
    match: str = "*"
    requests: int = Field(60, ge=1)
    window_seconds: float = Field(60.0, gt=0)
    operations: list[str] = Field(default_factory=list)  # empty = all (chat, stream, embed, rerank)
    enabled: bool = True


class BudgetRule(BaseModel):
    """Token / cost budget per period for one scope; ``block`` refuses with 429, ``warn`` only records."""

    scope: Scope = "team"
    match: str = "*"
    period: Literal["day", "month"] = "day"
    max_tokens: int | None = Field(None, ge=0)
    max_cost: float | None = Field(None, ge=0)
    action: Literal["block", "warn"] = "block"
    enabled: bool = True


class GuardrailStepConfig(BaseModel):
    enabled: bool = True
    screen_roles: list[str] = Field(default_factory=lambda: ["user", "tool"])  # system prompts are the app's own
    apply_masking: bool = True              # forward the masked text, not the original
    restore_masked_in_output: bool = True   # put the user's own values back in the answer they receive
    block_checks: list[str] = Field(default_factory=lambda: ["injection"])  # any finding of these blocks
    fail_closed: bool = True                # guardrail service unreachable -> refuse rather than skip


class CacheStepConfig(BaseModel):
    enabled: bool = True
    exact: bool = True
    semantic: bool = True
    exact_ttl_seconds: float = 600
    semantic_threshold: float = Field(0.92, ge=0, le=1)
    semantic_ttl_seconds: float = 3600
    ttl_by_content_type: dict[str, float] = Field(default_factory=dict)
    max_temperature: float = 0.3            # sampled answers above this are not reused
    embedding_model: str = "embedding"      # catalog model used to embed questions for the semantic cache
    max_entries_per_scope: int = 5000


class RouteMatch(BaseModel):
    model: list[str] = Field(default_factory=list)    # request model aliases ("auto", "chat"...)
    task: list[str] = Field(default_factory=list)     # metadata.task
    feature: list[str] = Field(default_factory=list)  # metadata.feature
    app: list[str] = Field(default_factory=list)


class RouteConfig(BaseModel):
    """Pick the model by task, cost and policy.

    ``candidates`` are the models that may serve the route; ``eval_pass`` the subset that passed the
    evaluation for this task (when set, only these are used). Among eligible ones the smallest tier, then the
    cheapest, is preferred; ``fallback`` names models tried first when the chosen one fails.
    """

    description: str = ""
    priority: int = 0
    match: RouteMatch = Field(default_factory=RouteMatch)
    candidates: list[str] = Field(default_factory=list)
    eval_pass: list[str] | None = None
    fallback: list[str] = Field(default_factory=list)
    escalate_prompt_tokens: int | None = None   # prompts larger than this skip the small tier
    max_tokens: int | None = None               # cap on output tokens for this route
    response_mode: ResponseMode | None = None
    content_type: str = "default"               # semantic-cache TTL class
    cache: bool = True
    enabled: bool = True


class CallConfig(BaseModel):
    timeout_seconds: float = Field(60.0, gt=0)
    stream_first_token_timeout_seconds: float = Field(30.0, gt=0)
    retries: int = Field(1, ge=0)               # per model, before falling back
    retry_backoff_seconds: float = Field(0.2, ge=0)
    max_fallbacks: int = Field(2, ge=0)
    reserve_output_tokens: int = Field(1024, ge=0)  # kept free in the context window when trimming
    prefix_cache_ordering: bool = True          # system prompts first, tools sorted: stable prefixes hit the cache


class ValidateConfig(BaseModel):
    enabled: bool = True
    on_failure: Literal["regenerate_once", "reject", "flag"] = "regenerate_once"


class EgressConfig(BaseModel):
    """Public (outside) models. Closed by default: even with this enabled a public model is used only when the
    model's ``data_classes`` include the request's data class AND policy allows ``egress.send``."""

    enabled: bool = False


class TargetConfig(BaseModel):
    """Where embeddings / rerank go: a model pool in the catalog, or the embedding / rerank service."""

    target: Literal["pool", "service"] = "pool"
    model: str = "embedding"
    max_items: int = Field(2048, ge=1)


class BatchConfig(BaseModel):
    enabled: bool = True
    window: str = "22:00-06:00"   # local time when queued work runs; "always" = whenever the worker polls
    poll_seconds: float = Field(30.0, gt=0)
    max_requests_per_batch: int = Field(1000, ge=1)
    queue: dict[str, Any] | None = None  # queue spec; default: the platform `queue:` section
    topic: str = "gateway.batches"


class UsageConfig(BaseModel):
    url: str | None = None   # SQL database for the usage table; default: the platform state database


class AdminConfig(BaseModel):
    required_role: str = "platform-admin"
    overlay_file: str | None = None   # also persist admin changes as a YAML overlay (mounted overlay dir)
    sync_seconds: float = Field(5.0, ge=0)  # how often a data-plane replica pulls admin changes (0 = every request)


class StreamingConfig(BaseModel):
    default_mode: ResponseMode = "stream_then_verify"


class GatewayConfig(BaseModel):
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limits: dict[str, RateLimitRule] = Field(default_factory=dict)
    budgets: dict[str, BudgetRule] = Field(default_factory=dict)
    guardrails: GuardrailStepConfig = Field(default_factory=GuardrailStepConfig)
    cache: CacheStepConfig = Field(default_factory=CacheStepConfig)
    routes: dict[str, RouteConfig] = Field(default_factory=dict)
    default_route: str = "auto"
    allow_direct_models: bool = True   # a request may name a catalog model instead of a route
    call: CallConfig = Field(default_factory=CallConfig)
    validate_output: ValidateConfig = Field(default_factory=ValidateConfig, alias="validate")
    egress: EgressConfig = Field(default_factory=EgressConfig)
    embedding: TargetConfig = Field(default_factory=lambda: TargetConfig(model="embedding"))
    rerank: TargetConfig = Field(default_factory=lambda: TargetConfig(model="reranker"))
    batches: BatchConfig = Field(default_factory=BatchConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    kv: dict[str, Any] | None = None   # counter / cache store; default: the platform `kv:` section
    currency: str = "USD"

    model_config = {"populate_by_name": True}

    @field_validator("routes", "budgets", "rate_limits", mode="before")
    @classmethod
    def _none_is_empty(cls, v: Any) -> Any:
        return v or {}
