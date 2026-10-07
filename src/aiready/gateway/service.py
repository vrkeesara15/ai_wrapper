"""The AI gateway data plane: the only path from the platform to a model.

Every chat request passes the same eight steps, in this order (slide "Internal API gateway"):

    1 Authenticate  service identity + user token
    2 Limit         rate limits and token budgets (per user / team / app / feature)
    3 Screen        injection and personal-data checks; masking; the data class feeds routing and egress
    4 Reuse         exact and semantic cache, scoped to app, tenant, access groups, prompt version and model
    5 Route         pick the model by task, cost and policy (smallest model that passed the evaluation)
    6 Call          timeouts, retries, fallback chain, context trimming, prefix-cache friendly ordering
    7 Validate      schema and safety checks on the output
    8 Record        tokens, latency and cost per feature/team/model; usage row; audit record; trace

Streaming supports both response modes of the latency-budget slide: *check then send* (Mode A: the answer is
generated and validated before the first word is sent) and *stream then verify* (Mode B: words flow as they
are generated; the check runs at the end and can withdraw the answer with an ``aiready.withdrawn`` event).

The service is stateless: counters and caches live in the KV store, usage rows in the state database, so the
data plane runs as several identical replicas.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from aiready.cache.semantic import CacheScope, ExactCache, SemanticCache
from aiready.core.audit import get_audit
from aiready.core.config import ConfigStore, get_config
from aiready.core.config import get_config
from aiready.core.errors import (
    AIReadyError,
    AuthenticationError,
    ConfigError,
    GuardrailBlocked,
    UpstreamError,
    ValidationFailed,
)
from aiready.core.identity import Principal
from aiready.core.interfaces import Finding
from aiready.core.stores import KVStore, make_kv
from aiready.core.telemetry import current_trace_id, metrics, new_trace_id, set_trace_id, span
from aiready.core.textutil import sha256
from aiready.core.tokens import count_messages, count_tokens
from aiready.core.types import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DataClass,
    RerankResponse,
    RerankResult,
    ToolCall,
    ToolCallFunction,
    Usage,
    new_id,
)
from aiready.core.wiring import connect
from aiready.gateway.config import GatewayConfig
from aiready.gateway.context import prefix_friendly, trim_to_window
from aiready.gateway.keys import ClientIdentity, KeyService
from aiready.gateway.limits import BudgetTracker, RateLimiter, Subjects
from aiready.gateway.routing import RoutePlan, check_model, endpoint_of, find_route, plan_route
from aiready.gateway.usage import UsageRecord, UsageStore
from aiready.models.backends import BackendError
from aiready.models.catalog import LiveCatalog, ModelCatalog, ModelSpec
from aiready.models.pool import InferenceRouter

log = logging.getLogger(__name__)
SERVICE = "gateway"
STEPS = ("authenticate", "limit", "screen", "reuse", "route", "call", "validate", "record")


@dataclass
class RequestContext:
    """Everything the pipeline learns about one request; the record step writes it out."""

    principal: Principal
    operation: str
    req: ChatRequest | None = None
    client: ClientIdentity | None = None
    trace_id: str = ""
    started: float = field(default_factory=time.perf_counter)
    steps: list[str] = field(default_factory=list)
    subjects: Subjects | None = None
    feature: str = "default"
    task: str | None = None
    data_class: DataClass = DataClass.INTERNAL
    findings: list[Finding] = field(default_factory=list)
    masked: dict[str, str] = field(default_factory=dict)
    plan: RoutePlan | None = None
    model: ModelSpec | None = None
    fallbacks: list[dict[str, str]] = field(default_factory=list)
    cacheable: bool = False
    cache_scope: CacheScope | None = None
    cache_payload: dict[str, Any] | None = None
    cached: str | None = None
    failed_check: bool = False
    flagged: bool = False
    decision: str = "allow"
    error: AIReadyError | None = None
    estimated_tokens: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    trimmed: int = 0
    ttft: float | None = None
    budget_warnings: list[dict[str, Any]] = field(default_factory=list)
    rate_status: list[dict[str, Any]] = field(default_factory=list)
    response_mode: str | None = None
    recorded: bool = False

    @property
    def team(self) -> str:
        return self.subjects.team if self.subjects else "default"

    @property
    def app(self) -> str:
        return self.subjects.app if self.subjects else self.principal.app


class GatewayService:
    """In-process implementation of ``aiready.core.interfaces.Gateway``."""

    def __init__(self, store: ConfigStore | None = None, *, kv: KVStore | None = None,
                 usage: UsageStore | None = None, keys: KeyService | None = None):
        self._fixed_store = store
        self.catalogs = LiveCatalog(store)
        self.router = InferenceRouter()
        self.catalogs.on_change(lambda cat: self.router.sync(cat.pools))
        self._cfg: GatewayConfig | None = None
        self._cfg_version: str | None = None
        cfg = self.config()
        self.kv: KVStore = kv or make_kv(cfg.kv or self.store.get("kv") or {"type": "memory"})
        self.rate_limiter = RateLimiter(self.kv)
        self.budgets = BudgetTracker(self.kv)
        self.exact_cache = ExactCache(self.kv, ttl=cfg.cache.exact_ttl_seconds, service=SERVICE)
        self.semantic_cache = SemanticCache(self.kv, None, threshold=cfg.cache.semantic_threshold,
                                            default_ttl=cfg.cache.semantic_ttl_seconds,
                                            ttl_by_content_type=cfg.cache.ttl_by_content_type,
                                            max_entries_per_scope=cfg.cache.max_entries_per_scope, service=SERVICE)
        self._usage = usage
        self.keys = keys or KeyService()
        self._batches: Any = None
        self._last_sync = 0.0
        self.catalog()  # build pools now so configuration errors surface at start-up

    # ================================================================== configuration
    @property
    def store(self) -> ConfigStore:
        return self._fixed_store or get_config()

    def config(self) -> GatewayConfig:
        store = self.store
        if self._cfg is None or self._cfg_version != store.version:
            try:
                self._cfg = store.section("gateway", GatewayConfig)
            except ConfigError:
                if self._cfg is None:
                    raise
                log.exception("gateway configuration rejected; keeping the previous version")
            self._cfg_version = store.version
            self._apply_cache_settings()
        return self._cfg

    def _apply_cache_settings(self) -> None:
        if not hasattr(self, "semantic_cache") or self._cfg is None:
            return
        c = self._cfg.cache
        self.exact_cache.ttl = c.exact_ttl_seconds
        self.semantic_cache.threshold = c.semantic_threshold
        self.semantic_cache.default_ttl = c.semantic_ttl_seconds
        self.semantic_cache.ttl_by_content_type = c.ttl_by_content_type

    def catalog(self) -> ModelCatalog:
        cat = self.catalogs.current()
        if set(self.router.pools) != set(cat.pools) or any(
                self.router.pools[n].fingerprint != s.fingerprint() for n, s in cat.pools.items()):
            self.router.sync(cat.pools)
        return cat

    def _sync_admin(self) -> None:
        """Pull control-plane changes made by the admin API in another process (multi-replica deployments)."""
        cfg = self.config()
        if self._last_sync and time.monotonic() - self._last_sync < cfg.admin.sync_seconds:
            return
        self._last_sync = time.monotonic()
        try:
            from aiready.gateway.control import ControlPlane

            ControlPlane(self.store).sync()
        except Exception:
            log.debug("admin sync failed", exc_info=True)

    @property
    def usage(self) -> UsageStore:
        if self._usage is None:
            self._usage = UsageStore(self.config().usage.url)
        return self._usage

    @property
    def batches(self) -> Any:
        if self._batches is None:
            from aiready.gateway.batches import BatchManager

            self._batches = BatchManager(self)
        return self._batches

    # ================================================================== helpers
    def _new_ctx(self, principal: Principal, operation: str, req: ChatRequest | None = None,
                 client: ClientIdentity | None = None) -> RequestContext:
        self._sync_admin()
        meta = (req.metadata if req else {}) or {}
        ctx = RequestContext(principal=principal, operation=operation, req=req, client=client,
                             trace_id=current_trace_id() or new_trace_id(),
                             feature=str(meta.get("feature") or "default"), task=meta.get("task"))
        declared = meta.get("data_class")
        if declared:
            try:
                ctx.data_class = DataClass(declared)
            except ValueError:
                raise ValidationFailed(f"unknown data_class '{declared}'") from None
        return ctx

    @contextlib.contextmanager
    def _step(self, ctx: RequestContext, name: str, **attrs: Any) -> Iterator[Any]:
        """One pipeline stage: trace span + stage latency metric. Never spans a ``yield`` of a stream."""
        ctx.steps.append(name)
        token = set_trace_id(ctx.trace_id) if current_trace_id() != ctx.trace_id else None
        try:
            with span(f"gateway.{name}", **attrs) as s, metrics.time_stage(SERVICE, name):
                yield s
        finally:
            if token is not None:
                from aiready.core.telemetry import _trace_id

                _trace_id.reset(token)

    def _subjects(self, ctx: RequestContext) -> Subjects:
        cfg = self.config()
        p = ctx.principal
        team = (p.claims or {}).get(cfg.auth.team_claim) or (ctx.client.team if ctx.client else None) \
            or (p.groups[0] if p.groups else None) or "default"
        app = (ctx.client.app if ctx.client and ctx.client.app else None) or p.app
        return Subjects(user=p.sub, team=str(team), app=app, feature=ctx.feature, tenant=p.tenant)

    # ================================================================== 1 authenticate
    async def _authenticate(self, ctx: RequestContext) -> None:
        cfg = self.config()
        with self._step(ctx, "authenticate", operation=ctx.operation) as s:
            p = ctx.principal
            if p is None:
                raise AuthenticationError("missing caller identity")
            if p.is_service_only and ctx.operation not in cfg.auth.service_only_operations:
                raise AuthenticationError(f"'{ctx.operation}' needs a signed-in user: services act for a user")
            if cfg.auth.require_client_identity and not (p.service or ctx.client):
                raise AuthenticationError("a service identity (X-Service-Token or API key) is required")
            ctx.subjects = self._subjects(ctx)
            s.set(user=p.sub, tenant=p.tenant, app=ctx.app, team=ctx.team, service=p.service or "",
                  client=ctx.client.name if ctx.client else "")

    # ================================================================== 2 limit
    async def _limit(self, ctx: RequestContext) -> None:
        cfg = self.config()
        with self._step(ctx, "limit") as s:
            assert ctx.subjects is not None
            ctx.rate_status = self.rate_limiter.check(cfg.rate_limits, ctx.subjects, ctx.operation)
            if ctx.req is not None:
                ctx.estimated_tokens = count_messages(ctx.req.messages)
            ctx.budget_warnings = self.budgets.check(cfg.budgets, ctx.subjects, ctx.estimated_tokens)
            s.set(estimated_tokens=ctx.estimated_tokens, budget_warnings=len(ctx.budget_warnings))

    # ================================================================== 3 screen
    async def _screen(self, ctx: RequestContext) -> None:
        cfg = self.config().guardrails
        assert ctx.req is not None
        with self._step(ctx, "screen", enabled=cfg.enabled) as s:
            if not cfg.enabled:
                return
            guard = connect("guardrails")
            messages = list(ctx.req.messages)
            for i, m in enumerate(messages):
                if m.role not in cfg.screen_roles or not m.content:
                    continue
                try:
                    res = await guard.screen_input(m.content, principal=ctx.principal, context={
                        "app": ctx.app, "feature": ctx.feature, "role": m.role, "operation": ctx.operation})
                except Exception as exc:
                    if cfg.fail_closed:
                        raise UpstreamError(f"guardrails unavailable: {exc}") from exc
                    log.warning("guardrails unavailable, continuing unscreened (fail_closed=false): %s", exc)
                    continue
                ctx.findings.extend(res.findings)
                ctx.data_class = DataClass.max(ctx.data_class, res.data_class)
                blocking = [f for f in res.findings if f.check in cfg.block_checks]
                for f in res.findings:
                    metrics.guardrail.labels(SERVICE, f.check, "blocked" if f in blocking else "flagged").inc()
                if not res.allowed or blocking:
                    ctx.decision = "blocked"
                    s.set(blocked=True, checks=[f.check for f in res.findings])
                    raise GuardrailBlocked("the request was blocked by input screening", details={
                        "findings": [{"check": f.check, "severity": f.severity, "message": f.message}
                                     for f in (blocking or res.findings)]})
                if res.masked_entities:
                    ctx.masked.update(res.masked_entities)
                if cfg.apply_masking and res.text != m.content:
                    messages[i] = m.model_copy(update={"content": res.text})
            ctx.req = ctx.req.model_copy(update={"messages": messages})
            metrics.guardrail.labels(SERVICE, "screen", "passed").inc()
            s.set(data_class=ctx.data_class.value, findings=len(ctx.findings), masked=len(ctx.masked))

    # ================================================================== 4 reuse
    def _cache_scope(self, ctx: RequestContext) -> CacheScope:
        req = ctx.req
        assert req is not None
        last_user = max((i for i, m in enumerate(req.messages) if m.role == "user"), default=-1)
        context = [m.model_dump(exclude_none=True) for i, m in enumerate(req.messages) if i != last_user]
        pv = str((req.metadata or {}).get("prompt_version") or "v1")
        # the rest of the conversation (system prompt, retrieved passages, history) is part of the scope:
        # the same question over different context is a different question
        return CacheScope(app=ctx.app, tenant=ctx.principal.tenant, access_groups=ctx.principal.access_groups,
                          prompt_version=f"{pv}:{sha256(json.dumps(context, sort_keys=True))[:16]}",
                          model=req.model, feature=ctx.feature)

    def _is_cacheable(self, ctx: RequestContext) -> tuple[bool, str]:
        cfg = self.config()
        req = ctx.req
        assert req is not None
        meta = req.metadata or {}
        if not cfg.cache.enabled:
            return False, "disabled"
        if req.tools or meta.get("write_action"):
            return False, "tools / write action"
        if meta.get("no_cache") or meta.get("cache") is False:
            return False, "caller opted out"
        if ctx.data_class.rank >= DataClass.PERSONAL.rank or ctx.masked:
            return False, "personal data"
        if (req.temperature or 0) > cfg.cache.max_temperature:
            return False, "sampled"
        try:
            _, route = find_route(cfg, self.catalog(), req, ctx.app)
        except AIReadyError:
            return False, "no route"
        if route is not None and not route.cache:
            return False, "route opts out"
        return True, ""

    async def _cache_vector(self, text: str) -> list[float] | None:
        """Embed a question for the semantic cache on a private embedding model; None disables semantic reuse."""
        cfg = self.config()
        spec = self.catalog().find(cfg.cache.embedding_model)
        if spec is None or spec.is_public or not spec.has("embedding"):
            return None
        try:
            vecs = await asyncio.wait_for(self.router.embed(spec.pool, [text], self.catalog().upstream_model(spec)),
                                          timeout=cfg.call.timeout_seconds)
            return vecs[0]
        except Exception:
            log.debug("semantic cache embedding failed", exc_info=True)
            return None

    async def _reuse(self, ctx: RequestContext) -> ChatResponse | None:
        cfg = self.config()
        assert ctx.req is not None
        with self._step(ctx, "reuse") as s:
            ok, why = self._is_cacheable(ctx)
            ctx.cacheable = ok
            if not ok:
                s.set(cacheable=False, reason=why)
                return None
            ctx.cache_scope = self._cache_scope(ctx)
            req = ctx.req
            ctx.cache_payload = {**req.model_dump(mode="json", exclude={"stream", "metadata", "user"}),
                                 "task": ctx.task}
            hit: dict[str, Any] | None = None
            if cfg.cache.exact:
                hit = self.exact_cache.get(ctx.cache_payload, ctx.cache_scope)
                if hit is not None:
                    ctx.cached = "exact"
            if hit is None and cfg.cache.semantic and req.last_user_text():
                vec = await self._cache_vector(req.last_user_text())
                entry = await self.semantic_cache.lookup(req.last_user_text(), ctx.cache_scope, vector=vec)
                if entry is not None:
                    hit, ctx.cached = entry.answer, "semantic"
            s.set(cacheable=True, hit=ctx.cached or "miss")
            if hit is None:
                return None
            resp = ChatResponse.model_validate(hit)
            resp.id, resp.created, resp.usage = new_id("chatcmpl-"), int(time.time()), Usage()
            ctx.model = self.catalog().find(resp.model)
            return resp

    async def _store_cache(self, ctx: RequestContext, resp: ChatResponse) -> None:
        cfg = self.config()
        if not ctx.cacheable or ctx.cached or ctx.cache_scope is None or ctx.cache_payload is None:
            return
        tool_calls = bool(resp.choices and resp.choices[0].message.tool_calls)
        no_cache = bool(ctx.plan and ctx.plan.obligations.get("no_cache"))
        if ctx.failed_check or ctx.flagged or tool_calls or not resp.text or no_cache:
            metrics.cache.labels(SERVICE, "store", "skip").inc()
            return
        answer = resp.model_dump(mode="json", exclude={"aiready"})
        if cfg.cache.exact:
            self.exact_cache.set(ctx.cache_payload, ctx.cache_scope, answer)
        if cfg.cache.semantic and ctx.req is not None and ctx.req.last_user_text():
            q = ctx.req.last_user_text()
            content_type = ctx.plan.route_config.content_type if ctx.plan and ctx.plan.route_config else "default"
            await self.semantic_cache.store(q, answer, ctx.cache_scope, vector=await self._cache_vector(q),
                                            content_type=content_type,
                                            contains_personal_data=ctx.data_class.rank >= DataClass.PERSONAL.rank,
                                            is_write_action=tool_calls, failed_check=ctx.failed_check)

    # ================================================================== 5 route
    async def _route(self, ctx: RequestContext) -> None:
        cfg = self.config()
        assert ctx.req is not None
        with self._step(ctx, "route") as s:
            try:
                ctx.plan = await plan_route(
                    cfg=cfg, catalog=self.catalog(), req=ctx.req, principal=ctx.principal, data_class=ctx.data_class,
                    context={"app": ctx.app, "feature": ctx.feature, "team": ctx.team, "task": ctx.task,
                             "tenant": ctx.principal.tenant, "operation": ctx.operation},
                    prompt_tokens=ctx.estimated_tokens, healthy=self.router.healthy)
            except AIReadyError as exc:
                ctx.decision = "deny" if exc.status_code == 403 else ctx.decision
                s.set(denied=True, reason=exc.message)
                raise
            obligations = ctx.plan.obligations
            if obligations.get("mask_pii") and not ctx.masked:
                await self._mask_for_obligation(ctx)
            s.set(route=ctx.plan.route or "", chain=[m.name for m in ctx.plan.chain],
                  rejected=ctx.plan.rejected, data_class=ctx.data_class.value)

    async def _mask_for_obligation(self, ctx: RequestContext) -> None:
        """Policy obligation ``mask_pii``: mask user content even when screening did not."""
        assert ctx.req is not None
        guard = connect("guardrails")
        msgs = []
        for m in ctx.req.messages:
            if m.role in ("user", "tool") and m.content:
                res = await guard.mask(m.content)
                ctx.masked.update(res.masked_entities)
                m = m.model_copy(update={"content": res.text})
            msgs.append(m)
        ctx.req = ctx.req.model_copy(update={"messages": msgs})

    # ================================================================== 6 call
    def _upstream_request(self, ctx: RequestContext, spec: ModelSpec, *, stream: bool) -> ChatRequest:
        cfg = self.config()
        assert ctx.req is not None
        req = ctx.req
        caps = [spec.max_output_tokens, req.max_tokens]
        if ctx.plan and ctx.plan.route_config and ctx.plan.route_config.max_tokens:
            caps.append(ctx.plan.route_config.max_tokens)
        if ctx.plan and ctx.plan.obligations.get("max_tokens"):
            caps.append(int(ctx.plan.obligations["max_tokens"]))
        max_tokens = min(c for c in caps if c)
        up = req.model_copy(update={"model": self.catalog().upstream_model(spec), "metadata": {}, "stream": stream,
                                    "max_tokens": max_tokens})
        if cfg.call.prefix_cache_ordering:
            up = prefix_friendly(up)
        up, ctx.trimmed = trim_to_window(up, spec.context_length, min(max_tokens, cfg.call.reserve_output_tokens))
        return up

    def _fallback(self, ctx: RequestContext, spec: ModelSpec, exc: BaseException) -> None:
        reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else str(exc)[:300]
        ctx.fallbacks.append({"model": spec.name, "error": reason})
        metrics.fallbacks.labels(SERVICE, "model").inc()
        log.warning("model %s failed (%s); falling back", spec.name, reason)

    async def _call(self, ctx: RequestContext) -> ChatResponse:
        cfg = self.config()
        assert ctx.plan is not None
        with self._step(ctx, "call") as s:
            for spec in ctx.plan.chain:
                up = self._upstream_request(ctx, spec, stream=False)
                for attempt in range(cfg.call.retries + 1):
                    try:
                        resp = await asyncio.wait_for(self.router.chat(spec.pool, up), cfg.call.timeout_seconds)
                    except (TimeoutError, BackendError, UpstreamError) as exc:
                        retryable = getattr(exc, "retryable", True)
                        if retryable and attempt < cfg.call.retries:
                            await asyncio.sleep(cfg.call.retry_backoff_seconds * (2 ** attempt))
                            continue
                        self._fallback(ctx, spec, exc)
                        break
                    ctx.model = spec
                    resp.model = spec.name
                    if not resp.usage.total_tokens:
                        pt = count_messages(up.messages)
                        ct = count_tokens(resp.text)
                        resp.usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
                    ctx.usage = Usage(prompt_tokens=ctx.usage.prompt_tokens + resp.usage.prompt_tokens,
                                      completion_tokens=ctx.usage.completion_tokens + resp.usage.completion_tokens,
                                      total_tokens=ctx.usage.total_tokens + resp.usage.total_tokens)
                    ctx.cost += spec.cost(resp.usage.prompt_tokens, resp.usage.completion_tokens)
                    s.set(model=spec.name, fallbacks=len(ctx.fallbacks), trimmed=ctx.trimmed,
                          prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)
                    return resp
            ctx.decision = "error"
            raise UpstreamError("every model in the route failed", details={"attempts": ctx.fallbacks})

    # ================================================================== 7 validate
    @staticmethod
    def _schema(req: ChatRequest) -> dict[str, Any] | None:
        rf = req.response_format or {}
        if rf.get("type") == "json_schema":
            return (rf.get("json_schema") or {}).get("schema") or {"type": "object"}
        if rf.get("type") == "json_object":
            return {"type": "object"}
        return None

    async def _check_output(self, ctx: RequestContext, text: str) -> Any:
        assert ctx.req is not None
        return await connect("guardrails").validate_output(text, schema=self._schema(ctx.req), context={
            "app": ctx.app, "feature": ctx.feature, "model": ctx.model.name if ctx.model else None,
            "data_class": ctx.data_class.value})

    async def _validate(self, ctx: RequestContext, resp: ChatResponse) -> ChatResponse:
        cfg = self.config().validate_output
        with self._step(ctx, "validate", enabled=cfg.enabled) as s:
            if not cfg.enabled or not resp.text:
                return resp
            vr = await self._check_output(ctx, resp.text)
            if not vr.valid and cfg.on_failure == "regenerate_once":
                metrics.guardrail.labels(SERVICE, "output", "regenerate").inc()
                s.event("regenerate", findings=[f.check for f in vr.findings])
                resp = await self._call(ctx)
                ctx.steps.append("validate")
                vr = await self._check_output(ctx, resp.text) if resp.text else vr.model_copy(update={"valid": True})
            if vr.valid:
                metrics.guardrail.labels(SERVICE, "output", "passed").inc()
                if vr.text and vr.text != resp.text:
                    resp.choices[0].message.content = vr.text
                return resp
            ctx.failed_check = True
            findings = [{"check": f.check, "severity": f.severity, "message": f.message} for f in vr.findings]
            s.set(valid=False, findings=findings)
            if cfg.on_failure == "flag":
                ctx.flagged = True
                metrics.guardrail.labels(SERVICE, "output", "flagged").inc()
                resp.aiready["flagged"] = True
                resp.aiready["findings"] = findings
                return resp
            metrics.guardrail.labels(SERVICE, "output", "rejected").inc()
            ctx.decision = "rejected"
            raise ValidationFailed("the model output failed validation", details={"findings": findings})

    def _restore_masked(self, ctx: RequestContext, text: str) -> str:
        if not ctx.masked or not self.config().guardrails.restore_masked_in_output:
            return text
        for placeholder, original in ctx.masked.items():
            text = text.replace(placeholder, original)
        return text

    # ================================================================== 8 record
    async def _record(self, ctx: RequestContext, resp: ChatResponse | None = None,
                      error: BaseException | None = None) -> None:
        if ctx.recorded:
            return
        ctx.recorded = True
        cfg = self.config()
        if error is not None and ctx.decision == "allow":
            ctx.decision = getattr(error, "code", "error")
        with self._step(ctx, "record") as s:
            latency = time.perf_counter() - ctx.started
            model = ctx.model.name if ctx.model else "-"
            subj = ctx.subjects or self._subjects(ctx)
            pt, ct = ctx.usage.prompt_tokens, ctx.usage.completion_tokens
            if subj and pt + ct > 0:
                self.budgets.charge(cfg.budgets, subj, pt + ct, ctx.cost)
            labels = (SERVICE, subj.app, ctx.feature, subj.team, model)
            metrics.tokens.labels(*labels, "input").inc(pt)
            metrics.tokens.labels(*labels, "output").inc(ct)
            if ctx.cost:
                metrics.cost.labels(*labels).inc(ctx.cost)
            metrics.latency.labels(*labels).observe(latency)
            if ctx.ttft is not None:
                metrics.ttft.labels(*labels).observe(ctx.ttft)
            metrics.requests.labels(*labels, ctx.decision).inc()
            try:
                self.usage.add(UsageRecord(
                    tenant=subj.tenant, app=subj.app, team=subj.team, feature=ctx.feature, user=subj.user, model=model,
                    route=(ctx.plan.route or "") if ctx.plan else "", operation=ctx.operation, status=ctx.decision,
                    prompt_tokens=pt, completion_tokens=ct, cost=ctx.cost, latency_ms=latency * 1000,
                    cached=bool(ctx.cached), trace_id=ctx.trace_id))
            except Exception:  # usage reporting must not fail the user's request; the audit record still holds it
                log.exception("usage store write failed")
            prompt_hash = sha256(ctx.req.all_text())[:32] if ctx.req else None
            get_audit().for_principal(
                ctx.principal, "gateway.request", trace_id=ctx.trace_id, action=f"model.{ctx.operation}",
                resource=model, decision=ctx.decision,
                prompt_hash=prompt_hash, route=ctx.plan.route if ctx.plan else None, data_class=ctx.data_class.value,
                feature=ctx.feature, team=subj.team, prompt_tokens=pt, completion_tokens=ct, cost=round(ctx.cost, 8),
                latency_ms=round(latency * 1000, 1), cached=ctx.cached, fallbacks=ctx.fallbacks,
                rejected_models=ctx.plan.rejected if ctx.plan else {}, findings=[f.check for f in ctx.findings],
                masked_entities=len(ctx.masked), failed_check=ctx.failed_check, client=ctx.client.key_id if ctx.client else None,
                policy_versions=sorted(set(ctx.plan.policy_versions)) if ctx.plan else [],
                error=ctx.error.code if isinstance(ctx.error, AIReadyError) else (type(error).__name__ if error else None),
                budget_warnings=ctx.budget_warnings or None)
            s.set(decision=ctx.decision, model=model, latency_ms=round(latency * 1000, 1), cost=ctx.cost)
        if resp is not None:
            resp.aiready.update(self._meta(ctx))

    def _meta(self, ctx: RequestContext) -> dict[str, Any]:
        return {"trace_id": ctx.trace_id, "steps": list(ctx.steps), "route": ctx.plan.route if ctx.plan else None,
                "model": ctx.model.name if ctx.model else None, "cached": ctx.cached,
                "fallbacks": [f["model"] for f in ctx.fallbacks], "data_class": ctx.data_class.value,
                "cost": round(ctx.cost, 8), "currency": self.config().currency, "decision": ctx.decision,
                "trimmed_messages": ctx.trimmed, "budget_warnings": ctx.budget_warnings or None}

    # ================================================================== public API: chat
    async def chat(self, req: ChatRequest, principal: Principal, *,
                   client: ClientIdentity | None = None) -> ChatResponse:
        ctx = self._new_ctx(principal, "chat", req, client)
        token = set_trace_id(ctx.trace_id) if current_trace_id() is None else None
        resp: ChatResponse | None = None
        try:
            with span("gateway.request", feature=ctx.feature, user=principal.sub) as root:
                await self._authenticate(ctx)
                await self._limit(ctx)
                await self._screen(ctx)
                if ctx.req is not None and get_config().get("gateway.usage.record_request_in_trace", True):
                    # the screened (masked) request, so an admin can replay it from the trace viewer;
                    # the trace store masks personal data again before storage
                    root.set(request=ctx.req.model_dump(mode="json", exclude_none=True, exclude={"metadata"}))
                resp = await self._reuse(ctx)
                if resp is None:
                    await self._route(ctx)
                    resp = await self._call(ctx)
                    resp = await self._validate(ctx, resp)
                    await self._store_cache(ctx, resp)
                if resp.choices and resp.choices[0].message.content:
                    resp.choices[0].message.content = self._restore_masked(ctx, resp.choices[0].message.content)
                await self._record(ctx, resp)
                return resp
        except AIReadyError as exc:
            ctx.error = exc
            await self._record(ctx, None, exc)
            raise
        except BaseException as exc:
            await self._record(ctx, None, exc)
            raise
        finally:
            if token is not None:
                from aiready.core.telemetry import _trace_id

                _trace_id.reset(token)

    # ================================================================== public API: streaming
    def _response_mode(self, ctx: RequestContext) -> str:
        meta = (ctx.req.metadata if ctx.req else {}) or {}
        mode = meta.get("response_mode")
        if mode not in ("check_then_send", "stream_then_verify"):
            route = ctx.plan.route_config if ctx.plan else None
            mode = (route.response_mode if route else None) or self.config().streaming.default_mode
        return mode

    @staticmethod
    def _chunks_of(resp: ChatResponse) -> Iterator[dict[str, Any]]:
        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {"id": resp.id, "object": "chat.completion.chunk", "created": resp.created, "model": resp.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        yield chunk({"role": "assistant", "content": ""})
        msg = resp.choices[0].message if resp.choices else ChatMessage(role="assistant", content="")
        for i, tc in enumerate(msg.tool_calls or []):
            yield chunk({"tool_calls": [{"index": i, "id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
        words = (msg.content or "").split(" ")
        for i, w in enumerate(words):
            if w or i < len(words) - 1:
                yield chunk({"content": w if i == len(words) - 1 else w + " "})
        yield chunk({}, resp.choices[0].finish_reason if resp.choices else "stop")

    def _meta_chunk(self, ctx: RequestContext, cid: str, obj: str = "chat.completion.chunk",
                    **extra: Any) -> dict[str, Any]:
        return {"id": cid, "object": obj, "created": int(time.time()),
                "model": ctx.model.name if ctx.model else None, "choices": [],
                "usage": ctx.usage.model_dump(), "aiready": {**self._meta(ctx), **extra}}

    async def _open_stream(self, ctx: RequestContext) -> tuple[ModelSpec, dict[str, Any] | None, AsyncIterator]:
        """Open a stream on the first model of the chain that produces a first chunk (fallback before first word)."""
        cfg = self.config()
        assert ctx.plan is not None
        with self._step(ctx, "call", streaming=True) as s:
            for spec in ctx.plan.chain:
                up = self._upstream_request(ctx, spec, stream=True)
                for attempt in range(cfg.call.retries + 1):
                    agen = self.router.chat_stream(spec.pool, up).__aiter__()
                    try:
                        first = await asyncio.wait_for(agen.__anext__(), cfg.call.stream_first_token_timeout_seconds)
                    except StopAsyncIteration:
                        first = None
                    except (TimeoutError, BackendError, UpstreamError) as exc:
                        with contextlib.suppress(Exception):
                            await agen.aclose()  # type: ignore[attr-defined]
                        if getattr(exc, "retryable", True) and attempt < cfg.call.retries:
                            await asyncio.sleep(cfg.call.retry_backoff_seconds * (2 ** attempt))
                            continue
                        self._fallback(ctx, spec, exc)
                        break
                    ctx.model = spec
                    ctx.ttft = time.perf_counter() - ctx.started
                    s.set(model=spec.name, fallbacks=len(ctx.fallbacks), ttft_ms=round(ctx.ttft * 1000, 1))
                    return spec, first, agen
            ctx.decision = "error"
            raise UpstreamError("every model in the route failed", details={"attempts": ctx.fallbacks})

    async def chat_stream(self, req: ChatRequest, principal: Principal, *,
                          client: ClientIdentity | None = None) -> AsyncIterator[dict[str, Any]]:
        ctx = self._new_ctx(principal, "stream", req, client)
        resp: ChatResponse | None = None
        try:
            await self._authenticate(ctx)
            await self._limit(ctx)
            await self._screen(ctx)
            hit = await self._reuse(ctx)
            if hit is not None:
                hit.choices[0].message.content = self._restore_masked(ctx, hit.text)
                await self._record(ctx, hit)
                for c in self._chunks_of(hit):
                    yield c
                yield self._meta_chunk(ctx, hit.id)
                return
            await self._route(ctx)
            mode = ctx.response_mode = self._response_mode(ctx)
            if mode == "check_then_send":
                # Mode A: generate and check the whole answer before the first word is sent
                resp = await self._call(ctx)
                resp = await self._validate(ctx, resp)
                await self._store_cache(ctx, resp)
                if resp.choices and resp.choices[0].message.content:
                    resp.choices[0].message.content = self._restore_masked(ctx, resp.choices[0].message.content)
                ctx.ttft = time.perf_counter() - ctx.started
                await self._record(ctx, resp)
                for c in self._chunks_of(resp):
                    yield c
                yield self._meta_chunk(ctx, resp.id, response_mode=mode)
                return

            # Mode B: stream, then verify at the end
            spec, first, agen = await self._open_stream(ctx)
            cid = new_id("chatcmpl-")
            text: list[str] = []
            tools: dict[int, dict[str, Any]] = {}
            finish = "stop"
            usage: dict[str, Any] | None = None
            pending = [first] if first is not None else []
            while True:
                if pending:
                    chunk = pending.pop()
                else:
                    try:
                        chunk = await asyncio.wait_for(agen.__anext__(), self.config().call.timeout_seconds)
                    except StopAsyncIteration:
                        break
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        text.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = tools.setdefault(int(tc.get("index", 0)), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if not chunk.get("choices"):
                    continue  # usage-only chunk: the gateway sends its own totals at the end
                yield {**chunk, "id": cid, "model": spec.name}
            full = "".join(text)
            up = self._upstream_request(ctx, spec, stream=True)
            pt = int((usage or {}).get("prompt_tokens") or count_messages(up.messages))
            ct = int((usage or {}).get("completion_tokens") or count_tokens(full))
            ctx.usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
            ctx.cost = spec.cost(pt, ct)
            msg = ChatMessage(role="assistant", content=full, tool_calls=[
                ToolCall(id=t["id"] or new_id("call_"), function=ToolCallFunction(name=t["name"], arguments=t["arguments"]))
                for _, t in sorted(tools.items())] or None)
            resp = ChatResponse(id=cid, model=spec.name, choices=[ChatChoice(message=msg, finish_reason=finish)],
                                usage=ctx.usage)
            withdrawn: dict[str, Any] | None = None
            vcfg = self.config().validate_output
            with self._step(ctx, "validate", enabled=vcfg.enabled, mode=mode) as s:
                if vcfg.enabled and full:
                    vr = await self._check_output(ctx, full)
                    if not vr.valid:
                        ctx.failed_check = True
                        findings = [{"check": f.check, "severity": f.severity, "message": f.message}
                                    for f in vr.findings]
                        if vcfg.on_failure == "flag":
                            ctx.flagged = True
                            metrics.guardrail.labels(SERVICE, "output", "flagged").inc()
                        else:
                            ctx.decision = "withdrawn"
                            metrics.guardrail.labels(SERVICE, "output", "withdrawn").inc()
                        withdrawn = {"findings": findings}
                        s.set(valid=False, findings=findings)
                    else:
                        metrics.guardrail.labels(SERVICE, "output", "passed").inc()
            await self._store_cache(ctx, resp)
            await self._record(ctx, resp)
            if withdrawn is not None:
                if ctx.flagged:
                    yield self._meta_chunk(ctx, cid, "aiready.flag", response_mode=mode, flagged=True, **withdrawn)
                else:
                    yield self._meta_chunk(ctx, cid, "aiready.withdrawn", response_mode=mode, withdrawn=True,
                                           reason="the answer failed the final check", **withdrawn)
            else:
                yield self._meta_chunk(ctx, cid, response_mode=mode, verified=True)
        except AIReadyError as exc:
            ctx.error = exc
            await self._record(ctx, None, exc)
            raise
        except GeneratorExit:
            ctx.decision = "client_closed" if ctx.decision == "allow" else ctx.decision
            await self._record(ctx, resp)
            raise
        except BaseException as exc:
            await self._record(ctx, None, exc)
            raise

    # ================================================================== public API: embeddings / rerank
    async def _governed_model(self, ctx: RequestContext, name: str, capability: str) -> ModelSpec:
        cfg = self.config()
        cat = self.catalog()
        spec = cat.find(name)
        if spec is None or not spec.has(capability):
            from aiready.core.errors import NotFound

            raise NotFound(f"no {capability} model named '{name}'")
        plan = RoutePlan(route=None, chain=[])
        ok = await check_model(spec, principal=ctx.principal, data_class=ctx.data_class,
                               egress_enabled=cfg.egress.enabled, action=f"model.{ctx.operation}",
                               endpoint=endpoint_of(cat, spec),
                               context={"app": ctx.app, "feature": ctx.feature, "team": ctx.team,
                                        "operation": ctx.operation}, plan=plan)
        ctx.plan = plan
        if not ok:
            from aiready.core.errors import PolicyViolation

            ctx.decision = "deny"
            raise PolicyViolation(f"model '{name}' may not be used", details={"rejected": plan.rejected})
        plan.chain = [spec]
        return spec

    async def _pool_call(self, ctx: RequestContext, spec: ModelSpec, fn: Any) -> Any:
        cfg = self.config()
        for attempt in range(cfg.call.retries + 1):
            try:
                return await asyncio.wait_for(fn(), cfg.call.timeout_seconds)
            except (TimeoutError, BackendError, UpstreamError) as exc:
                if getattr(exc, "retryable", True) and attempt < cfg.call.retries:
                    await asyncio.sleep(cfg.call.retry_backoff_seconds * (2 ** attempt))
                    continue
                ctx.decision = "error"
                raise UpstreamError(f"{spec.name} failed: {exc}") from exc
        raise UpstreamError(f"{spec.name} failed")

    async def embed(self, texts: list[str], principal: Principal, *, model: str = "embedding",
                    priority: str = "live", client: ClientIdentity | None = None,
                    feature: str | None = None) -> list[list[float]]:
        cfg = self.config()
        target = cfg.embedding
        name = target.model if model in ("embedding", "", None) else model
        ctx = self._new_ctx(principal, "embed", None, client)
        ctx.feature = feature or "embedding"
        try:
            with span("gateway.embed", texts=len(texts), priority=priority):
                if len(texts) > target.max_items:
                    raise ValidationFailed(f"at most {target.max_items} texts per request")
                await self._authenticate(ctx)
                ctx.estimated_tokens = sum(count_tokens(t) for t in texts)
                await self._limit(ctx)
                if target.target == "service":
                    with self._step(ctx, "call", target="service"):
                        out = await connect("embedding").embed(texts, principal=principal, model=name, priority=priority)
                    # the embedding service may return bare vectors or an OpenAI-shaped EmbeddingResponse
                    vectors = [list(d.embedding) for d in out.data] if hasattr(out, "data") else [list(v) for v in out]
                else:
                    with self._step(ctx, "route", model=name):
                        spec = await self._governed_model(ctx, name, "embedding")
                    upstream = self.catalog().upstream_model(spec)
                    with self._step(ctx, "call", model=spec.name, pool=spec.pool):
                        vectors = await self._pool_call(ctx, spec, lambda: self.router.embed(spec.pool, texts, upstream))
                    ctx.model = spec
                    ctx.cost = spec.cost(ctx.estimated_tokens, 0)
                ctx.usage = Usage(prompt_tokens=ctx.estimated_tokens, total_tokens=ctx.estimated_tokens)
                await self._record(ctx)
                return vectors
        except BaseException as exc:
            if isinstance(exc, AIReadyError):
                ctx.error = exc
            await self._record(ctx, None, exc)
            raise

    async def rerank(self, query: str, documents: list[str], principal: Principal, *, top_n: int | None = None,
                     model: str = "reranker", client: ClientIdentity | None = None,
                     feature: str | None = None) -> RerankResponse:
        cfg = self.config()
        target = cfg.rerank
        name = target.model if model in ("reranker", "", None) else model
        ctx = self._new_ctx(principal, "rerank", None, client)
        ctx.feature = feature or "rerank"
        try:
            with span("gateway.rerank", documents=len(documents)):
                if len(documents) > target.max_items:
                    raise ValidationFailed(f"at most {target.max_items} documents per request")
                await self._authenticate(ctx)
                ctx.estimated_tokens = count_tokens(query) * max(1, len(documents)) + sum(count_tokens(d) for d in documents)
                await self._limit(ctx)
                if target.target == "service":
                    with self._step(ctx, "call", target="service"):
                        out = await connect("rerank").rerank(query, documents, principal=principal, top_n=top_n, model=name)
                    ctx.usage = Usage(prompt_tokens=ctx.estimated_tokens, total_tokens=ctx.estimated_tokens)
                    await self._record(ctx)
                    return out
                with self._step(ctx, "route", model=name):
                    spec = await self._governed_model(ctx, name, "rerank")
                upstream = self.catalog().upstream_model(spec)
                with self._step(ctx, "call", model=spec.name, pool=spec.pool):
                    scores = await self._pool_call(ctx, spec, lambda: self.router.rerank(spec.pool, query, documents,
                                                                                          upstream))
                ranked = sorted(enumerate(scores), key=lambda x: -x[1])
                if top_n:
                    ranked = ranked[:top_n]
                ctx.model = spec
                ctx.usage = Usage(prompt_tokens=ctx.estimated_tokens, total_tokens=ctx.estimated_tokens)
                ctx.cost = spec.cost(ctx.estimated_tokens, 0)
                await self._record(ctx)
                return RerankResponse(model=spec.name, results=[RerankResult(index=i, relevance_score=float(s))
                                                                for i, s in ranked])
        except BaseException as exc:
            if isinstance(exc, AIReadyError):
                ctx.error = exc
            await self._record(ctx, None, exc)
            raise

    # ================================================================== catalogue, usage, operations
    async def list_models(self, principal: Principal) -> list[dict[str, Any]]:
        """Routes and models this caller may use (policy evaluated for the default data class)."""
        cfg = self.config()
        cat = self.catalog()
        out: list[dict[str, Any]] = []
        allowed: set[str] = set()
        ctx_info = {"app": principal.app, "operation": "list_models"}
        for spec in cat.list():
            plan = RoutePlan(route=None, chain=[])
            if await check_model(spec, principal=principal, data_class=DataClass.INTERNAL,
                                 egress_enabled=cfg.egress.enabled, context=ctx_info, plan=plan):
                allowed.add(spec.name)
                if cfg.allow_direct_models:
                    out.append({"id": spec.name, "object": "model", "owned_by": "aiready", "kind": "model",
                                **{k: v for k, v in spec.public_view().items() if k not in ("name", "pool")}})
        for name, route in cfg.routes.items():
            if route.enabled and any(c in allowed for c in route.candidates + route.fallback):
                out.append({"id": name, "object": "model", "owned_by": "aiready", "kind": "route",
                            "description": route.description, "aliases": route.match.model})
        return out

    def usage_report(self, *, group_by: list[str] | None = None, start: float | None = None, end: float | None = None,
                     filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        return self.usage.report(group_by=group_by, start=start, end=end, filters=filters)

    def usage_for(self, principal: Principal, *, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        cfg = self.config()
        ctx = RequestContext(principal=principal, operation="usage")
        subj = self._subjects(ctx)
        return {"user": principal.sub, "team": subj.team, "currency": cfg.currency,
                "usage": self.usage.report(group_by=["model", "feature"], start=start, end=end,
                                           filters={"user": principal.sub, "tenant": principal.tenant}),
                "budgets": self.budgets.status(cfg.budgets, subj)}

    def budget_status(self, principal: Principal) -> list[dict[str, Any]]:
        ctx = RequestContext(principal=principal, operation="usage")
        return self.budgets.status(self.config().budgets, self._subjects(ctx))

    def pool_stats(self) -> dict[str, Any]:
        self.catalog()
        return self.router.stats()

    def flush_cache(self) -> int:
        """Drop every exact and semantic cache entry (admin action, e.g. after a prompt or model change)."""
        n = 0
        for pattern in ("xc:*", "sc:*"):
            for k in self.kv.keys(pattern):
                self.kv.delete(k)
                n += 1
        return n

    async def readiness(self) -> dict[str, Any]:
        cat = self.catalog()
        stats = self.router.stats()
        unhealthy = [n for n, p in stats.items() if p["healthy_replicas"] == 0]
        return {"models": len(cat.models), "pools": len(stats), "unhealthy_pools": unhealthy}
