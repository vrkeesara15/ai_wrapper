"""RAG service: grounded answers with sources, filtered by what the user may see.

Runtime execution flow (slides "Runtime execution flow", "The path of one request", "Semantic cache"):

    screen input -> policy (data.read) -> plan (rewrite / split) -> embed -> semantic cache lookup
    -> hybrid search (50, user's access tags) -> weak retrieval? "not found", never a guess
    -> rerank (5) -> build prompt -> generate via the gateway (slow model -> smaller model)
    -> citations + hallucination check (fail -> regenerate once, stricter -> sources only)
    -> output validation -> cache store (never personal data / failed checks) -> audit -> answer

Response modes (slide "Latency budget and response modes"):

* Mode A, check-then-send (``answer``): the full answer is written and checked before the user sees it.
* Mode B, stream-then-verify (``answer_stream``): tokens are streamed as they are written; the check runs at
  the end and a final ``verdict`` event can flag or withdraw the answer.

Every stage is traced (``rag.<stage>`` spans), timed against the latency budget in configuration and the
outcome is written to the audit log (``rag.answer`` with the retrieved chunk ids and the trace id).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from aiready.cache import CacheScope, SemanticCache
from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import (
    AIReadyError,
    ApprovalRequired,
    GuardrailBlocked,
    PermissionDenied,
    PolicyViolation,
    UpstreamError,
)
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyDecision, PolicyInput, RagAnswer, RagQuery
from aiready.core.plugins import create
from aiready.core.stores import make_kv
from aiready.core.telemetry import _trace_id, current_trace_id, metrics, new_trace_id, set_trace_id, span
from aiready.core.types import ChatMessage, ChatRequest, DataClass, SearchHit
from aiready.core.wiring import connect
from aiready.rag import citations as _citations  # noqa: F401  (registers built-in components)
from aiready.rag import context as _context  # noqa: F401
from aiready.rag import grounding as _grounding  # noqa: F401
from aiready.rag import planner as _planner  # noqa: F401
from aiready.rag import rerank_client as _rerank  # noqa: F401
from aiready.rag import retriever as _retriever  # noqa: F401
from aiready.rag.citations import CitationMapper, CitationResult
from aiready.rag.config import RagConfig
from aiready.rag.context import ContextBuilder, ContextBundle
from aiready.rag.grounding import GroundingChecker, GroundingResult
from aiready.rag.planner import QueryPlan, QueryPlanner
from aiready.rag.prompts import PromptStore, PromptTemplate
from aiready.rag.rerank_client import RerankClient
from aiready.rag.retriever import Retriever

log = logging.getLogger(__name__)
SERVICE = "rag"


def principal_facts(p: Principal) -> dict[str, Any]:
    """What a policy may know about the caller (never the token)."""
    return p.model_dump(include={"sub", "tenant", "app", "roles", "groups", "scopes", "service", "actor"})


class _Run:
    """Per-request state: timings, fallbacks, governance flags."""

    def __init__(self, svc: RagServiceImpl, q: RagQuery, principal: Principal):
        self.svc, self.q, self.principal = svc, q, principal
        self.cfg = svc.cfg
        self.trace_id = current_trace_id() or new_trace_id()
        self.stage_ms: dict[str, float] = {}
        self.over_budget: list[str] = []
        self.fallbacks: list[str] = []
        self.question = q.question
        self.personal = False
        self.no_cache = False
        self.obligations: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = list(q.history)
        self.collection = ""
        self.plan: QueryPlan | None = None
        self.vector: list[float] | None = None
        self.sub_vectors: dict[str, list[float]] = {}
        self.candidates: list[SearchHit] = []
        self.passages: list[SearchHit] = []
        self.prompt: PromptTemplate | None = None
        self.model = q.model if q.model and q.model != "auto" else self.cfg.generation.model
        self.scope: CacheScope | None = None

    @contextlib.contextmanager
    def trace(self) -> Iterator[None]:
        """Bind this run's trace id (used around non-yielding sections of the stream generator)."""
        token = set_trace_id(self.trace_id)
        try:
            yield
        finally:
            _trace_id.reset(token)

    @contextlib.contextmanager
    def stage(self, name: str, **attrs: Any) -> Iterator[Any]:
        t0 = time.perf_counter()
        with span(f"rag.stage.{name}", **attrs) as sp, metrics.time_stage(SERVICE, name):
            yield sp
            ms = (time.perf_counter() - t0) * 1000
            budget = self.cfg.latency_budget_ms.get(name)
            if budget is not None and ms > budget:
                sp.event("over_budget", budget_ms=budget, ms=round(ms, 2))
                self.over_budget.append(name)
        self.stage_ms[name] = round(self.stage_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000, 2)

    def fallback(self, kind: str) -> None:
        self.fallbacks.append(kind)
        metrics.fallbacks.labels(SERVICE, kind).inc()


class _Outcome:
    """Result of generation + checks, before caching and audit."""

    def __init__(self, answer: RagAnswer, *, decision: str, failed_check: bool = False, personal: bool = False,
                 grounding: GroundingResult | None = None):
        self.answer, self.decision = answer, decision
        self.failed_check, self.personal, self.grounding = failed_check, personal, grounding


class RagServiceImpl:
    """In-process implementation of ``core.interfaces.RagService``."""

    def __init__(self, config: RagConfig | None = None, *, cache: SemanticCache | None = None,
                 planner: QueryPlanner | None = None, retriever: Retriever | None = None,
                 reranker: RerankClient | None = None, context_builder: ContextBuilder | None = None,
                 citation_mapper: CitationMapper | None = None, grounding: GroundingChecker | None = None,
                 prompts: PromptStore | None = None, follow_config: bool = False):
        self._follow = follow_config
        self._cfg_version = get_config().version
        self._overrides = {"planner": planner, "retriever": retriever, "reranker": reranker,
                           "context_builder": context_builder, "citation_mapper": citation_mapper,
                           "grounding": grounding, "prompts": prompts}
        self._cache_override = cache
        self._configure(config or get_config().section("rag", RagConfig))

    @classmethod
    def from_config(cls) -> RagServiceImpl:
        """Built from the ``rag`` section; picks up configuration changes without a restart."""
        return cls(follow_config=True)

    # ------------------------------------------------------------------ construction
    def _configure(self, cfg: RagConfig) -> None:
        self.cfg = cfg
        o = self._overrides
        self.prompts: PromptStore = o["prompts"] or PromptStore(cfg.prompts.source, cfg.prompts.prompts_dir)
        pv = cfg.prompts.versions
        comp = cfg.components
        self.planner = o["planner"] or create("rag_planner", comp.get("planner", "default"), config=cfg.planner,
                                              prompts=self.prompts, prompt_name=cfg.prompts.plan,
                                              prompt_version=pv.get(cfg.prompts.plan))
        self.retriever = o["retriever"] or create("rag_retriever", comp.get("retriever", "default"))
        self.reranker = o["reranker"] or create("rag_reranker", comp.get("reranker", "default"), config=cfg.rerank)
        self.context_builder = o["context_builder"] or create("rag_context_builder",
                                                              comp.get("context_builder", "default"),
                                                              config=cfg.context)
        self.citation_mapper = o["citation_mapper"] or create("rag_citation_mapper",
                                                              comp.get("citation_mapper", "default"))
        self.grounding = o["grounding"] or create("rag_grounding", comp.get("grounding", "default"),
                                                  config=cfg.grounding, prompts=self.prompts,
                                                  judge_prompt=cfg.prompts.judge,
                                                  judge_version=pv.get(cfg.prompts.judge))
        if self._cache_override is not None:
            self.cache: SemanticCache | None = self._cache_override
        elif cfg.cache.enabled:
            self.cache = SemanticCache(make_kv(cfg.cache.kv), threshold=cfg.cache.threshold,
                                       default_ttl=cfg.cache.default_ttl_seconds,
                                       ttl_by_content_type=cfg.cache.ttl_by_content_type,
                                       max_entries_per_scope=cfg.cache.max_entries_per_scope, service=SERVICE)
        else:
            self.cache = None
        self._not_found_res = [re.compile(p, re.IGNORECASE) for p in cfg.not_found_patterns]

    def _maybe_reload(self) -> None:
        if not self._follow:
            return
        store = get_config()
        if store.version != self._cfg_version:
            self._cfg_version = store.version
            try:
                self._configure(store.section("rag", RagConfig))
            except AIReadyError:
                log.exception("rag configuration rejected; keeping the previous one")

    # ================================================================== public API
    async def answer(self, q: RagQuery, principal: Principal) -> RagAnswer:
        """Mode A: check, then send."""
        self._maybe_reload()
        run = _Run(self, q, principal)
        with run.trace(), span("rag.answer", feature=q.feature, mode="check_then_send") as root:
            early = await self._prepare(run)
            if early is not None:
                return early
            outcome = await self._generate_checked(run)
            answer = await self._finish(run, outcome)
            root.set(decision=outcome.decision, grounded=answer.grounded, stage_ms=run.stage_ms)
            return answer

    async def retrieve(self, q: RagQuery, principal: Principal) -> list[SearchHit]:
        """Screened, access-filtered, reranked passages without generation (used by agents and tools)."""
        self._maybe_reload()
        run = _Run(self, q, principal)
        with run.trace(), span("rag.retrieve", feature=q.feature):
            await self._screen_and_authorise(run)
            await self._plan_and_embed(run)
            await self._search(run)
            weak = self._weak(run)
            if not weak:
                await self._rerank(run)
                weak = self._weak_after_rerank(run)
            passages = [] if weak else run.passages
            get_audit().for_principal(principal, "rag.retrieve", action="data.read", resource=run.collection,
                                      decision="not_found" if weak else "retrieved",
                                      chunk_ids=[h.chunk.id for h in passages],
                                      candidate_ids=[h.chunk.id for h in run.candidates[:50]],
                                      query=run.plan.standalone if run.plan else run.question,
                                      fallbacks=run.fallbacks, stage_ms=run.stage_ms)
            return passages

    async def answer_stream(self, q: RagQuery, principal: Principal) -> AsyncIterator[dict[str, Any]]:
        """Mode B: stream, then verify.

        Events: ``start`` (passages), ``token`` (text deltas), ``verdict`` (ok / flag / withdraw with grounding
        details), ``done`` (the final ``RagAnswer``; after a withdraw it holds the sources-only reply).
        """
        self._maybe_reload()
        run = _Run(self, q, principal)
        with run.trace(), span("rag.answer_stream.prepare", feature=q.feature):
            early = await self._prepare(run)
        if early is not None:
            yield self._start_event(run, early.passages, cached=early.cached)
            yield {"event": "token", "text": early.answer}
            yield {"event": "verdict", "action": "ok", "grounded": early.grounded, "score": early.grounding_score,
                   "unsupported": [], "citations": [c.model_dump() for c in early.citations]}
            yield {"event": "done", "answer": early.model_dump()}
            return

        with run.trace():
            tpl = run.prompt or await self._prompt(self.cfg.prompts.answer)
            bundle = self._bundle(run, tpl)
        yield self._start_event(run, bundle.passages, cached=False)

        parts: list[str] = []
        failed = False
        t_gen = time.perf_counter()
        first_seen = False
        try:
            async for delta in self._stream_tokens(run, bundle.messages, tpl):
                if not first_seen:
                    first_seen = True
                    ms = (time.perf_counter() - t_gen) * 1000
                    run.stage_ms["first_token"] = round(ms, 2)
                    metrics.ttft.labels(SERVICE, principal.app, q.feature, "-", run.model).observe(ms / 1000)
                    if ms > self.cfg.latency_budget_ms.get("first_token", float("inf")):
                        run.over_budget.append("first_token")
                parts.append(delta)
                yield {"event": "token", "text": delta}
        except (TimeoutError, UpstreamError) as exc:
            log.warning("streaming generation failed: %s", exc)
            failed = True
        run.stage_ms["generate"] = round((time.perf_counter() - t_gen) * 1000, 2)

        with run.trace(), span("rag.answer_stream.verify") as sp:
            if failed and not parts:
                run.fallback("generation_failed")
                outcome = self._sources_only(run, bundle.passages, decision="sources_only")
                action = "withdraw"
            else:
                text = "".join(parts).strip()
                outcome, action = await self._verify_streamed(run, text, bundle)
            sp.set(action=action, decision=outcome.decision)
            answer = await self._finish(run, outcome, mode="stream_then_verify", verdict=action)
        g = outcome.grounding
        yield {"event": "verdict", "action": action, "grounded": answer.grounded, "score": answer.grounding_score,
               "unsupported": g.unsupported if g else [], "citations": [c.model_dump() for c in answer.citations]}
        yield {"event": "done", "answer": answer.model_dump()}

    # ================================================================== preparation
    async def _prepare(self, run: _Run) -> RagAnswer | None:
        """Everything up to the prompt. Returns an answer when the request ends early (cache hit, not found)."""
        await self._screen_and_authorise(run)
        await self._load_history(run)
        run.prompt = await self._prompt(self.cfg.prompts.answer)
        await self._plan_and_embed(run)

        hit = await self._cache_lookup(run)
        if hit is not None:
            await self._record_turns(run, hit.answer)
            self._audit(run, "cached", hit)
            return hit

        await self._search(run)
        if self._weak(run):
            return await self._not_found(run, "weak_retrieval")
        await self._rerank(run)
        if self._weak_after_rerank(run):
            return await self._not_found(run, "weak_rerank")
        return None

    async def _screen_and_authorise(self, run: _Run) -> None:
        q, p = run.q, run.principal
        with run.stage("screen"):
            res = await connect("guardrails").screen_input(q.question, principal=p,
                                                          context={"feature": q.feature, "app": p.app,
                                                                   "service": SERVICE})
        for f in res.findings:
            metrics.guardrail.labels(SERVICE, f.check, "blocked" if not res.allowed else "flagged").inc()
        if not res.allowed:
            get_audit().for_principal(p, "rag.answer", action="guardrails.screen", decision="blocked",
                                      findings=[f.model_dump() for f in res.findings])
            raise GuardrailBlocked("the question was blocked by input screening",
                                   details={"checks": sorted({f.check for f in res.findings})})
        run.question = res.text or q.question
        run.personal = res.data_class.rank >= DataClass.PERSONAL.rank or bool(res.masked_entities)
        run.collection = self._collection(q, p)

        decision = await self._policy(run, "data.read", {"type": "collection", "collection": run.collection},
                                      {"data_class": res.data_class.value})
        run.obligations = decision.obligations or {}
        run.no_cache = bool(run.obligations.get("no_cache"))

    async def _policy(self, run: _Run, action: str, resource: dict[str, Any], context: dict[str, Any]) -> PolicyDecision:
        p = run.principal
        inp = PolicyInput(action=action, principal=principal_facts(p), resource=resource,
                          context={"feature": run.q.feature, "app": p.app, "service": SERVICE, **context})
        with span("rag.policy", action=action) as sp:
            decision = await connect("policy").decide(inp)
            sp.set(allow=decision.allow, effect=decision.effect)
        metrics.policy.labels(SERVICE, action, decision.effect if decision.allow else "deny").inc()
        if not decision.allow or decision.effect == "deny":
            get_audit().for_principal(p, "rag.policy_denied", action=action, resource=json.dumps(resource),
                                      decision="deny", reasons=decision.reasons, rules=decision.matched_rules,
                                      policy_version=decision.policy_version)
            raise PolicyViolation(f"policy denied {action}", details={"reasons": decision.reasons})
        if decision.effect == "require_approval":
            get_audit().for_principal(p, "rag.policy_denied", action=action, resource=json.dumps(resource),
                                      decision="require_approval", reasons=decision.reasons)
            raise ApprovalRequired(f"{action} requires approval", details={"reasons": decision.reasons})
        return decision

    async def _load_history(self, run: _Run) -> None:
        q = run.q
        if run.history or not q.session_id or not self.cfg.memory.enabled:
            return
        try:
            with span("rag.memory.context"):
                run.history = await connect("memory").session_context(q.session_id, run.principal,
                                                                      max_tokens=self.cfg.context.history_tokens)
        except Exception as exc:  # memory is optional context; the answer must not depend on it
            log.info("session memory unavailable: %s", exc)
            run.fallback("memory")

    async def _prompt(self, name: str) -> PromptTemplate:
        return await self.prompts.get(name, self.cfg.prompts.versions.get(name))

    async def _plan_and_embed(self, run: _Run) -> None:
        with run.stage("plan"):
            run.plan = await self.planner.plan(run.question, run.history, run.principal)
        queries = [run.plan.standalone] + [s for s in run.plan.sub_queries if s != run.plan.standalone]
        if not self.cfg.retrieval.embed_query:
            return
        with run.stage("embed", texts=len(queries)):
            try:
                vecs = await connect("gateway").embed(queries, run.principal,
                                                      model=self.cfg.retrieval.embedding_model)
                run.vector = vecs[0]
                run.sub_vectors = dict(zip(queries, vecs))
            except PermissionDenied:
                raise
            except Exception as exc:
                log.warning("query embedding failed (%s); search falls back to its own embedding/keywords", exc)
                run.fallback("embed")

    async def _search(self, run: _Run) -> None:
        assert run.plan is not None
        queries = run.plan.queries
        sub_vecs = run.sub_vectors
        top_k = run.q.top_k or self.cfg.retrieval.top_k
        with run.stage("search", collection=run.collection):
            run.candidates = await self.retriever.retrieve(
                queries, [sub_vecs.get(qq) for qq in queries], run.principal, collection=run.collection,
                top_k=top_k, filters=run.q.filters)

    def _weak(self, run: _Run) -> bool:
        if not run.candidates:
            return True
        field = self.cfg.retrieval.score_field
        best = max((getattr(h, field) if getattr(h, field) is not None else h.score) for h in run.candidates)
        return best < self.cfg.retrieval.min_score

    def _weak_after_rerank(self, run: _Run) -> bool:
        if not run.passages:
            return True
        floor = self.cfg.retrieval.min_rerank_score
        if floor is None:
            return False
        scores = [h.rerank_score for h in run.passages if h.rerank_score is not None]
        return bool(scores) and max(scores) < floor

    async def _rerank(self, run: _Run) -> None:
        assert run.plan is not None
        top_n = run.q.top_n or self.cfg.retrieval.top_n
        cap = run.obligations.get("max_passages")
        if isinstance(cap, int) and cap > 0:
            top_n = min(top_n, cap)
        with run.stage("rerank"):
            out = await self.reranker.rerank(run.plan.standalone, run.candidates, run.principal, top_n=top_n)
        if out.fallback_reason and out.fallback_reason != "disabled":
            run.fallbacks.append("rerank")  # already counted by the rerank client
        run.passages = out.hits

    async def _cache_lookup(self, run: _Run) -> RagAnswer | None:
        if self.cache is None or run.no_cache:
            return None
        assert run.prompt is not None
        run.scope = self._scope(run)
        with run.stage("cache"):
            entry = await self.cache.lookup(run.plan.standalone if run.plan else run.question, run.scope,
                                            vector=run.vector)
        if entry is None:
            return None
        ans = RagAnswer.model_validate(entry.answer)
        return ans.model_copy(update={"cached": True, "trace_id": run.trace_id})

    def _scope(self, run: _Run) -> CacheScope:
        assert run.prompt is not None
        p = run.principal
        fil = hashlib.sha256(json.dumps(run.q.filters, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return CacheScope(app=p.app, tenant=p.tenant, access_groups=p.access_groups, prompt_version=run.prompt.ref,
                          model=run.model, feature=f"{run.q.feature}|{run.collection}|{fil}")

    async def _not_found(self, run: _Run, reason: str) -> RagAnswer:
        best = max((h.score for h in run.candidates), default=None)
        ans = RagAnswer(answer=self.cfg.not_found_message, found=False, grounded=True, grounding_score=None,
                        passages=[], trace_id=run.trace_id, model=None, fallbacks=run.fallbacks)
        get_audit().for_principal(run.principal, "rag.answer", action="rag.answer", resource=run.collection,
                                  decision="not_found", reason=reason, best_score=best,
                                  candidate_ids=[h.chunk.id for h in run.candidates[:50]],
                                  question=run.question[:2000], stage_ms=run.stage_ms)
        await self._record_turns(run, ans.answer)
        return ans

    # ================================================================== generation + checks
    def _bundle(self, run: _Run, tpl: PromptTemplate, **extra: Any) -> ContextBundle:
        return self.context_builder.build(tpl, question=run.question, passages=run.passages, history=run.history,
                                          app=run.principal.app, not_found_message=self.cfg.not_found_message,
                                          **extra)

    def _request(self, run: _Run, messages: list[ChatMessage], tpl: PromptTemplate, model: str,
                 stream: bool = False) -> ChatRequest:
        g = self.cfg.generation
        max_tokens = g.max_tokens
        ob = run.obligations.get("max_tokens")
        if isinstance(ob, int) and ob > 0:
            max_tokens = min(max_tokens, ob)
        return ChatRequest(model=model, messages=messages, stream=stream, max_tokens=max_tokens,
                           temperature=tpl.params.get("temperature", g.temperature),
                           metadata={"feature": run.q.feature, "task": "rag", "app": run.principal.app,
                                     "prompt": tpl.ref, "trace_id": run.trace_id, "service": SERVICE})

    async def _generate(self, run: _Run, messages: list[ChatMessage], tpl: PromptTemplate) -> tuple[str, str] | None:
        """(text, model) - falls back to the smaller model when the main one is slow or failing."""
        g = self.cfg.generation
        models = [run.model] + ([g.fallback_model] if g.fallback_model and g.fallback_model != run.model else [])
        gateway = connect("gateway")
        for i, model in enumerate(models):
            with run.stage("generate", model=model):
                try:
                    resp = await asyncio.wait_for(gateway.chat(self._request(run, messages, tpl, model), run.principal),
                                                  g.timeout_seconds)
                    return resp.text, resp.model or model
                except (TimeoutError, UpstreamError) as exc:
                    log.warning("generation with %s failed (%s)", model, type(exc).__name__)
                    run.fallback("model_timeout" if not isinstance(exc, UpstreamError) else "model_error")
        return None

    async def _stream_tokens(self, run: _Run, messages: list[ChatMessage], tpl: PromptTemplate) -> AsyncIterator[str]:
        g = self.cfg.generation
        models = [run.model] + ([g.fallback_model] if g.fallback_model and g.fallback_model != run.model else [])
        gateway = connect("gateway")
        last_exc: Exception | None = None
        for model in models:
            agen = gateway.chat_stream(self._request(run, messages, tpl, model, stream=True), run.principal).__aiter__()
            try:
                first = await asyncio.wait_for(agen.__anext__(), g.first_token_timeout_seconds)
            except StopAsyncIteration:
                return
            except (TimeoutError, UpstreamError) as exc:
                last_exc = exc
                run.fallback("model_timeout" if not isinstance(exc, UpstreamError) else "model_error")
                with contextlib.suppress(Exception):
                    await agen.aclose()  # type: ignore[attr-defined]
                continue
            run.model = model
            text = _delta_text(first)
            if text:
                yield text
            async for chunk in agen:
                text = _delta_text(chunk)
                if text:
                    yield text
            return
        raise UpstreamError(f"no model produced a stream: {last_exc}")

    def _is_not_found(self, text: str) -> bool:
        return text.strip() == self.cfg.not_found_message or any(p.search(text) for p in self._not_found_res)

    async def _check(self, run: _Run, text: str, bundle: ContextBundle) -> tuple[CitationResult, GroundingResult]:
        with run.stage("check"):
            cit = self.citation_mapper.map(text, bundle.passages)
            g = await self.grounding.check(cit.claims, bundle.passages, run.principal, sources_text=bundle.sources_text)
        return cit, g

    async def _generate_checked(self, run: _Run) -> _Outcome:
        assert run.prompt is not None
        bundle = self._bundle(run, run.prompt)
        gen = await self._generate(run, bundle.messages, run.prompt)
        if gen is None:
            return self._sources_only(run, bundle.passages, decision="sources_only")
        text, model = gen
        if self._is_not_found(text):
            return self._model_not_found(run, model)
        cit, g = await self._check(run, text, bundle)
        if not g.grounded:
            run.fallback("regenerate")
            strict = await self._prompt(self.cfg.prompts.answer_strict)
            unsupported = "\n".join(f"- {u}" for u in g.unsupported) or "- (claims without a source)"
            bundle = self._bundle(run, strict, unsupported=unsupported)
            gen = await self._generate(run, bundle.messages, strict)
            if gen is None:
                return self._sources_only(run, bundle.passages, decision="sources_only", grounding=g)
            text, model = gen
            if self._is_not_found(text):
                return self._model_not_found(run, model)
            cit, g = await self._check(run, text, bundle)
            if not g.grounded:
                run.fallback("sources_only")
                return self._sources_only(run, bundle.passages, decision="sources_only", grounding=g)
        return await self._validated(run, cit, g, bundle, model)

    async def _validated(self, run: _Run, cit: CitationResult, g: GroundingResult, bundle: ContextBundle,
                         model: str) -> _Outcome:
        val = await connect("guardrails").validate_output(cit.answer, context={"feature": run.q.feature,
                                                                               "service": SERVICE})
        personal = any(f.check == "pii" for f in val.findings)
        if not val.valid:
            run.fallback("output_invalid")
            return self._sources_only(run, bundle.passages, decision="sources_only", grounding=g)
        ans = RagAnswer(answer=val.text or cit.answer, citations=cit.citations, found=True, grounded=True,
                        grounding_score=g.score, passages=bundle.passages, trace_id=run.trace_id, model=model,
                        fallbacks=run.fallbacks)
        return _Outcome(ans, decision="answered", personal=personal, grounding=g)

    async def _verify_streamed(self, run: _Run, text: str, bundle: ContextBundle) -> tuple[_Outcome, str]:
        if self._is_not_found(text):
            return self._model_not_found(run, run.model), "ok"
        cit, g = await self._check(run, text, bundle)
        if g.grounded:
            out = await self._validated(run, cit, g, bundle, run.model)
            return out, "ok" if out.decision == "answered" else "withdraw"
        if g.score >= self.cfg.stream.flag_above and cit.claims:
            ans = RagAnswer(answer=cit.answer, citations=cit.citations, found=True, grounded=False,
                            grounding_score=g.score, passages=bundle.passages, trace_id=run.trace_id,
                            model=run.model, fallbacks=run.fallbacks)
            return _Outcome(ans, decision="flagged", failed_check=True, grounding=g), "flag"
        run.fallback("withdrawn")
        return self._sources_only(run, bundle.passages, decision="withdrawn", grounding=g), "withdraw"

    def _model_not_found(self, run: _Run, model: str) -> _Outcome:
        ans = RagAnswer(answer=self.cfg.not_found_message, found=False, grounded=True, passages=[],
                        trace_id=run.trace_id, model=model, fallbacks=run.fallbacks)
        return _Outcome(ans, decision="not_found", failed_check=True)  # failed_check: never cache a not-found

    def _sources_only(self, run: _Run, passages: list[SearchHit], *, decision: str,
                      grounding: GroundingResult | None = None) -> _Outcome:
        lines = [self.cfg.sources_only_message]
        for i, h in enumerate(passages, start=1):
            where = h.chunk.source_uri or h.chunk.source
            lines.append(f"[{i}] {h.chunk.title or h.chunk.doc_id}" + (f" ({where})" if where else ""))
        ans = RagAnswer(answer="\n".join(lines), citations=self.citation_mapper.sources_only(passages), found=True,
                        grounded=False, grounding_score=grounding.score if grounding else None, passages=passages,
                        trace_id=run.trace_id, model=None, fallbacks=run.fallbacks)
        return _Outcome(ans, decision=decision, failed_check=True, grounding=grounding)

    # ================================================================== finish
    async def _finish(self, run: _Run, out: _Outcome, *, mode: str = "check_then_send",
                      verdict: str | None = None) -> RagAnswer:
        ans = out.answer.model_copy(update={"fallbacks": list(run.fallbacks), "trace_id": run.trace_id})
        await self._store(run, out, ans)
        await self._record_turns(run, ans.answer)
        self._audit(run, out.decision, ans, mode=mode, verdict=verdict, grounding=out.grounding)
        return ans

    async def _store(self, run: _Run, out: _Outcome, ans: RagAnswer) -> None:
        if self.cache is None or run.no_cache or out.decision != "answered":
            return
        never = set(self.cfg.cache.never_cache_data_classes)
        personal = run.personal or out.personal or any(h.chunk.data_class in never for h in ans.passages)
        ctypes = {str(h.chunk.metadata.get(self.cfg.cache.content_type_field, "default")) for h in ans.passages}
        # the shortest-lived content type governs the entry's expiry
        ttl = self.cfg.cache.ttl_by_content_type
        ctype = min(ctypes, key=lambda c: ttl.get(c, self.cfg.cache.default_ttl_seconds)) if ctypes else "default"
        scope = run.scope or self._scope(run)
        with run.stage("cache_store"):
            await self.cache.store(run.plan.standalone if run.plan else run.question,
                                   ans.model_dump(exclude={"trace_id", "cached"}), scope,
                                   source_doc_ids=sorted({h.chunk.doc_id for h in ans.passages}),
                                   content_type=ctype, vector=run.vector, contains_personal_data=personal,
                                   failed_check=out.failed_check)

    async def _record_turns(self, run: _Run, answer_text: str) -> None:
        q = run.q
        if not q.session_id or not self.cfg.memory.enabled or not self.cfg.memory.record_turns:
            return
        try:
            mem = connect("memory")
            await mem.append_turn(q.session_id, run.principal, "user", run.question)
            await mem.append_turn(q.session_id, run.principal, "assistant", answer_text)
        except Exception as exc:
            log.info("could not record the turn in session memory: %s", exc)
            run.fallback("memory")

    def _audit(self, run: _Run, decision: str, ans: RagAnswer, *, mode: str = "check_then_send",
               verdict: str | None = None, grounding: GroundingResult | None = None) -> None:
        get_audit().for_principal(
            run.principal, "rag.answer", action="rag.answer", resource=run.collection, decision=decision,
            mode=mode, verdict=verdict, question=run.question[:2000], answer=ans.answer[:4000],
            chunk_ids=[h.chunk.id for h in ans.passages], doc_ids=sorted({h.chunk.doc_id for h in ans.passages}),
            candidate_ids=[h.chunk.id for h in run.candidates[:50]], citations=[c.marker for c in ans.citations],
            grounded=ans.grounded, grounding_score=ans.grounding_score,
            unsupported=grounding.unsupported if grounding else [], cached=ans.cached, model=ans.model,
            prompt=run.prompt.ref if run.prompt else None, fallbacks=run.fallbacks, stage_ms=run.stage_ms,
            over_budget=run.over_budget, session_id=run.q.session_id, feature=run.q.feature)

    # ================================================================== helpers
    def _collection(self, q: RagQuery, p: Principal) -> str:
        """Physical collection: ``<base>__<app>__<tenant>`` as the search service builds and enforces it."""
        store = get_config()
        base = (q.collection or self.cfg.default_collection or store.get(f"apps.{p.app}.collection")
                or store.get("search.base_collection") or "knowledge")
        tpl = self.cfg.collection_template
        if not tpl or "__" in base:  # already a physical name
            return base
        return tpl.format(app=_safe(p.app), tenant=_safe(p.tenant), collection=_safe(base))

    @staticmethod
    def _start_event(run: _Run, passages: list[SearchHit], *, cached: bool) -> dict[str, Any]:
        return {"event": "start", "trace_id": run.trace_id, "cached": cached,
                "passages": [{"marker": f"[{i}]", "chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id,
                              "title": h.chunk.title, "source_uri": h.chunk.source_uri}
                             for i, h in enumerate(passages, start=1)]}


_SAFE = re.compile(r"[^a-z0-9_-]+")


def _safe(part: str) -> str:
    """Same normalisation as the search service's collection names."""
    return _SAFE.sub("-", (part or "default").lower()).replace("__", "_-")


def _delta_text(chunk: Any) -> str:
    """Text of one OpenAI-style stream chunk (``choices[0].delta.content``)."""
    if isinstance(chunk, str):
        return chunk
    try:
        choice = (chunk.get("choices") or [{}])[0]
        return (choice.get("delta") or {}).get("content") or choice.get("text") or ""
    except (AttributeError, IndexError, TypeError):
        return ""
