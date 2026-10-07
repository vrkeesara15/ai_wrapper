"""Reranking service.

    question + top 50 candidates -> (question, passage) pairs trimmed to the model limit -> score in batches
    -> sort -> drop under threshold -> keep top 5 -> passages go to the prompt in score order

Fallback: if the scorer is slower than ``timeout_seconds`` or unavailable, the search order is returned
unchanged (truncated to top_n), the event is logged and counted in
``aiready_fallbacks_total{service="rerank"}``.
A reranker outage therefore degrades answer quality slightly instead of failing every question.

Like the embedding service, this is part of the model-serving layer: it calls its scorer directly and the
gateway routes ``rerank`` requests here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import ConfigError, NotFound, PolicyViolation
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyInput
from aiready.core.plugins import create
from aiready.core.telemetry import metrics, span
from aiready.core.types import RerankResponse, RerankResult, SearchHit
from aiready.core.wiring import connect
from aiready.rerank.config import RerankConfig
from aiready.rerank.scorers import Scorer

log = logging.getLogger(__name__)
SERVICE = "rerank"
FALLBACK_MODEL = "search-order-fallback"


class RerankOutcome(BaseModel):
    """Result of reranking search hits. ``fallback`` names why the search order was kept, if it was."""

    hits: list[SearchHit]
    model: str
    fallback: str | None = None  # "timeout" | "error" | None
    candidates: int = 0
    dropped_below_threshold: int = 0


class RerankService:
    def __init__(self, config: RerankConfig | None = None, *, scorers: dict[str, Scorer] | None = None):
        self.config = config or get_config().section("rerank", RerankConfig)
        self.scorers: dict[str, Scorer] = (
            scorers
            if scorers is not None
            else {name: create("reranker", spec) for name, spec in self.config.models.items()}
        )
        if self.config.default_model not in self.scorers:
            raise ConfigError(f"rerank.default_model '{self.config.default_model}' is not in rerank.models")

    def resolve(self, model: str | None) -> tuple[str, Scorer]:
        name = model or self.config.default_model
        if name in self.scorers:
            return name, self.scorers[name]
        for logical, sc in self.scorers.items():
            if sc.model_id == name:
                return logical, sc
        raise NotFound(f"unknown reranker '{name}'", details={"available": sorted(self.scorers)})

    # ------------------------------------------------------------------ core
    async def _score(
        self, query: str, passages: list[str], scorer: Scorer
    ) -> tuple[list[float] | None, str | None]:
        """Scores, or (None, reason) when falling back to the search order."""
        try:
            with metrics.time_stage(SERVICE, "score"):
                return await asyncio.wait_for(
                    scorer.score(query, passages), timeout=self.config.timeout_seconds
                ), None
        except TimeoutError:
            reason = "timeout"
        except Exception as exc:
            if not self.config.fallback_on_error:
                raise
            log.warning("reranker %s failed: %s", scorer.model_id, exc)
            reason = "error"
        metrics.fallbacks.labels(SERVICE, reason).inc()
        log.warning("rerank fallback (%s): returning search order for %d candidates", reason, len(passages))
        return None, reason

    async def _rank(
        self,
        query: str,
        passages: list[str],
        *,
        top_n: int | None,
        threshold: float | None,
        max_candidates: int | None,
        model: str | None,
        principal: Principal | None,
        app: str | None,
    ) -> tuple[list[tuple[int, float | None]], str, str | None, int, int]:
        """Returns ([(index, score)], model_id, fallback_reason, n_candidates, n_dropped)."""
        sel = self.config.selection(app or (principal.app if principal else None))
        top_n = top_n or sel.top_n
        threshold = sel.threshold if threshold is None else threshold
        n_in = min(len(passages), max_candidates or sel.max_candidates)
        name, scorer = self.resolve(model)
        with span("rerank.rerank", model=scorer.model_id, candidates=n_in, top_n=top_n) as sp:
            if principal is not None and self.config.policy_check and not principal.is_service_only:
                await self._check_policy(principal, name, scorer.model_id)
            scores, reason = await self._score(query, passages[:n_in], scorer) if n_in else ([], None)
            if scores is None:
                sp.event("fallback", reason=reason)
                return [(i, None) for i in range(min(n_in, top_n))], FALLBACK_MODEL, reason, n_in, 0
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])
            kept = [(i, s) for i, s in ranked if threshold is None or s >= threshold]
            dropped = len(ranked) - len(kept)
            sp.set(dropped=dropped, kept=min(len(kept), top_n))
            return kept[:top_n], scorer.model_id, None, n_in, dropped

    # ------------------------------------------------------------------ public
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        model: str | None = None,
        threshold: float | None = None,
        principal: Principal | None = None,
        app: str | None = None,
    ) -> RerankResponse:
        """Cohere-style: indexes into ``documents`` with relevance scores, best first."""
        ranked, model_id, reason, n_in, dropped = await self._rank(
            query,
            documents,
            top_n=top_n,
            threshold=threshold,
            max_candidates=None,
            model=model,
            principal=principal,
            app=app,
        )
        self._audit(principal, model_id, reason, n_in, dropped, [str(i) for i, _ in ranked])
        return RerankResponse(
            model=model_id,
            results=[RerankResult(index=i, relevance_score=s if s is not None else 0.0) for i, s in ranked],
        )

    async def rerank_hits(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_n: int | None = None,
        threshold: float | None = None,
        max_candidates: int | None = None,
        model: str | None = None,
        principal: Principal | None = None,
        app: str | None = None,
    ) -> RerankOutcome:
        """Rerank search hits; each kept hit gets ``rerank_score`` (None on fallback)."""
        ranked, model_id, reason, n_in, dropped = await self._rank(
            query,
            [h.chunk.text for h in hits],
            top_n=top_n,
            threshold=threshold,
            max_candidates=max_candidates,
            model=model,
            principal=principal,
            app=app,
        )
        out = []
        for i, s in ranked:
            h = hits[i].model_copy()
            h.rerank_score = s
            out.append(h)
        self._audit(principal, model_id, reason, n_in, dropped, [h.chunk.id for h in out])
        return RerankOutcome(
            hits=out, model=model_id, fallback=reason, candidates=n_in, dropped_below_threshold=dropped
        )

    # ------------------------------------------------------------------ governance
    async def _check_policy(self, principal: Principal, name: str, model_id: str) -> None:
        decision = await connect("policy").decide(
            PolicyInput(
                action="model.invoke",
                principal=principal.model_dump(),
                resource={"type": "model", "kind": "reranker", "model": name, "model_id": model_id},
                context={"app": principal.app},
            )
        )
        metrics.policy.labels(SERVICE, "model.invoke", decision.effect).inc()
        if not decision.allow:
            get_audit().for_principal(
                principal,
                "rerank.denied",
                action="model.invoke",
                resource=model_id,
                decision=decision.effect,
                reasons=decision.reasons,
            )
            raise PolicyViolation("reranker not allowed by policy", details={"reasons": decision.reasons})

    @staticmethod
    def _audit(
        principal: Principal | None,
        model_id: str,
        reason: str | None,
        n_in: int,
        dropped: int,
        kept: list[str],
    ) -> None:
        kw: dict[str, Any] = dict(
            action="model.invoke",
            resource=model_id,
            decision="fallback" if reason else "allow",
            candidates=n_in,
            dropped=dropped,
            kept=kept,
            fallback=reason,
        )
        if principal is not None:
            get_audit().for_principal(principal, "rerank.score", **kw)
        elif reason:  # anonymous internal call: still record fallbacks
            get_audit().record("rerank.score", actor="svc:rerank", **kw)

    async def ready(self) -> dict[str, Any]:
        return {"models": {n: s.model_id for n, s in self.scorers.items()}}

    async def aclose(self) -> None:
        for s in self.scorers.values():
            closer = getattr(s, "aclose", None)
            if closer is not None:
                await closer()


class RerankHitsRequest(BaseModel):
    query: str
    hits: list[SearchHit]
    top_n: int | None = None
    threshold: float | None = None
    max_candidates: int | None = None
    model: str | None = None
    app: str | None = None


class RerankApiRequest(BaseModel):
    """Cohere-compatible body (``query``, ``documents``, ``top_n``, ``model``) plus optional extensions."""

    query: str
    documents: list[str | dict[str, Any]] = Field(default_factory=list)
    top_n: int | None = None
    model: str | None = None
    threshold: float | None = None
    app: str | None = None
    return_documents: bool = False

    def texts(self) -> list[str]:
        return [d if isinstance(d, str) else str(d.get("text", "")) for d in self.documents]
