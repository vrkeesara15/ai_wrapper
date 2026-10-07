"""Rerank client: the top 50 candidates in, the best 5 out.

Calls the reranking service (``connect("rerank")``) or the gateway's rerank route, both of which expose
``rerank(query, documents, principal, top_n=, model=) -> RerankResponse``. Passages are trimmed to the
reranker's length limit before the call. If the reranker is slow or unavailable the search order is used and
the event is logged and counted (slide "Reranking service" - Fallback).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.identity import Principal
from aiready.core.plugins import register
from aiready.core.telemetry import metrics, span
from aiready.core.tokens import truncate_to_tokens
from aiready.core.types import SearchHit
from aiready.rag.config import RerankConfig

log = logging.getLogger(__name__)


class RerankOutcome(BaseModel):
    hits: list[SearchHit]
    reranked: bool
    fallback_reason: str | None = None
    dropped: int = 0  # passages under the score threshold
    model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@register("rag_reranker", "default")
class RerankClient:
    def __init__(self, config: RerankConfig | None = None, backend: Any = None):
        self.cfg = config or RerankConfig()
        self._backend = backend

    @property
    def backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        from aiready.core.wiring import connect

        return connect("rerank" if self.cfg.via == "rerank" else "gateway")

    async def _call(self, query: str, docs: list[str], principal: Principal) -> Any:
        """Both backends return ``RerankResponse``; the reranking service takes the principal by keyword, the
        gateway positionally and needs a route name."""
        if self.cfg.via == "rerank":
            kw: dict[str, Any] = {"top_n": len(docs), "principal": principal}
            if self.cfg.model:
                kw["model"] = self.cfg.model
            return await self.backend.rerank(query, docs, **kw)
        return await self.backend.rerank(query, docs, principal, top_n=len(docs), model=self.cfg.model or "reranker")

    async def rerank(self, query: str, hits: list[SearchHit], principal: Principal, *, top_n: int) -> RerankOutcome:
        if not hits:
            return RerankOutcome(hits=[], reranked=False)
        if not self.cfg.enabled:
            return RerankOutcome(hits=hits[:top_n], reranked=False, fallback_reason="disabled")
        docs = [truncate_to_tokens(h.chunk.text, self.cfg.max_passage_tokens) for h in hits]
        with span("rag.rerank", via=self.cfg.via, candidates=len(hits), top_n=top_n) as sp:
            try:
                resp = await asyncio.wait_for(self._call(query, docs, principal), self.cfg.timeout_seconds)
            except Exception as exc:
                reason = "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else type(exc).__name__
                log.warning("reranker unavailable (%s: %s); using search order", reason, exc)
                sp.event("fallback", reason=reason)
                metrics.fallbacks.labels("rag", "rerank").inc()
                return RerankOutcome(hits=hits[:top_n], reranked=False, fallback_reason=reason)
            ordered: list[SearchHit] = []
            dropped = 0
            for r in sorted(resp.results, key=lambda r: -r.relevance_score):
                if not 0 <= r.index < len(hits):
                    continue
                if self.cfg.min_score is not None and r.relevance_score < self.cfg.min_score:
                    dropped += 1
                    continue
                ordered.append(hits[r.index].model_copy(update={"rerank_score": r.relevance_score}))
            sp.set(kept=min(len(ordered), top_n), dropped=dropped)
            return RerankOutcome(hits=ordered[:top_n], reranked=True, dropped=dropped, model=resp.model)
