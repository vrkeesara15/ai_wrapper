"""Verifier: test questions per source after each load confirm new content can be found.

An index that accepted every upsert can still fail users (wrong collection, broken access tags, an
embedding model mismatch, a chunker that cut the answer in half). After each load the verifier asks the
source's configured questions through the real search path - with access groups - and checks that the
expected document is in the top k. Results become a pipeline metric, a run record entry, an audit record
and, on failure, an alert.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from aiready.core.audit import get_audit
from aiready.core.identity import Principal
from aiready.core.telemetry import metrics, span
from aiready.core.types import SearchRequest
from aiready.ingestion.config import AlertConfig, VerifyQuestion
from aiready.ingestion.registry import DocumentRegistry

log = logging.getLogger(__name__)


class VerifyResult(BaseModel):
    question: str
    passed: bool
    expected: str
    rank: int | None = None
    top_doc_ids: list[str] = []
    error: str | None = None


class Alerter:
    """Logs alerts and optionally POSTs them to a webhook (on-call tool, chat channel, SIEM)."""

    def __init__(self, cfg: AlertConfig | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg or AlertConfig()
        self.transport = transport
        self.sent: list[dict[str, Any]] = []
        self.hooks: list[Callable[[dict[str, Any]], None]] = []

    async def alert(self, kind: str, **details: Any) -> None:
        payload = {"kind": kind, "service": "ingestion", **details}
        self.sent.append(payload)
        log.log(logging.getLevelName(self.cfg.log_level), "ingestion alert %s: %s", kind, details)
        for hook in self.hooks:
            hook(payload)
        if self.cfg.webhook_url:
            try:
                async with httpx.AsyncClient(timeout=5.0, transport=self.transport) as c:
                    await c.post(self.cfg.webhook_url, json=payload)
            except httpx.HTTPError as exc:
                log.warning("alert webhook failed: %s", exc)


class Verifier:
    def __init__(self, *, search: Any, gateway: Any, principal: Principal, registry: DocumentRegistry,
                 alerter: Alerter | None = None, model: str = "embedding"):
        self.search, self.gateway, self.principal = search, gateway, principal
        self.registry = registry
        self.alerter = alerter or Alerter()
        self.model = model

    def _groups(self, q: VerifyQuestion, source: str) -> list[str]:
        if q.as_groups is not None:
            return list(q.as_groups)
        if q.expect_doc_id:
            entry = self.registry.get(q.expect_doc_id, source) or self.registry.get(q.expect_doc_id)
            if entry is not None:
                return list(entry.access_tags)
        return sorted({t for e in self.registry.entries(source) for t in e.access_tags})

    @staticmethod
    def _matches(q: VerifyQuestion, hit: Any) -> bool:
        c = hit.chunk
        if q.expect_doc_id:
            return c.doc_id == q.expect_doc_id
        needle = (q.expect_doc_contains or "").lower()
        return bool(needle) and any(needle in (s or "").lower() for s in (c.doc_id, c.title, c.source_uri))

    async def verify(self, source: str, collection: str, questions: list[VerifyQuestion],
                     run_id: str | None = None) -> list[VerifyResult]:
        results: list[VerifyResult] = []
        with span("ingestion.verify", source=source, questions=len(questions)) as sp:
            for q in questions:
                expected = q.expect_doc_id or q.expect_doc_contains or ""
                try:
                    vec = (await self.gateway.embed([q.question], self.principal, model=self.model, priority="live"))[0]
                    hits = await self.search.search(SearchRequest(
                        collection=collection, query_text=q.question, query_vector=vec,
                        access_groups=self._groups(q, source), top_k=max(q.top_k, 1)), self.principal)
                    top = hits[: q.top_k]
                    rank = next((i + 1 for i, h in enumerate(top) if self._matches(q, h)), None)
                    res = VerifyResult(question=q.question, passed=rank is not None, expected=expected, rank=rank,
                                       top_doc_ids=[h.chunk.doc_id for h in top])
                except Exception as exc:  # a failing check is a failed verification, not a crashed run
                    res = VerifyResult(question=q.question, passed=False, expected=expected, error=str(exc))
                results.append(res)
                metrics.pipeline.labels(source, "verify", "pass" if res.passed else "fail").inc()
            failed = [r for r in results if not r.passed]
            sp.set(passed=len(results) - len(failed), failed=len(failed))
        if questions:
            get_audit().for_principal(self.principal, "ingestion.verify", action="verify", resource=source,
                                      decision="pass" if not failed else "fail", run_id=run_id,
                                      results=[r.model_dump() for r in results])
        if failed:
            await self.alerter.alert("verification_failed", source=source, run_id=run_id,
                                     failed=[r.model_dump() for r in failed])
        return results
