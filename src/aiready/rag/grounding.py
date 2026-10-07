"""Hallucination check: every claim must be backed by a retrieved passage.

Per claim (sentence) a support score in [0, 1] is computed against the passages it cites - or, when it cites
none, against all passages:

* ``lexical``: share of the claim's content words found in the passage (claim coverage). Cheap, explainable,
  catches invented numbers and names - the commonest RAG failure;
* ``embedding``: cosine similarity of claim and passage embeddings (gateway embeddings, or hash embeddings);
* ``hybrid``: the higher of the two.

Optionally an LLM judge (gateway route ``judge``) reviews the claims; a claim it flags is unsupported even if
the lexical score passed. A judge that fails or answers in an unexpected format is ignored (and logged), so
the check never depends on it. Result: grounded flag, overall score and the unsupported claims.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.identity import Principal
from aiready.core.plugins import register
from aiready.core.telemetry import metrics, span
from aiready.core.textutil import cosine, hash_embed, tokenize
from aiready.core.types import ChatMessage, ChatRequest, SearchHit
from aiready.rag.citations import Claim
from aiready.rag.config import GroundingConfig
from aiready.rag.prompts import PromptStore

log = logging.getLogger(__name__)


class ClaimSupport(BaseModel):
    index: int
    text: str
    score: float
    supported: bool
    best_passage: int | None = None  # 1-based source number
    checked: bool = True  # False for sentences too short to be a claim


class GroundingResult(BaseModel):
    grounded: bool
    score: float  # share of checked claims that are supported
    claims: list[ClaimSupport] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    judge_used: bool = False
    method: str = "lexical"


def coverage(claim: str, passage: str) -> float:
    c = set(tokenize(claim))
    if not c:
        return 1.0
    return len(c & set(tokenize(passage))) / len(c)


@register("rag_grounding", "default")
class GroundingChecker:
    def __init__(self, config: GroundingConfig | None = None, prompts: PromptStore | None = None,
                 judge_prompt: str = "rag.judge", judge_version: str | None = None):
        self.cfg = config or GroundingConfig()
        self.prompts = prompts or PromptStore()
        self.judge_prompt, self.judge_version = judge_prompt, judge_version

    async def check(self, claims: list[Claim], passages: list[SearchHit], principal: Principal, *,
                    sources_text: str = "") -> GroundingResult:
        with span("rag.check", method=self.cfg.method, claims=len(claims), passages=len(passages)) as sp:
            texts = [p.chunk.text for p in passages]
            vectors: dict[str, list[float]] = {}
            if self.cfg.method in ("embedding", "hybrid") and claims and texts:
                vectors = await self._embed([c.text for c in claims] + texts, principal)
            results: list[ClaimSupport] = []
            for c in claims:
                if len(tokenize(c.text, drop_stopwords=False)) < self.cfg.min_claim_words:
                    results.append(ClaimSupport(index=c.index, text=c.text, score=1.0, supported=True, checked=False))
                    continue
                candidates = c.markers or list(range(1, len(texts) + 1))
                best, best_n = 0.0, None
                for n in candidates:
                    s = self._score(c.text, texts[n - 1], vectors)
                    if s > best:
                        best, best_n = s, n
                ok = best >= self.cfg.min_support and (bool(c.markers) or not self.cfg.require_citations)
                results.append(ClaimSupport(index=c.index, text=c.text, score=round(best, 4), supported=ok,
                                            best_passage=best_n))
            judge_used = False
            if self.cfg.use_llm_judge and any(r.checked for r in results):
                flagged = await self._judge([r for r in results if r.checked], sources_text, principal)
                if flagged is not None:
                    judge_used = True
                    for r in results:
                        if r.index in flagged:
                            r.supported = False
            checked = [r for r in results if r.checked]
            ratio = sum(r.supported for r in checked) / len(checked) if checked else (1.0 if claims else 0.0)
            grounded = bool(claims) and ratio >= self.cfg.min_grounded_ratio
            res = GroundingResult(grounded=grounded, score=round(ratio, 4), claims=results,
                                  unsupported=[r.text for r in checked if not r.supported], judge_used=judge_used,
                                  method=self.cfg.method)
            sp.set(grounded=grounded, score=res.score, unsupported=len(res.unsupported))
            metrics.guardrail.labels("rag", "grounding", "pass" if grounded else "fail").inc()
            return res

    # ------------------------------------------------------------------ scoring
    def _score(self, claim: str, passage: str, vectors: dict[str, list[float]]) -> float:
        lex = coverage(claim, passage)
        if self.cfg.method == "lexical":
            return lex
        emb = cosine(vectors.get(claim) or hash_embed(claim, self.cfg.embedding_dim),
                     vectors.get(passage) or hash_embed(passage, self.cfg.embedding_dim))
        return emb if self.cfg.method == "embedding" else max(lex, emb)

    async def _embed(self, texts: list[str], principal: Principal) -> dict[str, list[float]]:
        from aiready.core.wiring import connect

        uniq = list(dict.fromkeys(texts))
        try:
            vecs = await connect("gateway").embed(uniq, principal)
            return dict(zip(uniq, vecs))
        except Exception as exc:  # hash embeddings keep the check running
            log.info("grounding embeddings unavailable (%s); using hash embeddings", exc)
            metrics.fallbacks.labels("rag", "grounding_embed").inc()
            return {t: hash_embed(t, self.cfg.embedding_dim) for t in uniq}

    async def _judge(self, claims: list[ClaimSupport], sources_text: str, principal: Principal) -> set[int] | None:
        """Claim indexes the judge says are unsupported, or None when the judge gave no usable verdict."""
        from aiready.core.wiring import connect

        numbered = {i + 1: c for i, c in enumerate(claims)}
        with span("rag.check.judge", route=self.cfg.judge_route) as sp:
            try:
                tpl = await self.prompts.get(self.judge_prompt, self.judge_version)
                claims_text = "\n".join(f"C{i}: {c.text}" for i, c in numbered.items())
                req = ChatRequest(
                    model=self.cfg.judge_route,
                    messages=[ChatMessage(role="system", content=tpl.render_system(claims=claims_text,
                                                                                    sources=sources_text)),
                              ChatMessage(role="user", content=tpl.render_user())],
                    temperature=0.0, max_tokens=200, response_format={"type": "json_object"},
                    metadata={"task": "judge", "feature": "rag.check", "prompt": tpl.ref})
                resp = await asyncio.wait_for(connect("gateway").chat(req, principal), self.cfg.judge_timeout_seconds)
                data: Any = json.loads(re.search(r"\{.*\}", resp.text, re.DOTALL).group(0))  # type: ignore[union-attr]
                nums = {int(re.sub(r"\D", "", str(x)) or 0) for x in data.get("unsupported", [])}
            except Exception as exc:
                log.info("grounding judge gave no usable verdict: %s", exc)
                sp.event("judge_failed", reason=str(exc)[:200])
                metrics.fallbacks.labels("rag", "judge").inc()
                return None
            return {numbered[n].index for n in nums if n in numbered}
