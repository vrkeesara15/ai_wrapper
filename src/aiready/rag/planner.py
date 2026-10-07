"""Query planner: turns what the user typed into what the index should be searched for.

Two jobs (slide "Retrieval platform architecture" - "Rewrites, splits the question"):

* **rewrite** a follow-up ("and for cars?") into a standalone question using the conversation, because the
  index knows nothing about the conversation;
* **split** a multi-part question ("what is the deadline and who approves it?") into sub-queries, because one
  embedding of two questions retrieves neither well.

The model (gateway route ``small``, task ``rewrite``) does both in one call. A deterministic heuristic runs
when the model is not needed, times out, or returns something unparsable, so planning never blocks an answer.
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
from aiready.core.textutil import tokenize
from aiready.core.types import ChatMessage, ChatRequest
from aiready.rag.config import PlannerConfig
from aiready.rag.prompts import PromptStore

log = logging.getLogger(__name__)

_FOLLOW_UP = re.compile(
    r"^\s*(and|but|also|what about|how about|and what|what if|why|so|then|ok|okay)\b|"
    r"\b(it|its|that|this|those|these|they|them|their|he|she|his|her|there|same|above|previous|former|latter)\b",
    re.IGNORECASE)
_SPLIT = re.compile(r"\?\s+|;\s+|\s+(?:and also|as well as|and then|additionally|also,)\s+", re.IGNORECASE)
_MULTI_HINT = re.compile(r"\?.+\?|;|\b(and also|as well as|additionally)\b", re.IGNORECASE | re.DOTALL)


class QueryPlan(BaseModel):
    original: str
    standalone: str
    sub_queries: list[str] = Field(default_factory=list)
    method: str = "heuristic"  # heuristic | llm

    @property
    def queries(self) -> list[str]:
        return self.sub_queries or [self.standalone]


@register("rag_planner", "default")
class QueryPlanner:
    def __init__(self, config: PlannerConfig | None = None, prompts: PromptStore | None = None,
                 prompt_name: str = "rag.plan", prompt_version: str | None = None):
        self.cfg = config or PlannerConfig()
        self.prompts = prompts or PromptStore()
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version

    # ------------------------------------------------------------------ public
    async def rewrite(self, question: str, history: list[dict[str, Any]], principal: Principal) -> QueryPlan:
        """Standalone form of the question (and sub-queries when the model produced them)."""
        hist = self._recent(history)
        if not hist:
            return QueryPlan(original=question, standalone=question)
        follow_up = self._looks_follow_up(question)
        if self.cfg.use_llm == "always" or (self.cfg.use_llm == "auto" and follow_up):
            plan = await self._llm_plan(question, hist, principal)
            if plan is not None:
                return plan
        if not follow_up:  # self-contained question: nothing to resolve
            return QueryPlan(original=question, standalone=question)
        return QueryPlan(original=question, standalone=self._heuristic_rewrite(question, hist))

    def needs_split(self, question: str) -> bool:
        return bool(_MULTI_HINT.search(question.strip().rstrip("?")) or question.count("?") > 1)

    async def split(self, plan: QueryPlan, principal: Principal) -> QueryPlan:
        """Fill ``sub_queries`` when the standalone question asks several things."""
        if plan.sub_queries:
            return plan
        q = plan.standalone
        if self.cfg.use_llm == "always" or (self.cfg.use_llm == "auto" and self.needs_split(q)):
            llm = await self._llm_plan(q, [], principal)
            if llm is not None and len(llm.sub_queries) > 1:
                return plan.model_copy(update={"sub_queries": llm.sub_queries, "method": "llm"})
        parts = self._heuristic_split(q)
        return plan.model_copy(update={"sub_queries": parts if len(parts) > 1 else []})

    async def plan(self, question: str, history: list[dict[str, Any]], principal: Principal) -> QueryPlan:
        return await self.split(await self.rewrite(question, history, principal), principal)

    # ------------------------------------------------------------------ model
    async def _llm_plan(self, question: str, history: list[dict[str, Any]], principal: Principal) -> QueryPlan | None:
        from aiready.core.wiring import connect

        with span("rag.plan.llm", route=self.cfg.route) as sp:
            try:
                tpl = await self.prompts.get(self.prompt_name, self.prompt_version)
                hist_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history) or "(none)"
                req = ChatRequest(
                    model=self.cfg.route,
                    messages=[
                        ChatMessage(role="system", content=tpl.render_system(
                            history=hist_text, max_subqueries=self.cfg.max_subqueries)),
                        ChatMessage(role="user", content=tpl.render_user(question=question)),
                    ],
                    temperature=tpl.params.get("temperature", 0.0), max_tokens=300,
                    response_format={"type": "json_object"},
                    metadata={"task": self.cfg.task, "feature": "rag.plan", "prompt": tpl.ref},
                )
                resp = await asyncio.wait_for(connect("gateway").chat(req, principal), self.cfg.timeout_seconds)
                plan = self._parse(question, resp.text)
            except Exception as exc:  # planning is an optimisation; never fail the request on it
                log.info("planner model unavailable, using heuristic: %s", exc)
                sp.event("fallback", reason=str(exc)[:200])
                metrics.fallbacks.labels("rag", "planner").inc()
                return None
            if plan is None:
                sp.event("fallback", reason="unparsable plan")
                metrics.fallbacks.labels("rag", "planner").inc()
            return plan

    def _parse(self, question: str, text: str) -> QueryPlan | None:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return None
        standalone = str(data.get("standalone") or "").strip()
        if not standalone:
            return None
        subs = [str(s).strip() for s in data.get("sub_queries") or [] if str(s).strip()]
        subs = subs[: self.cfg.max_subqueries]
        if len(subs) == 1 and subs[0].lower() == standalone.lower():
            subs = []
        return QueryPlan(original=question, standalone=standalone, sub_queries=subs, method="llm")

    # ------------------------------------------------------------------ heuristics
    def _recent(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        turns = [h for h in history or [] if h.get("content") and h.get("role") in ("user", "assistant", "system")]
        return turns[-self.cfg.max_history_turns:]

    @staticmethod
    def _looks_follow_up(question: str) -> bool:
        return bool(_FOLLOW_UP.search(question)) or len(tokenize(question)) <= 1

    def _heuristic_rewrite(self, question: str, history: list[dict[str, Any]]) -> str:
        """Append the subject words of the previous user question that the follow-up does not repeat."""
        if not self._looks_follow_up(question):
            return question
        prev = next((h["content"] for h in reversed(history) if h.get("role") == "user"), None)
        if not prev:
            prev = next((h["content"] for h in reversed(history) if h.get("role") == "system"), "")
        own = set(tokenize(question))
        carry: list[str] = []
        for tok in tokenize(prev):
            if tok not in own and tok not in carry:
                carry.append(tok)
        if not carry:
            return question
        return f"{question.strip()} ({' '.join(carry[:12])})"

    def _heuristic_split(self, question: str) -> list[str]:
        parts = [p.strip(" ,.?") for p in _SPLIT.split(question) if p and p.strip(" ,.?")]
        parts = [p + "?" for p in parts if len(tokenize(p, drop_stopwords=False)) >= self.cfg.min_subquery_words]
        if len(parts) <= 1:
            return []
        return parts[: self.cfg.max_subqueries]
