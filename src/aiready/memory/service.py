"""Memory service: session memory and consented long-term memory (slide "Knowledge and memory").

* **Session** - the current conversation, summarised as it grows: when the stored turns pass a token
  threshold the older ones are folded into a running summary (gateway route ``small``; a deterministic
  first-sentence summary if the model is unavailable), the newest turns stay verbatim.
* **Long-term** - facts and preferences per user, stored ONLY with consent and ALWAYS with an expiry date.
  Every write passes the ``memory.write`` policy and input screening and is audited. ``recall`` ranks by
  relevance; ``forget`` erases one item or everything (right to erasure); ``purge_expired`` drops expired items.

Isolation: every key starts with ``<tenant>/<user>/`` taken from the verified principal, so a user can only
read, list or delete their own memory - there is no API that takes another user's id.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any
from urllib.parse import quote

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import (
    ApprovalRequired,
    GuardrailBlocked,
    NotFound,
    PermissionDenied,
    PolicyViolation,
    ValidationFailed,
)
from aiready.core.identity import Principal
from aiready.core.interfaces import MemoryItem, PolicyInput
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics, span
from aiready.core.textutil import cosine, hash_embed, split_sentences
from aiready.core.tokens import count_messages, count_tokens, truncate_to_tokens
from aiready.core.types import ChatMessage, ChatRequest
from aiready.core.wiring import connect
from aiready.memory.config import MemoryConfig
from aiready.rag.prompts import PromptStore

log = logging.getLogger(__name__)
SERVICE = "memory"
DAY = 86400.0
ROLES = {"user", "assistant", "system", "tool"}


def _owner(p: Principal) -> str:
    return f"{quote(p.tenant, safe='')}/{quote(p.sub, safe='')}/"


def _principal_facts(p: Principal) -> dict[str, Any]:
    return p.model_dump(include={"sub", "tenant", "app", "roles", "groups", "scopes", "service", "actor"})


class MemoryServiceImpl:
    """In-process implementation of ``core.interfaces.MemoryService``."""

    def __init__(self, config: MemoryConfig | None = None, *, store: DocStore | None = None,
                 prompts: PromptStore | None = None):
        self.cfg = config or get_config().section("memory", MemoryConfig)
        self._store = store
        self.prompts = prompts or PromptStore(self.cfg.prompts_source)

    @classmethod
    def from_config(cls) -> MemoryServiceImpl:
        return cls()

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    # ================================================================== session memory
    def _session_key(self, session_id: str, p: Principal) -> str:
        return _owner(p) + quote(session_id, safe="")

    def _load_session(self, session_id: str, p: Principal) -> dict[str, Any]:
        data = self.store.get(self.cfg.namespace_sessions, self._session_key(session_id, p))
        now = time.time()
        if data and data.get("expires_at", now + 1) < now:
            self.store.delete(self.cfg.namespace_sessions, self._session_key(session_id, p))
            data = None
        return data or {"session_id": session_id, "tenant": p.tenant, "user": p.sub, "app": p.app, "summary": "",
                        "summarized_turns": 0, "turns": []}

    async def append_turn(self, session_id: str, principal: Principal, role: str, content: str) -> None:
        if role not in ROLES:
            raise ValidationFailed(f"role must be one of {sorted(ROLES)}")
        s = self.cfg.session
        with span("memory.append_turn", session=session_id) as sp:
            data = self._load_session(session_id, principal)
            data["turns"].append({"role": role, "content": (content or "")[: s.max_turn_chars], "ts": time.time()})
            if count_messages(data["turns"]) > s.summarize_over_tokens and len(data["turns"]) > s.keep_recent_turns:
                old, recent = data["turns"][: -s.keep_recent_turns], data["turns"][-s.keep_recent_turns:]
                with metrics.time_stage(SERVICE, "summarize"):
                    data["summary"], method = await self._summarize(data.get("summary", ""), old, principal)
                data["turns"] = recent
                data["summarized_turns"] = int(data.get("summarized_turns", 0)) + len(old)
                sp.set(summarized=len(old), method=method)
            data["updated_at"] = time.time()
            data["expires_at"] = data["updated_at"] + s.idle_ttl_hours * 3600
            self.store.put(self.cfg.namespace_sessions, self._session_key(session_id, principal), data)

    async def session_context(self, session_id: str, principal: Principal, max_tokens: int = 1500) -> list[dict[str, Any]]:
        """Summary (as a system message) plus the newest turns that fit ``max_tokens``."""
        with span("memory.session_context", session=session_id):
            data = self._load_session(session_id, principal)
            out: list[dict[str, Any]] = []
            budget = max_tokens
            summary = data.get("summary") or ""
            if summary:
                text = truncate_to_tokens(f"Summary of the earlier conversation: {summary}", max(budget // 2, 1))
                out.append({"role": "system", "content": text})
                budget -= count_tokens(text) + 4
            recent: list[dict[str, Any]] = []
            for t in reversed(data["turns"]):
                cost = count_tokens(t["content"]) + 4
                if cost > budget:
                    break
                recent.append({"role": t["role"], "content": t["content"]})
                budget -= cost
            return out + list(reversed(recent))

    async def _summarize(self, previous: str, turns: list[dict[str, Any]], principal: Principal) -> tuple[str, str]:
        s = self.cfg.session
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        try:
            tpl = await self.prompts.get(s.prompt, s.prompt_version)
            req = ChatRequest(
                model=s.summary_route,
                messages=[ChatMessage(role="system", content=tpl.render_system(max_words=s.summary_max_words)),
                          ChatMessage(role="user", content=tpl.render_user(previous_summary=previous or "(none)",
                                                                           transcript=transcript))],
                temperature=0.0, max_tokens=s.summary_max_words * 2,
                metadata={"task": "summarize", "feature": "memory.session", "prompt": tpl.ref, "service": SERVICE})
            resp = await asyncio.wait_for(connect("gateway").chat(req, principal), s.summary_timeout_seconds)
            text = (resp.text or "").strip()
            if text:
                return self._cap(text), "model"
            raise ValueError("empty summary")
        except (PolicyViolation, PermissionDenied):
            raise
        except Exception as exc:  # the conversation must keep working without the model
            log.info("session summarisation fell back to the deterministic summary: %s", exc)
            metrics.fallbacks.labels(SERVICE, "summarize").inc()
            return self._deterministic_summary(previous, turns), "deterministic"

    def _deterministic_summary(self, previous: str, turns: list[dict[str, Any]]) -> str:
        parts = [previous] if previous else []
        for t in turns:
            first = (split_sentences(t["content"]) or [t["content"]])[0].strip()
            if first:
                parts.append(f"{t['role']}: {first}")
        return self._cap(" | ".join(parts), keep_end=True)

    def _cap(self, text: str, keep_end: bool = False) -> str:
        words = text.split()
        limit = self.cfg.session.summary_max_words
        if len(words) <= limit:
            return text
        return " ".join(words[-limit:] if keep_end else words[:limit])

    # ================================================================== long-term memory
    def _item_key(self, p: Principal, item_id: str) -> str:
        return _owner(p) + item_id

    def _embed_tag(self) -> str:
        e = self.cfg.embedding
        return f"hash:{e.dim}" if e.type == "hash" else f"gateway:{e.model}"

    async def _embed(self, texts: list[str], principal: Principal) -> list[list[float]]:
        e = self.cfg.embedding
        if e.type == "gateway":
            try:
                return await connect("gateway").embed(texts, principal, model=e.model)
            except (PolicyViolation, PermissionDenied):
                raise
            except Exception as exc:
                log.warning("memory embedding via gateway failed (%s); using hash embeddings", exc)
                metrics.fallbacks.labels(SERVICE, "embed").inc()
        return [hash_embed(t, e.dim) for t in texts]

    async def remember(self, principal: Principal, text: str, *, kind: str = "fact", ttl_days: int | None = None,
                       consent: bool = False) -> MemoryItem:
        lt = self.cfg.long_term
        audit = get_audit()
        with span("memory.remember", kind=kind) as sp:
            if kind not in lt.allowed_kinds:
                raise ValidationFailed(f"kind must be one of {lt.allowed_kinds}")
            if not (text or "").strip():
                raise ValidationFailed("text is empty")
            if lt.require_consent and not consent:
                audit.for_principal(principal, "memory.write", action="memory.write", decision="denied",
                                    reason="no_consent", kind=kind)
                metrics.policy.labels(SERVICE, "memory.write", "no_consent").inc()
                raise PermissionDenied("long-term memory is stored only with the user's consent (consent=true)")

            decision = await connect("policy").decide(PolicyInput(
                action="memory.write", principal=_principal_facts(principal),
                resource={"type": "memory", "kind": kind},
                context={"consent": consent, "ttl_days": ttl_days, "app": principal.app, "service": SERVICE}))
            metrics.policy.labels(SERVICE, "memory.write", decision.effect if decision.allow else "deny").inc()
            if not decision.allow or decision.effect != "allow":
                audit.for_principal(principal, "memory.write", action="memory.write", decision=decision.effect,
                                    reasons=decision.reasons, rules=decision.matched_rules, kind=kind)
                if decision.allow and decision.effect == "require_approval":
                    raise ApprovalRequired("memory write requires approval", details={"reasons": decision.reasons})
                raise PolicyViolation("policy denied memory.write", details={"reasons": decision.reasons})

            if lt.screen_writes:
                res = await connect("guardrails").screen_input(text, principal=principal,
                                                              context={"feature": "memory.write", "service": SERVICE})
                if not res.allowed:
                    audit.for_principal(principal, "memory.write", action="memory.write", decision="blocked",
                                        findings=[f.check for f in res.findings], kind=kind)
                    raise GuardrailBlocked("the memory item was blocked by input screening",
                                           details={"checks": sorted({f.check for f in res.findings})})

            max_days = lt.max_ttl_days
            ob = (decision.obligations or {}).get("max_ttl_days")
            if isinstance(ob, (int, float)) and ob > 0:
                max_days = min(max_days, int(ob))
            days = min(int(ttl_days or lt.default_ttl_days), max_days)
            if days <= 0:
                raise ValidationFailed("ttl_days must be positive: long-term memory always expires")

            now = time.time()
            item = MemoryItem(id=uuid.uuid4().hex, user=principal.sub, kind=kind, text=text.strip(), consent=True,  # type: ignore[arg-type]
                              created_at=now, expires_at=now + days * DAY, source=principal.actor or principal.service,
                              app=principal.app)
            vector = (await self._embed([item.text], principal))[0]
            self._evict_if_full(principal)
            self.store.put(self.cfg.namespace_items, self._item_key(principal, item.id),
                           {**item.model_dump(), "tenant": principal.tenant, "vector": vector,
                            "embed": self._embed_tag()})
            audit.for_principal(principal, "memory.write", action="memory.write", decision="allow",
                                resource=item.id, kind=kind, expires_at=item.expires_at, ttl_days=days, consent=True,
                                chars=len(item.text), policy_version=decision.policy_version)
            sp.set(item=item.id, ttl_days=days)
            return item

    def _evict_if_full(self, p: Principal) -> None:
        rows = self.store.list(self.cfg.namespace_items, _owner(p))
        excess = len(rows) - self.cfg.long_term.max_items_per_user + 1
        if excess <= 0:
            return
        victims = sorted(rows, key=lambda kv: kv[1].get("expires_at") or 0)[:excess]
        for key, _ in victims:
            self.store.delete(self.cfg.namespace_items, key)
        get_audit().for_principal(p, "memory.forget", action="memory.evict", decision="evicted",
                                  item_ids=[v[1].get("id") for v in victims], reason="max_items_per_user")

    def _live_items(self, p: Principal) -> list[dict[str, Any]]:
        now = time.time()
        out = []
        for key, val in self.store.list(self.cfg.namespace_items, _owner(p)):
            if val.get("expires_at") is not None and val["expires_at"] < now:
                self.store.delete(self.cfg.namespace_items, key)  # lazily purge
                continue
            # defence in depth: the key prefix already isolates; also check the stored owner
            if val.get("user") != p.sub or val.get("tenant") != p.tenant:
                continue
            if not self.cfg.long_term.share_across_apps and val.get("app", "default") != p.app:
                continue
            out.append(val)
        return out

    async def recall(self, principal: Principal, query: str | None = None, limit: int = 10) -> list[MemoryItem]:
        with span("memory.recall", has_query=bool(query)) as sp:
            items = self._live_items(principal)
            if query and items:
                qv = (await self._embed([query], principal))[0]
                tag = self._embed_tag()
                stale = [it for it in items if it.get("embed") != tag or not it.get("vector")]
                if stale:  # embedding model changed since they were stored
                    vecs = await self._embed([it["text"] for it in stale], principal)
                    for it, v in zip(stale, vecs):
                        it["vector"], it["embed"] = v, tag
                        self.store.put(self.cfg.namespace_items, self._item_key(principal, it["id"]), it)
                scored = [(cosine(qv, it["vector"]), it) for it in items]
                scored = [s for s in scored if s[0] >= self.cfg.long_term.min_recall_score]
                scored.sort(key=lambda s: -s[0])
                ranked = [it for _, it in scored]
            else:
                ranked = sorted(items, key=lambda it: -it.get("created_at", 0))
            result = [MemoryItem.model_validate({k: v for k, v in it.items()
                                                 if k in MemoryItem.model_fields}) for it in ranked[:limit]]
            sp.set(returned=len(result))
            get_audit().for_principal(principal, "memory.recall", action="memory.read", decision="allow",
                                      item_ids=[m.id for m in result])
            return result

    async def forget(self, principal: Principal, item_id: str | None = None) -> int:
        """Erase one item, or (``item_id=None``) every long-term item and session of this user."""
        with span("memory.forget", all=item_id is None) as sp:
            ns_items, ns_sessions = self.cfg.namespace_items, self.cfg.namespace_sessions
            if item_id is not None:
                key = self._item_key(principal, item_id)
                if self.store.get(ns_items, key) is None:
                    raise NotFound(f"memory item {item_id} not found")
                self.store.delete(ns_items, key)
                count = 1
            else:
                count = 0
                for ns in (ns_items, ns_sessions):
                    for key, _ in self.store.list(ns, _owner(principal)):
                        self.store.delete(ns, key)
                        count += 1
            sp.set(deleted=count)
            get_audit().for_principal(principal, "memory.forget", action="memory.delete", decision="deleted",
                                      resource=item_id or "*", count=count,
                                      scope="item" if item_id else "all")
            return count

    async def purge_expired(self, now: float | None = None) -> int:
        """Delete expired long-term items and idle sessions of every user (run on a schedule)."""
        now = now or time.time()
        purged = 0
        with span("memory.purge_expired"):
            for ns in (self.cfg.namespace_items, self.cfg.namespace_sessions):
                for key, val in self.store.list(ns):
                    exp = val.get("expires_at") if isinstance(val, dict) else None
                    if exp is not None and exp < now:
                        self.store.delete(ns, key)
                        purged += 1
        get_audit().record("memory.purge", actor="svc:memory", action="memory.delete", decision="purged",
                           count=purged)
        return purged
