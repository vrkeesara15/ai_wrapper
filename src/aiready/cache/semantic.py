"""Semantic cache.

Request -> embed -> nearest stored question in the *same scope* -> similarity over threshold and entry not
expired -> stored answer, no model call. Otherwise the caller runs retrieval and the model and writes back.

Scope: an entry is reused only for the same application, tenant, access groups, prompt version and model.
Expiry: a time limit per content type; entries are dropped when a source document changes.
Never cached: answers with personal data, write actions, and answers that failed a check.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Awaitable, Callable

import numpy as np
from pydantic import BaseModel, Field

from aiready.core.stores import KVStore, MemoryKV
from aiready.core.telemetry import metrics


class CacheScope(BaseModel):
    app: str = "default"
    tenant: str = "default"
    access_groups: list[str] = Field(default_factory=list)
    prompt_version: str = "v1"
    model: str = "auto"
    feature: str = "default"

    def key(self) -> str:
        raw = json.dumps({**self.model_dump(), "access_groups": sorted(self.access_groups)}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class CacheEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: str
    vector: list[float] | None = None
    answer: dict[str, Any]
    source_doc_ids: list[str] = Field(default_factory=list)
    content_type: str = "default"
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    hits: int = 0
    similarity: float | None = None


class SemanticCache:
    def __init__(self, kv: KVStore | None = None, embed: Callable[[str], Awaitable[list[float]]] | None = None, *,
                 threshold: float = 0.92, default_ttl: float = 3600, ttl_by_content_type: dict[str, float] | None = None,
                 max_entries_per_scope: int = 5000, service: str = "cache"):
        self.kv = kv or MemoryKV()
        self.embed = embed
        self.threshold = threshold
        self.default_ttl = default_ttl
        self.ttl_by_content_type = ttl_by_content_type or {}
        self.max_entries = max_entries_per_scope
        self.service = service

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _index_key(scope_key: str) -> str:
        return f"sc:{scope_key}:index"

    @staticmethod
    def _entry_key(scope_key: str, entry_id: str) -> str:
        return f"sc:{scope_key}:e:{entry_id}"

    @staticmethod
    def _doc_key(doc_id: str) -> str:
        return f"sc:doc:{doc_id}"

    # ------------------------------------------------------------------ lookup
    async def lookup(self, question: str, scope: CacheScope, vector: list[float] | None = None) -> CacheEntry | None:
        sk = scope.key()
        ids: list[str] = self.kv.get(self._index_key(sk)) or []
        if not ids:
            metrics.cache.labels(self.service, "semantic", "miss").inc()
            return None
        if vector is None and self.embed is not None:
            vector = await self.embed(question)
        now = time.time()
        best: CacheEntry | None = None
        best_sim = -1.0
        live_ids = []
        q = np.asarray(vector, dtype=np.float32) if vector is not None else None
        for eid in ids:
            raw = self.kv.get(self._entry_key(sk, eid))
            if raw is None:
                continue
            entry = CacheEntry.model_validate(raw)
            if entry.expires_at < now:
                self.kv.delete(self._entry_key(sk, eid))
                continue
            live_ids.append(eid)
            if q is not None and entry.vector is not None:
                v = np.asarray(entry.vector, dtype=np.float32)
                denom = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
                sim = float(q @ v) / denom
            else:
                sim = 1.0 if _norm(entry.question) == _norm(question) else 0.0
            if sim > best_sim:
                best, best_sim = entry, sim
        if len(live_ids) != len(ids):
            self.kv.set(self._index_key(sk), live_ids)
        if best is not None and best_sim >= self.threshold:
            best.hits += 1
            best.similarity = best_sim
            self.kv.set(self._entry_key(sk, best.id), best.model_dump(), ttl=max(1.0, best.expires_at - now))
            metrics.cache.labels(self.service, "semantic", "hit").inc()
            return best
        metrics.cache.labels(self.service, "semantic", "miss").inc()
        return None

    # ------------------------------------------------------------------ store
    async def store(self, question: str, answer: dict[str, Any], scope: CacheScope, *,
                    source_doc_ids: list[str] | None = None, content_type: str = "default",
                    vector: list[float] | None = None, contains_personal_data: bool = False,
                    is_write_action: bool = False, failed_check: bool = False) -> CacheEntry | None:
        if contains_personal_data or is_write_action or failed_check:
            metrics.cache.labels(self.service, "semantic", "skip").inc()
            return None
        if vector is None and self.embed is not None:
            vector = await self.embed(question)
        ttl = float(self.ttl_by_content_type.get(content_type, self.default_ttl))
        entry = CacheEntry(question=question, vector=vector, answer=answer, source_doc_ids=source_doc_ids or [],
                           content_type=content_type, expires_at=time.time() + ttl)
        sk = scope.key()
        self.kv.set(self._entry_key(sk, entry.id), entry.model_dump(), ttl=ttl)
        ids: list[str] = self.kv.get(self._index_key(sk)) or []
        ids.append(entry.id)
        if len(ids) > self.max_entries:  # evict oldest
            for old in ids[: len(ids) - self.max_entries]:
                self.kv.delete(self._entry_key(sk, old))
            ids = ids[-self.max_entries:]
        self.kv.set(self._index_key(sk), ids)
        for doc_id in entry.source_doc_ids:
            refs: list[str] = self.kv.get(self._doc_key(doc_id)) or []
            refs.append(f"{sk}|{entry.id}")
            self.kv.set(self._doc_key(doc_id), refs[-10000:])
        return entry

    # ------------------------------------------------------------------ invalidation
    def invalidate_document(self, doc_id: str) -> int:
        """Drop every entry built from a document that changed. Called by the ingestion pipeline."""
        refs: list[str] = self.kv.get(self._doc_key(doc_id)) or []
        for ref in refs:
            sk, eid = ref.split("|", 1)
            self.kv.delete(self._entry_key(sk, eid))
        self.kv.delete(self._doc_key(doc_id))
        return len(refs)

    def clear_scope(self, scope: CacheScope) -> None:
        sk = scope.key()
        for eid in self.kv.get(self._index_key(sk)) or []:
            self.kv.delete(self._entry_key(sk, eid))
        self.kv.delete(self._index_key(sk))


class ExactCache:
    """Exact-match cache keyed on the full normalised request (model, messages, parameters, scope)."""

    def __init__(self, kv: KVStore | None = None, ttl: float = 600, service: str = "cache"):
        self.kv = kv or MemoryKV()
        self.ttl = ttl
        self.service = service

    @staticmethod
    def key(payload: dict[str, Any], scope: CacheScope) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return f"xc:{scope.key()}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, payload: dict[str, Any], scope: CacheScope) -> Any | None:
        val = self.kv.get(self.key(payload, scope))
        metrics.cache.labels(self.service, "exact", "hit" if val is not None else "miss").inc()
        return val

    def set(self, payload: dict[str, Any], scope: CacheScope, value: Any, ttl: float | None = None) -> None:
        self.kv.set(self.key(payload, scope), value, ttl=ttl or self.ttl)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())
