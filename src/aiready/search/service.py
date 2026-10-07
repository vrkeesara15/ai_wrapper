"""Vector + keyword search service (implements ``core.interfaces.SearchService``).

Search request = query vector + query text + the user's access groups + filters.

    policy (data.read on the collection) -> access + metadata filter -> dense kNN  \
                                                                      -> BM25       -> rank fusion -> top 50

* **Filter by access and metadata first, then score.** The filter is pushed into each index, so a user's
  top-k is computed only over chunks they may see. A chunk is visible only if its ``access_tags`` intersect
  the principal's ``access_groups``; untagged chunks are visible to nobody. The groups always come from the
  verified principal - a request may narrow them, never widen them. Returned hits are re-checked before
  leaving the service (defence in depth against a misbehaving custom index plugin).
* **One collection per application and tenant**: ``collection_name(app, tenant, base)``. Users cannot search
  another tenant's collection.
* **One model version per index**: the first upsert records the vector dimension and the embedding
  fingerprint (``chunk.metadata["embedding_fingerprint"]``); vectors from any other model are refused.
* **Write path**: upserts and deletes from ingestion are applied to both indexes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, TypeVar
from collections.abc import Callable

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import NotFound, PermissionDenied, PolicyViolation, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyInput
from aiready.core.plugins import create
from aiready.core.stores import get_docstore
from aiready.core.telemetry import metrics, span
from aiready.core.types import Chunk, DataClass, SearchHit, SearchRequest
from aiready.core.wiring import connect
from aiready.search.config import SearchConfig
from aiready.search.filters import SearchFilter, access_allowed
from aiready.search.fusion import fuse
from aiready.search.keyword import KeywordIndex
from aiready.search.stores import VectorStore

log = logging.getLogger(__name__)
SERVICE = "search"
META_NS = "search.collections"
SNAP_NS = "search.snapshots"
T = TypeVar("T")
_SAFE = re.compile(r"[^a-z0-9_\-]")


def _safe(part: str) -> str:
    return _SAFE.sub("-", (part or "default").lower()).replace("__", "_-")


def collection_name(app: str, tenant: str, base: str = "knowledge") -> str:
    """``<base>__<app>__<tenant>`` - one collection per application and tenant."""
    return f"{_safe(base)}__{_safe(app)}__{_safe(tenant)}"


def parse_collection(name: str) -> tuple[str, str, str] | None:
    """(base, app, tenant) of a name built by ``collection_name``; None for custom names."""
    parts = name.split("__")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def service_principal() -> Principal:
    return Principal(sub="svc:search", roles=["service"], service="search")


class SearchService:
    def __init__(
        self,
        config: SearchConfig | None = None,
        *,
        vector_store: VectorStore | None = None,
        keyword_index: KeywordIndex | None = None,
    ):
        self.config = config or get_config().section("search", SearchConfig)
        c = self.config
        self.vectors: VectorStore | None = vector_store or (
            create("vector_store", c.vector_store) if c.dense_enabled else None
        )
        self.keywords: KeywordIndex | None = keyword_index or (
            create("keyword_index", c.keyword_index) if c.keyword_enabled else None
        )
        if self.vectors is None and self.keywords is None:
            raise ValidationFailed("search needs at least one of dense_enabled / keyword_enabled")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def collection_name(app: str, tenant: str, base: str = "knowledge") -> str:
        return collection_name(app, tenant, base)

    @staticmethod
    async def _call(index: Any, fn: Callable[..., T], *args: Any, **kw: Any) -> T:
        """Run blocking backends (network clients) in a worker thread so the event loop stays responsive."""
        if getattr(index, "blocking", True):
            return await asyncio.to_thread(fn, *args, **kw)
        return fn(*args, **kw)

    def meta(self, collection: str) -> dict[str, Any]:
        return get_docstore().get(META_NS, collection) or {}

    def _check_tenant(self, collection: str, principal: Principal) -> None:
        if not self.config.enforce_tenant_collections or principal.is_service_only:
            return
        parsed = parse_collection(collection)
        if parsed and parsed[2] != _safe(principal.tenant):
            get_audit().for_principal(
                principal,
                "search.denied",
                action="data.read",
                resource=collection,
                decision="deny",
                reasons=["collection belongs to another tenant"],
            )
            raise PermissionDenied(f"collection '{collection}' belongs to another tenant")

    async def _policy(self, collection: str, principal: Principal, req: SearchRequest) -> dict[str, Any]:
        if not self.config.policy_check:
            return {}
        parsed = parse_collection(collection)
        decision = await connect("policy").decide(
            PolicyInput(
                action="data.read",
                principal=principal.model_dump(),
                resource={
                    "type": "collection",
                    "collection": collection,
                    "app": parsed[1] if parsed else None,
                    "tenant": parsed[2] if parsed else None,
                },
                context={"app": principal.app, "filters": req.filters, "top_k": req.top_k},
            )
        )
        metrics.policy.labels(SERVICE, "data.read", decision.effect).inc()
        if not decision.allow:
            get_audit().for_principal(
                principal,
                "search.denied",
                action="data.read",
                resource=collection,
                decision=decision.effect,
                reasons=decision.reasons,
                policy_version=decision.policy_version,
            )
            raise PolicyViolation(
                f"search on '{collection}' denied by policy", details={"reasons": decision.reasons}
            )
        return decision.obligations

    @staticmethod
    def _apply_obligations(
        metadata: dict[str, Any], top_k: int, ob: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Policy obligations narrow the search: ``max_data_class``, ``max_results``, extra ``filters``."""
        metadata = dict(metadata)
        if ob.get("max_data_class"):
            ceiling = DataClass(ob["max_data_class"])
            allowed = [d.value for d in DataClass if d.rank <= ceiling.rank]
            prev = metadata.get("data_class")
            if prev is not None:  # keep the caller's own restriction too: intersect, never widen
                if isinstance(prev, dict):
                    wanted = prev.get("in") or ([prev["eq"]] if "eq" in prev else allowed)
                else:
                    wanted = prev if isinstance(prev, (list, tuple, set)) else [prev]
                wanted_vals = {str(getattr(w, "value", w)) for w in wanted}
                allowed = [a for a in allowed if a in wanted_vals]
            metadata["data_class"] = {"in": allowed}
        for k, v in (ob.get("filters") or {}).items():
            metadata[k] = v  # policy-imposed filters override caller filters on the same key
        if ob.get("max_results"):
            top_k = min(top_k, int(ob["max_results"]))
        return metadata, top_k

    # ------------------------------------------------------------------ read path
    async def search(self, req: SearchRequest, principal: Principal) -> list[SearchHit]:
        cfg = self.config
        collection = req.collection or collection_name(principal.app, principal.tenant, cfg.base_collection)
        query = (req.query_text or "")[: cfg.max_query_chars]
        top_k = max(1, min(req.top_k or cfg.top_k, cfg.max_top_k))
        with span("search.query", collection=collection, top_k=top_k) as sp:
            self._check_tenant(collection, principal)
            obligations = await self._policy(collection, principal, req)
            metadata, top_k = self._apply_obligations(req.filters, top_k, obligations)
            # access groups come from the verified identity; a request may only narrow them
            groups = set(principal.access_groups)
            if req.access_groups:
                groups &= set(req.access_groups)
            flt = SearchFilter(access_groups=sorted(groups), metadata=metadata)
            n = max(cfg.candidates_per_index, top_k)

            dense: list[tuple[Chunk, float]] = []
            keyword: list[tuple[Chunk, float]] = []
            if self.vectors is not None:
                vector = req.query_vector
                if vector is None and query.strip():
                    with metrics.time_stage(SERVICE, "embed_query"):
                        vector = (
                            await connect("gateway").embed(
                                [query], principal, model=cfg.embedding_model, priority="live"
                            )
                        )[0]
                if vector is not None:
                    with metrics.time_stage(SERVICE, "dense"):
                        dense = await self._call(
                            self.vectors, self.vectors.search, collection, vector, flt, n
                        )
                    if cfg.min_dense_score is not None:
                        dense = [(c, s) for c, s in dense if s >= cfg.min_dense_score]
            if self.keywords is not None and query.strip():
                with metrics.time_stage(SERVICE, "keyword"):
                    keyword = await self._call(self.keywords, self.keywords.search, collection, query, flt, n)

            with metrics.time_stage(SERVICE, "fuse"):
                hits = self._fuse(dense, keyword, top_k)
            leaked = [h for h in hits if not access_allowed(h.chunk.access_tags, flt.access_groups)]
            if leaked:  # an index plugin ignored the filter: never return those chunks
                log.error("index returned %d chunks outside the caller's access groups; dropped", len(leaked))
                metrics.fallbacks.labels(SERVICE, "access_filter_violation").inc()
                hits = [h for h in hits if access_allowed(h.chunk.access_tags, flt.access_groups)]
            sp.set(dense=len(dense), keyword=len(keyword), returned=len(hits))

        get_audit().for_principal(
            principal,
            "search.query",
            action="data.read",
            resource=collection,
            decision="allow",
            query=query[:500],
            top_k=top_k,
            filters=metadata,
            dense_candidates=len(dense),
            keyword_candidates=len(keyword),
            chunk_ids=[h.chunk.id for h in hits],
            doc_ids=sorted({h.chunk.doc_id for h in hits}),
        )
        return hits

    def _fuse(
        self, dense: list[tuple[Chunk, float]], keyword: list[tuple[Chunk, float]], top_k: int
    ) -> list[SearchHit]:
        chunks: dict[str, Chunk] = {}
        d_scores: dict[str, float] = {}
        k_scores: dict[str, float] = {}
        for c, s in dense:
            chunks.setdefault(c.id, c)
            d_scores[c.id] = s
        for c, s in keyword:
            chunks.setdefault(c.id, c)
            k_scores[c.id] = s
        lists = []
        weights = []
        fw = self.config.fusion.weights + [1.0] * (2 - len(self.config.fusion.weights))
        if dense:
            lists.append([(c.id, s) for c, s in dense])
            weights.append(fw[0])
        if keyword:
            lists.append([(c.id, s) for c, s in keyword])
            weights.append(fw[1])
        fused = fuse(lists, method=self.config.fusion.method, k=self.config.fusion.k, weights=weights)
        out = []
        for cid, score in fused[:top_k]:
            chunk = chunks[cid]
            if not self.config.return_vectors:
                chunk = chunk.model_copy(update={"vector": None})
            out.append(
                SearchHit(
                    chunk=chunk, score=score, dense_score=d_scores.get(cid), keyword_score=k_scores.get(cid)
                )
            )
        return out

    # ------------------------------------------------------------------ write path
    async def upsert(
        self, collection: str, chunks: list[Chunk], *, principal: Principal | None = None
    ) -> int:
        if not chunks:
            return 0
        with span("search.upsert", collection=collection, count=len(chunks)):
            untagged = [c.id for c in chunks if not c.access_tags]
            if untagged:
                log.warning(
                    "%d chunks in %s have no access_tags and will be invisible to every user",
                    len(untagged),
                    collection,
                )
            if self.vectors is not None:
                missing = [c for c in chunks if c.vector is None]
                if missing:
                    if not self.config.embed_missing_vectors:
                        raise ValidationFailed(f"{len(missing)} chunks have no vector")
                    with metrics.time_stage(SERVICE, "embed_chunks"):
                        vecs = await connect("gateway").embed(
                            [c.text for c in missing],
                            principal or service_principal(),
                            model=self.config.embedding_model,
                            priority="bulk",
                        )
                    by_id = {c.id: v for c, v in zip(missing, vecs)}
                    chunks = [
                        c.model_copy(update={"vector": by_id[c.id]}) if c.id in by_id else c for c in chunks
                    ]
                self._check_model(collection, chunks)
                with metrics.time_stage(SERVICE, "upsert_dense"):
                    await self._call(self.vectors, self.vectors.upsert, collection, chunks)
            if self.keywords is not None:
                with metrics.time_stage(SERVICE, "upsert_keyword"):
                    await self._call(self.keywords, self.keywords.upsert, collection, chunks)
        self._audit_write(
            principal,
            "search.upsert",
            collection,
            count=len(chunks),
            doc_ids=sorted({c.doc_id for c in chunks})[:1000],
            untagged=len(untagged),
        )
        return len(chunks)

    def _check_model(self, collection: str, chunks: list[Chunk]) -> None:
        """One model version per index: same dimension and embedding fingerprint as the collection."""
        meta = self.meta(collection)
        dims = {len(c.vector or []) for c in chunks}
        fps = {c.metadata.get("embedding_fingerprint") for c in chunks} - {None}
        if len(dims) > 1 or len(fps) > 1:
            raise ValidationFailed(
                "one upsert mixes vector dimensions or embedding models",
                details={"dims": sorted(dims), "fingerprints": sorted(fps)},
            )
        dim = dims.pop()
        fp = fps.pop() if fps else None
        if meta.get("dim") and meta["dim"] != dim:
            raise ValidationFailed(
                f"collection '{collection}' holds {meta['dim']}-dim vectors, got {dim}: "
                "changing the embedding model needs a full re-embed into a new collection"
            )
        if fp and meta.get("embedding_fingerprint") and meta["embedding_fingerprint"] != fp:
            raise ValidationFailed(
                f"collection '{collection}' was embedded with {meta['embedding_fingerprint']}, "
                f"got {fp}: re-embed the whole collection with one model version",
                details={"collection_model": meta["embedding_fingerprint"], "chunk_model": fp},
            )
        if not meta or (fp and not meta.get("embedding_fingerprint")):
            parsed = parse_collection(collection)
            get_docstore().put(
                META_NS,
                collection,
                {
                    "collection": collection,
                    "dim": dim,
                    "embedding_fingerprint": fp or meta.get("embedding_fingerprint"),
                    "created_at": meta.get("created_at", time.time()),
                    "app": parsed[1] if parsed else None,
                    "tenant": parsed[2] if parsed else None,
                },
            )

    async def delete(
        self,
        collection: str,
        *,
        doc_ids: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        principal: Principal | None = None,
    ) -> int:
        if not (doc_ids or chunk_ids):
            return 0
        removed = 0
        with span("search.delete", collection=collection):
            if self.vectors is not None:
                removed = await self._call(
                    self.vectors, self.vectors.delete, collection, doc_ids=doc_ids, chunk_ids=chunk_ids
                )
            if self.keywords is not None:
                kw = await self._call(
                    self.keywords, self.keywords.delete, collection, doc_ids=doc_ids, chunk_ids=chunk_ids
                )
                removed = max(removed, kw)
        self._audit_write(
            principal,
            "search.delete",
            collection,
            doc_ids=doc_ids or [],
            chunk_ids=chunk_ids or [],
            removed=removed,
        )
        return removed

    @staticmethod
    def _audit_write(principal: Principal | None, event: str, collection: str, **details: Any) -> None:
        action = "data.write" if event.endswith("upsert") else "data.delete"
        if principal is not None:
            get_audit().for_principal(
                principal, event, action=action, resource=collection, decision="allow", **details
            )
        else:
            get_audit().record(
                event, actor="svc:search", action=action, resource=collection, decision="allow", **details
            )

    # ------------------------------------------------------------------ operations
    async def stats(self, collection: str) -> dict[str, Any]:
        meta = self.meta(collection)
        out: dict[str, Any] = {
            "collection": collection,
            "dim": meta.get("dim"),
            "embedding_fingerprint": meta.get("embedding_fingerprint"),
            "vector_store": self.config.vector_store.get("type") or self.config.vector_store.get("class"),
            "keyword_index": self.config.keyword_index.get("type") or self.config.keyword_index.get("class"),
        }
        if self.vectors is not None:
            out["vectors"] = await self._call(self.vectors, self.vectors.count, collection)
        if self.keywords is not None:
            out["keyword_docs"] = await self._call(self.keywords, self.keywords.count, collection)
        return out

    async def snapshot(self, collection: str, *, principal: Principal | None = None) -> dict[str, Any]:
        """Snapshot both indexes; the record is kept in the doc store so ``restore`` can find it by id."""
        snap_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        directory = f"{self.config.snapshot_dir.rstrip('/')}/{collection}/{snap_id}"
        rec: dict[str, Any] = {
            "id": snap_id,
            "collection": collection,
            "created_at": time.time(),
            "meta": self.meta(collection),
        }
        with span("search.snapshot", collection=collection):
            if self.vectors is not None:
                rec["vector"] = await self._call(self.vectors, self.vectors.snapshot, collection, directory)
            if self.keywords is not None:
                rec["keyword"] = await self._call(
                    self.keywords, self.keywords.snapshot, collection, directory
                )
        rec["counts"] = await self.stats(collection)
        get_docstore().put(SNAP_NS, f"{collection}/{snap_id}", rec)
        self._audit_write(principal, "search.snapshot", collection, snapshot_id=snap_id)
        return rec

    async def list_snapshots(self, collection: str) -> list[dict[str, Any]]:
        return get_docstore().values(SNAP_NS, prefix=f"{collection}/")

    async def restore(
        self, collection: str, snapshot_id: str, *, principal: Principal | None = None
    ) -> dict[str, Any]:
        rec = get_docstore().get(SNAP_NS, f"{collection}/{snapshot_id}")
        if rec is None:
            raise NotFound(f"snapshot '{snapshot_id}' of '{collection}' not found")
        with span("search.restore", collection=collection, snapshot=snapshot_id):
            if self.vectors is not None and rec.get("vector"):
                await self._call(self.vectors, self.vectors.restore, collection, rec["vector"])
            if self.keywords is not None and rec.get("keyword"):
                await self._call(self.keywords, self.keywords.restore, collection, rec["keyword"])
        if rec.get("meta"):
            get_docstore().put(META_NS, collection, rec["meta"])
        self._audit_write(principal, "search.restore", collection, snapshot_id=snapshot_id)
        return await self.stats(collection)

    async def ready(self) -> dict[str, Any]:
        return {
            "vector_store": type(self.vectors).__name__ if self.vectors else None,
            "keyword_index": type(self.keywords).__name__ if self.keywords else None,
        }
