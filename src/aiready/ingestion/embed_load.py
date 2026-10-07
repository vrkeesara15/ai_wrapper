"""Load: batched embedding through the gateway and the index writer.

* Embeddings are requested through ``connect("gateway").embed(..., priority="bulk")``: the gateway applies
  identity, limits and tracing, and the embedding service keeps bulk traffic on its own queue so a large
  re-index never slows down live questions. The pipeline never calls a model server directly.
* The index writer upserts vectors + text (the search service maintains both the vector and the keyword
  index), removes the stale chunks of a changed document after the new version is in (so there is no window
  where the document cannot be found), removes deleted documents, and invalidates semantic-cache entries
  built from a document that changed - otherwise the cache would keep serving the old answer.
"""

from __future__ import annotations

import logging
from typing import Any

from aiready.core.identity import Principal
from aiready.core.telemetry import metrics, span
from aiready.core.types import Chunk
from aiready.ingestion.enrich import embedding_text

log = logging.getLogger(__name__)


def service_principal() -> Principal:
    """The ingestion pipeline's own identity. It carries no user token and no data rights: it only proves
    *which service* calls the gateway/search (the service token is added by the HTTP client)."""
    return Principal(sub="svc:ingestion", roles=["service"], service="ingestion", app="ingestion")


class IndexWriter:
    def __init__(self, *, search: Any = None, gateway: Any = None, cache: Any = None, model: str = "embedding",
                 batch_size: int = 64, priority: str = "bulk", fingerprint: str | None = None,
                 include_title: bool = True, principal: Principal | None = None):
        self._search, self._gateway = search, gateway
        self.cache = cache
        self.model, self.batch_size, self.priority = model, max(1, batch_size), priority
        self._fingerprint = fingerprint
        self.include_title = include_title
        self.principal = principal or service_principal()
        self._dim: int | None = None

    @property
    def search(self) -> Any:
        if self._search is None:
            from aiready.core.wiring import connect

            self._search = connect("search")
        return self._search

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            from aiready.core.wiring import connect

            self._gateway = connect("gateway")
        return self._gateway

    @property
    def fingerprint(self) -> str:
        """Embedding model version stamped on every chunk: vectors from different models are not comparable."""
        if self._fingerprint:
            return self._fingerprint
        return f"{self.model}:dim{self._dim}" if self._dim else self.model

    async def embed(self, chunks: list[Chunk], *, source: str = "") -> list[Chunk]:
        with span("ingestion.embed", chunks=len(chunks), model=self.model, priority=self.priority), \
                metrics.time_stage("ingestion", "embed"):
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                vectors = await self.gateway.embed([embedding_text(c, self.include_title) for c in batch],
                                                   self.principal, model=self.model, priority=self.priority)
                if len(vectors) != len(batch):
                    raise RuntimeError(f"embedding returned {len(vectors)} vectors for {len(batch)} texts")
                for c, v in zip(batch, vectors):
                    c.vector = list(v)
                    self._dim = len(v)
                metrics.pipeline.labels(source, "embed", "ok").inc(len(batch))
        for c in chunks:
            c.metadata["embedding_model"] = self.model
            c.metadata["embedding_fingerprint"] = self.fingerprint
        return chunks

    async def write(self, collection: str, chunks: list[Chunk], *, stale_chunk_ids: list[str] | None = None,
                    doc_id: str | None = None, changed: bool = False, source: str = "") -> int:
        """Upsert the new chunks first, then delete the previous version's chunks that no longer exist."""
        with span("ingestion.index", collection=collection, chunks=len(chunks)), \
                metrics.time_stage("ingestion", "index"):
            n = await self.search.upsert(collection, chunks) if chunks else 0
            keep = {c.id for c in chunks}
            stale = [cid for cid in (stale_chunk_ids or []) if cid not in keep]
            if stale:
                await self.search.delete(collection, chunk_ids=stale)
            metrics.pipeline.labels(source, "index", "ok").inc(len(chunks))
        if changed and doc_id:
            self.invalidate(doc_id)
        return int(n or 0)

    async def remove(self, collection: str, doc_id: str, *, source: str = "") -> int:
        with span("ingestion.index_delete", collection=collection, doc_id=doc_id):
            n = await self.search.delete(collection, doc_ids=[doc_id])
            metrics.pipeline.labels(source, "index", "deleted").inc()
        self.invalidate(doc_id)
        return int(n or 0)

    def invalidate(self, doc_id: str) -> int:
        if self.cache is None:
            return 0
        try:
            return int(self.cache.invalidate_document(doc_id) or 0)
        except Exception:  # a cache outage must not fail indexing; entries also expire by TTL
            log.warning("semantic cache invalidation failed for %s", doc_id, exc_info=True)
            return 0
