"""Lineage store: every chunk traced back to its source record.

For each chunk id we keep the document id and version, checksum, source uri, the run that produced it, the
parser and chunker used and the embedding model fingerprint. This answers the questions auditors and
incident responders ask ("where did this sentence in the answer come from, and when was it indexed?") and
tells operators which chunks must be rebuilt when a parser or embedding model changes.

Chunk ids start with a hash of the document id (see ``enrich.chunk_id``), so all chunks of a document are a
prefix scan away.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.stores import DocStore, get_docstore
from aiready.core.types import Chunk

NS_LINEAGE = "ingestion.lineage"


class LineageRecord(BaseModel):
    chunk_id: str
    doc_id: str
    doc_version: int
    checksum: str | None = None
    source: str
    source_uri: str | None = None
    collection: str
    run_id: str | None = None
    parser: str
    chunker: str
    embedding_model: str
    embedding_fingerprint: str
    landing_uri: str | None = None
    position: int = 0
    section: str | None = None
    page: int | None = None
    content_hash: str | None = None
    created_at: float = Field(default_factory=time.time)


class LineageStore:
    def __init__(self, store: DocStore | None = None):
        self._store = store

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def record(self, chunks: list[Chunk], *, doc_version: int, checksum: str | None, collection: str,
               run_id: str | None, parser: str, chunker: str, embedding_model: str, embedding_fingerprint: str,
               landing_uri: str | None) -> list[LineageRecord]:
        out = []
        for c in chunks:
            rec = LineageRecord(
                chunk_id=c.id, doc_id=c.doc_id, doc_version=doc_version, checksum=checksum, source=c.source,
                source_uri=c.source_uri, collection=collection, run_id=run_id, parser=parser, chunker=chunker,
                embedding_model=embedding_model, embedding_fingerprint=embedding_fingerprint,
                landing_uri=landing_uri, position=c.position, section=c.section, page=c.metadata.get("page"),
                content_hash=c.metadata.get("content_hash"),
            )
            self.store.put(NS_LINEAGE, c.id, rec.model_dump())
            out.append(rec)
        return out

    def get(self, chunk_id: str) -> LineageRecord | None:
        raw = self.store.get(NS_LINEAGE, chunk_id)
        return LineageRecord.model_validate(raw) if raw else None

    def for_document(self, doc_id: str) -> list[LineageRecord]:
        from aiready.ingestion.enrich import doc_prefix

        return [LineageRecord.model_validate(v) for v in self.store.values(NS_LINEAGE, doc_prefix(doc_id))]

    def remove(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self.store.delete(NS_LINEAGE, cid)

    def remove_document(self, doc_id: str) -> int:
        recs = self.for_document(doc_id)
        self.remove([r.chunk_id for r in recs])
        return len(recs)

    def by_fingerprint(self, fingerprint: str) -> list[dict[str, Any]]:
        """Chunks embedded with a given model fingerprint (what a model change must re-embed)."""
        return [v for v in self.store.values(NS_LINEAGE) if v.get("embedding_fingerprint") == fingerprint]
