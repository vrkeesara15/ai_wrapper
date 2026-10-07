"""Document registry: version, checksum and status of every item, and change detection.

The registry is what makes runs incremental ("only new, changed and deleted items since the last run"):
an item whose checksum matches the indexed version is skipped before parsing or embedding, which is where
the cost is. It also records which chunk ids each version produced, so a changed document's stale chunks
can be removed from the index precisely, and keeps run records for the API and pipeline metrics.

Stored in the shared ``DocStore`` (SQLite in the pilot, PostgreSQL in production).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.stores import DocStore, get_docstore
from aiready.core.types import Document

Status = Literal["pending", "indexed", "failed", "deleted", "quarantined", "blocked", "empty"]
Change = Literal["new", "changed", "unchanged", "deleted"]

NS_DOCS = "ingestion.registry"
NS_RUNS = "ingestion.runs"


class RegistryEntry(BaseModel):
    doc_id: str
    source: str
    version: int = 0
    checksum: str | None = None
    status: Status = "pending"
    title: str | None = None
    uri: str | None = None
    collection: str | None = None
    access_tags: list[str] = Field(default_factory=list)
    owner: str | None = None
    source_updated_at: float | None = None
    indexed_at: float | None = None
    last_seen_run: str | None = None
    last_run_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    landing_uri: str | None = None
    error: str | None = None
    lag_seconds: float | None = None  # indexed_at - source_updated_at: freshness of the index
    history: list[dict[str, Any]] = Field(default_factory=list)  # last few versions (checksum, at, run)


class RunRecord(BaseModel):
    run_id: str
    source: str
    trigger: str = "manual"  # manual / schedule / event / replay
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    status: Literal["running", "succeeded", "failed", "partial"] = "running"
    counts: dict[str, int] = Field(default_factory=dict)  # new / changed / unchanged / deleted / indexed / failed
    verification: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


class DocumentRegistry:
    def __init__(self, store: DocStore | None = None, history: int = 5):
        self._store = store
        self.history = history

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    @staticmethod
    def _key(source: str, doc_id: str) -> str:
        return f"{source}|{doc_id}"

    # ------------------------------------------------------------------ documents
    def get(self, doc_id: str, source: str | None = None) -> RegistryEntry | None:
        if source is not None:
            raw = self.store.get(NS_DOCS, self._key(source, doc_id))
            return RegistryEntry.model_validate(raw) if raw else None
        for _, raw in self.store.list(NS_DOCS):
            if raw.get("doc_id") == doc_id:
                return RegistryEntry.model_validate(raw)
        return None

    def put(self, entry: RegistryEntry) -> None:
        self.store.put(NS_DOCS, self._key(entry.source, entry.doc_id), entry.model_dump())

    def entries(self, source: str | None = None) -> list[RegistryEntry]:
        prefix = f"{source}|" if source else ""
        return [RegistryEntry.model_validate(v) for v in self.store.values(NS_DOCS, prefix)]

    def classify(self, doc: Document) -> Change:
        """Compare an incoming item with the registry: new, changed, unchanged or deleted."""
        if doc.deleted:
            return "deleted"
        entry = self.get(doc.id, doc.source)
        if entry is None or entry.status == "deleted":
            return "new"
        if entry.checksum == doc.checksum and entry.status in ("indexed", "empty", "quarantined", "blocked"):
            return "unchanged"  # blocked / quarantined versions are not retried until they change
        return "changed"  # content changed, or a failed / pending item retried on the next run

    def touch(self, doc: Document, run_id: str) -> None:
        """Record that a full-listing run still sees this item (used for delete detection)."""
        entry = self.get(doc.id, doc.source)
        if entry is not None:
            entry.last_seen_run = run_id
            self.put(entry)

    def missing(self, source: str, seen_ids: set[str]) -> list[RegistryEntry]:
        """Live items of a full-listing source that a run did not see: they were deleted at the source."""
        return [e for e in self.entries(source) if e.status != "deleted" and e.doc_id not in seen_ids]

    def next_version(self, doc: Document) -> int:
        entry = self.get(doc.id, doc.source)
        if entry is None:
            return 1
        return entry.version if entry.checksum == doc.checksum and entry.status == "indexed" else entry.version + 1

    def mark_indexed(self, doc: Document, *, run_id: str | None, chunk_ids: list[str], collection: str,
                     landing_uri: str | None = None, status: Status = "indexed",
                     version: int | None = None) -> RegistryEntry:
        entry = self.get(doc.id, doc.source) or RegistryEntry(doc_id=doc.id, source=doc.source)
        now = time.time()
        if version is not None:
            entry.version = version
        elif entry.checksum != doc.checksum or entry.status != status:
            entry.version += 1
        entry.history = (entry.history + [{"version": entry.version, "checksum": doc.checksum, "at": now,
                                           "run_id": run_id}])[-self.history:]
        entry.checksum, entry.status, entry.error = doc.checksum, status, None
        entry.title, entry.uri, entry.collection = doc.title, doc.uri, collection
        entry.access_tags, entry.owner = list(doc.access_tags), doc.owner
        entry.source_updated_at, entry.indexed_at = doc.updated_at, now
        entry.lag_seconds = max(0.0, now - doc.updated_at) if doc.updated_at else None
        entry.last_run_id = entry.last_seen_run = run_id
        entry.chunk_ids = list(chunk_ids)
        entry.landing_uri = landing_uri or entry.landing_uri
        self.put(entry)
        return entry

    def mark(self, doc_id: str, source: str, status: Status, *, error: str | None = None,
             run_id: str | None = None, checksum: str | None = None) -> RegistryEntry:
        entry = self.get(doc_id, source) or RegistryEntry(doc_id=doc_id, source=source)
        entry.status, entry.error = status, error
        if run_id:
            entry.last_run_id = run_id
        if checksum and status in ("quarantined", "blocked"):
            entry.checksum = checksum  # a blocked version is not retried until it changes
        if status == "deleted":
            entry.chunk_ids = []
            entry.indexed_at = time.time()
        self.put(entry)
        return entry

    # ------------------------------------------------------------------ runs
    def save_run(self, run: RunRecord) -> None:
        self.store.put(NS_RUNS, run.run_id, run.model_dump())

    def get_run(self, run_id: str) -> RunRecord | None:
        raw = self.store.get(NS_RUNS, run_id)
        return RunRecord.model_validate(raw) if raw else None

    def runs(self, source: str | None = None, limit: int = 50) -> list[RunRecord]:
        out = [RunRecord.model_validate(v) for v in self.store.values(NS_RUNS)]
        if source:
            out = [r for r in out if r.source == source]
        return sorted(out, key=lambda r: -r.started_at)[:limit]
