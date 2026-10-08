"""The ingestion pipeline: Extract -> Transform -> Load.

    Extract    1 connect (connector worker)   2 detect changes (registry)   3 parse (worker)
    Transform  4 clean and classify           5 chunk                       6 enrich
    Load       7 embed (gateway, bulk)        8 index (search upsert/delete) 9 verify

Extraction runs per source (scheduled or on demand) and puts **one message per new / changed / deleted
document** on the work queue; the worker pool (``worker.py``) runs steps 3-8 per message with retries and a
dead-letter queue; verification runs after each load. Change capture and the file crawler can put messages
on the same queue directly (event-driven runs) - the handler re-checks the registry so duplicates are cheap.

Governance at every decision point: a policy decision before a source is read (``data.ingest``) and before
each document is indexed (``data.index``, with its sensitivity), an audit record for every run, index,
deletion, quarantine and dead letter, a trace span per stage and pipeline metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PolicyViolation, ValidationFailed
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.plugins import create
from aiready.core.queues import make_queue
from aiready.core.telemetry import metrics, span
from aiready.core.types import DataClass, Document
from aiready.ingestion.chunking import Chunker
from aiready.ingestion.cleaning import Cleaner
from aiready.ingestion.config import IngestionConfig, SourceConfig, load_config
from aiready.ingestion.documents import checksum_of, decode_document, work_message
from aiready.ingestion.embed_load import IndexWriter, service_principal
from aiready.ingestion.enrich import enrich
from aiready.ingestion.landing import LandingStore, make_landing
from aiready.ingestion.lineage import LineageStore
from aiready.ingestion.parsers import ParserSelector
from aiready.ingestion.registry import DocumentRegistry, RunRecord
from aiready.ingestion.verify import Alerter, Verifier

log = logging.getLogger(__name__)
SERVICE = "ingestion"


class IngestionPipeline:
    """Builds every stage from configuration; any dependency can be injected (tests, embedding in a process)."""

    def __init__(self, cfg: IngestionConfig | None = None, *, queue: Any = None,
                 registry: DocumentRegistry | None = None, landing: LandingStore | None = None,
                 lineage: LineageStore | None = None, search: Any = None, gateway: Any = None,
                 guardrails: Any = None, policy: Any = None, cache: Any = None, alerter: Alerter | None = None):
        self.cfg = cfg or load_config()
        self.queue = queue or make_queue(self.cfg.queue.spec)
        self.topic = self.cfg.queue.topic
        self.registry = registry or DocumentRegistry()
        self.landing = landing or make_landing(self.cfg.landing.model_dump())
        self.lineage = lineage or LineageStore()
        self.principal = service_principal()
        self._policy = policy
        e = self.cfg.embedding
        self.writer = IndexWriter(search=search, gateway=gateway, cache=cache, model=e.model, batch_size=e.batch_size,
                                  priority=e.priority, fingerprint=e.fingerprint, include_title=e.include_title,
                                  principal=self.principal)
        self.cleaner = Cleaner(self.cfg.cleaning, guardrails=guardrails)
        self.alerter = alerter or Alerter(self.cfg.alerts)
        self._selectors: dict[str, ParserSelector] = {}
        self._chunkers: dict[str, tuple[str, Chunker]] = {}
        self._run_lock = asyncio.Lock()
        self._events: list[tuple[float, str, str]] = []  # (ts, source, outcome) for items-per-hour metrics

    # ------------------------------------------------------------------ dependencies
    @property
    def policy(self) -> Any:
        if self._policy is None:
            from aiready.core.wiring import connect

            self._policy = connect("policy")
        return self._policy

    def source(self, name: str) -> SourceConfig:
        src = self.cfg.source(name)
        if src is None:
            raise NotFound(f"unknown ingestion source '{name}'")
        return src

    def _source_or_default(self, name: str) -> SourceConfig:
        """Event-driven producers may use a source name with no connector entry (e.g. change capture)."""
        return self.cfg.source(name) or SourceConfig(name=name, connector={"type": "external"},
                                                     collection=self.cfg.default_collection)

    def _selector(self, src: SourceConfig) -> ParserSelector:
        if src.name not in self._selectors:
            self._selectors[src.name] = ParserSelector(src.parsers)
        return self._selectors[src.name]

    def _chunker(self, src: SourceConfig, mime: str) -> tuple[str, Chunker]:
        spec = src.chunker or self.cfg.chunking.by_mime.get(mime.split(";")[0]) or self.cfg.chunking.default
        key = repr(sorted(spec.items()))
        if key not in self._chunkers:
            ch = create("chunker", spec)
            self._chunkers[key] = (getattr(ch, "name", spec.get("type", "custom")), ch)
        return self._chunkers[key]

    async def _decide(self, action: str, resource: dict[str, Any], context: dict[str, Any]) -> PolicyDecision:
        decision: PolicyDecision = await self.policy.decide(PolicyInput(
            action=action, principal=self.principal.model_dump(), resource=resource, context=context))
        metrics.policy.labels(SERVICE, action, decision.effect if not decision.allow else "allow").inc()
        return decision

    def _audit(self, event: str, **kw: Any) -> None:
        get_audit().for_principal(self.principal, event, **kw)

    async def _bump(self, run_id: str | None, key: str) -> None:
        self._events.append((time.time(), run_id or "-", key))
        if len(self._events) > 100_000:
            self._events = self._events[-50_000:]
        if not run_id:
            return
        async with self._run_lock:
            run = self.registry.get_run(run_id)
            if run is not None:
                run.bump(key)
                self.registry.save_run(run)

    # ------------------------------------------------------------------ extract
    async def extract(self, name: str, *, trigger: str = "manual", run_id: str | None = None) -> RunRecord:
        """Steps 1-2: read the source and enqueue only new, changed and deleted items."""
        src = self.source(name)
        if not src.enabled:
            raise ValidationFailed(f"ingestion source '{name}' is disabled")
        run = RunRecord(run_id=run_id or uuid.uuid4().hex, source=name, trigger=trigger)
        self.registry.save_run(run)
        with span("ingestion.extract", source=name, run_id=run.run_id, trigger=trigger) as sp, \
                metrics.time_stage(SERVICE, "extract"):
            decision = await self._decide("data.ingest", {"source": name, "collection": src.collection,
                                                          "connector": src.connector.get("type") or src.connector.get("class")},
                                          {"data_class": src.data_class.value, "trigger": trigger})
            if not decision.allow:
                run.status, run.finished_at = "failed", time.time()
                run.error = "policy denied: " + "; ".join(decision.reasons)
                self.registry.save_run(run)
                self._audit("ingestion.run", action="data.ingest", resource=name, decision="deny",
                            run_id=run.run_id, reasons=decision.reasons)
                raise PolicyViolation(run.error)
            try:
                connector = create("connector", src.connector, source=name)
                seen: set[str] = set()
                async for doc in connector.documents():
                    if doc.checksum is None:
                        doc.checksum = checksum_of(doc.content)
                    seen.add(doc.id)
                    change = self.registry.classify(doc)
                    run.bump(change)
                    metrics.pipeline.labels(name, "detect", change).inc()
                    if change == "unchanged":
                        self.registry.touch(doc, run.run_id)
                        continue
                    payload = work_message(name, doc, run_id=run.run_id, origin="connector")
                    payload["change"] = change
                    if src.landing:
                        payload["landing_uri"] = self.landing.put(doc)
                    await self.queue.put(self.topic, payload)
                    run.bump("enqueued")
                if getattr(connector, "full_listing", False) and src.detect_deletes:
                    await self._enqueue_deletions(src, run, seen)
            except Exception as exc:
                run.status, run.finished_at, run.error = "failed", time.time(), str(exc)
                self.registry.save_run(run)
                metrics.pipeline.labels(name, "extract", "error").inc()
                self._audit("ingestion.run", action="extract", resource=name, decision="error", run_id=run.run_id,
                            error=str(exc))
                await self.alerter.alert("extract_failed", source=name, run_id=run.run_id, error=str(exc))
                raise
            sp.set(**run.counts)
        self.registry.save_run(run)
        self._audit("ingestion.run", action="extract", resource=name, decision="ok", run_id=run.run_id,
                    trigger=trigger, counts=run.counts)
        return run

    async def _enqueue_deletions(self, src: SourceConfig, run: RunRecord, seen: set[str]) -> None:
        missing = self.registry.missing(src.name, seen)
        live = [e for e in self.registry.entries(src.name) if e.status != "deleted"]
        if missing and live and len(missing) / len(live) > src.max_delete_fraction and len(live) > 2:
            # An empty or truncated listing is far more often an outage than a mass deletion.
            run.bump("deletes_refused", len(missing))
            await self.alerter.alert("mass_delete_refused", source=src.name, run_id=run.run_id,
                                     missing=len(missing), live=len(live))
            return
        for entry in missing:
            tomb = Document(id=entry.doc_id, source=src.name, uri=entry.uri, deleted=True)
            payload = work_message(src.name, tomb, run_id=run.run_id, origin="connector")
            payload["change"] = "deleted"
            await self.queue.put(self.topic, payload)
            run.bump("deleted")
            run.bump("enqueued")

    # ------------------------------------------------------------------ transform + load (one message)
    async def handle(self, payload: dict[str, Any]) -> str:
        """Steps 3-8 for one work message. Raises on failure so the worker can retry / dead-letter.

        Outcomes (counted on the run next to the detection counts new / changed / unchanged / deleted):
        ``indexed``, ``removed``, ``duplicate`` (event for an unchanged item), ``blocked``, ``quarantined``,
        ``empty`` (``dead_lettered`` is counted by the worker)."""
        doc = decode_document(payload["document"])
        source = payload.get("source") or doc.source
        run_id = payload.get("run_id")
        src = self._source_or_default(source)
        with span("ingestion.document", source=source, doc_id=doc.id, run_id=run_id) as sp:
            outcome = await self._handle(doc, src, run_id, payload)
            sp.set(outcome=outcome)
        metrics.pipeline.labels(source, "document", outcome).inc()
        await self._bump(run_id, outcome)
        return outcome

    async def _handle(self, doc: Document, src: SourceConfig, run_id: str | None, payload: dict[str, Any]) -> str:
        if doc.deleted:
            return await self._delete(doc, src, run_id)
        if doc.checksum is None:
            doc.checksum = checksum_of(doc.content)
        if self.registry.classify(doc) == "unchanged" and not payload.get("force"):
            return "duplicate"  # duplicate event (change capture + scheduled run): nothing to do
        previous = self.registry.get(doc.id, doc.source)
        collection = src.collection
        landing_uri = payload.get("landing_uri")
        if landing_uri is None and src.landing:
            landing_uri = self.landing.put(doc)

        # 3 parse
        filename = str(doc.metadata.get("path") or PurePosixPath(doc.uri or doc.id).name)
        with metrics.time_stage(SERVICE, "parse"), span("ingestion.parse", mime=doc.mime_type):
            parser = self._selector(src).get(doc.mime_type, filename)
            parsed = await asyncio.to_thread(parser.parse, doc.content, mime_type=doc.mime_type, filename=filename)
        parser_name = parsed.parser or getattr(parser, "name", type(parser).__name__)

        # 4 clean and classify
        floor = DataClass.max(src.data_class, DataClass(doc.metadata.get("data_class", src.data_class.value)))
        with metrics.time_stage(SERVICE, "clean"), span("ingestion.clean"):
            cleaned = await self.cleaner.clean(parsed, floor=floor, source=src.name)
        if cleaned.quarantined:
            await self._withdraw(doc, previous, collection, src)
            self.registry.mark(doc.id, doc.source, "quarantined", error="indirect prompt injection suspected",
                               run_id=run_id, checksum=doc.checksum)
            self._audit("ingestion.quarantine", action="data.index", resource=doc.id, decision="quarantined",
                        source=src.name, indicators=cleaned.injection)
            await self.alerter.alert("document_quarantined", source=src.name, doc_id=doc.id,
                                     indicators=cleaned.injection)
            return "quarantined"

        access_tags = sorted(set(doc.access_tags) | set(src.access_tags))
        if not access_tags and src.require_access_tags:
            await self._withdraw(doc, previous, collection, src)
            self.registry.mark(doc.id, doc.source, "blocked", error="no access tags", run_id=run_id,
                               checksum=doc.checksum)
            self._audit("ingestion.index", action="data.index", resource=doc.id, decision="deny",
                        reason="no access tags", source=src.name)
            return "blocked"

        decision = await self._decide("data.index", {"source": src.name, "collection": collection, "doc_id": doc.id},
                                      {"data_class": cleaned.data_class.value, "access_tags": access_tags,
                                       "personal_data": bool(cleaned.pii_entities), "pii_masked": cleaned.pii_masked,
                                       "injection_suspected": bool(cleaned.injection)})
        if not decision.allow:
            await self._withdraw(doc, previous, collection, src)
            self.registry.mark(doc.id, doc.source, "blocked", error="policy: " + "; ".join(decision.reasons),
                               run_id=run_id, checksum=doc.checksum)
            self._audit("ingestion.index", action="data.index", resource=doc.id, decision="deny",
                        reasons=decision.reasons, policy_version=decision.policy_version, source=src.name)
            return "blocked"
        if decision.obligations.get("mask_pii") and cleaned.pii_entities and not cleaned.pii_masked:
            cleaned = await self.cleaner.clean(parsed, floor=floor, force_mask=True, source=src.name)

        # 5 chunk
        chunker_name, chunker = self._chunker(src, doc.mime_type)
        with metrics.time_stage(SERVICE, "chunk"), span("ingestion.chunk", chunker=chunker_name):
            drafts = chunker.chunk(cleaned.blocks, title=parsed.title or doc.title)
        if not drafts:
            await self._withdraw(doc, previous, collection, src)
            self.registry.mark_indexed(doc, run_id=run_id, chunk_ids=[], collection=collection,
                                       landing_uri=landing_uri, status="empty")
            return "empty"

        # 6 enrich
        version = self.registry.next_version(doc)
        flags: dict[str, Any] = {}
        if cleaned.injection:
            flags["injection_suspected"] = True
        if cleaned.pii_entities:
            flags["pii_entities"] = cleaned.pii_entities
            flags["pii_masked"] = cleaned.pii_masked
        if parsed.metadata.get("scanned_pages"):
            flags["scanned_pages_missing"] = parsed.metadata["scanned_pages"]
        chunks = enrich(doc, drafts, title=parsed.title or doc.title, data_class=cleaned.data_class,
                        access_tags=access_tags, run_id=run_id, doc_version=version, parser=parser_name,
                        chunker=chunker_name, flags=flags)

        # 7 embed + 8 index
        await self.writer.embed(chunks, source=src.name)
        stale = list(previous.chunk_ids) if previous else []
        await self.writer.write(collection, chunks, stale_chunk_ids=stale, doc_id=doc.id,
                                changed=previous is not None, source=src.name)
        entry = self.registry.mark_indexed(doc, run_id=run_id, chunk_ids=[c.id for c in chunks],
                                           collection=collection, landing_uri=landing_uri, version=version)
        keep = {c.id for c in chunks}
        self.lineage.remove([cid for cid in stale if cid not in keep])
        self.lineage.record(chunks, doc_version=version, checksum=doc.checksum, collection=collection, run_id=run_id,
                            parser=parser_name, chunker=chunker_name, embedding_model=self.writer.model,
                            embedding_fingerprint=self.writer.fingerprint, landing_uri=landing_uri)
        self._audit("ingestion.index", action="data.index", resource=doc.id, decision="allow", source=src.name,
                    collection=collection, version=entry.version, chunk_ids=[c.id for c in chunks],
                    data_class=cleaned.data_class.value, access_tags=access_tags, run_id=run_id)
        return "indexed"

    async def _withdraw(self, doc: Document, previous: Any, collection: str, src: SourceConfig) -> None:
        """A document that may no longer be indexed (blocked, quarantined, empty) loses its old chunks too."""
        if previous is not None and previous.chunk_ids:
            await self.writer.remove(previous.collection or collection, doc.id, source=src.name)
            self.lineage.remove(previous.chunk_ids)

    async def _delete(self, doc: Document, src: SourceConfig, run_id: str | None) -> str:
        entry = self.registry.get(doc.id, doc.source)
        collection = (entry.collection if entry and entry.collection else src.collection)
        await self.writer.remove(collection, doc.id, source=src.name)
        removed = self.lineage.remove_document(doc.id)
        self.registry.mark(doc.id, doc.source, "deleted", run_id=run_id)
        self._audit("ingestion.delete", action="data.delete", resource=doc.id, decision="ok", source=src.name,
                    collection=collection, chunks=removed, run_id=run_id)
        return "removed"

    # ------------------------------------------------------------------ runs
    async def run_source(self, name: str, *, trigger: str = "manual", process: bool = True,
                         verify: bool = True) -> RunRecord:
        """Extract, then (in-process mode) drain the queue through a worker and verify.

        With ``process=False`` only extraction happens and the separate worker pool does the rest; the run
        is then completed by ``finish_run`` (the worker calls it when the queue is drained)."""
        run = await self.extract(name, trigger=trigger)
        if not process:
            return run
        from aiready.ingestion.worker import Worker

        await Worker(self).drain()
        await self._wait_for_outcomes(run.run_id)
        return await self.finish_run(run.run_id, verify=verify)

    OUTCOMES = ("indexed", "removed", "duplicate", "blocked", "quarantined", "empty", "dead_lettered")

    async def _wait_for_outcomes(self, run_id: str, timeout: float = 120.0) -> None:
        """When separate workers share the queue, this process's drain returns once the queue is empty while
        other workers may still be processing the run's documents. Verification must only start after every
        enqueued document has an outcome, so the new content is really in the index."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = self.registry.get_run(run_id)
            if run is None:
                return
            done = sum(run.counts.get(k, 0) for k in self.OUTCOMES)
            if done >= run.counts.get("enqueued", 0):
                return
            await asyncio.sleep(0.25)
        log.warning("run %s: not every document had an outcome after %ss; verifying anyway", run_id, timeout)

    async def finish_run(self, run_id: str, *, verify: bool = True) -> RunRecord:
        run = self.registry.get_run(run_id)
        if run is None:
            raise NotFound(f"unknown run '{run_id}'")
        src = self.source(run.source)
        if verify and src.verify:
            results = await self.verifier().verify(src.name, src.collection, src.verify, run_id=run_id)
            run = self.registry.get_run(run_id) or run
            run.verification = [r.model_dump() for r in results]
        failed = run.counts.get("failed", 0) + run.counts.get("dead_lettered", 0)
        verify_failed = any(not v.get("passed") for v in run.verification)
        run.status = "partial" if failed or verify_failed else "succeeded"
        run.finished_at = time.time()
        self.registry.save_run(run)
        self._audit("ingestion.run", action="load", resource=run.source, decision=run.status, run_id=run_id,
                    counts=run.counts)
        return run

    def verifier(self) -> Verifier:
        return Verifier(search=self.writer.search, gateway=self.writer.gateway, principal=self.principal,
                        registry=self.registry, alerter=self.alerter, model=self.writer.model)

    # ------------------------------------------------------------------ dead letters
    async def dead_letters(self) -> list[dict[str, Any]]:
        out = []
        for m in await self.queue.dead_letters(self.topic):
            d = m.payload.get("document", {})
            out.append({"message_id": m.id, "source": m.payload.get("source"), "doc_id": d.get("id"),
                        "attempts": m.attempts, "error": m.last_error, "enqueued_at": m.enqueued_at,
                        "run_id": m.payload.get("run_id")})
        return out

    async def replay_dead_letters(self, *, process: bool = True) -> dict[str, Any]:
        n = await self.queue.replay_dead_letters(self.topic)
        self._audit("ingestion.dlq_replay", action="replay", resource=self.topic, decision="ok", count=n)
        processed = 0
        if process and n:
            from aiready.ingestion.worker import Worker

            processed = await Worker(self).drain()
        return {"replayed": n, "processed": processed}

    # ------------------------------------------------------------------ metrics
    async def metrics_summary(self) -> dict[str, Any]:
        """Lag, failures and items per hour - the pipeline metrics of the platform slide."""
        now = time.time()
        entries = self.registry.entries()
        status = Counter(e.status for e in entries)
        recent = [e for e in entries if e.indexed_at and e.indexed_at > now - 3600 and e.status == "indexed"]
        lags = [e.lag_seconds for e in entries if e.lag_seconds is not None and e.indexed_at and e.indexed_at > now - 86400]
        last_hour = [o for t, _, o in self._events if t > now - 3600]
        depth = await self.queue.depth(self.topic)
        dlq = len(await self.queue.dead_letters(self.topic))
        metrics.queue_depth.labels(SERVICE, self.topic).set(depth)
        metrics.queue_depth.labels(SERVICE, self.topic + ":dlq").set(dlq)
        per_source: dict[str, Any] = {}
        for s in self.cfg.sources:
            runs = self.registry.runs(s.name, limit=1)
            per_source[s.name] = {
                "documents": sum(1 for e in entries if e.source == s.name and e.status == "indexed"),
                "last_run": runs[0].model_dump() if runs else None,
            }
        return {
            "documents_by_status": dict(status),
            "items_indexed_last_hour": len(recent),
            "events_last_hour": dict(Counter(last_hour)),
            "failures_last_hour": sum(1 for o in last_hour if o in ("failed", "dead_lettered")),
            "lag_seconds": {"avg": sum(lags) / len(lags) if lags else None, "max": max(lags) if lags else None},
            "queue_depth": depth,
            "dead_letters": dlq,
            "sources": per_source,
        }
