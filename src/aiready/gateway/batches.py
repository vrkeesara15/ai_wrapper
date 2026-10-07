"""Batch and async work: requests nobody is waiting on run off-peak on spare capacity.

A batch is a list of chat requests submitted at once. It is stored in the state store and queued; a worker
drains the queue only inside the configured window (e.g. ``22:00-06:00``), unless the batch asked to run
``now``. Every item still passes the full gateway pipeline as the submitting user - limits, screening,
policy, audit and budgets apply to batch work exactly as to live traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from aiready.core.errors import AIReadyError, NotFound, PermissionDenied, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.queues import Queue, make_queue
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import span
from aiready.core.types import ChatRequest

if TYPE_CHECKING:
    from aiready.gateway.service import GatewayService

log = logging.getLogger(__name__)
NAMESPACE = "gateway.batches"


class BatchItem(BaseModel):
    custom_id: str
    body: dict[str, Any]


class BatchRecord(BaseModel):
    id: str = Field(default_factory=lambda: "batch_" + uuid.uuid4().hex[:20])
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    run: Literal["off_peak", "now"] = "off_peak"
    owner: str
    principal: dict[str, Any]
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return self.model_dump(exclude={"principal", "results"})


def in_window(window: str, now: datetime | None = None) -> bool:
    """``HH:MM-HH:MM`` in local time, wrapping past midnight; ``always`` is always open."""
    if window.strip().lower() in ("always", "*", ""):
        return True
    now = now or datetime.now()
    start_s, end_s = window.split("-", 1)
    sh, sm = (int(x) for x in start_s.strip().split(":"))
    eh, em = (int(x) for x in end_s.strip().split(":"))
    cur, start, end = now.hour * 60 + now.minute, sh * 60 + sm, eh * 60 + em
    return start <= cur < end if start <= end else (cur >= start or cur < end)


class BatchManager:
    def __init__(self, service: GatewayService, *, queue: Queue | None = None, docstore: DocStore | None = None):
        self.service = service
        cfg = service.config().batches
        self.queue = queue or make_queue(cfg.queue or service.store.get("queue") or {"type": "memory"})
        self._docstore = docstore
        self._task: asyncio.Task | None = None

    @property
    def docstore(self) -> DocStore:
        return self._docstore or get_docstore()

    def _save(self, rec: BatchRecord) -> None:
        self.docstore.put(NAMESPACE, rec.id, rec.model_dump())

    def get(self, batch_id: str, principal: Principal) -> BatchRecord:
        raw = self.docstore.get(NAMESPACE, batch_id)
        if raw is None:
            raise NotFound(f"batch '{batch_id}' not found")
        rec = BatchRecord.model_validate(raw)
        if rec.owner != principal.sub and not principal.has_role(self.service.config().admin.required_role):
            raise NotFound(f"batch '{batch_id}' not found")  # do not reveal other users' batches
        return rec

    def list(self, principal: Principal) -> list[dict[str, Any]]:
        return [BatchRecord.model_validate(v).summary() for v in self.docstore.values(NAMESPACE)
                if v.get("owner") == principal.sub]

    async def submit(self, items: list[dict[str, Any]], principal: Principal, *, run: str = "off_peak",
                     metadata: dict[str, Any] | None = None) -> BatchRecord:
        cfg = self.service.config().batches
        if not cfg.enabled:
            raise PermissionDenied("batch processing is disabled")
        if principal.is_service_only and "batch" not in self.service.config().auth.service_only_operations:
            raise PermissionDenied("batches need a signed-in user")
        if not items or len(items) > cfg.max_requests_per_batch:
            raise ValidationFailed(f"a batch holds 1 to {cfg.max_requests_per_batch} requests")
        parsed = []
        for i, it in enumerate(items):
            item = BatchItem.model_validate(it if "body" in it else {"custom_id": str(i), "body": it})
            try:
                ChatRequest.model_validate(item.body)
            except Exception as exc:
                raise ValidationFailed(f"request '{item.custom_id}' is not a valid chat request: {exc}") from exc
            parsed.append(item)
        # identity is kept (not the token, which will have expired by the off-peak window) so items run as this user
        ident = {**principal.model_dump(), "claims": {k: v for k, v in (principal.claims or {}).items()
                                                      if k not in ("exp", "iat", "nbf", "jti")}}
        rec = BatchRecord(owner=principal.sub, principal=ident, run="now" if run == "now" else "off_peak",
                          total=len(parsed), metadata=metadata or {})
        self._save(rec)
        for item in parsed:
            await self.queue.put(cfg.topic, {"batch_id": rec.id, "custom_id": item.custom_id, "body": item.body,
                                             "run": rec.run})
        return rec

    async def run_once(self, *, force: bool = False, max_items: int | None = None) -> int:
        """Process queued items. Off-peak items wait until the window opens (or ``force``)."""
        cfg = self.service.config().batches
        open_now = force or in_window(cfg.window)
        processed = 0
        deferred = []
        limit = max_items or 10_000
        while processed + len(deferred) < limit:
            msg = await self.queue.get(cfg.topic)
            if msg is None:
                break
            if msg.payload.get("run") != "now" and not open_now:
                deferred.append(msg)
                continue
            await self._process(msg.payload)
            await self.queue.ack(msg)
            processed += 1
        for msg in deferred:  # put waiting work back untouched
            await self.queue.ack(msg)
            await self.queue.put(cfg.topic, msg.payload)
        return processed

    async def _process(self, payload: dict[str, Any]) -> None:
        raw = self.docstore.get(NAMESPACE, payload["batch_id"])
        if raw is None:
            return
        rec = BatchRecord.model_validate(raw)
        if rec.status == "cancelled":
            return
        if rec.status == "queued":
            rec.status, rec.started_at = "running", time.time()
        principal = Principal.model_validate(rec.principal)
        body = dict(payload["body"])
        meta = {**(body.get("metadata") or {}), "priority": "batch", "batch_id": rec.id}
        body["metadata"] = meta
        body["stream"] = False
        with span("gateway.batch_item", batch=rec.id, item=payload["custom_id"]):
            try:
                resp = await self.service.chat(ChatRequest.model_validate(body), principal)
                result = {"custom_id": payload["custom_id"], "status": "ok", "response": resp.model_dump(mode="json")}
            except AIReadyError as exc:
                result = {"custom_id": payload["custom_id"], "status": "error", "error": exc.to_dict()["error"]}
            except Exception as exc:  # one bad item must not stop the batch
                log.exception("batch item failed")
                result = {"custom_id": payload["custom_id"], "status": "error",
                          "error": {"code": "internal_error", "message": str(exc)[:300]}}
        # re-read: another worker may have recorded items meanwhile
        latest = BatchRecord.model_validate(self.docstore.get(NAMESPACE, rec.id) or rec.model_dump())
        latest.results.append(result)
        latest.succeeded, latest.failed = (latest.succeeded + (result["status"] == "ok"),
                                           latest.failed + (result["status"] != "ok"))
        latest.started_at = latest.started_at or rec.started_at
        latest.status = "running"
        if len(latest.results) >= latest.total:
            latest.status = "completed" if latest.succeeded else "failed"
            latest.completed_at = time.time()
        self._save(latest)

    def cancel(self, batch_id: str, principal: Principal) -> BatchRecord:
        rec = self.get(batch_id, principal)
        if rec.status in ("queued", "running"):
            rec.status = "cancelled"
            self._save(rec)
        return rec

    # ------------------------------------------------------------------ background worker
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="gateway-batch-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception:
                log.exception("batch worker iteration failed")
            await asyncio.sleep(self.service.config().batches.poll_seconds)
