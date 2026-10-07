"""Worker pool: one message per document from the work queue, with retries and a dead-letter queue.

Run as its own process (scaled on queue depth) with::

    python -m aiready.ingestion.worker            # workers + scheduler
    python -m aiready.ingestion.worker --no-scheduler

A failed message is retried with backoff (``queue.max_attempts``); permanent failures - a file no parser
can read - go straight to the dead-letter queue instead of burning retries. Dead letters are held for
inspection and replay (``POST /v1/dlq/replay`` on the ingestion API) once the cause is fixed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, ValidationFailed
from aiready.core.queues import Message
from aiready.core.telemetry import metrics

log = logging.getLogger(__name__)

PERMANENT_ERRORS: tuple[type[BaseException], ...] = (ValidationFailed, NotFound, UnicodeDecodeError)


class Worker:
    def __init__(self, pipeline: Any, *, concurrency: int | None = None, max_attempts: int | None = None):
        self.pipeline = pipeline
        q = pipeline.cfg.queue
        self.queue, self.topic = pipeline.queue, pipeline.topic
        self.concurrency = max(1, concurrency or q.concurrency)
        self.max_attempts = max(1, max_attempts or q.max_attempts)
        self.poll_timeout = q.poll_timeout
        self.processed = 0

    async def process(self, msg: Message) -> str:
        """Handle one message; on failure requeue or dead-letter it. Never raises."""
        payload = msg.payload
        doc = payload.get("document", {})
        source = payload.get("source", "?")
        try:
            outcome = await self.pipeline.handle(payload)
            await self.queue.ack(msg)
            self.processed += 1
            return outcome
        except Exception as exc:  # noqa: BLE001 - every failure must end in a retry or the DLQ
            error = f"{type(exc).__name__}: {exc}"
            attempts = msg.attempts + 1 if isinstance(exc, PERMANENT_ERRORS) else self.max_attempts
            dead = await self.queue.nack(msg, error, max_attempts=attempts)
            self.pipeline.registry.mark(doc.get("id", "?"), doc.get("source") or source, "failed", error=error,
                                        run_id=payload.get("run_id"))
            if dead:
                metrics.pipeline.labels(source, "document", "dead_lettered").inc()
                await self.pipeline._bump(payload.get("run_id"), "dead_lettered")
                get_audit().for_principal(self.pipeline.principal, "ingestion.dead_letter", action="data.index",
                                          resource=doc.get("id"), decision="failed", source=source, error=error,
                                          attempts=msg.attempts)
                await self.pipeline.alerter.alert("dead_lettered", source=source, doc_id=doc.get("id"), error=error)
                return "dead_lettered"
            metrics.pipeline.labels(source, "document", "retry").inc()
            log.warning("retrying %s after error: %s", doc.get("id"), error)
            return "retry"

    async def drain(self) -> int:
        """Process until the queue is empty (in-process runs, tests, replay). Returns messages handled."""
        handled = 0
        while True:
            batch: list[Message] = []
            for _ in range(self.concurrency):
                m = await self.queue.get(self.topic)
                if m is None:
                    break
                batch.append(m)
            if not batch:
                return handled
            await asyncio.gather(*(self.process(m) for m in batch))
            handled += len(batch)

    async def run(self, stop: asyncio.Event) -> None:
        """Long-running pool: ``concurrency`` consumers until ``stop`` is set."""
        async def consumer(i: int) -> None:
            while not stop.is_set():
                m = await self.queue.get(self.topic, timeout=self.poll_timeout)
                if m is None:
                    continue
                await self.process(m)
                metrics.queue_depth.labels("ingestion", self.topic).set(await self.queue.depth(self.topic))

        await asyncio.gather(*(consumer(i) for i in range(self.concurrency)))


async def main_async(with_scheduler: bool = True) -> None:
    from aiready.core.telemetry import setup_telemetry
    from aiready.ingestion.pipeline import IngestionPipeline
    from aiready.ingestion.scheduler import Scheduler

    setup_telemetry("ingestion-worker")
    pipeline = IngestionPipeline()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    tasks = [Worker(pipeline).run(stop)]
    if with_scheduler:
        tasks.append(Scheduler(pipeline, process=False).run(stop))
    await asyncio.gather(*tasks)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AI-Ready ingestion worker pool")
    ap.add_argument("--no-scheduler", action="store_true", help="only consume the work queue")
    args = ap.parse_args(argv)
    asyncio.run(main_async(with_scheduler=not args.no_scheduler))


if __name__ == "__main__":  # pragma: no cover
    main()
