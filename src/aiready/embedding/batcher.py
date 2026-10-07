"""Dynamic batcher with a live and a bulk queue.

GPU throughput depends on batch size, but a single question cannot wait for a full batch. The batcher
therefore holds the first pending text a few milliseconds (``max_wait_ms``) so concurrent requests can share
one model call, and closes the batch early once ``max_batch_size`` texts are waiting.

Two queues:
  * ``live``  - query texts a user is waiting for. Always served first, with a shorter wait;
  * ``bulk``  - ingestion batches that can wait. They fill whatever room live traffic leaves.

A large ingestion job therefore never sits in front of a question: each new batch takes live texts first.
Queue depth is exported (``aiready_queue_depth{service="embedding",queue="<model>:<priority>"}``) so model
replicas can autoscale on it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Awaitable, Callable

from aiready.core.errors import RateLimited
from aiready.core.telemetry import metrics

log = logging.getLogger(__name__)
Priority = Literal["live", "bulk"]
EncodeFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class _Item:
    text: str
    future: asyncio.Future
    enqueued: float = field(default_factory=time.perf_counter)


class DynamicBatcher:
    def __init__(
        self,
        encode: EncodeFn,
        *,
        max_batch_size: int = 64,
        max_wait_ms: float = 5.0,
        live_max_wait_ms: float | None = None,
        max_concurrent_batches: int = 1,
        max_queue: int = 100_000,
        name: str = "embedding",
        service: str = "embedding",
    ):
        self.encode = encode
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.live_max_wait = (live_max_wait_ms if live_max_wait_ms is not None else max_wait_ms) / 1000.0
        self.max_concurrent = max_concurrent_batches
        self.max_queue = max_queue
        self.name, self.service = name, service
        self._queues: dict[str, deque[_Item]] = {"live": deque(), "bulk": deque()}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._slots: asyncio.Semaphore | None = None
        self._worker: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()
        self.batches: list[list[str]] = []  # recent batch contents (bounded) - observability and tests
        self.batch_history = 100

    # ------------------------------------------------------------------ public
    def depth(self, priority: Priority | None = None) -> int:
        if priority:
            return len(self._queues[priority])
        return sum(len(q) for q in self._queues.values())

    async def submit(self, texts: list[str], priority: Priority = "live") -> list[list[float]]:
        if not texts:
            return []
        if priority not in self._queues:
            raise ValueError(f"unknown priority '{priority}'")
        self._ensure_worker()
        if priority == "bulk" and self.depth() + len(texts) > self.max_queue:
            metrics.fallbacks.labels(self.service, "queue_full").inc()
            raise RateLimited(
                f"embedding queue full ({self.depth()} pending); retry later",
                details={"queue_depth": self.depth()},
            )
        loop = asyncio.get_running_loop()
        items = [_Item(t, loop.create_future()) for t in texts]
        self._queues[priority].extend(items)
        self._report_depth()
        assert self._wake is not None
        self._wake.set()
        return list(await asyncio.gather(*(i.future for i in items)))

    async def aclose(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        for t in list(self._inflight):
            t.cancel()
        self._worker = None

    # ------------------------------------------------------------------ internals
    def _ensure_worker(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._worker is None or self._worker.done():
            if (
                self._loop is not loop
            ):  # new event loop (tests, reload): items of the old loop cannot complete
                for q in self._queues.values():
                    q.clear()
            self._loop = loop
            self._wake = asyncio.Event()
            self._slots = asyncio.Semaphore(self.max_concurrent)
            self._worker = loop.create_task(self._run(), name=f"embedding-batcher-{self.name}")

    def _report_depth(self) -> None:
        for p, q in self._queues.items():
            metrics.queue_depth.labels(self.service, f"{self.name}:{p}").set(len(q))

    async def _run(self) -> None:
        assert self._wake is not None and self._slots is not None
        while True:
            if not self.depth():
                self._wake.clear()
                await self._wake.wait()
            await self._fill_window()
            await self._slots.acquire()
            batch = self._take_batch()
            if not batch:
                self._slots.release()
                continue
            task = asyncio.get_running_loop().create_task(self._execute(batch))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _fill_window(self) -> None:
        """Wait until the batch is full or the oldest pending text has waited long enough."""
        assert self._wake is not None
        while self.depth() < self.max_batch_size:
            live = self._queues["live"]
            if live:
                deadline = live[0].enqueued + self.live_max_wait
            else:
                deadline = self._queues["bulk"][0].enqueued + self.max_wait
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)
            except TimeoutError:
                return

    def _take_batch(self) -> list[_Item]:
        batch: list[_Item] = []
        for p in ("live", "bulk"):  # live first: bulk only fills the remaining room
            q = self._queues[p]
            while q and len(batch) < self.max_batch_size:
                item = q.popleft()
                if not item.future.done():  # caller may have been cancelled
                    batch.append(item)
        self._report_depth()
        return batch

    async def _execute(self, batch: list[_Item]) -> None:
        assert self._slots is not None
        texts = [i.text for i in batch]
        self.batches.append(texts)
        del self.batches[: -self.batch_history]
        try:
            with metrics.time_stage(self.service, "encode_batch"):
                vectors = await self.encode(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(f"encoder returned {len(vectors)} vectors for {len(batch)} texts")
            for item, vec in zip(batch, vectors):
                if not item.future.done():
                    item.future.set_result(vec)
        except Exception as exc:  # every waiter of the batch gets the error; the worker keeps running
            log.warning("embedding batch of %d failed: %s", len(batch), exc)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
        finally:
            self._slots.release()
