"""Model pools and the inference router.

The gateway picks a *pool* (by routing on task, cost and policy); the inference router then picks a *replica*
inside that pool:

* least loaded wins: score = (requests in flight + 1) x smoothed latency (EWMA), so a slow replica gets less
  traffic even when it has free slots;
* a circuit breaker ejects a replica after N consecutive failures for a cooldown, then lets one trial request
  through (half-open) before trusting it again;
* optional per-replica concurrency caps queue the excess; the queue depth and tokens/second per pool are
  exported as autoscaling signals ("scales on queue depth and tokens per second").
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from aiready.core.errors import UpstreamError
from aiready.core.telemetry import metrics
from aiready.core.types import ChatRequest, ChatResponse
from aiready.models.backends import BackendError, ModelBackend, make_backend
from aiready.models.catalog import PoolSpec

log = logging.getLogger(__name__)
T = TypeVar("T")

_DEFAULT_LATENCY_S = 0.05  # optimistic prior so new replicas receive traffic


class Replica:
    """One model server, with the load and health bookkeeping the router needs."""

    def __init__(self, name: str, backend: ModelBackend, spec: dict[str, Any], alpha: float = 0.3):
        self.name = name
        self.backend = backend
        self.spec = spec
        self.alpha = alpha
        self.in_flight = 0
        self.ewma_latency: float | None = None
        self.consecutive_failures = 0
        self.ejected_until = 0.0
        self.half_open = False
        self.requests = 0
        self.failures = 0

    def healthy(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) >= self.ejected_until

    def score(self) -> float:
        return (self.in_flight + 1) * (self.ewma_latency or _DEFAULT_LATENCY_S)

    def on_success(self, latency: float) -> None:
        self.ewma_latency = latency if self.ewma_latency is None else (
            self.alpha * latency + (1 - self.alpha) * self.ewma_latency)
        self.consecutive_failures = 0
        self.half_open = False

    def on_failure(self, eject_after: int, cooldown: float) -> bool:
        """Returns True when this failure ejected the replica. A failure also counts as a slow response, so
        traffic shifts away from a flaky replica before the breaker opens."""
        penalty = max(1.0, (self.ewma_latency or _DEFAULT_LATENCY_S) * 2)
        self.ewma_latency = penalty if self.ewma_latency is None else (
            self.alpha * penalty + (1 - self.alpha) * self.ewma_latency)
        self.failures += 1
        self.consecutive_failures += 1
        if self.half_open or self.consecutive_failures >= eject_after:
            self.ejected_until = time.monotonic() + cooldown
            self.half_open = True  # the next request after the cooldown is a trial
            return True
        return False

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        return {"name": self.name, "backend": self.spec.get("type") or self.spec.get("class"),
                "healthy": self.healthy(now), "in_flight": self.in_flight,
                "ewma_latency_ms": round((self.ewma_latency or 0) * 1000, 2), "requests": self.requests,
                "failures": self.failures, "consecutive_failures": self.consecutive_failures,
                "ejected_for_s": round(max(0.0, self.ejected_until - now), 2)}


class ModelPool:
    """Replicas serving the same model. Thread-safe enough for one event loop (no awaits while choosing)."""

    def __init__(self, name: str, spec: PoolSpec, backends: list[ModelBackend] | None = None):
        self.name = name
        self.spec = spec
        self.fingerprint = spec.fingerprint()
        bks = backends or [make_backend(r) for r in spec.replicas]
        self.replicas = [Replica(f"{name}-{i}", b, spec.replicas[i] if i < len(spec.replicas) else {},
                                 spec.ewma_alpha) for i, b in enumerate(bks)]
        if not self.replicas:
            raise UpstreamError(f"pool '{name}' has no replicas")
        self.waiting = 0
        self._tokens: deque[tuple[float, int]] = deque()

    # ------------------------------------------------------------------ selection
    def _candidates(self, exclude: set[str]) -> list[Replica]:
        now = time.monotonic()
        cap = self.spec.max_in_flight_per_replica
        return [r for r in self.replicas if r.name not in exclude and r.healthy(now)
                and (cap == 0 or r.in_flight < cap)]

    def has_healthy(self) -> bool:
        now = time.monotonic()
        return any(r.healthy(now) for r in self.replicas)

    async def acquire(self, exclude: set[str] | None = None) -> Replica:
        """Pick the least-loaded healthy replica, queueing when every replica is at its cap."""
        exclude = exclude or set()
        deadline = time.monotonic() + self.spec.queue_timeout_seconds
        queued = False
        try:
            while True:
                cands = self._candidates(exclude)
                if cands:
                    best = min(cands, key=Replica.score)
                    best.in_flight += 1
                    best.requests += 1
                    return best
                now = time.monotonic()
                if not any(r.healthy(now) for r in self.replicas if r.name not in exclude):
                    raise BackendError(f"pool '{self.name}': no healthy replica")
                if now >= deadline:
                    raise BackendError(f"pool '{self.name}': queue timeout after {self.spec.queue_timeout_seconds}s")
                if not queued:
                    queued = True
                    self.waiting += 1
                    metrics.queue_depth.labels("models", self.name).set(self.waiting)
                await asyncio.sleep(0.005)
        finally:
            if queued:
                self.waiting -= 1
                metrics.queue_depth.labels("models", self.name).set(self.waiting)

    def release(self, replica: Replica, *, ok: bool, latency: float, tokens: int = 0) -> None:
        replica.in_flight = max(0, replica.in_flight - 1)
        if ok:
            replica.on_success(latency)
            if tokens:
                self._tokens.append((time.monotonic(), tokens))
        elif replica.on_failure(self.spec.eject_after_failures, self.spec.cooldown_seconds):
            log.warning("replica %s ejected for %.0fs after %d failures", replica.name, self.spec.cooldown_seconds,
                        replica.consecutive_failures)
            metrics.fallbacks.labels("models", "replica_ejected").inc()

    # ------------------------------------------------------------------ calls
    async def run(self, fn: Callable[[ModelBackend], Awaitable[T]], *, tokens_of: Callable[[T], int] | None = None,
                  attempts: int | None = None) -> T:
        """Run ``fn`` on a replica; on a retryable failure try one other replica before giving up."""
        tried: set[str] = set()
        attempts = attempts or min(2, len(self.replicas))
        last: Exception | None = None
        for _ in range(attempts):
            try:
                rep = await self.acquire(tried)
            except BackendError as exc:
                if last is not None:
                    raise last from exc
                raise
            tried.add(rep.name)
            t0 = time.perf_counter()
            try:
                out = await fn(rep.backend)
            except BackendError as exc:
                self.release(rep, ok=not exc.retryable, latency=time.perf_counter() - t0)
                if not exc.retryable:
                    raise
                last = exc
                continue
            except asyncio.CancelledError:
                self.release(rep, ok=True, latency=time.perf_counter() - t0)
                raise
            except Exception as exc:
                self.release(rep, ok=False, latency=time.perf_counter() - t0)
                last = BackendError(f"pool '{self.name}' replica {rep.name} failed: {exc}")
                continue
            self.release(rep, ok=True, latency=time.perf_counter() - t0, tokens=tokens_of(out) if tokens_of else 0)
            return out
        assert last is not None
        raise last

    async def stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream from one replica. The replica's slot is held until the stream ends."""
        rep = await self.acquire()
        t0 = time.perf_counter()
        ok, tokens = False, 0
        try:
            async for chunk in rep.backend.chat_stream(req):
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                if usage:
                    tokens = int(usage.get("completion_tokens") or 0)
                elif chunk.get("choices"):
                    tokens += 1 if chunk["choices"][0].get("delta", {}).get("content") else 0
                yield chunk
            ok = True
        except (GeneratorExit, asyncio.CancelledError):
            ok = True  # the consumer stopped reading; not the replica's fault
            raise
        except BackendError as exc:
            ok = not exc.retryable
            raise
        except Exception as exc:
            raise BackendError(f"pool '{self.name}' replica {rep.name} stream failed: {exc}") from exc
        finally:
            self.release(rep, ok=ok, latency=time.perf_counter() - t0, tokens=tokens)

    # ------------------------------------------------------------------ signals
    def tokens_per_second(self, window: float = 60.0) -> float:
        cutoff = time.monotonic() - window
        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()
        return round(sum(t for _, t in self._tokens) / window, 3)

    def stats(self) -> dict[str, Any]:
        reps = [r.stats() for r in self.replicas]
        return {"pool": self.name, "model": self.spec.model, "replicas": reps,
                "healthy_replicas": sum(1 for r in reps if r["healthy"]),
                "in_flight": sum(r["in_flight"] for r in reps), "queue_depth": self.waiting,
                "tokens_per_second": self.tokens_per_second()}


def _completion_tokens(resp: ChatResponse) -> int:
    return resp.usage.completion_tokens or 0


class InferenceRouter:
    """Holds every pool and dispatches calls to the least-loaded healthy replica of the chosen pool.

    ``sync(specs)`` rebuilds only the pools whose specification changed, so health and latency history of
    untouched pools survive a configuration reload.
    """

    def __init__(self, specs: dict[str, PoolSpec] | None = None):
        self.pools: dict[str, ModelPool] = {}
        if specs:
            self.sync(specs)

    def sync(self, specs: dict[str, PoolSpec]) -> None:
        new: dict[str, ModelPool] = {}
        for name, spec in specs.items():
            cur = self.pools.get(name)
            new[name] = cur if cur is not None and cur.fingerprint == spec.fingerprint() else ModelPool(name, spec)
        self.pools = new

    def pool(self, name: str) -> ModelPool:
        try:
            return self.pools[name]
        except KeyError:
            raise UpstreamError(f"unknown model pool '{name}'") from None

    def healthy(self, name: str) -> bool:
        p = self.pools.get(name)
        return bool(p and p.has_healthy())

    async def chat(self, pool: str, req: ChatRequest) -> ChatResponse:
        return await self.pool(pool).run(lambda b: b.chat(req), tokens_of=_completion_tokens)

    def chat_stream(self, pool: str, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        return self.pool(pool).stream(req)

    async def embed(self, pool: str, texts: list[str], model: str) -> list[list[float]]:
        return await self.pool(pool).run(lambda b: b.embed(texts, model))

    async def rerank(self, pool: str, query: str, docs: list[str], model: str) -> list[float]:
        return await self.pool(pool).run(lambda b: b.rerank(query, docs, model))

    def stats(self) -> dict[str, Any]:
        return {name: p.stats() for name, p in self.pools.items()}

    async def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, p in self.pools.items():
            out[name] = [await r.backend.health() for r in p.replicas]
        return out
