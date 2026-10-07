"""Vector cache: text hash + model id + model version -> vector.

Repeated text (unchanged chunks on re-ingestion, popular questions) skips the GPU. The key includes the
model version, so upgrading a model can never serve a vector from the old space.
"""

from __future__ import annotations

from aiready.core.stores import KVStore, MemoryKV
from aiready.core.telemetry import metrics
from aiready.core.textutil import sha256


class VectorCache:
    def __init__(
        self, kv: KVStore | None = None, *, ttl_seconds: float | None = None, service: str = "embedding"
    ):
        self.kv = kv or MemoryKV()
        self.ttl = ttl_seconds
        self.service = service

    @staticmethod
    def key(text: str, model_id: str, version: str) -> str:
        return f"emb:{model_id}:{version}:{sha256(text)}"

    def get_many(self, texts: list[str], model_id: str, version: str) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        for t in texts:
            v = self.kv.get(self.key(t, model_id, version))
            metrics.cache.labels(self.service, "vector", "hit" if v is not None else "miss").inc()
            out.append(v)
        return out

    def set_many(self, texts: list[str], vectors: list[list[float]], model_id: str, version: str) -> None:
        for t, v in zip(texts, vectors):
            self.kv.set(self.key(t, model_id, version), v, ttl=self.ttl)
