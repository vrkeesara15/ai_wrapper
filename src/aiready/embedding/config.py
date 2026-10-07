"""Configuration of the embedding service (top-level section ``embedding:``).

Defaults run offline: one logical model ``embedding`` served by the deterministic ``hash`` encoder, an
in-memory vector cache and a dynamic batcher tuned for a CPU.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BatcherConfig(BaseModel):
    """Dynamic batching. Waiting a few milliseconds lets concurrent requests share one GPU batch."""

    max_batch_size: int = Field(64, ge=1)
    # how long the first bulk text waits for company
    max_wait_ms: float = Field(5.0, ge=0)
    # shorter wait once a live text is pending (None = max_wait_ms)
    live_max_wait_ms: float | None = Field(2.0, ge=0)
    # >1 for remote servers that batch internally / several replicas
    max_concurrent_batches: int = Field(1, ge=1)
    # back-pressure: bulk submissions beyond this are rejected (429)
    max_queue: int = Field(100_000, ge=1)


class CacheConfig(BaseModel):
    enabled: bool = True
    # any kv_store spec (redis in prod)
    kv: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    # vectors never go stale for a fixed model version; the TTL only bounds memory
    ttl_seconds: float | None = 7 * 24 * 3600


class EmbeddingConfig(BaseModel):
    # logical model name -> embedder spec (plugin kind ``embedder``). The gateway routes "embedding" here.
    models: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {"embedding": {"type": "hash", "dim": 384}}
    )
    default_model: str = "embedding"
    cache: CacheConfig = Field(default_factory=CacheConfig)
    batcher: BatcherConfig = Field(default_factory=BatcherConfig)
    max_texts_per_request: int = Field(4096, ge=1)
    # longer texts are truncated (the model would truncate anyway)
    max_chars_per_text: int = Field(32_000, ge=1)
    policy_check: bool = True  # ask the policy engine (model.invoke) when a user principal is known
