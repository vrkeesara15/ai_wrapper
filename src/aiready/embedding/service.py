"""Embedding service: one internal endpoint behind the gateway.

    caller -> vector cache (text hash + model version) -> dynamic batcher (live / bulk queues) -> encoder

* **One model version per index.** ``index_model_fingerprint()`` returns ``model_id@version/dim``; the search
  service stores it with each collection and refuses vectors from any other fingerprint.
* **Same model on both paths.** Ingestion (``priority="bulk"``) and queries (``priority="live"``) resolve the
  same logical name to the same encoder instance with the same settings.
* This *is* the model-serving layer, so it calls its encoder directly (CONVENTIONS 3: the exception to "no
  direct model calls"). Policy is still asked (``model.invoke``) when a user principal is known.
"""

from __future__ import annotations

import logging
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import ConfigError, NotFound, PolicyViolation, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyInput
from aiready.core.plugins import create
from aiready.core.stores import make_kv
from aiready.core.telemetry import metrics, span
from aiready.core.wiring import connect
from aiready.embedding.batcher import DynamicBatcher, Priority
from aiready.embedding.cache import VectorCache
from aiready.embedding.config import EmbeddingConfig
from aiready.embedding.encoders import Encoder, fingerprint

log = logging.getLogger(__name__)
SERVICE = "embedding"


class EmbeddingService:
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        encoders: dict[str, Encoder] | None = None,
        cache: VectorCache | None = None,
    ):
        self.config = config or get_config().section("embedding", EmbeddingConfig)
        self.encoders: dict[str, Encoder] = (
            encoders
            if encoders is not None
            else {name: create("embedder", spec) for name, spec in self.config.models.items()}
        )
        if not self.encoders:
            raise ConfigError("embedding.models is empty")
        if self.config.default_model not in self.encoders:
            raise ConfigError(
                f"embedding.default_model '{self.config.default_model}' is not in embedding.models"
            )
        c = self.config.cache
        self.cache = (
            cache
            if cache is not None
            else (VectorCache(make_kv(c.kv), ttl_seconds=c.ttl_seconds) if c.enabled else None)
        )
        b = self.config.batcher
        self.batchers: dict[str, DynamicBatcher] = {
            name: DynamicBatcher(
                enc.encode,
                max_batch_size=min(b.max_batch_size, enc.max_batch_size),
                max_wait_ms=b.max_wait_ms,
                live_max_wait_ms=b.live_max_wait_ms,
                max_concurrent_batches=b.max_concurrent_batches,
                max_queue=b.max_queue,
                name=name,
            )
            for name, enc in self.encoders.items()
        }

    # ------------------------------------------------------------------ model resolution
    def resolve(self, model: str | None = None) -> tuple[str, Encoder]:
        """Accept a logical name (``embedding``) or the concrete model id the encoder reports."""
        name = model or self.config.default_model
        if name in self.encoders:
            return name, self.encoders[name]
        for logical, enc in self.encoders.items():
            if enc.model_id == name:
                return logical, enc
        raise NotFound(f"unknown embedding model '{name}'", details={"available": sorted(self.encoders)})

    def index_model_fingerprint(self, model: str | None = None) -> str:
        """``model_id@version/dim`` of the encoder serving ``model``. Store it with every index."""
        return fingerprint(self.resolve(model)[1])

    async def fingerprint(self, model: str | None = None) -> str:
        """Async form, same signature as the HTTP client."""
        return self.index_model_fingerprint(model)

    async def info(self, model: str | None = None) -> dict[str, Any]:
        names = [self.resolve(model)[0]] if model else list(self.encoders)
        out = []
        for name in names:
            enc = self.encoders[name]
            out.append(
                {
                    "name": name,
                    "model_id": enc.model_id,
                    "version": enc.version,
                    "dim": enc.dim,
                    "fingerprint": fingerprint(enc),
                    "max_batch_size": self.batchers[name].max_batch_size,
                    "queue_depth": {
                        "live": self.batchers[name].depth("live"),
                        "bulk": self.batchers[name].depth("bulk"),
                    },
                }
            )
        return {"default_model": self.config.default_model, "models": out}

    # ------------------------------------------------------------------ embed
    async def embed(
        self,
        texts: list[str],
        *,
        priority: Priority = "live",
        model: str | None = None,
        principal: Principal | None = None,
    ) -> list[list[float]]:
        if priority not in ("live", "bulk"):
            raise ValidationFailed(f"priority must be 'live' or 'bulk', got '{priority}'")
        if len(texts) > self.config.max_texts_per_request:
            raise ValidationFailed(f"at most {self.config.max_texts_per_request} texts per request")
        if not texts:
            return []
        name, enc = self.resolve(model)
        fp = fingerprint(enc)
        texts = [t[: self.config.max_chars_per_text] for t in texts]
        with (
            span("embedding.embed", model=fp, priority=priority, count=len(texts)) as sp,
            metrics.time_stage(SERVICE, f"embed_{priority}"),
        ):
            if principal is not None and self.config.policy_check and not principal.is_service_only:
                await self._check_policy(principal, name, fp, priority)
            vectors: list[list[float] | None] = (
                self.cache.get_many(texts, enc.model_id, enc.version) if self.cache else [None] * len(texts)
            )
            # dedupe misses: the same text in one request is encoded once
            missing = list(dict.fromkeys(t for t, v in zip(texts, vectors) if v is None))
            hits = len(texts) - sum(1 for v in vectors if v is None)
            sp.set(cache_hits=hits, encoded=len(missing))
            if missing:
                fresh = await self.batchers[name].submit(missing, priority)
                by_text = dict(zip(missing, fresh))
                if self.cache:
                    self.cache.set_many(missing, fresh, enc.model_id, enc.version)
                vectors = [v if v is not None else by_text[t] for t, v in zip(texts, vectors)]
        if principal is not None:
            metrics.tokens.labels(SERVICE, principal.app, "-", "-", enc.model_id, "input").inc(
                sum(len(t.split()) for t in missing)
            )
            get_audit().for_principal(
                principal,
                "embedding.embed",
                action="model.invoke",
                resource=fp,
                decision="allow",
                count=len(texts),
                cache_hits=hits,
                encoded=len(missing),
                priority=priority,
            )
        return vectors  # type: ignore[return-value]

    async def _check_policy(self, principal: Principal, name: str, fp: str, priority: str) -> None:
        decision = await connect("policy").decide(
            PolicyInput(
                action="model.invoke",
                principal=principal.model_dump(),
                resource={"type": "model", "kind": "embedding", "model": name, "fingerprint": fp},
                context={"app": principal.app, "priority": priority},
            )
        )
        metrics.policy.labels(SERVICE, "model.invoke", decision.effect).inc()
        if not decision.allow:
            get_audit().for_principal(
                principal,
                "embedding.denied",
                action="model.invoke",
                resource=fp,
                decision=decision.effect,
                reasons=decision.reasons,
            )
            raise PolicyViolation(
                "embedding model not allowed by policy", details={"reasons": decision.reasons}
            )

    async def ready(self) -> dict[str, Any]:
        return {"models": {n: fingerprint(e) for n, e in self.encoders.items()}}

    async def aclose(self) -> None:
        for b in self.batchers.values():
            await b.aclose()
        for enc in self.encoders.values():
            closer = getattr(enc, "aclose", None)
            if closer is not None:
                await closer()
