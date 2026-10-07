"""Embedding model service: vector cache, dynamic batcher (live/bulk queues), pluggable encoders.

Wiring name ``embedding``: in-process ``EmbeddingService`` or ``EmbeddingClient`` when
``services.embedding.url`` is set. Imports stay light; the service module loads on first ``connect``.
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


@provider("embedding", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.embedding.service import EmbeddingService

    return EmbeddingService()


@provider("embedding", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.embedding.client import EmbeddingClient

    return EmbeddingClient(
        cfg["url"],
        service_token=service_token(os.environ.get("AIREADY_SERVICE", "aiready")),
        timeout=float(cfg.get("timeout", 60.0)),
    )
