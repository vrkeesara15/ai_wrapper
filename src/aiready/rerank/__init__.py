"""Reranking service: cross-encoder scoring of the search shortlist, with search-order fallback.

Wiring name ``rerank``: in-process ``RerankService`` or ``RerankClient`` when ``services.rerank.url`` is set.
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


@provider("rerank", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.rerank.service import RerankService

    return RerankService()


@provider("rerank", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.rerank.client import RerankClient

    return RerankClient(
        cfg["url"],
        service_token=service_token(os.environ.get("AIREADY_SERVICE", "aiready")),
        timeout=float(cfg.get("timeout", 10.0)),
    )
