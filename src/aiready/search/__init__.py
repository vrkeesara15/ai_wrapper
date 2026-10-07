"""Vector + keyword search service: filter-first hybrid retrieval with rank fusion.

Wiring name ``search``: in-process ``SearchService`` or ``SearchClient`` when ``services.search.url`` is set.
``collection_name(app, tenant, base)`` builds the per-application, per-tenant collection name.
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


def collection_name(app: str, tenant: str, base: str = "knowledge") -> str:
    from aiready.search.service import collection_name as _cn

    return _cn(app, tenant, base)


@provider("search", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.search.service import SearchService

    return SearchService()


@provider("search", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.search.client import SearchClient

    return SearchClient(
        cfg["url"],
        service_token=service_token(os.environ.get("AIREADY_SERVICE", "aiready")),
        timeout=float(cfg.get("timeout", 30.0)),
    )
