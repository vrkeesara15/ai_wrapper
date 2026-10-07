"""Structured-data query service (port 8112, wiring name ``query``), exposed to agents as an MCP tool."""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


@provider("query", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.query.service import QueryServiceImpl

    return QueryServiceImpl.from_config()


@provider("query", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.query.client import QueryClient

    return QueryClient(cfg["url"], service_token=service_token(os.environ.get("AIREADY_SERVICE", "query-client")),
                       timeout=float(cfg.get("timeout", 60.0)))
