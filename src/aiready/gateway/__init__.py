"""AI gateway: the only path to a model (authenticate, limit, screen, reuse, route, call, validate, record).

``connect("gateway")`` returns the in-process ``GatewayService`` or, when ``services.gateway.url`` is set, the
HTTP ``GatewayClient``. Imports are kept light: the service module loads on first use.
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


@provider("gateway", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.gateway.service import GatewayService

    return GatewayService()


@provider("gateway", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.gateway.client import GatewayClient

    return GatewayClient(cfg["url"], service_token=cfg.get("service_token") or service_token(
                             cfg.get("caller") or os.environ.get("AIREADY_SERVICE", "aiready")),
                         timeout=float(cfg.get("timeout", 120)), retries=int(cfg.get("retries", 1)),
                         api_key=cfg.get("api_key"))
