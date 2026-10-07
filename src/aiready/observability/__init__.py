"""Observability: trace store and viewer with replay, cost and chargeback reports, latency against the
budget, live scorers, review queue, feedback capture, alerts and dashboards.

Wiring name ``observability`` (port 8170). Imports stay light: the service is built on first ``connect``.
"""

from __future__ import annotations

from typing import Any

from aiready.core.wiring import provider, service_token


@provider("observability", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.observability.service import ObservabilityService

    return ObservabilityService.from_config()


@provider("observability", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.observability.client import ObservabilityClient

    return ObservabilityClient(cfg["url"], service_token=service_token("observability"),
                               timeout=float(cfg.get("timeout", 15.0)))
