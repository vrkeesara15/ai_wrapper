"""Agent runtime (wiring name ``agents``, port 8120): versioned workflows, stateless workers, checkpoints,
limits, approval gate and approval inbox."""

from __future__ import annotations

from typing import Any

from aiready.core.wiring import provider


@provider("agents", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.agents.service import AgentService

    return AgentService()


@provider("agents", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.agents.client import AgentsClient

    return AgentsClient(cfg["url"], timeout=float(cfg.get("timeout", 60.0)))
