"""Memory service: session and consented long-term memory (port 8111, wiring name ``memory``)."""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


@provider("memory", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.memory.service import MemoryServiceImpl

    return MemoryServiceImpl.from_config()


@provider("memory", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.memory.client import MemoryClient

    return MemoryClient(cfg["url"], service_token=service_token(os.environ.get("AIREADY_SERVICE", "memory-client")),
                        timeout=float(cfg.get("timeout", 10.0)))
