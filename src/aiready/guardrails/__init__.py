"""Guardrails: screen inputs and retrieved documents, mask personal data, validate outputs.

Wiring name ``guardrails`` (``core.interfaces.Guardrails``). In-process by default; set
``services.guardrails.url`` to call the guardrails pod (port 8090).
"""

from __future__ import annotations

from typing import Any

from aiready.core.wiring import provider


@provider("guardrails", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.guardrails.service import GuardrailService

    return GuardrailService.from_config()


@provider("guardrails", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.guardrails.client import GuardrailsClient

    return GuardrailsClient(cfg["url"], timeout=float(cfg.get("timeout", 5.0)))
