"""HTTP client for the tool executor (``services.tools.url`` set): same signatures as ``ToolService``."""

from __future__ import annotations

from typing import Any

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.interfaces import ToolDefinition, ToolResult
from aiready.core.wiring import service_token


class ToolsClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None,
                 service: str = "agents"):
        self._http = ServiceClient(base_url, service_token=service_token(service), timeout=timeout,
                                   transport=transport, retries=0)

    async def list_tools(self, principal: Principal) -> list[ToolDefinition]:
        data = await self._http.json("GET", "/v1/tools", principal=principal)
        return [ToolDefinition.model_validate(t) for t in data["tools"]]

    async def call(self, name: str, arguments: dict[str, Any], principal: Principal, *,
                   approval_id: str | None = None, context: dict[str, Any] | None = None) -> ToolResult:
        # writes are never retried by the transport: a duplicate POST could act twice
        r = await self._http.request("POST", f"/v1/tools/{name}:call", principal=principal, retry=False,
                                     json={"arguments": arguments, "approval_id": approval_id,
                                           "context": context or {}})
        if r.status_code in (200, 202):
            return ToolResult.model_validate(r.json())
        try:
            msg = r.json().get("error", {}).get("message", r.text[:300])
        except ValueError:
            msg = r.text[:300]
        return ToolResult(ok=False, error=f"tool service returned {r.status_code}: {msg}")

    async def aclose(self) -> None:
        await self._http.aclose()
