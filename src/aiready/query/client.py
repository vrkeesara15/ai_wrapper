"""HTTP client for the query service - same method signature as ``QueryServiceImpl.query``."""

from __future__ import annotations

from typing import Any

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.interfaces import QueryRequest, QueryResult


class QueryClient:
    def __init__(self, base_url: str, *, service_token: str | None = None, timeout: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.http = ServiceClient(base_url, service_token=service_token, timeout=timeout, retries=0,
                                  transport=transport)

    async def query(self, req: QueryRequest, principal: Principal) -> QueryResult:
        data = await self.http.json("POST", "/v1/query", principal=principal, json=req.model_dump(mode="json"))
        return QueryResult.model_validate(data)

    async def list_metrics(self, principal: Principal) -> list[dict[str, Any]]:
        return await self.http.json("GET", "/v1/query/metrics", principal=principal)

    async def describe_catalog(self, principal: Principal) -> str:
        data = await self.http.json("GET", "/v1/query/catalog", principal=principal)
        return (data or {}).get("catalog", "")

    async def aclose(self) -> None:
        await self.http.aclose()
