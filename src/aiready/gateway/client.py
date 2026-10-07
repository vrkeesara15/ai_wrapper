"""HTTP client for the gateway with the same method signatures as ``GatewayService``.

Returned by ``connect("gateway")`` when ``services.gateway.url`` is configured. The user's token, this
service's own token and the trace id travel with every call.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from aiready.core.errors import AIReadyError, UpstreamError
from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.types import ChatRequest, ChatResponse, RerankResponse


class GatewayClient:
    def __init__(self, base_url: str, *, service_token: str | None = None, timeout: float = 120.0,
                 retries: int = 1, api_key: str | None = None, transport: Any = None):
        self.http = ServiceClient(base_url, service_token=service_token, timeout=timeout, retries=retries,
                                  transport=transport)
        self.extra = {"X-API-Key": api_key} if api_key else None

    async def chat(self, req: ChatRequest, principal: Principal, **_: Any) -> ChatResponse:
        body = req.model_dump(mode="json", exclude_none=True)
        body["stream"] = False
        data = await self.http.json("POST", "/v1/chat/completions", principal=principal, json=body, headers=self.extra)
        return ChatResponse.model_validate(data)

    async def chat_stream(self, req: ChatRequest, principal: Principal, **_: Any) -> AsyncIterator[dict[str, Any]]:
        body = req.model_dump(mode="json", exclude_none=True)
        body["stream"] = True
        async with self.http.stream("POST", "/v1/chat/completions", principal=principal, json=body) as r:
            if r.status_code >= 400:
                await r.aread()
                try:
                    detail = r.json().get("error", {})
                except ValueError:
                    detail = {"message": r.text[:300]}
                err = UpstreamError(detail.get("message", f"gateway returned {r.status_code}"), details=detail)
                err.status_code, err.code = r.status_code, detail.get("code", err.code)
                raise err
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("object") == "error":
                    e = obj.get("error", {})
                    err2 = AIReadyError(e.get("message", "gateway stream error"), details=e.get("details"))
                    err2.code = e.get("code", "internal_error")
                    raise err2
                yield obj

    async def embed(self, texts: list[str], principal: Principal, *, model: str = "embedding",
                    priority: str = "live", **_: Any) -> list[list[float]]:
        data = await self.http.json("POST", "/v1/embeddings", principal=principal, headers=self.extra,
                                    json={"model": model, "input": texts, "priority": priority})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    async def rerank(self, query: str, documents: list[str], principal: Principal, *, top_n: int | None = None,
                     model: str = "reranker", **_: Any) -> RerankResponse:
        data = await self.http.json("POST", "/v1/rerank", principal=principal, headers=self.extra,
                                    json={"query": query, "documents": documents, "top_n": top_n, "model": model})
        return RerankResponse.model_validate(data)

    async def list_models(self, principal: Principal) -> list[dict[str, Any]]:
        return (await self.http.json("GET", "/v1/models", principal=principal))["data"]

    async def usage_for(self, principal: Principal) -> dict[str, Any]:
        return await self.http.json("GET", "/v1/usage/me", principal=principal)

    async def pool_stats(self, principal: Principal | None = None) -> dict[str, Any]:
        return await self.http.json("GET", "/v1/pools/stats", principal=principal)

    async def flush_cache(self, principal: Principal | None = None) -> int:
        return (await self.http.json("POST", "/v1/cache/flush", principal=principal))["flushed"]

    async def aclose(self) -> None:
        await self.http.aclose()
