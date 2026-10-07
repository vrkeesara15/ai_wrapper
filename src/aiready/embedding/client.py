"""HTTP client for the embedding service, same method signatures as ``EmbeddingService``."""

from __future__ import annotations

from typing import Any

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal


class EmbeddingClient:
    def __init__(
        self,
        url: str,
        *,
        service_token: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.http = ServiceClient(url, service_token=service_token, timeout=timeout, transport=transport)

    async def embed(
        self,
        texts: list[str],
        *,
        priority: str = "live",
        model: str | None = None,
        principal: Principal | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        body = {"input": texts, "model": model or "embedding", "priority": priority}
        # embeddings are idempotent: retry transient failures
        data = await self.http.json("POST", "/v1/embeddings", principal=principal, json=body, retry=True)
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    async def info(self, model: str | None = None, principal: Principal | None = None) -> dict[str, Any]:
        params = {"model": model} if model else None
        return await self.http.json("GET", "/v1/embeddings/info", params=params, principal=principal)

    async def fingerprint(self, model: str | None = None) -> str:
        info = await self.info(model)
        return info["models"][0]["fingerprint"]

    async def aclose(self) -> None:
        await self.http.aclose()
