"""HTTP client for the reranking service, same method signatures as ``RerankService``."""

from __future__ import annotations

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.types import RerankResponse, SearchHit
from aiready.rerank.service import RerankOutcome


class RerankClient:
    def __init__(
        self,
        url: str,
        *,
        service_token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.http = ServiceClient(
            url, service_token=service_token, timeout=timeout, retries=1, transport=transport
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        model: str | None = None,
        threshold: float | None = None,
        principal: Principal | None = None,
        app: str | None = None,
    ) -> RerankResponse:
        body = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "model": model,
            "threshold": threshold,
            "app": app,
        }
        return RerankResponse.model_validate(
            await self.http.json("POST", "/v1/rerank", principal=principal, json=body, retry=True)
        )

    async def rerank_hits(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_n: int | None = None,
        threshold: float | None = None,
        max_candidates: int | None = None,
        model: str | None = None,
        principal: Principal | None = None,
        app: str | None = None,
    ) -> RerankOutcome:
        body = {
            "query": query,
            "hits": [h.model_dump(mode="json") for h in hits],
            "top_n": top_n,
            "threshold": threshold,
            "max_candidates": max_candidates,
            "model": model,
            "app": app,
        }
        return RerankOutcome.model_validate(
            await self.http.json("POST", "/v1/rerank/hits", principal=principal, json=body, retry=True)
        )

    async def aclose(self) -> None:
        await self.http.aclose()
