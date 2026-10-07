"""HTTP client for the RAG service - same method signatures as ``RagServiceImpl``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from aiready.core.errors import UpstreamError
from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.interfaces import RagAnswer, RagQuery
from aiready.core.types import SearchHit


class RagClient:
    def __init__(self, base_url: str, *, service_token: str | None = None, timeout: float = 60.0,
                 retries: int = 1, transport: httpx.AsyncBaseTransport | None = None):
        self.http = ServiceClient(base_url, service_token=service_token, timeout=timeout, retries=retries,
                                  transport=transport)

    async def answer(self, q: RagQuery, principal: Principal) -> RagAnswer:
        data = await self.http.json("POST", "/v1/rag/answer", principal=principal, json=q.model_dump(mode="json"))
        return RagAnswer.model_validate(data)

    async def retrieve(self, q: RagQuery, principal: Principal) -> list[SearchHit]:
        data = await self.http.json("POST", "/v1/rag/retrieve", principal=principal, json=q.model_dump(mode="json"))
        return [SearchHit.model_validate(h) for h in data or []]

    async def answer_stream(self, q: RagQuery, principal: Principal) -> AsyncIterator[dict[str, Any]]:
        """Server-sent events decoded to the same event dicts the in-process service yields."""
        async with self.http.stream("POST", "/v1/rag/answer/stream", principal=principal,
                                    json=q.model_dump(mode="json")) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")[:500]
                err = UpstreamError(f"rag stream returned {r.status_code}", details={"body": body})
                err.status_code = r.status_code if r.status_code < 500 else 502
                raise err
            data: list[str] = []
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line.strip() and data:
                    yield json.loads("\n".join(data))
                    data = []
            if data:
                yield json.loads("\n".join(data))

    async def aclose(self) -> None:
        await self.http.aclose()
