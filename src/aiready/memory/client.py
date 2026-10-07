"""HTTP client for the memory service - same method signatures as ``MemoryServiceImpl``."""

from __future__ import annotations

from typing import Any

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.interfaces import MemoryItem


class MemoryClient:
    def __init__(self, base_url: str, *, service_token: str | None = None, timeout: float = 10.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.http = ServiceClient(base_url, service_token=service_token, timeout=timeout, transport=transport)

    async def append_turn(self, session_id: str, principal: Principal, role: str, content: str) -> None:
        await self.http.json("POST", f"/v1/memory/sessions/{session_id}/turns", principal=principal,
                             json={"role": role, "content": content})

    async def session_context(self, session_id: str, principal: Principal, max_tokens: int = 1500) -> list[dict[str, Any]]:
        return await self.http.json("GET", f"/v1/memory/sessions/{session_id}/context", principal=principal,
                                    params={"max_tokens": max_tokens})

    async def remember(self, principal: Principal, text: str, *, kind: str = "fact", ttl_days: int | None = None,
                       consent: bool = False) -> MemoryItem:
        data = await self.http.json("POST", "/v1/memory/items", principal=principal,
                                    json={"text": text, "kind": kind, "ttl_days": ttl_days, "consent": consent})
        return MemoryItem.model_validate(data)

    async def recall(self, principal: Principal, query: str | None = None, limit: int = 10) -> list[MemoryItem]:
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        data = await self.http.json("GET", "/v1/memory/items", principal=principal, params=params)
        return [MemoryItem.model_validate(d) for d in data or []]

    async def forget(self, principal: Principal, item_id: str | None = None) -> int:
        path = f"/v1/memory/items/{item_id}" if item_id else "/v1/memory/items"
        data = await self.http.json("DELETE", path, principal=principal)
        return int((data or {}).get("deleted", 0))

    async def purge_expired(self) -> int:
        data = await self.http.json("POST", "/v1/memory/purge")
        return int((data or {}).get("purged", 0))

    async def aclose(self) -> None:
        await self.http.aclose()
