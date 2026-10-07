"""Python client for the assistant API: chat turns as parsed SSE events (scripts, tests, server-side hosts)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal


class AssistantClient:
    def __init__(self, base_url: str, *, app: str | None = None, timeout: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.app = app
        self._http = ServiceClient(base_url, timeout=timeout, transport=transport, retries=0)

    async def chat(self, message: str, principal: Principal, *, history: list[dict[str, Any]] | None = None,
                   feature: str | None = None, session_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        body = {"message": message, "history": history or [], "app": self.app, "feature": feature,
                "session_id": session_id}
        async with self._http.stream("POST", "/v1/assistant/chat", principal=principal, json=body) as r:
            if r.status_code >= 400:
                await r.aread()
                raise RuntimeError(f"assistant returned {r.status_code}: {r.text[:300]}")
            event, data = "message", []
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif line == "" and data:
                    yield {"event": event, "data": json.loads("\n".join(data))}
                    event, data = "message", []

    async def ask(self, message: str, principal: Principal, **kw: Any) -> dict[str, Any]:
        """Convenience: run a turn to its end and return the final ``done`` (or ``approval``) event data."""
        last: dict[str, Any] = {}
        async for ev in self.chat(message, principal, **kw):
            if ev["event"] in ("done", "approval", "waiting", "error"):
                last = {"event": ev["event"], **ev["data"]}
        return last

    async def feedback(self, principal: Principal, **fields: Any) -> dict[str, Any]:
        return await self._http.json("POST", "/v1/assistant/feedback", principal=principal,
                                     json={"app": self.app, **fields}, retry=False)

    async def aclose(self) -> None:
        await self._http.aclose()
