"""Memory service API (port 8111).

    uvicorn aiready.memory.api:create_app --factory --port 8111

Every route acts on the caller's own memory only (tenant + user from the verified token).

* ``POST   /v1/memory/sessions/{session_id}/turns``    append a turn
* ``GET    /v1/memory/sessions/{session_id}/context``  summary + recent turns (``?max_tokens=``)
* ``POST   /v1/memory/items``                          remember (requires ``consent: true``)
* ``GET    /v1/memory/items``                          recall (``?query=&limit=``)
* ``DELETE /v1/memory/items/{item_id}``                forget one item
* ``DELETE /v1/memory/items``                          forget everything (right to erasure)
* ``POST   /v1/memory/purge``                          drop expired items (service token or ``admin`` role)
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel

from aiready.core.app import create_service_app
from aiready.core.errors import PermissionDenied
from aiready.core.identity import Principal, current_principal, current_principal_or_service
from aiready.core.interfaces import MemoryItem
from aiready.core.wiring import local as _local


class TurnIn(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class RememberIn(BaseModel):
    text: str
    kind: str = "fact"
    ttl_days: int | None = None
    consent: bool = False


def create_app(service: Any | None = None) -> FastAPI:
    holder: dict[str, Any] = {"svc": service}

    def svc() -> Any:
        if holder["svc"] is None:

            holder["svc"] = _local("memory")
        return holder["svc"]

    app = create_service_app("memory", title="AI-Ready memory service")

    @app.post("/v1/memory/sessions/{session_id}/turns", status_code=204)
    async def append_turn(session_id: str, body: TurnIn, principal: Principal = Depends(current_principal)) -> None:
        await svc().append_turn(session_id, principal, body.role, body.content)

    @app.get("/v1/memory/sessions/{session_id}/context")
    async def session_context(session_id: str, max_tokens: int = Query(1500, ge=1, le=100_000),
                              principal: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        return await svc().session_context(session_id, principal, max_tokens=max_tokens)

    @app.post("/v1/memory/items", response_model=MemoryItem)
    async def remember(body: RememberIn, principal: Principal = Depends(current_principal)) -> MemoryItem:
        return await svc().remember(principal, body.text, kind=body.kind, ttl_days=body.ttl_days,
                                    consent=body.consent)

    @app.get("/v1/memory/items", response_model=list[MemoryItem])
    async def recall(query: str | None = None, limit: int = Query(10, ge=1, le=200),
                     principal: Principal = Depends(current_principal)) -> list[MemoryItem]:
        return await svc().recall(principal, query, limit)

    @app.delete("/v1/memory/items/{item_id}")
    async def forget_one(item_id: str, principal: Principal = Depends(current_principal)) -> dict[str, int]:
        return {"deleted": await svc().forget(principal, item_id)}

    @app.delete("/v1/memory/items")
    async def forget_all(principal: Principal = Depends(current_principal)) -> dict[str, int]:
        return {"deleted": await svc().forget(principal, None)}

    @app.post("/v1/memory/purge")
    async def purge(principal: Principal = Depends(current_principal_or_service)) -> dict[str, int]:
        if not (principal.is_service_only or principal.has_role("admin")):
            raise PermissionDenied("purge requires a service token or the admin role")
        return {"purged": await svc().purge_expired()}

    return app
