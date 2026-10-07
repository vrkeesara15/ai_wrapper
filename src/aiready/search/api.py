"""HTTP API of the search service (port 8102).

POST /v1/collections/{c}/search     user principal; access filtering uses the caller's groups
POST /v1/collections/{c}/upsert     ingestion only (role service or ingestion)
POST /v1/collections/{c}/delete     ingestion only
GET  /v1/collections/{c}/stats      any authenticated caller of the collection's tenant
POST /v1/collections/{c}/snapshot   ingestion / admin
GET  /v1/collections/{c}/snapshots  ingestion / admin
POST /v1/collections/{c}/restore    ingestion / admin

uvicorn aiready.search.api:create_app --factory --port 8102
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from aiready.core.app import create_service_app
from aiready.core.identity import Principal, current_principal, current_principal_or_service
from aiready.core.types import Chunk, SearchHit, SearchRequest
from aiready.core.wiring import local as _local
from aiready.search.service import SearchService

WRITE_ROLES = ("service", "ingestion")
OPS_ROLES = ("service", "ingestion", "admin")


class SearchBody(BaseModel):
    query_text: str = ""
    query_vector: list[float] | None = None
    access_groups: list[str] = Field(default_factory=list)  # may only narrow the caller's own groups
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 50


class SearchResult(BaseModel):
    collection: str
    hits: list[SearchHit]


class UpsertBody(BaseModel):
    chunks: list[Chunk]


class DeleteBody(BaseModel):
    doc_ids: list[str] | None = None
    chunk_ids: list[str] | None = None


class RestoreBody(BaseModel):
    snapshot_id: str


def create_app(service: SearchService | None = None) -> FastAPI:
    holder: dict[str, SearchService] = {}

    def svc() -> SearchService:
        if "svc" not in holder:
            holder["svc"] = service or _local("search")
        return holder["svc"]

    app = create_service_app("search", readiness=lambda: svc().ready())

    @app.post("/v1/collections/{collection}/search", response_model=SearchResult)
    async def search(
        collection: str, body: SearchBody, principal: Principal = Depends(current_principal)
    ) -> SearchResult:
        req = SearchRequest(collection=collection, **body.model_dump())
        return SearchResult(collection=collection, hits=await svc().search(req, principal))

    @app.post("/v1/collections/{collection}/upsert")
    async def upsert(
        collection: str, body: UpsertBody, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        principal.require_role(*WRITE_ROLES)
        return {
            "collection": collection,
            "upserted": await svc().upsert(collection, body.chunks, principal=principal),
        }

    @app.post("/v1/collections/{collection}/delete")
    async def delete(
        collection: str, body: DeleteBody, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        principal.require_role(*WRITE_ROLES)
        n = await svc().delete(
            collection, doc_ids=body.doc_ids, chunk_ids=body.chunk_ids, principal=principal
        )
        return {"collection": collection, "deleted": n}

    @app.get("/v1/collections/{collection}/stats")
    async def stats(
        collection: str, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        svc()._check_tenant(collection, principal)
        return await svc().stats(collection)

    @app.post("/v1/collections/{collection}/snapshot")
    async def snapshot(
        collection: str, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        principal.require_role(*OPS_ROLES)
        return await svc().snapshot(collection, principal=principal)

    @app.get("/v1/collections/{collection}/snapshots")
    async def snapshots(
        collection: str, principal: Principal = Depends(current_principal_or_service)
    ) -> list[dict[str, Any]]:
        principal.require_role(*OPS_ROLES)
        return await svc().list_snapshots(collection)

    @app.post("/v1/collections/{collection}/restore")
    async def restore(
        collection: str, body: RestoreBody, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        principal.require_role(*OPS_ROLES)
        return await svc().restore(collection, body.snapshot_id, principal=principal)

    return app
