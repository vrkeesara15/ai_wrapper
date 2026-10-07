"""HTTP client for the search service, same method signatures as ``SearchService``.

Write calls (upsert/delete/snapshot/restore) authenticate with this process's service token; the search
service requires the ``service`` or ``ingestion`` role for them.
"""

from __future__ import annotations

from typing import Any

import httpx

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.types import Chunk, SearchHit, SearchRequest


class SearchClient:
    def __init__(
        self,
        url: str,
        *,
        service_token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.http = ServiceClient(url, service_token=service_token, timeout=timeout, transport=transport)

    @staticmethod
    def _path(collection: str, op: str) -> str:
        return f"/v1/collections/{collection}/{op}"

    async def search(self, req: SearchRequest, principal: Principal) -> list[SearchHit]:
        body = req.model_dump(exclude={"collection"})
        data = await self.http.json(
            "POST", self._path(req.collection, "search"), principal=principal, json=body, retry=True
        )
        return [SearchHit.model_validate(h) for h in data["hits"]]

    async def upsert(
        self, collection: str, chunks: list[Chunk], *, principal: Principal | None = None
    ) -> int:
        body = {"chunks": [c.model_dump(mode="json") for c in chunks]}
        data = await self.http.json(
            "POST", self._path(collection, "upsert"), principal=principal, json=body, retry=True
        )  # idempotent by chunk id
        return int(data["upserted"])

    async def delete(
        self,
        collection: str,
        *,
        doc_ids: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        principal: Principal | None = None,
    ) -> int:
        data = await self.http.json(
            "POST",
            self._path(collection, "delete"),
            principal=principal,
            json={"doc_ids": doc_ids, "chunk_ids": chunk_ids},
            retry=True,
        )
        return int(data["deleted"])

    async def stats(self, collection: str) -> dict[str, Any]:
        return await self.http.json("GET", self._path(collection, "stats"))

    async def snapshot(self, collection: str, *, principal: Principal | None = None) -> dict[str, Any]:
        return await self.http.json("POST", self._path(collection, "snapshot"), principal=principal)

    async def list_snapshots(self, collection: str) -> list[dict[str, Any]]:
        return await self.http.json("GET", self._path(collection, "snapshots"))

    async def restore(
        self, collection: str, snapshot_id: str, *, principal: Principal | None = None
    ) -> dict[str, Any]:
        return await self.http.json(
            "POST", self._path(collection, "restore"), principal=principal, json={"snapshot_id": snapshot_id}
        )

    async def aclose(self) -> None:
        await self.http.aclose()
