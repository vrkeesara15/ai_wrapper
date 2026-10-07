"""HTTP API of the reranking service (port 8101).

* ``POST /v1/rerank`` - Cohere-style request/response (``core.types.RerankRequest``/``RerankResponse``);
  also accepts Cohere's ``{"text": ...}`` document objects and the extensions ``threshold`` and ``app``.
* ``POST /v1/rerank/hits`` - takes ``SearchHit`` lists from the search service and returns them reranked,
  with ``fallback`` set when the search order was kept.

    uvicorn aiready.rerank.api:create_app --factory --port 8101
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI

from aiready.core.app import create_service_app
from aiready.core.identity import Principal, current_principal_or_service
from aiready.core.wiring import local as _local
from aiready.rerank.service import RerankApiRequest, RerankHitsRequest, RerankOutcome, RerankService


def create_app(service: RerankService | None = None) -> FastAPI:
    holder: dict[str, RerankService] = {}

    def svc() -> RerankService:
        if "svc" not in holder:
            holder["svc"] = service or _local("rerank")
        return holder["svc"]

    async def shutdown() -> None:
        if "svc" in holder:
            await holder["svc"].aclose()

    app = create_service_app("rerank", on_shutdown=[shutdown], readiness=lambda: svc().ready())

    @app.post("/v1/rerank")
    async def rerank(
        body: RerankApiRequest, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        texts = body.texts()
        resp = await svc().rerank(
            body.query,
            texts,
            top_n=body.top_n,
            model=body.model,
            threshold=body.threshold,
            principal=principal,
            app=body.app,
        )
        out = resp.model_dump()
        if body.return_documents:
            for r in out["results"]:
                r["document"] = {"text": texts[r["index"]]}
        return out

    @app.post("/v1/rerank/hits", response_model=RerankOutcome)
    async def rerank_hits(
        body: RerankHitsRequest, principal: Principal = Depends(current_principal_or_service)
    ) -> RerankOutcome:
        return await svc().rerank_hits(
            body.query,
            body.hits,
            top_n=body.top_n,
            threshold=body.threshold,
            max_candidates=body.max_candidates,
            model=body.model,
            principal=principal,
            app=body.app,
        )

    return app
