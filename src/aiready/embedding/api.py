"""HTTP API of the embedding service (port 8100).

``POST /v1/embeddings`` is OpenAI-compatible, so the gateway's embedding pool (or any OpenAI client) can
point at it unchanged. The AI-Ready extension ``priority`` (body field or ``X-Priority`` header) selects the
live or bulk queue. ``GET /v1/embeddings/info`` returns each model's fingerprint for index bookkeeping.

    uvicorn aiready.embedding.api:create_app --factory --port 8100
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Request

from aiready.core.app import create_service_app
from aiready.core.identity import Principal, current_principal_or_service
from aiready.core.types import EmbeddingData, EmbeddingRequest, EmbeddingResponse, Usage
from aiready.core.wiring import local as _local
from aiready.embedding.service import EmbeddingService


def create_app(service: EmbeddingService | None = None) -> FastAPI:
    holder: dict[str, EmbeddingService] = {}

    def svc() -> EmbeddingService:
        if "svc" not in holder:
            holder["svc"] = service or _local("embedding")
        return holder["svc"]

    async def shutdown() -> None:
        if "svc" in holder:
            await holder["svc"].aclose()

    app = create_service_app("embedding", on_shutdown=[shutdown], readiness=lambda: svc().ready())

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def embeddings(
        body: EmbeddingRequest, request: Request, principal: Principal = Depends(current_principal_or_service)
    ) -> EmbeddingResponse:
        priority = request.headers.get("X-Priority", body.priority)
        texts = body.texts()
        s = svc()
        _, enc = s.resolve(body.model)
        vectors = await s.embed(texts, priority=priority, model=body.model, principal=principal)  # type: ignore[arg-type]
        n_tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResponse(
            data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)],
            model=enc.model_id,
            usage=Usage(prompt_tokens=n_tokens, total_tokens=n_tokens),
        )

    @app.get("/v1/embeddings/info")
    async def info(
        model: str | None = None, principal: Principal = Depends(current_principal_or_service)
    ) -> dict[str, Any]:
        return await svc().info(model)

    return app
