"""Gateway data plane HTTP API (port 8080), OpenAI-compatible.

    uvicorn aiready.gateway.api:create_app --factory --port 8080

Any OpenAI client works by pointing ``base_url`` at ``http://gateway:8080/v1`` and sending the user's AI-Ready
token as the bearer token; AI-Ready options go in the request's ``metadata`` (feature, task, data_class,
response_mode, prompt_version). A calling application may also present a gateway API key in ``X-API-Key``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aiready.core import wiring
from aiready.core.app import create_service_app
from aiready.core.errors import AIReadyError
from aiready.core.identity import Principal, current_principal, current_principal_or_service
from aiready.core.tokens import count_tokens
from aiready.core.types import (
    ChatRequest,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    Usage,
)
from aiready.core.wiring import local as _local
from aiready.gateway.keys import ClientIdentity
from aiready.gateway.service import GatewayService


class BatchSubmit(BaseModel):
    requests: list[dict[str, Any]]  # [{custom_id, body: <chat request>}] or bare chat requests
    run: str = "off_peak"            # off_peak | now
    metadata: dict[str, Any] = Field(default_factory=dict)


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


def create_app(service: GatewayService | None = None) -> FastAPI:
    svc = service or _local("gateway")
    # in-process callers (agents, RAG in a single-container deployment) share this instance
    wiring.override("gateway", svc)

    async def start_batches() -> None:
        if svc.config().batches.enabled:
            svc.batches.start()

    async def stop_batches() -> None:
        if svc._batches is not None:
            await svc.batches.stop()

    app = create_service_app("gateway", title="AI-Ready gateway", on_startup=[start_batches],
                             on_shutdown=[stop_batches], readiness=svc.readiness)
    app.state.gateway = svc

    def client_identity(request: Request) -> ClientIdentity | None:
        key = request.headers.get(svc.config().auth.api_key_header)
        return svc.keys.verify(key) if key else None

    def require_admin(p: Principal) -> None:
        p.require_role(svc.config().admin.required_role)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: ChatRequest, request: Request,
                               principal: Principal = Depends(current_principal_or_service)) -> Any:
        client = client_identity(request)
        if not req.stream:
            resp = await svc.chat(req, principal, client=client)
            return resp.model_dump(mode="json")
        agen = svc.chat_stream(req, principal, client=client).__aiter__()
        # pull the first event before answering, so limits, screening and routing errors keep their status code
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:
            first = None

        async def events() -> AsyncIterator[str]:
            try:
                if first is not None:
                    yield _sse(first)
                async for chunk in agen:
                    yield _sse(chunk)
            except AIReadyError as exc:  # failure after the stream started: report it in-band
                yield _sse({"object": "error", **exc.to_dict()})
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest, request: Request,
                         principal: Principal = Depends(current_principal_or_service)) -> EmbeddingResponse:
        texts = req.texts()
        vecs = await svc.embed(texts, principal, model=req.model, priority=req.priority,
                               client=client_identity(request), feature=request.headers.get("X-Feature"))
        n = sum(count_tokens(t) for t in texts)
        return EmbeddingResponse(model=req.model, data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vecs)],
                                 usage=Usage(prompt_tokens=n, total_tokens=n))

    @app.post("/v1/rerank")
    async def rerank(req: RerankRequest, request: Request,
                     principal: Principal = Depends(current_principal_or_service)) -> RerankResponse:
        return await svc.rerank(req.query, req.documents, principal, top_n=req.top_n, model=req.model,
                                client=client_identity(request), feature=request.headers.get("X-Feature"))

    @app.get("/v1/models")
    async def models(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
        return {"object": "list", "data": await svc.list_models(principal)}

    @app.get("/v1/usage/me")
    async def usage_me(principal: Principal = Depends(current_principal),
                       start: float | None = Query(None), end: float | None = Query(None)) -> dict[str, Any]:
        return svc.usage_for(principal, start=start, end=end)

    # ---------------------------------------------------------------- batch and async
    @app.post("/v1/batches")
    async def submit_batch(body: BatchSubmit, principal: Principal = Depends(current_principal_or_service)) -> dict[str, Any]:
        rec = await svc.batches.submit(body.requests, principal, run=body.run, metadata=body.metadata)
        return rec.summary()

    @app.get("/v1/batches")
    async def list_batches(principal: Principal = Depends(current_principal_or_service)) -> dict[str, Any]:
        return {"object": "list", "data": svc.batches.list(principal)}

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(batch_id: str, principal: Principal = Depends(current_principal_or_service)) -> dict[str, Any]:
        rec = svc.batches.get(batch_id, principal)
        return {**rec.summary(), "results": rec.results}

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str, principal: Principal = Depends(current_principal_or_service)) -> dict[str, Any]:
        return svc.batches.cancel(batch_id, principal).summary()

    # ---------------------------------------------------------------- operations (used by the admin API)
    @app.get("/v1/pools/stats")
    async def pool_stats(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
        require_admin(principal)
        return svc.pool_stats()

    @app.post("/v1/cache/flush")
    async def flush_cache(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
        require_admin(principal)
        return {"flushed": svc.flush_cache()}

    return app

