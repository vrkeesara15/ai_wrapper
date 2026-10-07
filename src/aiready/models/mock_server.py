"""OpenAI-compatible mock model server (port 8200, development only).

Lets the whole platform - and any OpenAI client - run without GPUs or network: point a pool replica at
``http://localhost:8200/v1`` with ``type: openai`` and every code path of the real backend (HTTP, SSE
streaming, embeddings, rerank) is exercised against deterministic answers.

    uvicorn aiready.models.mock_server:create_app --factory --port 8200

Environment: ``AIREADY_MOCK_LATENCY_MS`` (simulated model time), ``AIREADY_MOCK_MODELS`` (comma-separated
names listed by ``/v1/models``). No authentication: never expose it outside a dev network.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from aiready.core.app import create_service_app
from aiready.core.tokens import count_tokens
from aiready.core.types import (
    ChatRequest,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)
from aiready.models.backends import MockBackend


def create_app(backend: MockBackend | None = None) -> FastAPI:
    mock = backend or MockBackend(latency_ms=float(os.environ.get("AIREADY_MOCK_LATENCY_MS", "0")))
    names = [n.strip() for n in os.environ.get(
        "AIREADY_MOCK_MODELS", "mock-small,mock-mid,mock-large,mock-embedding,mock-reranker").split(",") if n.strip()]
    app = create_service_app("mock-model", title="AI-Ready mock model server")

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(req: ChatRequest) -> Any:
        if req.stream:
            async def events() -> AsyncIterator[str]:
                async for chunk in mock.chat_stream(req):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        resp = await mock.chat(req)
        return resp.model_dump(exclude={"aiready"})

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
        texts = req.texts()
        vecs = await mock.embed(texts, req.model)
        n = sum(count_tokens(t) for t in texts)
        return EmbeddingResponse(model=req.model, data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vecs)],
                                 usage=Usage(prompt_tokens=n, total_tokens=n))

    @app.post("/v1/rerank")
    @app.post("/rerank")
    async def rerank(req: RerankRequest) -> RerankResponse:
        scores = await mock.rerank(req.query, req.documents, req.model)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        if req.top_n:
            ranked = ranked[: req.top_n]
        return RerankResponse(model=req.model, results=[RerankResult(index=i, relevance_score=s) for i, s in ranked])

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        now = int(time.time())
        return {"object": "list", "data": [{"id": n, "object": "model", "created": now, "owned_by": "aiready-mock"}
                                           for n in names]}

    return app
