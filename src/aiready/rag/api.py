"""RAG service API (port 8110).

    uvicorn aiready.rag.api:create_app --factory --port 8110

* ``POST /v1/rag/answer``         Mode A, check then send -> ``RagAnswer`` (JSON)
* ``POST /v1/rag/answer/stream``  Mode B, stream then verify -> server-sent events
                                  (``start``, ``token``..., ``verdict``, ``done``; ``error`` if it fails mid-stream)
* ``POST /v1/rag/retrieve``       access-filtered, reranked passages only -> ``list[SearchHit]``
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI
from sse_starlette.sse import EventSourceResponse

from aiready.core.app import create_service_app
from aiready.core.errors import AIReadyError
from aiready.core.identity import Principal, current_principal
from aiready.core.interfaces import RagAnswer, RagQuery
from aiready.core.types import SearchHit

log = logging.getLogger(__name__)


def create_app(service: Any | None = None) -> FastAPI:
    """``service`` defaults to the in-process ``RagServiceImpl`` built from configuration."""
    holder: dict[str, Any] = {"svc": service}

    def svc() -> Any:
        if holder["svc"] is None:
            from aiready.rag.service import RagServiceImpl

            holder["svc"] = RagServiceImpl.from_config()
        return holder["svc"]

    app = create_service_app("rag", title="AI-Ready RAG service")

    @app.post("/v1/rag/answer", response_model=RagAnswer)
    async def answer(q: RagQuery, principal: Principal = Depends(current_principal)) -> RagAnswer:
        return await svc().answer(q, principal)

    @app.post("/v1/rag/answer/stream")
    async def answer_stream(q: RagQuery, principal: Principal = Depends(current_principal)) -> EventSourceResponse:
        events = svc().answer_stream(q, principal).__aiter__()
        # Pull the first event before the response starts, so screening/policy errors become proper HTTP errors.
        first = await events.__anext__()

        async def gen() -> AsyncIterator[dict[str, str]]:
            yield {"event": first["event"], "data": json.dumps(first, default=str)}
            try:
                async for ev in events:
                    yield {"event": ev["event"], "data": json.dumps(ev, default=str)}
            except AIReadyError as exc:
                log.warning("rag stream failed: %s", exc)
                yield {"event": "error", "data": json.dumps({"event": "error", **exc.to_dict()})}

        return EventSourceResponse(gen())

    @app.post("/v1/rag/retrieve", response_model=list[SearchHit])
    async def retrieve(q: RagQuery, principal: Principal = Depends(current_principal)) -> list[SearchHit]:
        return await svc().retrieve(q, principal)

    return app
