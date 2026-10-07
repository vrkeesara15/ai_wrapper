"""HTTP client for the observability service, mirroring the in-process methods other services use:
span ingest, feedback, trace lookup and replay."""

from __future__ import annotations

from typing import Any

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.telemetry import SpanRecord
from aiready.observability.feedback import FeedbackIn, FeedbackRecord
from aiready.observability.traces import TraceSummary, TraceTree


class ObservabilityClient:
    def __init__(self, url: str, *, service_token: str | None = None, timeout: float = 15.0, transport: Any = None):
        self.http = ServiceClient(url, service_token=service_token, timeout=timeout, transport=transport)

    async def ingest(self, spans: list[SpanRecord], principal: Principal | None = None) -> dict[str, Any]:
        return await self.http.json("POST", "/v1/spans", principal=principal, json=[s.model_dump() for s in spans])

    async def record_feedback(self, fb: FeedbackIn, principal: Principal) -> FeedbackRecord:
        return FeedbackRecord.model_validate(await self.http.json("POST", "/v1/feedback", principal=principal,
                                                                  json=fb.model_dump()))

    async def get_trace(self, trace_id: str, principal: Principal) -> TraceTree:
        return TraceTree.model_validate(await self.http.json("GET", f"/v1/traces/{trace_id}", principal=principal))

    async def search_traces(self, principal: Principal, **filters: Any) -> list[TraceSummary]:
        params = {k: v for k, v in filters.items() if v is not None}
        return [TraceSummary.model_validate(x) for x in await self.http.json("GET", "/v1/traces", principal=principal,
                                                                             params=params)]

    async def replay(self, trace_id: str, principal: Principal, *, model: str | None = None) -> dict[str, Any]:
        return await self.http.json("POST", f"/v1/traces/{trace_id}/replay", principal=principal,
                                    json={"model": model} if model else {})

    async def golden_candidates(self, principal: Principal, *, mark_exported: bool = False) -> list[dict[str, Any]]:
        return await self.http.json("GET", "/v1/review/golden-candidates", principal=principal,
                                    params={"mark_exported": str(mark_exported).lower()})

    async def aclose(self) -> None:
        await self.http.aclose()
