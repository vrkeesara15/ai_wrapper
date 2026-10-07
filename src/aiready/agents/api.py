"""Agent runtime API (port 8120): runs, event stream (SSE), resume/cancel, workflows and the approval inbox.

Run: ``uvicorn aiready.agents.api:create_app --factory --port 8120``. With ``agents.workers.mode: pool`` the
same process also runs the worker pool; scale workers by adding replicas - they share the queue and the
checkpoint store.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Body, Depends, FastAPI, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from aiready.agents.models import RunEvent
from aiready.agents.service import AgentService
from aiready.core.app import create_service_app
from aiready.core.identity import Principal, current_principal
from aiready.core.wiring import local as _local


class StartRunBody(BaseModel):
    workflow: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    version: str | None = None
    feature: str | None = None


class ResumeBody(BaseModel):
    input: dict[str, Any] | None = None


class DecisionBody(BaseModel):
    comment: str | None = None


def sse_events(events: AsyncIterator[RunEvent], request: Request) -> EventSourceResponse:
    async def gen() -> AsyncIterator[dict[str, Any]]:
        async for ev in events:
            if await request.is_disconnected():
                break
            item: dict[str, Any] = {"event": ev.type, "data": json.dumps(ev.model_dump(mode="json"))}
            if ev.seq is not None:
                item["id"] = str(ev.seq)
            yield item

    return EventSourceResponse(gen(), ping=15)


def create_app(service: AgentService | None = None) -> FastAPI:
    """Build the API around an in-process ``AgentService``.

    Never ``svc_()`` here: in Kubernetes ``services.agents.url`` is set for every pod, including
    this one, so wiring would return an HTTP client pointing back at this very service.
    """
    holder: dict[str, AgentService] = {}

    def svc_() -> AgentService:
        if "svc" not in holder:
            holder["svc"] = service or _local("agents")
        return holder["svc"]

    async def start_workers() -> None:
        runtime = svc_().runtime
        if runtime.cfg.workers.mode == "pool":
            await runtime.recover_stale()
            await runtime.run_workers()

    async def stop_workers() -> None:
        await svc_().aclose()

    app = create_service_app("agents", title="AI-Ready agent runtime", on_startup=[start_workers],
                             on_shutdown=[stop_workers])

    @app.post("/v1/runs", status_code=202)
    async def start_run(body: StartRunBody, principal: Principal = Depends(current_principal)) -> Any:
        view = await svc_().start_run(body.workflow, body.input, principal, session_id=body.session_id,
                                                 version=body.version, feature=body.feature)
        return view.model_dump(mode="json")

    @app.get("/v1/runs")
    async def list_runs(status: str | None = None, limit: int = Query(50, le=500),
                        principal: Principal = Depends(current_principal)) -> Any:
        runs = await svc_().list_runs(principal, status=status, limit=limit)
        return {"runs": [r.model_dump(mode="json") for r in runs]}

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, principal: Principal = Depends(current_principal)) -> Any:
        return (await svc_().get_run(run_id, principal)).model_dump(mode="json")

    @app.get("/v1/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, after: int = 0,
                         principal: Principal = Depends(current_principal)) -> EventSourceResponse:
        svc = svc_()
        await svc.get_run(run_id, principal)  # 404 before the stream opens, not inside it
        last = request.headers.get("Last-Event-ID")
        start = int(last) if last and last.isdigit() else after
        return sse_events(svc.stream_events(run_id, principal, after=start), request)

    @app.get("/v1/runs/{run_id}/history")
    async def run_history(run_id: str, principal: Principal = Depends(current_principal)) -> Any:
        events = await svc_().run_history(run_id, principal)
        return {"events": [e.model_dump(mode="json") for e in events]}

    @app.post("/v1/runs/{run_id}/resume", status_code=202)
    async def resume_run(run_id: str, body: ResumeBody = Body(default_factory=ResumeBody),
                         principal: Principal = Depends(current_principal)) -> Any:
        return (await svc_().resume_run(run_id, principal, input=body.input)).model_dump(mode="json")

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, principal: Principal = Depends(current_principal)) -> Any:
        return (await svc_().cancel_run(run_id, principal)).model_dump(mode="json")

    @app.get("/v1/workflows")
    async def workflows(principal: Principal = Depends(current_principal)) -> Any:
        return {"workflows": await svc_().list_workflows()}

    # ------------------------------------------------------------------ approval inbox
    @app.get("/v1/approvals")
    async def inbox(principal: Principal = Depends(current_principal)) -> Any:
        items = await svc_().list_approvals(principal)
        return {"approvals": [a.model_dump(mode="json") for a in items]}

    @app.get("/v1/approvals/mine")
    async def mine(status: str | None = None, principal: Principal = Depends(current_principal)) -> Any:
        items = await svc_().list_my_approvals(principal, status=status)
        return {"approvals": [a.model_dump(mode="json") for a in items]}

    @app.get("/v1/approvals/{approval_id}")
    async def get_approval(approval_id: str, principal: Principal = Depends(current_principal)) -> Any:
        return (await svc_().get_approval(approval_id, principal)).model_dump(mode="json")

    @app.post("/v1/approvals/{approval_id}/approve")
    async def approve(approval_id: str, body: DecisionBody = Body(default_factory=DecisionBody),
                      principal: Principal = Depends(current_principal)) -> Any:
        req = await svc_().decide_approval(approval_id, principal, approve=True, comment=body.comment)
        return req.model_dump(mode="json")

    @app.post("/v1/approvals/{approval_id}/reject")
    async def reject(approval_id: str, body: DecisionBody = Body(default_factory=DecisionBody),
                     principal: Principal = Depends(current_principal)) -> Any:
        req = await svc_().decide_approval(approval_id, principal, approve=False, comment=body.comment)
        return req.model_dump(mode="json")

    return app
