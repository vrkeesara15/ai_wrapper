"""Ingestion control API (port 8160).

    uvicorn aiready.ingestion.api:create_app --factory --port 8160

Trigger a run for a source, run status, document registry lookup, lineage of a chunk, dead-letter list and
replay, pipeline metrics. Callers need the ``ingestion-admin`` role (configurable) or must be a service
principal (scheduler, change notifications from another service). Every state-changing call is audited.
"""

from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Query, Request
from pydantic import BaseModel

from aiready.core.app import create_service_app
from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PermissionDenied
from aiready.core.identity import Principal, current_principal_or_service
from aiready.ingestion.pipeline import IngestionPipeline


class RunRequest(BaseModel):
    wait: bool = True  # run in the request (small sources) or in the background
    verify: bool = True


def create_app(pipeline: IngestionPipeline | None = None) -> FastAPI:
    holder: dict[str, IngestionPipeline] = {}

    def get_pipeline() -> IngestionPipeline:
        if "p" not in holder:
            holder["p"] = pipeline or IngestionPipeline()
        return holder["p"]

    async def readiness() -> dict[str, Any]:
        p = get_pipeline()
        return {"sources": len(p.cfg.sources), "queue_depth": await p.queue.depth(p.topic)}

    app = create_service_app("ingestion", title="AI-Ready ingestion", readiness=readiness)

    async def admin(request: Request) -> Principal:
        principal = await current_principal_or_service(request)
        roles = get_pipeline().cfg.admin_roles
        if not (principal.is_service_only or principal.has_role(*roles)):
            get_audit().for_principal(principal, "ingestion.api_denied", action=request.url.path, decision="deny")
            raise PermissionDenied(f"requires one of roles {roles} or a service principal")
        return principal

    @app.get("/v1/sources")
    async def sources(p: Principal = Depends(admin)) -> list[dict[str, Any]]:
        pl = get_pipeline()
        return [{"name": s.name, "connector": s.connector.get("type") or s.connector.get("class"),
                 "collection": s.collection, "schedule": s.schedule, "enabled": s.enabled,
                 "verify_questions": len(s.verify)} for s in pl.cfg.sources]

    @app.post("/v1/sources/{name}/runs")
    async def trigger(name: str, background: BackgroundTasks, body: RunRequest | None = None,
                      p: Principal = Depends(admin)) -> dict[str, Any]:
        body = body or RunRequest()
        pl = get_pipeline()
        src = pl.source(name)
        get_audit().for_principal(p, "ingestion.trigger", action="run", resource=name, decision="accepted",
                                  wait=body.wait)
        if body.wait:
            run = await pl.run_source(src.name, trigger=f"api:{p.sub}", verify=body.verify)
            return run.model_dump()
        run = await pl.extract(src.name, trigger=f"api:{p.sub}")

        async def finish() -> None:
            from aiready.ingestion.worker import Worker

            await Worker(pl).drain()
            await pl._wait_for_outcomes(run.run_id)  # other workers may still hold some of the run's documents
            await pl.finish_run(run.run_id, verify=body.verify)

        background.add_task(finish)
        return run.model_dump()

    @app.get("/v1/runs")
    async def runs(source: str | None = None, limit: int = Query(50, ge=1, le=500),
                   p: Principal = Depends(admin)) -> list[dict[str, Any]]:
        return [r.model_dump() for r in get_pipeline().registry.runs(source, limit)]

    @app.get("/v1/runs/{run_id}")
    async def run_status(run_id: str, p: Principal = Depends(admin)) -> dict[str, Any]:
        run = get_pipeline().registry.get_run(run_id)
        if run is None:
            raise NotFound(f"unknown run '{run_id}'")
        return run.model_dump()

    @app.get("/v1/documents")
    async def documents(source: str | None = None, status: str | None = None, limit: int = Query(200, ge=1, le=5000),
                        p: Principal = Depends(admin)) -> list[dict[str, Any]]:
        entries = get_pipeline().registry.entries(source)
        if status:
            entries = [e for e in entries if e.status == status]
        return [e.model_dump(exclude={"history"}) for e in entries[:limit]]

    @app.get("/v1/documents/{doc_id:path}")
    async def document(doc_id: str, source: str | None = None, p: Principal = Depends(admin)) -> dict[str, Any]:
        pl = get_pipeline()
        entry = pl.registry.get(doc_id, source)
        if entry is None:
            raise NotFound(f"document '{doc_id}' is not in the registry")
        return {**entry.model_dump(), "lineage": [r.model_dump() for r in pl.lineage.for_document(doc_id)]}

    @app.get("/v1/lineage/{chunk_id}")
    async def lineage(chunk_id: str, p: Principal = Depends(admin)) -> dict[str, Any]:
        rec = get_pipeline().lineage.get(chunk_id)
        if rec is None:
            raise NotFound(f"no lineage for chunk '{chunk_id}'")
        return rec.model_dump()

    @app.get("/v1/dlq")
    async def dlq(p: Principal = Depends(admin)) -> list[dict[str, Any]]:
        return await get_pipeline().dead_letters()

    @app.post("/v1/dlq/replay")
    async def replay(p: Principal = Depends(admin)) -> dict[str, Any]:
        get_audit().for_principal(p, "ingestion.dlq_replay_requested", action="replay", decision="accepted")
        return await get_pipeline().replay_dead_letters()

    @app.get("/v1/metrics/summary")
    async def summary(p: Principal = Depends(admin)) -> dict[str, Any]:
        return await get_pipeline().metrics_summary()

    return app
