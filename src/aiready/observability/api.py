"""HTTP API of the observability service (port 8170).

    uvicorn aiready.observability.api:create_app --factory --port 8170
"""

from __future__ import annotations
from aiready.core.wiring import local as _local

from typing import Any

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from aiready.core.app import create_service_app
from aiready.core.errors import AuthenticationError
from aiready.core.identity import Principal, current_principal, principal_from_headers
from aiready.core.telemetry import SpanRecord
from aiready.observability.dashboard import render_dashboard
from aiready.observability.exports import grafana_dashboard, prometheus_rules
from aiready.observability.feedback import FeedbackIn
from aiready.observability.review import ReviewItem, ReviewLabel
from aiready.observability.scoring import AnswerSample
from aiready.observability.service import ObservabilityService


def create_app(service: ObservabilityService | None = None) -> FastAPI:
    svc = service or _local("observability")
    app = create_service_app("observability", on_startup=[svc.start], on_shutdown=[svc.stop],
                             readiness=lambda: {"review_queue": svc.review.size()})
    app.state.service = svc

    async def ingest_principal(request: Request) -> Principal | None:
        """Span exporters authenticate with a service token; anonymous only when explicitly configured."""
        try:
            return principal_from_headers(request.headers, allow_service_only=True)
        except AuthenticationError:
            if svc.cfg.ingest.allow_anonymous and not request.headers.get("authorization") \
                    and not request.headers.get("x-service-token"):
                return None
            raise

    @app.post("/v1/spans")
    async def ingest(spans: list[SpanRecord] | dict[str, list[SpanRecord]] = Body(...),
                     p: Principal | None = Depends(ingest_principal)) -> dict[str, Any]:
        batch = spans.get("spans", []) if isinstance(spans, dict) else spans
        return await svc.ingest(batch, p)

    @app.post("/v1/answers")
    async def answers(sample: AnswerSample, force: bool = False,
                      p: Principal | None = Depends(ingest_principal)) -> dict[str, Any]:
        rec = await svc.submit_answer(sample, p, force=force)
        return {"scored": rec is not None, "score": rec.model_dump() if rec else None}

    @app.get("/v1/traces")
    async def traces(app_: str | None = Query(None, alias="app"), feature: str | None = None, user: str | None = None,
                     since: float | None = None, until: float | None = None, limit: int = Query(50, le=500),
                     p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        res = await svc.search_traces(p, app=app_, feature=feature, user=user, since=since, until=until, limit=limit)
        return [r.model_dump() for r in res]

    @app.get("/v1/traces/{trace_id}")
    async def trace(trace_id: str, p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return (await svc.get_trace(trace_id, p)).model_dump()

    @app.post("/v1/traces/{trace_id}/replay")
    async def replay(trace_id: str, body: dict[str, Any] | None = Body(None),
                     p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return await svc.replay(trace_id, p, model=(body or {}).get("model"))

    @app.post("/v1/feedback")
    async def feedback(fb: FeedbackIn, p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return (await svc.record_feedback(fb, p)).model_dump()

    @app.get("/v1/usage")
    async def usage(by: str = "team", since: float | None = None, until: float | None = None,
                    p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        dims = [d.strip() for d in by.split(",") if d.strip()]
        return [r.model_dump() for r in await svc.usage(p, by=dims, since=since, until=until)]

    @app.get("/v1/chargeback")
    async def chargeback(period: str | None = None, format: str = "json",
                         p: Principal = Depends(current_principal)) -> Response:
        report = await svc.chargeback(p, period)
        if format == "csv":
            return PlainTextResponse(report.to_csv(), media_type="text/csv", headers={
                "Content-Disposition": f'attachment; filename="chargeback-{report.period}.csv"'})
        return Response(report.model_dump_json(), media_type="application/json")

    @app.get("/v1/latency")
    async def latency(since: float | None = None, until: float | None = None,
                      p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        return [r.model_dump() for r in await svc.latency(p, since=since, until=until)]

    @app.get("/v1/quality")
    async def quality(since: float | None = None, app_: str | None = Query(None, alias="app"), feature: str | None = None,
                      model: str | None = None, bucket: str = "day", p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        return await svc.quality(p, since=since, app=app_, feature=feature, model=model, bucket=bucket)

    @app.get("/v1/review")
    async def review_list(status: str | None = "open", source: str | None = None,
                          p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        return [i.model_dump() for i in await svc.review_list(p, status=None if status == "all" else status, source=source)]

    @app.post("/v1/review")
    async def review_add(item: ReviewItem, p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return (await svc.review_enqueue(item, p)).model_dump()

    @app.post("/v1/review/{item_id}/label")
    async def review_label(item_id: str, label: ReviewLabel, p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return (await svc.review_label(item_id, label, p)).model_dump()

    @app.get("/v1/review/golden-candidates")
    async def golden(format: str = "json", mark_exported: bool = False, suite: str = "review_candidates",
                     p: Principal = Depends(current_principal)) -> Response:
        cases = await svc.golden_candidates(p, mark_exported=mark_exported)
        if format == "yaml":
            from aiready.evals.golden import dump_cases_yaml

            return PlainTextResponse(dump_cases_yaml(suite, cases), media_type="application/yaml")
        import json

        return Response(json.dumps(cases), media_type="application/json")

    @app.get("/v1/alerts")
    async def alerts(p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        from aiready.observability.governance import require_any_role

        require_any_role(p, svc.cfg.access.report_roles, what="alerts")
        return svc.alerts.active()

    @app.post("/v1/alerts/evaluate")
    async def alerts_eval(p: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        from aiready.observability.governance import require_any_role

        require_any_role(p, svc.cfg.access.report_roles, what="alerts")
        return [e.model_dump() for e in await svc.evaluate_alerts()]

    @app.get("/v1/dashboard/data")
    async def dashboard_data(days: float = 7, p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return await svc.dashboard_data(p, days=days)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(days: float = 7, p: Principal = Depends(current_principal)) -> HTMLResponse:
        return HTMLResponse(render_dashboard(await svc.dashboard_data(p, days=days)))

    @app.get("/v1/exports/grafana")
    async def export_grafana(p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return grafana_dashboard(svc.cfg)

    @app.get("/v1/exports/prometheus-rules")
    async def export_rules(p: Principal = Depends(current_principal)) -> dict[str, Any]:
        return prometheus_rules(svc.cfg)

    return app
