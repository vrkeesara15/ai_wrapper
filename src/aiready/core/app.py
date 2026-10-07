"""Service application factory.

Every AI-Ready service is a FastAPI app created here, so all of them share: health and readiness probes,
Prometheus ``/metrics``, trace-id propagation, uniform error bodies, request metrics and config hot reload.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest

from aiready import __version__
from aiready.core.config import get_config
from aiready.core.errors import AIReadyError
from aiready.core.telemetry import TRACE_HEADER, metrics, new_trace_id, set_trace_id, setup_telemetry

log = logging.getLogger(__name__)


def create_service_app(
    name: str,
    *,
    title: str | None = None,
    on_startup: list[Callable[[], Awaitable[None] | None]] | None = None,
    on_shutdown: list[Callable[[], Awaitable[None] | None]] | None = None,
    readiness: Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]] | None = None,
    setup_logging: bool | None = None,
) -> FastAPI:
    if setup_logging if setup_logging is not None else os.environ.get("AIREADY_SETUP_LOGGING", "1") == "1":
        setup_telemetry(name)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if os.environ.get("AIREADY_CONFIG_WATCH", "1") == "1":
            get_config().watch(float(os.environ.get("AIREADY_CONFIG_WATCH_INTERVAL", "2")))
        for fn in on_startup or []:
            res = fn()
            if hasattr(res, "__await__"):
                await res  # type: ignore[misc]
        yield
        for fn in on_shutdown or []:
            res = fn()
            if hasattr(res, "__await__"):
                await res  # type: ignore[misc]

    app = FastAPI(title=title or f"AI-Ready {name}", version=__version__, lifespan=lifespan)
    app.state.service_name = name

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next: Callable) -> Any:
        tid = request.headers.get(TRACE_HEADER) or new_trace_id()
        token = set_trace_id(tid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[TRACE_HEADER] = tid
            return response
        finally:
            if request.url.path not in ("/healthz", "/readyz", "/metrics"):
                metrics.requests.labels(name, request.headers.get("X-App", "-"), request.headers.get("X-Feature", "-"),
                                        request.headers.get("X-Team", "-"), "-", str(status)).inc()
                metrics.latency.labels(name, request.headers.get("X-App", "-"), request.headers.get("X-Feature", "-"),
                                       request.headers.get("X-Team", "-"), "-").observe(time.perf_counter() - t0)
            from aiready.core.telemetry import _trace_id

            _trace_id.reset(token)

    @app.exception_handler(AIReadyError)
    async def aiready_error(_: Request, exc: AIReadyError) -> JSONResponse:
        return JSONResponse(exc.to_dict(), status_code=exc.status_code)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": name, "version": __version__}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        detail: dict[str, Any] = {"service": name, "config_version": get_config().version[:12]}
        if readiness:
            try:
                res = readiness()
                if hasattr(res, "__await__"):
                    res = await res  # type: ignore[misc]
                detail.update(res or {})
            except Exception as exc:  # not ready
                return JSONResponse({"status": "not_ready", "error": str(exc), **detail}, status_code=503)
        return JSONResponse({"status": "ready", **detail})

    @app.get("/metrics", include_in_schema=False)
    async def prom() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(metrics.registry).decode(), media_type="text/plain; version=0.0.4")

    return app
