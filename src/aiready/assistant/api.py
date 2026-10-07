"""Assistant API (port 8130): the chat endpoint behind the embedded widget, plus the widget files.

    POST /v1/assistant/chat                 SSE stream of a chat turn
    POST /v1/assistant/runs/{id}/resume     resume a paused run (SSE)
    GET  /v1/assistant/runs/{id}            run status / answer (poll after an approval)
    GET  /v1/assistant/runs/{id}/events     re-attach to a run's stream (SSE)
    POST /v1/assistant/feedback             thumbs up/down, comments, edited answers
    POST /v1/assistant/session              app session (cookie) or bearer -> AI-Ready token
    GET  /v1/assistant/config?app=          public widget settings (title, greeting, theme)
    GET  /widget/assistant-widget.js        the one-script-tag widget
    GET  /widget/webmcp-tools.js            the WebMCP add-on

CORS is an allow-list per app (``assistant.apps.<app>.allowed_origins``): the browser may only call the API
from pages of applications that embed the widget.

Run: ``uvicorn aiready.assistant.api:create_app --factory --port 8130``
"""

from __future__ import annotations

import fnmatch
import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aiready.assistant.config import AssistantConfig
from aiready.assistant.service import AssistantService, ChatTurn, Feedback
from aiready.assistant.session import SessionExchanger
from aiready.core.app import create_service_app
from aiready.core.errors import AIReadyError, NotFound
from aiready.core.identity import Principal, current_principal
from aiready.core.wiring import connect

WIDGET_FILES = {"assistant-widget.js": ("widget", "assistant-widget.js"),
                "webmcp-tools.js": ("webmcp", "webmcp-tools.js")}


class ResumeBody(BaseModel):
    input: dict[str, Any] | None = None


def static_root(cfg: AssistantConfig) -> Path:
    candidates = [cfg.static_dir, os.environ.get("AIREADY_WEB_DIR"),
                  str(Path(__file__).resolve().parents[3] / "web"), str(Path.cwd() / "web")]
    for c in candidates:
        if c and (Path(c) / "widget").is_dir():
            return Path(c)
    raise NotFound("widget files not found: set assistant.static_dir")


# --------------------------------------------------------------------------- CORS


def allowed_origins(cfg: AssistantConfig, app: str | None) -> list[str]:
    if app and app in cfg.apps:
        return cfg.apps[app].allowed_origins + cfg.allowed_origins
    return [o for a in cfg.apps.values() for o in a.allowed_origins] + cfg.allowed_origins


def origin_ok(origin: str, patterns: list[str]) -> bool:
    return any(origin == p or fnmatch.fnmatchcase(origin, p) for p in patterns)


class AppCORS:
    """Per-app CORS. Preflights (which cannot carry X-App) are checked against every app's list; the actual
    request is checked against the list of the app named in ``X-App`` (or ``?app=``)."""

    def __init__(self, app: ASGIApp, cfg: Callable[[], AssistantConfig]):
        self.app = app
        self.cfg = cfg

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.datastructures import Headers, QueryParams

        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if not origin:
            await self.app(scope, receive, send)
            return
        cfg = self.cfg()
        if scope["method"] == "OPTIONS" and headers.get("access-control-request-method"):
            ok = origin_ok(origin, allowed_origins(cfg, None))
            resp_headers = [(b"vary", b"Origin")]
            if ok:
                resp_headers += [
                    (b"access-control-allow-origin", origin.encode()),
                    (b"access-control-allow-credentials", b"true"),
                    (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                    (b"access-control-allow-headers",
                     b"Authorization, Content-Type, X-App, X-Feature, X-Trace-Id, Last-Event-ID, X-CSRF-Token"),
                    (b"access-control-max-age", b"600")]
            await send({"type": "http.response.start", "status": 204 if ok else 403, "headers": resp_headers})
            await send({"type": "http.response.body", "body": b""})
            return
        app_id = headers.get("x-app") or QueryParams(scope.get("query_string", b"").decode()).get("app")
        if not origin_ok(origin, allowed_origins(cfg, app_id)):
            body = json.dumps({"error": {"code": "origin_not_allowed",
                                         "message": f"origin {origin} is not allowed for app {app_id or '-'}"}})
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"application/json"), (b"vary", b"Origin")]})
            await send({"type": "http.response.body", "body": body.encode()})
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                hs = list(message.get("headers", []))
                hs += [(b"access-control-allow-origin", origin.encode()), (b"access-control-allow-credentials", b"true"),
                       (b"access-control-expose-headers", b"X-Trace-Id"), (b"vary", b"Origin")]
                message = {**message, "headers": hs}
            await send(message)

        await self.app(scope, receive, send_with_cors)


# --------------------------------------------------------------------------- app


def _sse(gen: AsyncIterator[dict[str, Any]], request: Request, ping: int) -> EventSourceResponse:
    async def wrapped() -> AsyncIterator[dict[str, Any]]:
        try:
            async for item in gen:
                if await request.is_disconnected():
                    break
                yield {"event": item["event"], "data": json.dumps(item["data"], default=str)}
        except AIReadyError as exc:  # errors after the stream opened travel as an event, not a status
            yield {"event": "error", "data": json.dumps({"message": exc.message, "code": exc.code})}

    return EventSourceResponse(wrapped(), ping=ping, headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _first(gen: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Run the generator up to its first item *before* the response starts, so validation and permission
    errors become proper HTTP status codes instead of an event inside a 200 stream."""
    first = await gen.__anext__()

    async def rest() -> AsyncIterator[dict[str, Any]]:
        yield first
        async for item in gen:
            yield item

    return rest()


def create_app(service: AssistantService | None = None, *, session_transport: Any = None) -> FastAPI:
    svc = service or AssistantService()
    app = create_service_app("assistant", title="AI-Ready assistant API")
    app.add_middleware(AppCORS, cfg=lambda: svc.cfg)

    @app.post("/v1/assistant/chat")
    async def chat(turn: ChatTurn, request: Request, principal: Principal = Depends(current_principal)) -> Any:
        turn.app = turn.app or request.headers.get("X-App")
        turn.feature = turn.feature or request.headers.get("X-Feature")
        return _sse(await _first(svc.chat(turn, principal)), request, svc.cfg.stream_ping_seconds)

    @app.post("/v1/assistant/runs/{run_id}/resume")
    async def resume(run_id: str, request: Request, body: ResumeBody = Body(default_factory=ResumeBody),
                     principal: Principal = Depends(current_principal)) -> Any:
        return _sse(await _first(svc.resume(run_id, principal, input=body.input)), request,
                    svc.cfg.stream_ping_seconds)

    @app.get("/v1/assistant/runs/{run_id}/events")
    async def follow(run_id: str, request: Request, after: int = 0,
                     principal: Principal = Depends(current_principal)) -> Any:
        await connect("agents").get_run(run_id, principal)
        return _sse(svc.follow(run_id, principal, after=after), request, svc.cfg.stream_ping_seconds)

    @app.get("/v1/assistant/runs/{run_id}")
    async def get_run(run_id: str, principal: Principal = Depends(current_principal)) -> Any:
        run = await connect("agents").get_run(run_id, principal)
        result = run.result or {}
        return {"run_id": run.id, "status": run.status, "answer": result.get("answer"),
                "citations": result.get("citations") or [], "pending": run.pending, "error": run.error,
                "event_seq": run.event_seq}

    @app.post("/v1/assistant/feedback")
    async def feedback(fb: Feedback, request: Request, principal: Principal = Depends(current_principal)) -> Any:
        fb.app = fb.app or request.headers.get("X-App")
        fb.feature = fb.feature or request.headers.get("X-Feature")
        return await svc.feedback(fb, principal)

    @app.post("/v1/assistant/session")
    async def session(request: Request) -> Any:
        exchanger = SessionExchanger(svc.cfg.identity_bridge, transport=session_transport)
        return await exchanger.exchange(request.headers, dict(request.cookies), request.headers.get("X-App"))

    @app.get("/v1/assistant/config")
    async def public_config(app: str | None = None) -> Any:
        a = svc.cfg.apps.get(app or "")
        if a is None:
            return {"title": "Assistant", "greeting": "Hi! How can I help?", "theme": {}}
        return {"title": a.title, "greeting": a.greeting, "theme": a.theme}

    @app.get("/widget/{name}")
    async def widget_file(name: str) -> FileResponse:
        if name not in WIDGET_FILES:
            raise NotFound(f"no widget file {name}")
        sub, fname = WIDGET_FILES[name]
        path = static_root(svc.cfg) / sub / fname
        if not path.is_file():
            raise NotFound(f"widget file missing on server: {sub}/{fname}")
        return FileResponse(path, media_type="application/javascript",
                            headers={"Cache-Control": "public, max-age=300", "X-Content-Type-Options": "nosniff"})

    return app

