"""Identity bridge HTTP API (port 8150).

    uvicorn aiready.integration.bridge_api:create_app --factory --port 8150

These endpoints are *pre-authentication by nature*: their job is to authenticate an existing app session
(cookie, proxy headers, IdP token) and turn it into an AI-Ready token, so they cannot require an AI-Ready
token themselves. ``/v1/token/refresh`` requires the previous AI-Ready token instead. Every issuance and
every denial is audited by :class:`IdentityBridge`.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aiready.core.app import create_service_app
from aiready.core.errors import AuthenticationError, NotFound
from aiready.core.identity import USER_HEADER, bearer_token as _bearer
from aiready.integration.identity_bridge import (
    IdentityBridge,
    SessionCredentials,
    TokenResponse,
    build_bridge,
)


class ExchangeBody(BaseModel):
    validator: str | None = None
    session: str | None = None
    id_token: str | None = None


class RefreshBody(BaseModel):
    token: str | None = None
    session: str | None = None


async def _body(request: Request, model: type[BaseModel]) -> Any:
    raw = await request.body()
    if not raw:
        return model()
    return model.model_validate_json(raw)


def _creds(request: Request, *, session: str | None = None, id_token: str | None = None) -> SessionCredentials:
    return SessionCredentials(
        cookies=dict(request.cookies), headers={k.lower(): v for k, v in request.headers.items()},
        client_ip=request.client.host if request.client else None, session=session, id_token=id_token,
    )


def create_app(bridge: IdentityBridge | None = None) -> FastAPI:
    app = create_service_app("identity-bridge", title="AI-Ready identity bridge")
    app.state.bridge = bridge

    def get_bridge() -> IdentityBridge:
        if app.state.bridge is None:
            app.state.bridge = build_bridge()
        return app.state.bridge

    @app.post("/v1/token", response_model=TokenResponse)
    async def exchange(request: Request) -> TokenResponse:
        """Exchange the caller's app session (cookie / trusted headers / IdP token) for an AI-Ready token."""
        body: ExchangeBody = await _body(request, ExchangeBody)
        return await get_bridge().exchange(_creds(request, session=body.session, id_token=body.id_token),
                                           validator=body.validator)

    @app.post("/v1/token/refresh", response_model=TokenResponse)
    async def refresh(request: Request) -> TokenResponse:
        body: RefreshBody = await _body(request, RefreshBody)
        token = body.token or _bearer(request.headers.get(USER_HEADER))
        if not token:
            raise AuthenticationError("missing token to refresh")
        return await get_bridge().refresh(token, _creds(request, session=body.session))

    @app.get("/.well-known/jwks.json")
    async def jwks() -> JSONResponse:
        keys = get_bridge().jwks()
        if keys is None:
            raise NotFound("tokens are signed with a shared secret (HS*); no public keys to publish")
        return JSONResponse(keys, headers={"Cache-Control": "public, max-age=300"})

    @app.get("/v1/validators")
    async def validators() -> dict[str, Any]:
        return {"validators": [{"name": n, "type": type(v).__name__} for n, v in get_bridge().validators.items()]}

    return app
