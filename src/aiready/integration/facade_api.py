"""API facade HTTP app (port 8151).

    uvicorn aiready.integration.facade_api:create_app --factory --port 8151

Each declared endpoint is served at its configured ``path`` (default ``/v1/{name}``) with its configured
method (default GET for reads, POST otherwise). Arguments are the union of path parameters, query parameters
and the JSON body. ``/openapi.json`` describes only the declared endpoints (with ``x-aiready-*`` extensions)
so the MCP server can turn each one into a tool; ``/v1/tools`` returns the same as ``ToolDefinition`` list.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from aiready.core.app import create_service_app
from aiready.core.errors import ValidationFailed
from aiready.core.identity import Principal, current_principal
from aiready.integration.config import EndpointSpec, FacadeConfig
from aiready.integration.facade import FacadeService, build_facade


def _handler(facade: FacadeService, ep: EndpointSpec) -> Callable[..., Awaitable[JSONResponse]]:
    async def handle(request: Request, principal: Principal = Depends(current_principal)) -> JSONResponse:
        args: dict[str, Any] = dict(request.query_params)
        raw = await request.body()
        if raw:
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise ValidationFailed("request body must be JSON") from exc
            if not isinstance(body, dict):
                raise ValidationFailed("request body must be a JSON object")
            args.update(body)
        args.update(request.path_params)
        result = await facade.call(ep.name, args, principal,
                                   approval_id=request.headers.get(facade.config.approval_header))
        return JSONResponse(result)

    handle.__name__ = f"facade_{ep.name}"
    return handle


def create_app(facade: FacadeService | None = None, *, config: FacadeConfig | None = None,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    facade = facade or build_facade(config, transport=transport)
    app = create_service_app("api-facade", title=facade.config.title, on_shutdown=[facade.aclose])
    app.state.facade = facade

    for ep in facade.endpoints.values():
        app.add_api_route(ep.route_path, _handler(facade, ep), methods=[ep.route_method], name=ep.name,
                          summary=ep.description or ep.name)

    @app.get("/v1/tools")
    async def tools(principal: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        return [t.model_dump(mode="json") for t in facade.to_tool_definitions()
                if not t.required_roles or principal.has_role(*t.required_roles)]

    # Replace FastAPI's generated document: tools are described from the declarations, not the handlers.
    app.openapi = lambda: facade.openapi()  # type: ignore[method-assign]
    return app
