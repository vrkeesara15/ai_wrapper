"""Query service API (port 8112).

    uvicorn aiready.query.api:create_app --factory --port 8112

* ``POST /v1/query``          ``QueryRequest`` -> ``QueryResult`` (with the SQL that ran)
* ``GET  /v1/query/metrics``  approved metrics available to the caller
* ``GET  /v1/query/catalog``  tables and columns the caller's roles may query
* ``GET  /v1/query/tool``     the MCP tool definition
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI

from aiready.core.app import create_service_app
from aiready.core.identity import Principal, current_principal
from aiready.core.interfaces import QueryRequest, QueryResult, ToolDefinition
from aiready.core.wiring import local as _local


def create_app(service: Any | None = None) -> FastAPI:
    holder: dict[str, Any] = {"svc": service}

    def svc() -> Any:
        if holder["svc"] is None:

            holder["svc"] = _local("query")
        return holder["svc"]

    app = create_service_app("query", title="AI-Ready structured data query service")

    @app.post("/v1/query", response_model=QueryResult)
    async def query(req: QueryRequest, principal: Principal = Depends(current_principal)) -> QueryResult:
        return await svc().query(req, principal)

    @app.get("/v1/query/metrics")
    async def metrics_(principal: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
        return svc().list_metrics(principal)

    @app.get("/v1/query/catalog")
    async def catalog(principal: Principal = Depends(current_principal)) -> dict[str, str]:
        return {"catalog": await svc().describe_catalog(principal)}

    @app.get("/v1/query/tool", response_model=ToolDefinition)
    async def tool(principal: Principal = Depends(current_principal)) -> ToolDefinition:
        from aiready.query.tool import QUERY_TOOL

        return QUERY_TOOL

    return app
