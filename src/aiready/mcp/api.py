"""MCP server app (port 8140).

* ``/mcp`` - Model Context Protocol over Streamable HTTP for any MCP-capable assistant;
* ``GET /v1/tools`` - the caller's tools as JSON (``?format=webmcp`` for the browser add-on,
  ``?format=openai`` for function-calling specs);
* ``POST /v1/tools/{name}:call`` - execute a tool (used by the agent runtime and the WebMCP add-on). Returns
  202 with ``approval_id`` when the call waits for a person.

Run: ``uvicorn aiready.mcp.api:create_app --factory --port 8140``
"""

from __future__ import annotations
from aiready.core.wiring import local as _local

from typing import Any, Literal

from fastapi import Body, Depends, FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.routing import Route

from aiready.core.app import create_service_app
from aiready.core.config import get_config
from aiready.core.identity import Principal, current_principal
from aiready.core.interfaces import ToolDefinition
from aiready.mcp.service import ToolService
from aiready.mcp.config import ToolsConfig
from aiready.mcp.server import MCPMount, build_mcp_server


class ToolCallBody(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)  # run_id / step, recorded with the approval


def webmcp_manifest(tools: list[ToolDefinition], call_base: str = "/v1/tools") -> dict[str, Any]:
    """Shape for ``navigator.modelContext.registerTool`` (W3C WebMCP draft field names)."""
    return {
        "version": 1,
        "call_endpoint": call_base + "/{name}:call",
        "tools": [{"name": t.name, "description": t.description, "inputSchema": t.input_schema,
                   "annotations": {"readOnlyHint": t.effect == "read"}, "effect": t.effect} for t in tools],
    }


def openai_specs(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_schema}} for t in tools]


def create_app(service: ToolService | None = None) -> FastAPI:
    """Build the MCP server around an in-process ``ToolService``.

    Never ``svc_()`` here: in Kubernetes ``services.tools.url`` is set for every pod, including this
    one, so wiring would return an HTTP client pointing back at this very service (an infinite loop).
    """
    cfg = get_config().section("tools", ToolsConfig)
    holder: dict[str, ToolService] = {}

    def svc_() -> ToolService:
        if "svc" not in holder:
            holder["svc"] = service or _local("tools")
        return holder["svc"]

    server = build_mcp_server(svc_, name=cfg.mcp_server_name, instructions=cfg.mcp_instructions)
    mount = MCPMount(server)
    app = create_service_app("mcp", title="AI-Ready MCP server", on_startup=[mount.start], on_shutdown=[mount.stop])
    app.state.mcp = mount
    if cfg.cors_allowed_origins:  # the WebMCP add-on calls the REST surface from the app's pages
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_allowed_origins, allow_credentials=False,
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-App"])
    for path in {cfg.mcp_path.rstrip("/"), cfg.mcp_path.rstrip("/") + "/"}:
        app.router.routes.append(Route(path, endpoint=mount.asgi, methods=["GET", "POST", "DELETE"]))

    @app.get("/v1/tools")
    async def list_tools(format: Literal["json", "webmcp", "openai"] = Query("json"),
                         principal: Principal = Depends(current_principal)) -> Any:
        tools = await svc_().list_tools(principal)
        if format == "webmcp":
            return webmcp_manifest(tools)
        if format == "openai":
            return {"tools": openai_specs(tools)}
        return {"tools": [t.model_dump(mode="json") for t in tools]}

    @app.post("/v1/tools/{name}:call")
    async def call_tool(name: str, body: ToolCallBody = Body(default_factory=ToolCallBody),
                        principal: Principal = Depends(current_principal)) -> JSONResponse:
        executor = svc_()
        kwargs: dict[str, Any] = {"approval_id": body.approval_id}
        if body.context:
            kwargs["context"] = body.context
        res = await executor.call(name, body.arguments, principal, **kwargs)
        return JSONResponse(res.model_dump(mode="json"), status_code=202 if res.approval_id else 200)

    @app.post("/v1/tools:reload")
    async def reload(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
        principal.require_role("admin")
        errors = await svc_().reload()
        return {"reloaded": True, "errors": errors}

    return app
