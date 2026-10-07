"""Shared test data for the MCP tests (a module, not conftest, so tests can import it)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from aiready.core.interfaces import ToolDefinition

FACADE_SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Claims facade", "version": "1"},
    "servers": [{"url": "https://facade.test/api"}],
    "components": {
        "schemas": {
            "ClaimUpdate": {
                "type": "object",
                "properties": {"status": {"type": "string", "enum": ["open", "closed"]},
                               "note": {"type": "string", "nullable": True}},
                "required": ["status"],
            }
        },
        "parameters": {"ClaimId": {"name": "claim_id", "in": "path", "required": True, "schema": {"type": "string"}}},
    },
    "paths": {
        "/claims/{claim_id}": {
            "parameters": [{"$ref": "#/components/parameters/ClaimId"}],
            "get": {"operationId": "getClaim", "summary": "Read one claim", "tags": ["claims"]},
            "patch": {"operationId": "updateClaim", "summary": "Update a claim", "tags": ["claims"],
                      "requestBody": {"required": True, "content": {"application/json": {
                          "schema": {"$ref": "#/components/schemas/ClaimUpdate"}}}}},
            "delete": {"operationId": "deleteClaim", "summary": "Delete a claim", "tags": ["claims"]},
        },
        "/claims": {
            "get": {"operationId": "searchClaims", "summary": "Search claims", "tags": ["claims"],
                    "parameters": [{"name": "q", "in": "query", "schema": {"type": "string"}},
                                   {"name": "limit", "in": "query", "schema": {"type": "integer"}}]},
        },
        "/payments": {
            "post": {"operationId": "payClaim", "summary": "Pay out a claim", "tags": ["payments"],
                     "requestBody": {"content": {"application/json": {"schema": {
                         "type": "object", "properties": {"claim_id": {"type": "string"},
                                                          "amount": {"type": "number"}},
                         "required": ["claim_id", "amount"]}}}}},
        },
        "/admin/reindex": {
            "post": {"operationId": "adminReindex", "summary": "Internal", "tags": ["admin"]},
        },
    },
}


class FacadeRecorder:
    """In-memory API facade: records each request (to check the user's token is forwarded)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else None
        return httpx.Response(200, json={"method": request.method, "path": request.url.path,
                                         "query": dict(request.url.params), "body": body})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class EchoTool:
    """A client-specific tool plugged in with `class:` - no fork needed."""

    def __init__(self, greeting: str = "hello"):
        self.greeting = greeting
        self.definition = ToolDefinition(name="echo", description="echo", source="custom",
                                         input_schema={"type": "object", "properties": {"text": {"type": "string"}}})

    async def invoke(self, arguments, principal):
        return f"{self.greeting} {arguments['text']}"
