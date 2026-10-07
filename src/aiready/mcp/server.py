"""A real MCP server (official ``mcp`` SDK, low-level ``Server``) over Streamable HTTP, mounted in FastAPI.

The tool list is dynamic: ``tools/list`` asks the executor for the tools the *caller* may see (role-filtered),
and ``tools/call`` goes through the executor (schema validation, policy, approval gate, audit). The bearer
token of each HTTP request becomes the ``Principal`` for that call, so any MCP-capable assistant acts with the
signed-in user's permissions and nothing more.

Approval round trip for MCP clients: a write returns a result whose structured content carries
``approval_id``; once a person approves it, the client repeats the same call with that id either in the
request ``_meta`` (``{"aiready/approval_id": "apr_..."}``) or as the reserved argument ``_approval_id``.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from starlette.types import Receive, Scope, Send

from aiready import __version__
from aiready.core.errors import AIReadyError, AuthenticationError
from aiready.core.identity import Principal, principal_from_headers
from aiready.core.interfaces import ToolExecutor, ToolResult
from aiready.core.telemetry import TRACE_HEADER, new_trace_id, set_trace_id

log = logging.getLogger(__name__)
APPROVAL_META_KEY = "aiready/approval_id"
APPROVAL_ARG = "_approval_id"


def _principal(ctx: Any) -> Principal:
    request = getattr(ctx, "request", None)
    if request is None or not hasattr(request, "headers"):
        raise AuthenticationError("MCP calls must arrive over HTTP with the user's bearer token")
    return principal_from_headers(request.headers)


def _meta_value(meta: Any, key: str) -> Any:
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(key)
    extra = getattr(meta, "model_extra", None) or {}
    return extra.get(key) or getattr(meta, key, None)


def to_mcp_result(res: ToolResult) -> types.CallToolResult:
    if res.ok:
        content = res.content
        text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
        structured = content if isinstance(content, dict) else {"result": content}
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)],
                                    structured_content=structured, is_error=False)
    if res.approval_id:
        text = (f"NOT EXECUTED YET: {res.error}. Approval request {res.approval_id} was sent to an approver. "
                f"Repeat the same call with _meta['{APPROVAL_META_KEY}'] = '{res.approval_id}' once approved.")
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)],
                                    structured_content={"status": "pending_approval", "approval_id": res.approval_id,
                                                        **(res.content if isinstance(res.content, dict) else {})},
                                    is_error=False)
    return types.CallToolResult(content=[types.TextContent(type="text", text=res.error or "tool call failed")],
                                is_error=True)


def build_mcp_server(executor: Callable[[], ToolExecutor], *, name: str = "aiready",
                     instructions: str | None = None) -> Server:
    """Create the low-level MCP server. ``executor`` is resolved per call so wiring overrides apply."""

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        principal = _principal(ctx)
        tools = await executor().list_tools(principal)
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, description=t.description, input_schema=t.input_schema,
                       annotations=types.ToolAnnotations(read_only_hint=t.effect == "read",
                                                         destructive_hint=t.effect in ("delete", "payment"),
                                                         idempotent_hint=t.effect == "read",
                                                         open_world_hint=False))
            for t in tools])

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            principal = _principal(ctx)
        except AIReadyError as exc:
            return types.CallToolResult(content=[types.TextContent(type="text", text=exc.message)], is_error=True)
        arguments = dict(params.arguments or {})
        approval_id = arguments.pop(APPROVAL_ARG, None) or _meta_value(params.meta, APPROVAL_META_KEY)
        principal.actor = principal.actor or "mcp-client"
        res = await executor().call(params.name, arguments, principal, approval_id=approval_id)
        return to_mcp_result(res)

    return Server(name, version=__version__, instructions=instructions, on_list_tools=on_list_tools,
                  on_call_tool=on_call_tool)


class AuthenticatedMCP:
    """ASGI wrapper: rejects requests without a valid user token before the MCP transport sees them, and sets
    the trace id so every tool call made through MCP is traceable end to end."""

    def __init__(self, app: StreamableHTTPASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.datastructures import Headers

        headers = Headers(scope=scope)
        try:
            principal_from_headers(headers)
        except AIReadyError as exc:
            body = json.dumps(exc.to_dict()).encode()
            await send({"type": "http.response.start", "status": exc.status_code,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"www-authenticate", b'Bearer realm="aiready"')]})
            await send({"type": "http.response.body", "body": body})
            return
        token = set_trace_id(headers.get(TRACE_HEADER) or new_trace_id())
        try:
            await self.app(scope, receive, send)
        finally:
            from aiready.core.telemetry import _trace_id

            _trace_id.reset(token)


class MCPMount:
    """Holds the session manager whose task group must live for the whole app lifetime."""

    def __init__(self, server: Server, *, json_response: bool = True):
        # stateless: any replica can answer any request - the MCP server scales like the other services
        self.manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=json_response)
        self.asgi = AuthenticatedMCP(StreamableHTTPASGIApp(self.manager))
        self._stack: contextlib.AsyncExitStack | None = None

    async def start(self) -> None:
        self._stack = contextlib.AsyncExitStack()
        await self._stack.enter_async_context(self.manager.run())

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        async with self.manager.run():
            yield
