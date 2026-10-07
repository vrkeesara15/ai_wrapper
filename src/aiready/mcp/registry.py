"""Tool registry: every tool an agent or outside assistant may call, from three origins.

1. **OpenAPI-derived** - one tool per operation of the API facade (``sources`` in ``config/tools.yaml``);
2. **config-declared HTTP tools** - endpoints described by hand when no OpenAPI document exists;
3. **built-in tools** - ``search_knowledge`` (RAG), ``query_data`` (governed SQL / approved metrics),
   ``remember`` / ``recall`` (memory). They reach the other services only through ``wiring.connect``.

Client-specific tools plug in with ``custom: [{class: their_pkg.tools:MyTool, ...}]``: any class exposing a
``definition: ToolDefinition`` and ``async invoke(arguments, principal) -> Any`` works. The registry itself
never decides *whether* a call may run - that is the executor's job (policy, approval, audit).
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import quote

import httpx
import yaml

from aiready.core import plugins
from aiready.core.errors import ConfigError, PermissionDenied, UpstreamError, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.interfaces import QueryRequest, RagQuery, ToolDefinition
from aiready.core.telemetry import TRACE_HEADER, current_trace_id
from aiready.core.types import DataClass
from aiready.core.wiring import connect, service_token
from aiready.mcp.config import BuiltinToolConfig, HttpToolConfig, OpenAPISourceConfig, ToolsConfig
from aiready.mcp.openapi import HttpBinding, ParamBinding, apply_override, infer_effect, tools_from_openapi

log = logging.getLogger(__name__)

# Set by the executor around an approved call: HTTP tools forward it so the facade can verify the approval.
CURRENT_APPROVAL: contextvars.ContextVar[str | None] = contextvars.ContextVar("aiready_tool_approval", default=None)


@runtime_checkable
class Tool(Protocol):
    definition: ToolDefinition

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any: ...


# =========================================================================== HTTP tools


class ToolHttpError(UpstreamError):
    """The facade answered with an error status. Carries the status so callers can explain it."""


class HttpTool:
    """Calls one facade endpoint *as the user*: the user's bearer token is forwarded, never a service account.

    The service token (``X-Service-Token``) only proves that the MCP server relayed the call.
    """

    def __init__(self, definition: ToolDefinition, binding: HttpBinding, client: httpx.AsyncClient, *,
                 forward_user_token: bool = True, service: str = "mcp", approval_header: str = "X-AIReady-Approval"):
        self.definition = definition
        self.binding = binding
        self._client = client
        self.forward_user_token = forward_user_token
        self.service = service
        self.approval_header = approval_header

    def build_request(self, arguments: dict[str, Any], principal: Principal) -> httpx.Request:
        b = self.binding
        path = b.path
        query: dict[str, Any] = {}
        headers: dict[str, str] = {}
        body_fields: dict[str, Any] = {}
        body: Any = None
        for p in b.params:
            if p.name not in arguments:
                continue
            val = arguments[p.name]
            if p.location == "path":
                path = path.replace("{" + p.wire_name + "}", quote(str(val), safe=""))
            elif p.location == "query":
                query[p.wire_name] = json.dumps(val) if isinstance(val, (dict, list)) else val
            elif p.location == "header":
                headers[p.wire_name] = str(val)
            elif p.location == "body_field":
                body_fields[p.wire_name] = val
            elif p.location == "body":
                body = val
        if re.search(r"\{[^}]+\}", path):
            raise ValidationFailed(f"tool {self.definition.name}: missing path parameter in {path}")
        if body_fields:
            body = {**(body if isinstance(body, dict) else {}), **body_fields}

        if self.forward_user_token:
            if not principal.token:
                raise PermissionDenied("tool calls carry the signed-in user's token; none was provided")
        headers.update(principal.forward_headers(service_token(self.service)) if self.forward_user_token
                       else {"X-Service-Token": service_token(self.service) or ""})
        tid = current_trace_id()
        if tid:
            headers[TRACE_HEADER] = tid
        approval_id = CURRENT_APPROVAL.get()
        if approval_id and self.approval_header:
            headers[self.approval_header] = approval_id
        url = (b.base_url.rstrip("/") + path) if b.base_url else path
        kwargs: dict[str, Any] = {"params": query or None, "headers": headers}
        if body is not None:
            if "json" in b.content_type:
                kwargs["json"] = body
            else:
                kwargs["data"] = body
                headers["Content-Type"] = b.content_type
        return self._client.build_request(b.method, url, timeout=self.definition.timeout_seconds, **kwargs)

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any:
        req = self.build_request(arguments, principal)
        try:
            resp = await self._client.send(req)
        except httpx.TimeoutException as exc:
            raise UpstreamError(f"tool {self.definition.name} timed out") from exc
        except httpx.TransportError as exc:
            raise UpstreamError(f"tool {self.definition.name} could not reach {req.url.host}: {exc}") from exc
        ctype = resp.headers.get("content-type", "")
        try:
            payload: Any = resp.json() if "json" in ctype else resp.text
        except ValueError:
            payload = resp.text
        if resp.status_code >= 400:
            err = ToolHttpError(f"{self.binding.method} {self.binding.path} returned {resp.status_code}",
                                details={"status": resp.status_code, "body": payload if isinstance(payload, (dict, list))
                                         else str(payload)[:500]})
            err.status_code = resp.status_code if resp.status_code < 500 else 502
            raise err
        return payload


def http_tool_from_config(cfg: HttpToolConfig, client: httpx.AsyncClient, *, forward_user_token: bool,
                          approval_header: str = "X-AIReady-Approval") -> HttpTool:
    if not (cfg.url or cfg.path):
        raise ConfigError(f"http tool '{cfg.name}' needs 'url' or 'path'")
    method = cfg.method.upper()
    if cfg.url:
        base_url, path = "", cfg.url
    else:
        base_url, path = cfg.base_url or "", cfg.path or ""
    props = (cfg.input_schema or {}).get("properties", {}) or {}
    bindings: list[ParamBinding] = []
    for arg in props:
        if arg in cfg.path_params or "{" + arg + "}" in path:
            loc = "path"
        elif arg in cfg.header_params:
            bindings.append(ParamBinding(name=arg, wire_name=cfg.header_params[arg], location="header"))
            continue
        elif arg in cfg.query_params or method in ("GET", "DELETE", "HEAD"):
            loc = "query"
        else:
            loc = "body_field"
        bindings.append(ParamBinding(name=arg, wire_name=arg, location=loc))
    defn = ToolDefinition(name=cfg.name, description=cfg.description, input_schema=cfg.input_schema,
                          effect=cfg.effect or infer_effect(method), data_class=cfg.data_class,
                          required_roles=cfg.required_roles, timeout_seconds=cfg.timeout_seconds, source="config")
    return HttpTool(defn, HttpBinding(method=method, path=path, base_url=base_url, params=bindings,
                                      source="config"), client, forward_user_token=forward_user_token,
                  approval_header=approval_header)


# =========================================================================== built-in tools


class BuiltinTool:
    """Base for tools that call another AI-Ready service through wiring (never by import)."""

    name = ""
    description = ""
    effect = "read"
    data_class = DataClass.INTERNAL
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, description: str | None = None, effect: str | None = None,
                 required_roles: list[str] | None = None, timeout_seconds: float | None = None,
                 options: dict[str, Any] | None = None):
        self.options = options or {}
        self.definition = ToolDefinition(
            name=self.name, description=description or self.description, input_schema=self.input_schema,
            effect=effect or self.effect, data_class=self.data_class, required_roles=required_roles or [],
            timeout_seconds=timeout_seconds or 30.0, source="builtin")

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


@plugins.register("tool", "search_knowledge")
class SearchKnowledgeTool(BuiltinTool):
    name = "search_knowledge"
    description = ("Search the knowledge base the user is allowed to see. Returns a grounded answer with "
                   "citations, or the matching passages when answer=false.")
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What to look up, in natural language."},
            "collection": {"type": "string", "description": "Knowledge collection (default: the app's)."},
            "answer": {"type": "boolean", "default": True,
                       "description": "true: grounded answer with citations; false: raw passages."},
            "top_n": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["question"],
        "additionalProperties": False,
    }

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any:
        rag = connect("rag")
        q = RagQuery(question=arguments["question"], collection=arguments.get("collection")
                     or self.options.get("collection"), top_n=arguments.get("top_n"), feature="agent.tool")
        if arguments.get("answer", True):
            ans = await rag.answer(q, principal)
            return {"answer": ans.answer, "found": ans.found, "grounded": ans.grounded,
                    "citations": [c.model_dump() for c in ans.citations]}
        hits = await rag.retrieve(q, principal)
        return {"passages": [{"chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "title": h.chunk.title,
                              "source_uri": h.chunk.source_uri, "text": h.chunk.text, "score": h.score}
                             for h in hits]}


@plugins.register("tool", "query_data")
class QueryDataTool(BuiltinTool):
    name = "query_data"
    description = ("Answer a question from the application's structured data with governed, read-only SQL. "
                   "Prefer an approved metric name when one fits.")
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "metric": {"type": "string", "description": "Approved metric name, preferred to free questions."},
            "params": {"type": "object"},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "additionalProperties": False,
        "anyOf": [{"required": ["question"]}, {"required": ["metric"]}],
    }

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any:
        req = QueryRequest(question=arguments.get("question"), metric=arguments.get("metric"),
                           params=arguments.get("params") or {}, max_rows=arguments.get("max_rows") or 200)
        res = await connect("query").query(req, principal)
        return res.model_dump()


@plugins.register("tool", "remember")
class RememberTool(BuiltinTool):
    name = "remember"
    description = "Store a fact or preference about the user for later sessions (only with the user's consent)."
    effect = "write"  # a memory write: waits for confirmation unless config/tools.yaml says otherwise
    data_class = DataClass.PERSONAL
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["fact", "preference", "summary"], "default": "fact"},
            "ttl_days": {"type": "integer", "minimum": 1},
            "consent": {"type": "boolean", "description": "true only when the user explicitly agreed."},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any:
        item = await connect("memory").remember(principal, arguments["text"], kind=arguments.get("kind", "fact"),
                                                ttl_days=arguments.get("ttl_days"),
                                                consent=bool(arguments.get("consent", False)))
        return item.model_dump()


@plugins.register("tool", "recall")
class RecallTool(BuiltinTool):
    name = "recall"
    description = "Recall stored facts and preferences about the current user."
    data_class = DataClass.PERSONAL
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
        "additionalProperties": False,
    }

    async def invoke(self, arguments: dict[str, Any], principal: Principal) -> Any:
        items = await connect("memory").recall(principal, arguments.get("query"), limit=arguments.get("limit", 10))
        return [i.model_dump(include={"id", "kind", "text", "created_at", "expires_at"}) for i in items]


BUILTINS = ("search_knowledge", "query_data", "remember", "recall")


# =========================================================================== registry


class ToolRegistry:
    """Holds the merged tool set. Rebuilt lazily after a configuration change."""

    def __init__(self, cfg: ToolsConfig, *, transport: httpx.AsyncBaseTransport | None = None,
                 spec_loader: Callable[[OpenAPISourceConfig], Any] | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(transport=transport, timeout=cfg.default_timeout_seconds,
                                         follow_redirects=False)
        self._tools: dict[str, Tool] = {}
        self._loaded = False
        self._spec_loader = spec_loader
        self.errors: list[str] = []

    # ------------------------------------------------------------------ access
    async def ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def invalidate(self, cfg: ToolsConfig | None = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        self._loaded = False

    def add(self, tool: Tool, *, replace: bool = False) -> None:
        name = tool.definition.name
        if name in self._tools and not replace:
            raise ConfigError(f"duplicate tool name '{name}'")
        ov = self.cfg.overrides.get(name)
        if ov:
            if ov.get("enabled") is False:
                return
            tool.definition = apply_override(tool.definition, ov)
        self._tools[name] = tool

    # ------------------------------------------------------------------ loading
    async def load(self) -> None:
        """(Re)build the tool set. A broken source is reported and skipped - other tools keep working."""
        self._tools = {}
        self.errors = []
        cfg = self.cfg
        for src in cfg.sources:
            if not src.enabled:
                continue
            try:
                spec = await self._load_spec(src)
                for t in tools_from_openapi(spec, base_url=src.base_url, source=src.name, include=src.include,
                                            exclude=src.exclude, prefix=src.prefix,
                                            default_timeout=src.timeout_seconds, default_data_class=src.data_class,
                                            default_roles=src.required_roles, reserved_headers=[cfg.approval_header]):
                    self.add(HttpTool(t.definition, t.binding, self._client, forward_user_token=cfg.forward_user_token,
                                      approval_header=cfg.approval_header))
            except Exception as exc:
                msg = f"tool source '{src.name}' failed to load: {exc}"
                log.error(msg)
                self.errors.append(msg)
        for h in cfg.http_tools:
            if h.enabled:
                self.add(http_tool_from_config(h, self._client, forward_user_token=cfg.forward_user_token,
                                               approval_header=cfg.approval_header))
        for name in BUILTINS:
            bcfg = cfg.builtins.get(name, BuiltinToolConfig())
            if bcfg.enabled:
                self.add(plugins.create("tool", {"type": name, **bcfg.model_dump(exclude={"enabled"})}))
        for spec in cfg.custom:
            tool = plugins.create("tool", spec)
            if not isinstance(getattr(tool, "definition", None), ToolDefinition) or not hasattr(tool, "invoke"):
                raise ConfigError(f"custom tool {spec} must expose `definition: ToolDefinition` and `invoke()`")
            self.add(tool)
        self._loaded = True

    async def _load_spec(self, src: OpenAPISourceConfig) -> dict[str, Any]:
        if self._spec_loader is not None:
            res = self._spec_loader(src)
            return await res if hasattr(res, "__await__") else res
        if src.spec is not None:
            return src.spec
        if src.spec_path:
            text = Path(src.spec_path).read_text()
            return json.loads(text) if src.spec_path.endswith(".json") else yaml.safe_load(text)
        if src.spec_url:
            headers = {}
            tok = service_token("mcp")
            if tok:
                headers["X-Service-Token"] = tok
            r = await self._client.get(src.spec_url, headers=headers)
            r.raise_for_status()
            return r.json() if "json" in r.headers.get("content-type", "") else yaml.safe_load(r.text)
        raise ConfigError(f"tool source '{src.name}' needs spec, spec_path or spec_url")

    async def aclose(self) -> None:
        await self._client.aclose()
