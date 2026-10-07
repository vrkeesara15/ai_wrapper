"""API facade: clean, declared endpoints over the existing application's functions.

Legacy applications rarely have APIs an agent can use safely. The facade declares, in configuration, a small
set of endpoints (name, input schema, effect, required roles) and how each maps onto something that already
exists: an HTTP call to the app, a read-only SQL query on the replica, or a stored procedure. The MCP server
reads the facade's OpenAPI document and turns every endpoint into one tool.

Every call:
  1. requires a verified AI-Ready user token (the caller acts as the user);
  2. checks required roles and validates arguments against the input schema;
  3. for anything but ``read`` (when ``require_approval_header_for_writes``) requires the approval header
     ``X-AIReady-Approval`` set by the MCP executor once a person approved - otherwise **403**
     (``approval_required``). The facade does not run approvals; it refuses unapproved side effects;
  4. asks the policy engine (action ``facade.call``);
  5. forwards the call *with the user's identity* (their original app session cookie, a user header, or the
     AI-Ready bearer token) so the app enforces its own rules too;
  6. maps the result (select/rename fields), records an audit entry, a span and metrics.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote

import httpx
import jsonschema
import sqlglot
from sqlglot import exp

from aiready.core.audit import get_audit
from aiready.core.errors import (
    AuthenticationError,
    ConfigError,
    NotFound,
    PermissionDenied,
    PolicyViolation,
    UpstreamError,
    ValidationFailed,
)
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyInput, ToolDefinition
from aiready.core.telemetry import metrics, span
from aiready.integration.config import (
    EndpointSpec,
    FacadeConfig,
    HttpUpstream,
    OutputMapping,
    SqlUpstream,
    StoredProcedureUpstream,
)
from aiready.integration.jsonpath import extract

SERVICE = "api-facade"
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class ApprovalRequiredError(PermissionDenied):
    """A side-effecting call arrived without the approval header the MCP executor adds after approval."""

    code = "approval_required"


# =========================================================================== helpers


def render(template: Any, args: dict[str, Any]) -> Any:
    """Fill ``{name}`` placeholders. A value that is exactly one placeholder keeps the argument's type;
    a placeholder whose argument is missing yields None (dropped from query/body)."""
    if isinstance(template, str):
        whole = _PLACEHOLDER.fullmatch(template)
        if whole:
            return args.get(whole.group(1))
        return _PLACEHOLDER.sub(lambda m: "" if args.get(m.group(1)) is None else str(args[m.group(1)]), template)
    if isinstance(template, dict):
        return {k: render(v, args) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, args) for v in template]
    return template


def map_output(data: Any, mapping: OutputMapping) -> Any:
    root = extract(data, mapping.root) if mapping.root else data
    if mapping.fields is None:
        out = root
    elif isinstance(root, list):
        out = [{k: extract(item, p) for k, p in mapping.fields.items()} for item in root]
    else:
        out = {k: extract(root, p) for k, p in mapping.fields.items()}
    if mapping.max_items and isinstance(out, list):
        out = out[: mapping.max_items]
    return out


def ensure_read_only_select(sql: str) -> None:
    """Accept exactly one SELECT (incl. WITH / UNION) with no data-modifying node anywhere."""
    try:
        statements = [s for s in sqlglot.parse(sql) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise ConfigError(f"facade SQL does not parse: {exc}") from exc
    if len(statements) != 1:
        raise ConfigError("facade SQL must be a single statement")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.With)):
        raise ConfigError(f"facade SQL must be a read-only SELECT, got {type(stmt).__name__}")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Merge)
    for node in stmt.walk():
        if isinstance(node, forbidden):
            raise ConfigError(f"facade SQL contains a forbidden {type(node).__name__} clause")


def _coerce(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Query and path values arrive as strings; convert them to the schema's declared scalar types."""
    props = schema.get("properties", {}) or {}
    out = dict(args)
    for k, v in args.items():
        t = (props.get(k) or {}).get("type")
        if not isinstance(v, str):
            continue
        try:
            if t == "integer":
                out[k] = int(v)
            elif t == "number":
                out[k] = float(v)
            elif t == "boolean":
                out[k] = v.lower() in ("1", "true", "yes", "on")
        except ValueError:
            pass  # leave as is; schema validation reports it
    return out


# =========================================================================== service


class FacadeService:
    def __init__(self, config: FacadeConfig, *, transport: httpx.AsyncBaseTransport | None = None,
                 http_client: httpx.AsyncClient | None = None):
        self.config = config
        self.endpoints: dict[str, EndpointSpec] = {}
        for ep in config.endpoints:
            if ep.name in self.endpoints:
                raise ConfigError(f"duplicate facade endpoint '{ep.name}'")
            if isinstance(ep.upstream, SqlUpstream):
                ensure_read_only_select(ep.upstream.query)  # refuse to start with a writing query
            if isinstance(ep.upstream, StoredProcedureUpstream) and not _IDENT.match(ep.upstream.name):
                raise ConfigError(f"invalid stored procedure name '{ep.upstream.name}'")
            jsonschema.Draft202012Validator.check_schema(ep.input_schema)
            self.endpoints[ep.name] = ep
        self._http = http_client or httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport)
        self._engines: dict[str, Any] = {}

    # ------------------------------------------------------------------ public
    def get(self, name: str) -> EndpointSpec:
        if name not in self.endpoints:
            raise NotFound(f"unknown facade endpoint '{name}'")
        return self.endpoints[name]

    async def call(self, name: str, arguments: dict[str, Any], principal: Principal, *,
                   approval_id: str | None = None) -> Any:
        ep = self.get(name)
        t0 = time.perf_counter()
        audit = get_audit()
        base = {"action": "facade.call", "resource": name}
        with span("facade.call", endpoint=name, effect=ep.effect, upstream=ep.upstream.type) as sp, \
                metrics.time_stage(SERVICE, "call"):
            if ep.required_roles and not principal.has_role(*ep.required_roles):
                self._deny(principal, ep, "missing role", required_roles=ep.required_roles)
                raise PermissionDenied(f"'{name}' requires one of roles {ep.required_roles}")
            args = _coerce(arguments or {}, ep.input_schema)
            errors = sorted(jsonschema.Draft202012Validator(ep.input_schema).iter_errors(args), key=lambda e: e.path)
            if errors:
                raise ValidationFailed(f"invalid arguments for '{name}'",
                                       details={"errors": [e.message for e in errors[:10]]})
            if ep.effect != "read" and self.config.require_approval_header_for_writes and not approval_id:
                self._deny(principal, ep, "approval header missing")
                raise ApprovalRequiredError(
                    f"'{name}' has effect '{ep.effect}' and needs the {self.config.approval_header} header")
            await self._policy(principal, ep, args, approval_id)
            try:
                raw = await self._upstream(ep, args, principal)
            except Exception as exc:
                sp.error(exc)
                audit.for_principal(principal, "facade.call", decision="error", effect=ep.effect,
                                    approval_id=approval_id, error=str(exc)[:300], **base)
                metrics.pipeline.labels(SERVICE, name, "error").inc()
                raise
            result = map_output(raw, ep.output)
            audit.for_principal(principal, "facade.call", decision="allow", effect=ep.effect,
                                upstream=ep.upstream.type, approval_id=approval_id,
                                arguments=sorted(args), latency_ms=round((time.perf_counter() - t0) * 1000, 1), **base)
            metrics.pipeline.labels(SERVICE, name, "ok").inc()
            return result

    # ------------------------------------------------------------------ governance
    def _deny(self, principal: Principal, ep: EndpointSpec, reason: str, **details: Any) -> None:
        metrics.policy.labels(SERVICE, "facade.call", "deny").inc()
        get_audit().for_principal(principal, "facade.call", action="facade.call", resource=ep.name,
                                  decision="deny", effect=ep.effect, reason=reason, **details)

    async def _policy(self, principal: Principal, ep: EndpointSpec, args: dict[str, Any],
                      approval_id: str | None) -> None:
        from aiready.core.wiring import connect

        decision = await connect("policy").decide(PolicyInput(
            action="facade.call",
            principal={"sub": principal.sub, "tenant": principal.tenant, "app": principal.app,
                       "roles": principal.roles, "groups": principal.groups, "actor": principal.actor},
            resource={"endpoint": ep.name, "effect": ep.effect, "upstream": ep.upstream.type,
                      "data_class": ep.data_class.value},
            context={"arguments": sorted(args), "approved": bool(approval_id), "approval_id": approval_id},
        ))
        if not decision.allow or decision.effect == "deny":
            self._deny(principal, ep, "policy", reasons=decision.reasons, policy_version=decision.policy_version)
            raise PolicyViolation(f"policy denied '{ep.name}'", details={"reasons": decision.reasons})
        if decision.effect == "require_approval" and not approval_id:
            self._deny(principal, ep, "policy requires approval", reasons=decision.reasons)
            raise ApprovalRequiredError(f"policy requires approval for '{ep.name}'")
        metrics.policy.labels(SERVICE, "facade.call", "allow").inc()

    # ------------------------------------------------------------------ upstreams
    async def _upstream(self, ep: EndpointSpec, args: dict[str, Any], principal: Principal) -> Any:
        up = ep.upstream
        if isinstance(up, HttpUpstream):
            return await self._http_call(ep, up, args, principal)
        if isinstance(up, SqlUpstream):
            return self._sql_call(up, args)
        return self._procedure_call(up, args)

    def _identity(self, principal: Principal) -> tuple[dict[str, str], dict[str, str]]:
        """(headers, cookies) proving who the user is to the legacy app."""
        fwd = self.config.forward
        if fwd.mode == "session_cookie":
            session = principal.claims.get(fwd.claim)
            if not session:
                raise AuthenticationError(
                    "token carries no application session; obtain the token from the identity bridge")
            return {}, {fwd.cookie_name: str(session)}
        if fwd.mode == "header":
            return {fwd.header: render(fwd.value_template, principal.model_dump())}, {}
        if fwd.mode == "bearer":
            if not principal.token:
                raise AuthenticationError("no user token to forward")
            return {"Authorization": f"Bearer {principal.token}"}, {}
        return {}, {}

    async def _http_call(self, ep: EndpointSpec, up: HttpUpstream, args: dict[str, Any],
                         principal: Principal) -> Any:
        used = set(_PLACEHOLDER.findall(up.path))
        missing = [k for k in used if args.get(k) in (None, "")]
        if missing:
            raise ValidationFailed(f"missing path arguments {missing}")
        path_args = {k: str(args[k]) for k in used}

        path = _PLACEHOLDER.sub(lambda m: quote(path_args[m.group(1)], safe=""), up.path)
        query = {k: v for k, v in render(up.query, args).items() if v is not None}
        for tmpl in up.query.values():
            used.update(_PLACEHOLDER.findall(str(tmpl)))
        method = up.method.upper()
        body: Any = None
        if up.body is not None:
            body = render(up.body, args)
            if isinstance(body, dict):
                body = {k: v for k, v in body.items() if v is not None}
        elif method in ("POST", "PUT", "PATCH"):
            body = {k: v for k, v in args.items() if k not in used}
        headers, cookies = self._identity(principal)
        headers = {**up.headers, **headers}
        if cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
        url = (up.base_url or self.config.base_url).rstrip("/") + path
        try:
            r = await self._http.request(method, url, params=query or None, json=body, headers=headers,
                                         timeout=ep.timeout_seconds)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"legacy app unreachable for '{ep.name}': {exc}") from exc
        if r.status_code >= 400:
            detail = _safe_json(r)
            if r.status_code == 401:
                raise AuthenticationError("the application rejected the user's session", details=detail)
            if r.status_code == 403:
                raise PermissionDenied("the application refused this action for the user", details=detail)
            if r.status_code == 404:
                raise NotFound("not found in the application", details=detail)
            if r.status_code in (400, 409, 422):
                raise ValidationFailed("the application rejected the input", details=detail)
            raise UpstreamError(f"application returned HTTP {r.status_code}", details=detail)
        return _safe_json(r) if r.content else None

    def _engine(self, url: str | None) -> Any:
        url = url or self.config.replica_url
        if not url:
            raise ConfigError("SQL facade endpoints need a replica_url (never the live primary)")
        if url not in self._engines:
            from sqlalchemy import create_engine

            self._engines[url] = create_engine(url, future=True)
        return self._engines[url]

    def _sql_call(self, up: SqlUpstream, args: dict[str, Any]) -> list[dict[str, Any]]:
        from sqlalchemy import text

        ensure_read_only_select(up.query)
        stmt = text(up.query)
        params = {k: args.get(k) for k in stmt.compile().params}
        with self._engine(up.replica_url).connect() as c:
            rows = c.execute(stmt, params).mappings().fetchmany(up.max_rows)
        return [dict(r) for r in rows]

    def _procedure_call(self, up: StoredProcedureUpstream, args: dict[str, Any]) -> Any:
        from sqlalchemy import text

        engine = self._engine(up.url)
        if engine.dialect.name == "sqlite":
            raise ConfigError("SQLite has no stored procedures; use an http or sql upstream")
        if engine.dialect.name == "mssql":
            sql = f"EXEC {up.name} " + ", ".join(f"@{p} = :{p}" for p in up.params)
        else:
            sql = f"CALL {up.name}(" + ", ".join(f":{p}" for p in up.params) + ")"
        with engine.begin() as c:
            res = c.execute(text(sql), {p: args.get(p) for p in up.params})
            return [dict(r) for r in res.mappings().fetchall()] if res.returns_rows else {"ok": True}

    # ------------------------------------------------------------------ discovery
    def to_tool_definitions(self) -> list[ToolDefinition]:
        return [ToolDefinition(name=ep.name, description=ep.description or ep.name, input_schema=ep.input_schema,
                               effect=ep.effect, data_class=ep.data_class, required_roles=ep.required_roles,
                               timeout_seconds=ep.timeout_seconds, source="facade")
                for ep in self.endpoints.values()]

    def openapi(self, *, server_url: str | None = None) -> dict[str, Any]:
        """OpenAPI 3.1 document of the declared endpoints only; the MCP server turns each into a tool."""
        paths: dict[str, Any] = {}
        for ep in self.endpoints.values():
            props = ep.input_schema.get("properties", {}) or {}
            required = set(ep.input_schema.get("required", []) or [])
            path_params = set(_PLACEHOLDER.findall(ep.route_path))
            op: dict[str, Any] = {
                "operationId": ep.name, "summary": ep.description or ep.name, "description": ep.description,
                "tags": ["facade"], "security": [{"bearer": []}],
                "x-aiready-effect": ep.effect, "x-aiready-required-roles": ep.required_roles,
                "x-aiready-data-class": ep.data_class.value, "x-aiready-input-schema": ep.input_schema,
                "parameters": [], "responses": {
                    "200": {"description": "Result", "content": {"application/json": {"schema": {}}}},
                    "401": {"description": "Not authenticated"},
                    "403": {"description": "Missing role, policy denial or approval required"},
                    "422": {"description": "Invalid arguments"},
                },
            }
            for p in sorted(path_params):
                op["parameters"].append({"name": p, "in": "path", "required": True,
                                         "schema": props.get(p, {"type": "string"})})
            if ep.effect != "read" and self.config.require_approval_header_for_writes:
                op["parameters"].append({"name": self.config.approval_header, "in": "header", "required": True,
                                         "schema": {"type": "string"},
                                         "description": "Approval id issued after a person approved the action"})
            rest = {k: v for k, v in props.items() if k not in path_params}
            if ep.route_method in ("GET", "DELETE"):
                for k, v in rest.items():
                    op["parameters"].append({"name": k, "in": "query", "required": k in required, "schema": v})
            else:
                body_schema = {**ep.input_schema, "properties": rest,
                               "required": [r for r in ep.input_schema.get("required", []) if r in rest]}
                if not body_schema["required"]:
                    body_schema.pop("required")
                op["requestBody"] = {"required": bool(required - path_params),
                                     "content": {"application/json": {"schema": body_schema}}}
            paths.setdefault(ep.route_path, {})[ep.route_method.lower()] = op
        doc: dict[str, Any] = {
            "openapi": "3.1.0",
            "info": {"title": self.config.title, "version": "1.0.0",
                     "description": "Clean endpoints over the existing application. Each operation is one tool."},
            "paths": paths,
            "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}}},
        }
        if server_url:
            doc["servers"] = [{"url": server_url}]
        return doc

    async def aclose(self) -> None:
        await self._http.aclose()
        for e in self._engines.values():
            e.dispose()


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return {"text": r.text[:500]}


def build_facade(config: FacadeConfig | None = None, **kw: Any) -> FacadeService:
    if config is None:
        from aiready.integration.config import load_integration_config

        config = load_integration_config().facade
    return FacadeService(config, **kw)
