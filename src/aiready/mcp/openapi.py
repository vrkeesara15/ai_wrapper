"""OpenAPI 3 -> MCP tools.

Each tool is one facade endpoint ("Legacy integration architecture"): the API facade (or the application's
own API, when it is clean enough) already enforces the application's permissions, so a tool is nothing more
than a typed, described wrapper around one operation, called with the signed-in user's token.

Mapping rules:

* name: ``operationId`` (sanitised to MCP's ``[A-Za-z0-9_-]{1,64}``), else ``<method>_<path>``; an optional
  per-source ``prefix`` keeps names unique when several APIs are exposed;
* description: ``summary`` + ``description``;
* input schema: one JSON-schema object built from path/query/header parameters and the request body. When
  the body is an object whose properties do not collide with parameter names, they are merged at the top level
  (models fill flat arguments far more reliably than nested ones); otherwise the body is nested under ``body``;
* effect: ``GET/HEAD/OPTIONS`` -> ``read``, ``POST/PUT/PATCH`` -> ``write``, ``DELETE`` -> ``delete``; an
  ``x-aiready-effect`` extension on the operation, or an override in ``config/tools.yaml``, wins (this is
  how ``payment`` is declared);
* ``x-aiready-data-class`` / ``x-aiready-required-roles`` extensions set classification and required roles,
  ``x-aiready-input-schema`` (emitted by the AI-Ready API facade) is used verbatim as the input schema;
* reserved headers (the facade's approval header) are never tool arguments: the executor sets them.

Include/exclude filters accept glob patterns matched against the tool name, the operationId, ``"METHOD /path"``
and ``tag:<name>``, so a client exposes exactly the operations it wants without editing the spec.
"""

from __future__ import annotations

import copy
import fnmatch
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.errors import ConfigError
from aiready.core.interfaces import ToolDefinition
from aiready.core.types import DataClass

HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
EFFECT_BY_METHOD = {"get": "read", "head": "read", "options": "read", "post": "write", "put": "write",
                    "patch": "write", "delete": "delete"}
_NAME_OK = re.compile(r"[^A-Za-z0-9_-]+")


class ParamBinding(BaseModel):
    """Where one tool argument goes in the HTTP request."""

    name: str  # argument name in the tool's input schema
    wire_name: str  # name on the wire (path placeholder, query key, header name)
    location: str  # path | query | header | body_field | body


class HttpBinding(BaseModel):
    """Everything needed to turn tool arguments into an HTTP request."""

    method: str
    path: str  # template, e.g. /claims/{claim_id}
    base_url: str = ""
    params: list[ParamBinding] = Field(default_factory=list)
    content_type: str = "application/json"
    source: str = "facade"


class OpenAPITool(BaseModel):
    definition: ToolDefinition
    binding: HttpBinding
    operation_id: str | None = None
    tags: list[str] = Field(default_factory=list)


def sanitize_name(raw: str, prefix: str = "") -> str:
    name = _NAME_OK.sub("_", f"{prefix}{raw}").strip("_")
    return (name or "tool")[:64]


def infer_effect(method: str) -> str:
    return EFFECT_BY_METHOD.get(method.lower(), "write")


# --------------------------------------------------------------------------- $ref resolution


class _Resolver:
    """Resolves local ``#/...`` references; external references are rejected (no network at build time)."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec

    def ref(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            raise ConfigError(f"only local $ref is supported in OpenAPI specs: {ref}")
        node: Any = self.spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"unresolvable $ref {ref}")
            node = node[part]
        return node

    def deref(self, node: Any, _seen: tuple[str, ...] = ()) -> Any:
        """Inline every local reference. Recursive schemas are cut at the first repetition."""
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref = node["$ref"]
                if ref in _seen:
                    return {"type": "object", "description": f"(recursive {ref.rsplit('/', 1)[-1]})"}
                target = self.deref(copy.deepcopy(self.ref(ref)), _seen + (ref,))
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**target, **self.deref(extra, _seen)} if isinstance(target, dict) else target
            return {k: self.deref(v, _seen) for k, v in node.items()}
        if isinstance(node, list):
            return [self.deref(v, _seen) for v in node]
        return node


def _to_json_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """OpenAPI 3.0 schema objects are almost JSON schema; fix the differences that matter to validators."""
    if not isinstance(schema, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("xml", "externalDocs", "example", "discriminator", "readOnly", "writeOnly", "deprecated"):
            continue
        if k == "nullable":
            continue
        if k in ("properties", "patternProperties") and isinstance(v, dict):
            out[k] = {pk: _to_json_schema(pv) for pk, pv in v.items()}
        elif k in ("items", "additionalProperties", "not") and isinstance(v, dict):
            out[k] = _to_json_schema(v)
        elif k in ("allOf", "anyOf", "oneOf") and isinstance(v, list):
            out[k] = [_to_json_schema(x) for x in v]
        else:
            out[k] = v
    if schema.get("nullable") and "type" in out and isinstance(out["type"], str):
        out["type"] = [out["type"], "null"]
    return out


# --------------------------------------------------------------------------- filters


def _keys_for(name: str, op_id: str | None, method: str, path: str, tags: Iterable[str]) -> list[str]:
    keys = [name, f"{method.upper()} {path}"]
    if op_id:
        keys.append(op_id)
    keys += [f"tag:{t}" for t in tags]
    return keys


def matches(patterns: list[str], keys: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(k, p) for p in patterns for k in keys)


# --------------------------------------------------------------------------- conversion


def tools_from_openapi(spec: dict[str, Any], *, base_url: str | None = None, source: str = "facade",
                       include: list[str] | None = None, exclude: list[str] | None = None, prefix: str = "",
                       overrides: dict[str, dict[str, Any]] | None = None,
                       default_timeout: float = 30.0, default_data_class: DataClass = DataClass.INTERNAL,
                       default_roles: list[str] | None = None,
                       reserved_headers: Iterable[str] = ("X-AIReady-Approval",)) -> list[OpenAPITool]:
    """Generate one tool per operation of an OpenAPI 3.x document."""
    if not isinstance(spec, dict) or "paths" not in spec:
        raise ConfigError("not an OpenAPI document (missing 'paths')")
    version = str(spec.get("openapi", ""))
    if version and not version.startswith("3"):
        raise ConfigError(f"OpenAPI {version} is not supported (3.x only)")
    resolver = _Resolver(spec)
    if base_url is None:
        servers = spec.get("servers") or []
        base_url = servers[0].get("url", "") if servers and isinstance(servers[0], dict) else ""
    overrides = overrides or {}
    out: list[OpenAPITool] = []
    seen: set[str] = set()

    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        item = resolver.deref(item)
        shared_params = item.get("parameters") or []
        for method in HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            tags = list(op.get("tags") or [])
            name = sanitize_name(op_id or f"{method}_{path}", prefix)
            keys = _keys_for(name, op_id, method, path, tags)
            if include and not matches(include, keys):
                continue
            if exclude and matches(exclude, keys):
                continue
            if name in seen:  # duplicate operationIds would make tool calls ambiguous
                raise ConfigError(f"duplicate tool name '{name}' from {method.upper()} {path}")
            seen.add(name)
            tool = _operation_tool(name, method, path, op, shared_params, base_url=base_url, source=source,
                                   default_timeout=default_timeout, default_data_class=default_data_class,
                                   default_roles=default_roles or [],
                                   reserved={h.lower() for h in reserved_headers})
            tool.operation_id, tool.tags = op_id, tags
            ov = overrides.get(name) or (overrides.get(op_id) if op_id else None)
            if ov:
                tool.definition = apply_override(tool.definition, ov)
                if ov.get("enabled") is False:
                    continue
            out.append(tool)
    return out


def apply_override(defn: ToolDefinition, ov: dict[str, Any]) -> ToolDefinition:
    allowed = {"description", "effect", "data_class", "required_roles", "timeout_seconds"}
    patch = {k: v for k, v in ov.items() if k in allowed and v is not None}
    return ToolDefinition.model_validate({**defn.model_dump(), **patch})  # validates effect / data_class


def _operation_tool(name: str, method: str, path: str, op: dict[str, Any], shared_params: list[Any], *,
                    base_url: str, source: str, default_timeout: float, default_data_class: DataClass,
                    default_roles: list[str], reserved: set[str] = frozenset()) -> OpenAPITool:
    params: dict[tuple[str, str], dict[str, Any]] = {}
    for p in list(shared_params) + list(op.get("parameters") or []):  # operation-level wins
        if isinstance(p, dict) and "name" in p and "in" in p:
            params[(p["in"], p["name"])] = p

    properties: dict[str, Any] = {}
    required: list[str] = []
    bindings: list[ParamBinding] = []
    for (loc, wire), p in params.items():
        if loc == "cookie":  # cookies belong to the browser session, never to a tool argument
            continue
        if loc == "header" and wire.lower() in reserved:  # set by the executor (approval id), not by a model
            continue
        arg = wire if wire not in properties else f"{loc}_{wire}"
        arg = _NAME_OK.sub("_", arg)
        schema = _to_json_schema(p.get("schema") or {"type": "string"})
        if p.get("description"):
            schema = {**schema, "description": p["description"]}
        properties[arg] = schema
        if p.get("required") or loc == "path":
            required.append(arg)
        bindings.append(ParamBinding(name=arg, wire_name=wire, location=loc))

    content_type = "application/json"
    body = op.get("requestBody")
    if isinstance(body, dict):
        content = body.get("content") or {}
        content_type = next((ct for ct in content if "json" in ct), next(iter(content), "application/json"))
        body_schema = _to_json_schema((content.get(content_type) or {}).get("schema") or {})
        body_props = body_schema.get("properties") if body_schema.get("type", "object") == "object" else None
        if isinstance(body_props, dict) and body_props and not (set(body_props) & set(properties)):
            for prop, sch in body_props.items():
                properties[prop] = sch
                bindings.append(ParamBinding(name=prop, wire_name=prop, location="body_field"))
            if body.get("required"):
                required += [r for r in body_schema.get("required", []) if r in body_props]
        else:
            properties["body"] = body_schema or {"type": "object"}
            bindings.append(ParamBinding(name="body", wire_name="body", location="body"))
            if body.get("required"):
                required.append("body")

    input_schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        input_schema["required"] = sorted(set(required))
    declared = op.get("x-aiready-input-schema")  # the facade's own declared schema is authoritative
    if isinstance(declared, dict) and declared.get("type", "object") == "object":
        input_schema = {"additionalProperties": False, **_to_json_schema(declared)}

    desc = " - ".join(x for x in (op.get("summary"), op.get("description")) if x) or f"{method.upper()} {path}"
    effect = op.get("x-aiready-effect") or infer_effect(method)
    data_class = op.get("x-aiready-data-class") or default_data_class
    roles = op.get("x-aiready-required-roles") or op.get("x-aiready-roles") or default_roles
    timeout = float(op.get("x-aiready-timeout") or default_timeout)
    definition = ToolDefinition(name=name, description=desc[:1024], input_schema=input_schema, effect=effect,
                                data_class=data_class, required_roles=list(roles), timeout_seconds=timeout,
                                source=source)
    binding = HttpBinding(method=method.upper(), path=path, base_url=base_url, params=bindings,
                          content_type=content_type, source=source)
    return OpenAPITool(definition=definition, binding=binding)
