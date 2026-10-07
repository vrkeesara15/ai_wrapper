"""Configuration of the MCP server / tool executor (section ``tools`` in ``config/tools.yaml``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.types import DataClass


class OpenAPISourceConfig(BaseModel):
    """One OpenAPI document whose operations become tools (normally the API facade)."""

    name: str = "facade"
    base_url: str | None = None  # default: first entry of the spec's `servers`
    spec_url: str | None = None  # fetched at load time (with this service's token)
    spec_path: str | None = None  # local file (JSON or YAML)
    spec: dict[str, Any] | None = None  # inline document
    include: list[str] = Field(default_factory=list)  # globs on name / operationId / "METHOD /path" / tag:x
    exclude: list[str] = Field(default_factory=list)
    prefix: str = ""
    timeout_seconds: float = 30.0
    data_class: DataClass = DataClass.INTERNAL
    required_roles: list[str] = Field(default_factory=list)
    enabled: bool = True


class HttpToolConfig(BaseModel):
    """A tool declared directly in configuration (no OpenAPI document available)."""

    name: str
    description: str
    method: str = "GET"
    url: str | None = None  # absolute URL, or `path` relative to base_url
    path: str | None = None
    base_url: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    # where each argument goes; arguments not listed go to the query string (GET/DELETE) or JSON body
    path_params: list[str] = Field(default_factory=list)
    query_params: list[str] = Field(default_factory=list)
    header_params: dict[str, str] = Field(default_factory=dict)  # argument -> header name
    effect: Literal["read", "write", "payment", "delete"] | None = None  # default inferred from method
    data_class: DataClass = DataClass.INTERNAL
    required_roles: list[str] = Field(default_factory=list)
    timeout_seconds: float = 30.0
    enabled: bool = True


class BuiltinToolConfig(BaseModel):
    enabled: bool = True
    description: str | None = None
    effect: Literal["read", "write", "payment", "delete"] | None = None
    required_roles: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    sources: list[OpenAPISourceConfig] = Field(default_factory=list)
    http_tools: list[HttpToolConfig] = Field(default_factory=list)
    builtins: dict[str, BuiltinToolConfig] = Field(default_factory=dict)
    custom: list[dict[str, Any]] = Field(default_factory=list)  # plugin specs: {class: pkg.mod:Tool, ...}
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)  # per tool: effect, roles, ...
    default_timeout_seconds: float = 30.0
    max_result_bytes: int = 65536
    forward_user_token: bool = True  # never relax: the facade must see the signed-in user
    retry_failed_sources_seconds: float = 60.0  # reload a source that failed to load after this long
    http_retries: int = 0  # retries for read tools only; writes are never retried automatically
    approval_header: str = "X-AIReady-Approval"  # sent to the facade with the approval id of an approved call
    cors_allowed_origins: list[str] = Field(default_factory=list)  # pages using the WebMCP add-on
    mcp_path: str = "/mcp"
    mcp_server_name: str = "aiready"
    mcp_instructions: str = (
        "Tools act on the application as the signed-in user. Reads run immediately; writes, payments and "
        "deletions return an approval id and run only after a person approves them."
    )
