"""Configuration models for the ``integration`` section (identity bridge, API facade, change capture).

Defaults run locally against the sample legacy app (``examples/legacy_app``) with no other infrastructure.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.types import DataClass
from aiready.ingestion.documents import RowMapping

# =========================================================================== identity bridge


class RoleRule(BaseModel):
    """Maps something the app knows about a user to AI-Ready roles and groups.

    ``app_role`` matches one of the roles returned by the app; ``attributes`` matches attribute values
    (all must be equal). A rule with neither matches everyone.
    """

    app_role: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)


class AttributeGroup(BaseModel):
    """Derive a group from an attribute value, e.g. ``{attribute: region, template: "region:{value}"}``."""

    attribute: str
    template: str = "{value}"


class RoleMapping(BaseModel):
    rules: list[RoleRule] = Field(default_factory=list)
    attribute_groups: list[AttributeGroup] = Field(default_factory=list)
    default_roles: list[str] = Field(default_factory=lambda: ["user"])
    default_groups: list[str] = Field(default_factory=list)
    # Keep app roles that no rule maps, prefixed (e.g. "app:clerk"). False drops them: least privilege.
    keep_unmapped_roles: bool = False
    unmapped_prefix: str = "app:"


class IdentityBridgeConfig(BaseModel):
    # Validators tried in order (or the one named in the request). Each is a plugin spec of kind
    # ``session_validator`` with an optional ``name`` (defaults to its type).
    validators: list[dict[str, Any]] = Field(default_factory=lambda: [{
        "type": "introspect_http", "url": "http://localhost:8000/api/me", "session_cookie": "LEGACYSESSID",
    }])
    role_mapping: RoleMapping = Field(default_factory=RoleMapping)
    app: str = "default"
    default_tenant: str = "default"
    token_ttl_seconds: int = 900
    refresh_grace_seconds: int = 300  # an expired token may still be refreshed this long after expiry
    embed_app_session: bool = True  # store the app session in the token so the facade can forward it
    app_session_claim: str = "app_session"
    policy_check: bool = True


# =========================================================================== facade


class ForwardConfig(BaseModel):
    """How the facade proves *who the user is* to the legacy app on every forwarded call."""

    mode: Literal["session_cookie", "header", "bearer", "none"] = "session_cookie"
    cookie_name: str = "LEGACYSESSID"
    claim: str = "app_session"  # token claim holding the original session (set by the identity bridge)
    header: str = "X-Remote-User"
    value_template: str = "{sub}"


class HttpUpstream(BaseModel):
    type: Literal["http"] = "http"
    method: str = "GET"
    path: str  # template, e.g. /api/claims/{claim_number}
    query: dict[str, Any] = Field(default_factory=dict)  # name -> template ("{status}")
    body: dict[str, Any] | None = None  # template; None -> unused arguments for POST/PUT/PATCH
    base_url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class SqlUpstream(BaseModel):
    type: Literal["sql"] = "sql"
    query: str  # read-only SELECT with :named parameters
    replica_url: str | None = None
    max_rows: int = 500


class StoredProcedureUpstream(BaseModel):
    type: Literal["stored_procedure"] = "stored_procedure"
    name: str
    params: list[str] = Field(default_factory=list)
    url: str | None = None


class OutputMapping(BaseModel):
    """Select and rename fields of the upstream result. ``root`` picks a sub-tree (``$.items``); ``fields``
    maps output names to paths inside each item. Unmapped fields are dropped when ``fields`` is set."""

    root: str | None = None
    fields: dict[str, str] | None = None
    max_items: int | None = None


class EndpointSpec(BaseModel):
    name: str
    description: str = ""
    method: str | None = None  # facade route method; default GET for reads, POST otherwise
    path: str | None = None  # facade route path; default /v1/{name}
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    upstream: HttpUpstream | SqlUpstream | StoredProcedureUpstream = Field(discriminator="type")
    output: OutputMapping = Field(default_factory=OutputMapping)
    effect: Literal["read", "write", "payment", "delete"] = "read"
    required_roles: list[str] = Field(default_factory=list)
    data_class: DataClass = DataClass.INTERNAL
    timeout_seconds: float = 30.0

    @property
    def route_method(self) -> str:
        return (self.method or ("GET" if self.effect == "read" else "POST")).upper()

    @property
    def route_path(self) -> str:
        return self.path or f"/v1/{self.name}"


class FacadeConfig(BaseModel):
    base_url: str = "http://localhost:8000"
    replica_url: str | None = None
    forward: ForwardConfig = Field(default_factory=ForwardConfig)
    require_approval_header_for_writes: bool = True
    approval_header: str = "X-AIReady-Approval"
    timeout_seconds: float = 30.0
    title: str = "Legacy application facade"
    endpoints: list[EndpointSpec] = Field(default_factory=list)


# =========================================================================== change capture


class CaptureTable(RowMapping):
    watermark_column: str = "updated_at"  # updated_at timestamp or rowversion column
    delete_detection: Literal["tombstone", "key_diff", "none"] = "tombstone"
    key_diff_every: int = 1  # with key_diff: run the (expensive) key comparison every N runs


class TombstoneConfig(BaseModel):
    table: str = "deleted_records"
    id_column: str = "id"
    table_column: str = "table_name"
    key_column: str = "record_id"


class OutboxConfig(BaseModel):
    table: str = "outbox"
    id_column: str = "id"
    aggregate_column: str = "aggregate"
    aggregate_id_column: str = "aggregate_id"
    event_type_column: str = "event_type"
    delete_event_types: list[str] = Field(default_factory=lambda: ["deleted"])
    aggregate_tables: dict[str, str] = Field(default_factory=dict)  # aggregate name -> captured table


class KafkaConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    topics: list[str] = Field(default_factory=list)
    group_id: str = "aiready-change-capture"
    max_records: int = 500
    poll_timeout_ms: int = 1000


class ChangeCaptureConfig(BaseModel):
    type: str = "polling"  # polling | debezium | outbox | class: module:Class
    source: str = "legacy-db"
    replica_url: str | None = None  # REQUIRED: the read replica, never the live primary
    primary_url: str | None = None  # optional: refuse to start if replica_url equals it
    tables: list[CaptureTable] = Field(default_factory=list)
    tombstone: TombstoneConfig = Field(default_factory=TombstoneConfig)
    outbox: OutboxConfig = Field(default_factory=OutboxConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    queue: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    interval_seconds: float = 60.0
    batch_size: int = 1000
    policy_check: bool = True


class IntegrationConfig(BaseModel):
    identity_bridge: IdentityBridgeConfig = Field(default_factory=IdentityBridgeConfig)
    facade: FacadeConfig = Field(default_factory=FacadeConfig)
    change_capture: ChangeCaptureConfig = Field(default_factory=ChangeCaptureConfig)


def load_integration_config() -> IntegrationConfig:
    from aiready.core.config import get_config

    return get_config().section("integration", IntegrationConfig)
