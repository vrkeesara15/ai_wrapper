"""Configuration of the structured-data query service.

Sections: ``query`` (``config/query.yaml``: executor, validator, writer, access rules, catalog) and
``semantic_layer`` (``config/semantic_layer.yaml``: business terms and approved metrics).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------------- semantic layer


class MetricParam(BaseModel):
    type: Literal["string", "integer", "number", "date", "boolean"] = "string"
    required: bool = False
    default: Any = None
    description: str = ""
    allowed_values: list[Any] | None = None


class ValueRange(BaseModel):
    min: float | None = None
    max: float | None = None


class Metric(BaseModel):
    name: str = ""
    description: str = ""
    sql: str  # fixed, approved SQL; parameters are bound (:name), never formatted into the text
    params: dict[str, MetricParam] = Field(default_factory=dict)
    allowed_roles: list[str] = Field(default_factory=list)  # empty = every signed-in user
    synonyms: list[str] = Field(default_factory=list)
    expected_range: dict[str, ValueRange] = Field(default_factory=dict)  # column -> plausible values
    owner: str | None = None
    version: str = "1"


class SemanticLayerConfig(BaseModel):
    terms: dict[str, str] = Field(default_factory=dict)  # business term -> definition
    metrics: dict[str, Metric] = Field(default_factory=dict)


# ----------------------------------------------------------------------------- catalog


class ColumnConfig(BaseModel):
    type: str = ""
    description: str = ""
    personal: bool = False


class ForeignKey(BaseModel):
    column: str
    references: str  # "table.column"


class TableConfig(BaseModel):
    description: str = ""
    columns: dict[str, ColumnConfig] = Field(default_factory=dict)
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    sample_queries: list[str] = Field(default_factory=list)

    @field_validator("columns", mode="before")
    @classmethod
    def _columns(cls, v: Any) -> Any:
        """Accept a list of names or a mapping name -> description/config."""
        if isinstance(v, list):
            return {str(c): {} for c in v}
        if isinstance(v, dict):
            return {k: ({"description": c} if isinstance(c, str) else (c or {})) for k, c in v.items()}
        return v


class CatalogConfig(BaseModel):
    reflect: bool = True  # also read tables/columns/keys from the replica (SQLAlchemy inspect)
    reflect_schema: str | None = None
    tables: dict[str, TableConfig] = Field(default_factory=dict)


# ----------------------------------------------------------------------------- access


class TableAccess(BaseModel):
    columns: list[str] | Literal["*"] = "*"
    row_filter: str | None = None  # SQL condition with bound parameters, e.g. "region = :region"


class RoleAccess(BaseModel):
    tables: dict[str, TableAccess] = Field(default_factory=dict)

    @field_validator("tables", mode="before")
    @classmethod
    def _tables(cls, v: Any) -> Any:
        if isinstance(v, list):  # shorthand: list of fully readable tables
            return {str(t): {} for t in v}
        return v


class AccessConfig(BaseModel):
    roles: dict[str, RoleAccess] = Field(default_factory=dict)
    # conditions applied to a table for EVERY caller (e.g. tenant isolation); ANDed with role filters
    row_filters: dict[str, str] = Field(default_factory=dict)


# ----------------------------------------------------------------------------- pipeline


class ValidatorConfig(BaseModel):
    default_limit: int = 200
    max_rows: int = 1000  # hard cap whatever the request asks
    max_joins: int = 6
    forbid_cartesian: bool = True
    default_schemas: list[str] = Field(default_factory=lambda: ["public", "main", "dbo"])
    denied_functions: list[str] = Field(default_factory=lambda: [
        "pg_sleep", "sleep", "benchmark", "waitfor", "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
        "pg_stat_file", "lo_import", "lo_export", "dblink", "dblink_exec", "pg_terminate_backend",
        "pg_cancel_backend", "set_config", "current_setting", "load_file", "load_extension", "readfile",
        "writefile", "edit", "fts3_tokenizer", "xp_cmdshell", "openrowset", "opendatasource", "query_to_xml",
        "randomblob", "zeroblob"])
    reserved_params: list[str] = Field(default_factory=lambda: ["tenant", "user", "app"])


class ExecutorConfig(BaseModel):
    # The read replica or warehouse. REQUIRED: the service never runs a query on the live database.
    replica_url: str | None = None
    # URLs of live/primary databases; refusing them guards against a misconfigured replica_url
    live_db_urls: list[str] = Field(default_factory=list)
    timeout_seconds: float = 15.0
    # statements run before each query, as the user (bound :tenant, :user, :app), e.g. to drive
    # row-level security on the replica: "SELECT set_config('app.user', :user, true)"
    session_setup: list[str] = Field(default_factory=list)
    max_estimated_cost: float | None = None  # PostgreSQL EXPLAIN total cost limit (None = off)
    pool_size: int = 5


class WriterConfig(BaseModel):
    route: str = "auto"
    task: str = "sql"
    timeout_seconds: float = 20.0
    max_tokens: int = 600
    prompt: str = "query.sql_writer"
    prompt_version: str | None = None


class RoutingConfig(BaseModel):
    metric_match_threshold: float = 0.8  # share of a metric synonym's words found in the question


class QueryConfig(BaseModel):
    dialect: str | None = None  # sqlglot dialect; default derived from the replica URL
    prompts_source: Literal["auto", "registry", "files"] = "auto"
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    writer: WriterConfig = Field(default_factory=WriterConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    access: AccessConfig = Field(default_factory=AccessConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    allow_generated_sql: bool = True  # False: approved metrics only
