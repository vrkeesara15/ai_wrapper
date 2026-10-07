"""Configuration models for the governance service (sections ``governance`` and ``security.egress``).

Defaults are safe and run with no infrastructure: in-memory KV, the shared DocStore, rules read from the
``policies`` section of the merged configuration (``config/policies/*.yaml``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyConfig(BaseModel):
    # Which engine answers decide(): the built-in rule engine or an adapter such as OPA.
    engine: dict[str, Any] = Field(default_factory=lambda: {"type": "local"})
    # Where the local engine reads rules from (policy_source plugins); evaluated in order, rules concatenated.
    sources: list[dict[str, Any]] = Field(default_factory=lambda: [{"type": "config", "path": "policies"}])
    # Semver label of the bundle; the content hash is appended automatically ("1.0.0+3fa2b1c9d0e4").
    version: str = "0.0.0"
    # Effect when no rule matches, by action prefix (longest prefix wins; "" is the fallback).
    defaults: dict[str, Literal["allow", "deny", "require_approval"]] = Field(
        default_factory=lambda: {"": "deny"}
    )
    # Audit allow decisions too (denials and approval requests are always audited).
    audit_allows: bool = False
    # Seconds between modification-time checks of directory sources.
    reload_interval: float = 2.0


class RegistryConfig(BaseModel):
    # A model unknown to the registry is treated as "unregistered" (denied by policy) when true.
    require_registration: bool = False
    # Fields a model record must carry before it can be approved (release gate).
    approval_requires: list[str] = Field(default_factory=lambda: ["licence", "weights_checksum"])
    # The person who registered a model / adapter may not approve it (separation of duties).
    four_eyes: bool = True
    # An adapter must beat its base model on every shared eval metric before approval.
    adapter_must_beat_base: bool = True
    # Models declared in configuration and loaded on start (status as written; audited as "seed").
    models: list[dict[str, Any]] = Field(default_factory=list)


class AccessReviewConfig(BaseModel):
    # Group -> members, as exported from the client's directory (or kept here for a pilot).
    group_members: dict[str, list[str]] = Field(default_factory=dict)
    # Role -> members, for roles granted outside groups.
    role_members: dict[str, list[str]] = Field(default_factory=dict)
    # Data resources to include in the review (collection name -> access tags).
    collections: dict[str, list[str]] = Field(default_factory=dict)
    # Data classes evaluated for each model.
    data_classes: list[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential", "personal", "restricted"]
    )
    # A grant nobody used for this long is flagged stale.
    stale_after_days: int = 90
    # How often a review must be signed (the previous report is flagged overdue after this).
    review_period_days: int = 90


class ApiRoles(BaseModel):
    registry_write: list[str] = Field(default_factory=lambda: ["ai-engineer", "model-approver", "admin"])
    registry_approve: list[str] = Field(default_factory=lambda: ["model-approver"])
    policy_admin: list[str] = Field(default_factory=lambda: ["security-admin", "policy-admin"])
    security: list[str] = Field(default_factory=lambda: ["security-admin"])
    access_review: list[str] = Field(default_factory=lambda: ["security-admin", "access-reviewer"])


class GovernanceConfig(BaseModel):
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    access_review: AccessReviewConfig = Field(default_factory=AccessReviewConfig)
    api_roles: ApiRoles = Field(default_factory=ApiRoles)
    # KV store for the revocation deny-list and closed routes (memory for one process, redis for many).
    kv: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    # Default lifetime of a jti revocation entry: the longest token lifetime in use.
    revocation_ttl_seconds: int = 86400


class EgressDestination(BaseModel):
    name: str
    hosts: list[str]  # exact hosts or globs ("*.openai.azure.com")
    aliases: list[str] = Field(default_factory=list)  # other names callers use: model names, model categories
    ports: list[int] = Field(default_factory=lambda: [443])
    data_classes: list[str] = Field(default_factory=list)  # classes allowed to leave to this destination
    roles: list[str] = Field(default_factory=list)  # empty = any authenticated principal
    groups: list[str] = Field(default_factory=list)
    purpose: str = ""
    enabled: bool = True
    owner: str | None = None
    expires_at: float | None = None


class EgressConfig(BaseModel):
    # "closed": nothing leaves unless a destination below opens it. Anything else is rejected at load.
    default: Literal["closed"] = "closed"
    destinations: list[EgressDestination] = Field(default_factory=list)
    # Proxy settings echoed into the exported allow-list.
    proxy: dict[str, Any] = Field(default_factory=dict)
