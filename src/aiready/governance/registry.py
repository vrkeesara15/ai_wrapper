"""Registries: models, prompts and adapters.

* **Model registry** - one record per model *version* with the slots of the reference architecture (slot,
  availability, type, originality, parameter tier, base model, licence, context length, weights checksum,
  data classes allowed, evaluation scores). Status moves ``draft -> approved -> deprecated``; only an
  approved, non-disabled version is routable. Approved records are immutable: a change is a new version.
* **Prompt registry** - prompts and tool schemas are versioned; every version is kept, each environment
  (dev / staging / prod) has an *active* pointer, and any two versions can be diffed. Rollback is moving
  the pointer back.
* **Adapter registry** - a fine-tuned adapter is linked to the exact base model version and training
  dataset version it was built from, and may only be approved when its evaluation beats the base.

Everything persists in the shared ``DocStore`` and every state change writes an audit record, so a
regulator can see who approved which model, when, and on what evidence.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PermissionDenied, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics, span
from aiready.core.types import DataClass
from aiready.governance.config import RegistryConfig

NS_MODELS = "registry.models"
NS_PROMPTS = "registry.prompts"
NS_PROMPT_VERSIONS = "registry.prompt_versions"
NS_ADAPTERS = "registry.adapters"
NS_EVALUATIONS = "registry.evaluations"

Status = Literal["draft", "approved", "deprecated"]
Availability = Literal[
    "private_gpu", "private_shared_gpu", "private_endpoint", "public_api", "hosted_open_weight"
]
PUBLIC_AVAILABILITY = {"public_api", "hosted_open_weight"}
_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"approved", "deprecated"},
    "approved": {"deprecated"},
    "deprecated": set(),
}


def parameter_tier(parameters_b: float | None) -> str | None:
    """Working size tiers of the reference architecture: small < 10B, mid 10-40B, large > 40B."""
    if parameters_b is None:
        return None
    if parameters_b < 10:
        return "small"
    if parameters_b <= 40:
        return "mid"
    return "large"


def _now() -> float:
    return time.time()


def _actor(principal: Principal | None) -> str:
    return principal.sub if principal else "svc:governance"


def _audit(principal: Principal | None, event: str, *, resource: str, decision: str, **details: Any) -> None:
    audit = get_audit()
    if principal is not None:
        audit.for_principal(principal, event, action=event, resource=resource, decision=decision, **details)
    else:
        audit.record(
            event, actor="svc:governance", action=event, resource=resource, decision=decision, **details
        )


# =========================================================================== models


class ModelRecord(BaseModel):
    name: str
    version: str = "1"
    slot: str = "general"  # general / domain / router / embedding / reranker / escalation / guard
    availability: Availability = "private_gpu"
    type: str = "chat"  # chat / extraction / classification / embedding / rerank / reasoning / guard
    originality: Literal["original", "tuned_adapter", "original_quantised", "proprietary"] = "original"
    parameters_b: float | None = None  # billions of parameters; None when the vendor does not publish it
    parameter_tier: Literal["small", "mid", "large", "unpublished"] | None = None
    base_model: str | None = None
    licence: str | None = None
    context_length: int | None = None
    weights_checksum: str | None = None  # sha256 of the weights as served (pins exactly what was evaluated)
    endpoint: str | None = None  # where the model server / vendor API lives
    egress_destination: str | None = None  # egress route name for public models
    data_classes_allowed: list[DataClass] = Field(
        default_factory=lambda: [DataClass.PUBLIC, DataClass.INTERNAL]
    )
    eval_scores: dict[str, float] = Field(default_factory=dict)
    owner: str | None = None
    notes: str = ""
    status: Status = "draft"
    registered_by: str | None = None
    approved_by: str | None = None
    approved_at: float | None = None
    deprecated_reason: str | None = None
    disabled: bool = False  # closed by an incident action; overrides approval
    disabled_reason: str | None = None
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)
    history: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive(self) -> ModelRecord:
        if self.parameter_tier is None:
            self.parameter_tier = parameter_tier(self.parameters_b) or (
                "unpublished" if self.originality == "proprietary" else None
            )
        if "@" in self.name or not self.name:
            raise ValueError("model name must be non-empty and must not contain '@'")
        return self

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def boundary(self) -> Literal["private", "public"]:
        return "public" if self.availability in PUBLIC_AVAILABILITY else "private"

    @property
    def routable(self) -> bool:
        return self.status == "approved" and not self.disabled

    def facts(self) -> dict[str, Any]:
        """The registry facts the policy engine sees as ``resource.*``."""
        return {
            "registry_status": self.status,
            "registry_version": self.version,
            "availability": self.availability,
            "boundary": self.boundary,
            "data_classes_allowed": [d.value for d in self.data_classes_allowed],
            "disabled": self.disabled,
            "slot": self.slot,
            "model_type": self.type,
        }

    def public(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["boundary"], d["routable"], d["key"] = self.boundary, self.routable, self.key
        return d


_MODEL_MUTABLE = {
    "slot",
    "availability",
    "type",
    "originality",
    "parameters_b",
    "parameter_tier",
    "base_model",
    "licence",
    "context_length",
    "weights_checksum",
    "endpoint",
    "egress_destination",
    "data_classes_allowed",
    "eval_scores",
    "owner",
    "notes",
}


# =========================================================================== prompts


class PromptVersion(BaseModel):
    name: str
    version: int
    template: str
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    model: str | None = None  # model or route the prompt was evaluated with
    note: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str = ""
    created_by: str | None = None
    created_at: float = Field(default_factory=_now)

    def content_hash(self) -> str:
        body = {
            "template": self.template,
            "tool_schemas": self.tool_schemas,
            "variables": self.variables,
            "model": self.model,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class PromptHead(BaseModel):
    name: str
    latest: int = 0
    active: dict[str, int] = Field(default_factory=dict)  # environment -> version
    description: str = ""
    updated_at: float = Field(default_factory=_now)


# =========================================================================== adapters


class AdapterRecord(BaseModel):
    name: str
    version: str = "1"
    base_model: str
    base_model_version: str
    task: str  # the one task its gateway route serves
    dataset_version: str
    eval_scores: dict[str, float] = Field(default_factory=dict)
    base_eval_scores: dict[str, float] = Field(default_factory=dict)  # same held-out set, base model alone
    lower_is_better: list[str] = Field(
        default_factory=lambda: ["leak_rate", "latency_ms", "cost", "toxicity"]
    )
    artifact_uri: str | None = None
    checksum: str | None = None
    status: Status = "draft"
    registered_by: str | None = None
    approved_by: str | None = None
    approved_at: float | None = None
    deprecated_reason: str | None = None
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)
    history: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def comparison(self) -> dict[str, Any]:
        """Per-metric comparison with the base on the shared held-out metrics."""
        shared = sorted(set(self.eval_scores) & set(self.base_eval_scores))
        rows, worse, better = {}, [], []
        for m in shared:
            a, b = self.eval_scores[m], self.base_eval_scores[m]
            improved = a < b if m in self.lower_is_better else a > b
            regressed = a > b if m in self.lower_is_better else a < b
            rows[m] = {"adapter": a, "base": b, "delta": round(a - b, 6)}
            (better if improved else worse if regressed else []).append(m)
        return {
            "metrics": rows,
            "better": better,
            "worse": worse,
            "beats_base": bool(shared) and bool(better) and not worse,
        }


# =========================================================================== service


class RegistryService:
    """Model, prompt and adapter registries over the DocStore."""

    def __init__(
        self,
        store: DocStore | None = None,
        cfg: RegistryConfig | None = None,
        approver_roles: list[str] | None = None,
        writer_roles: list[str] | None = None,
    ):
        self._store = store
        self.cfg = cfg or RegistryConfig()
        self.approver_roles = approver_roles or ["model-approver"]
        self.writer_roles = writer_roles or ["ai-engineer", "model-approver", "admin"]

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    # ------------------------------------------------------------------ helpers
    def _require(self, principal: Principal | None, roles: list[str], what: str) -> None:
        if principal is None:  # internal call (seeding, incident tooling) - audited as svc:governance
            return
        if not principal.has_role(*roles):
            metrics.policy.labels("governance", what, "deny").inc()
            _audit(principal, "registry.denied", resource=what, decision="deny", required_roles=roles)
            raise PermissionDenied(f"{what} requires one of roles {roles}")

    @staticmethod
    def _transition(rec: ModelRecord | AdapterRecord, to: str) -> None:
        if to not in _TRANSITIONS[rec.status]:
            raise ValidationFailed(f"cannot move {rec.key} from {rec.status} to {to}")

    @staticmethod
    def _history(
        rec: ModelRecord | AdapterRecord, principal: Principal | None, event: str, **extra: Any
    ) -> None:
        rec.history.append({"ts": _now(), "by": _actor(principal), "event": event, **extra})
        rec.updated_at = _now()

    # ------------------------------------------------------------------ models
    def register_model(
        self, record: ModelRecord | dict[str, Any], principal: Principal | None = None, *, seed: bool = False
    ) -> ModelRecord:
        """Add a model version as ``draft`` (or with the declared status when ``seed`` from config)."""
        with span("registry.model.register"):
            rec = record if isinstance(record, ModelRecord) else ModelRecord.model_validate(record)
            self._require(principal, self.writer_roles, "registry.model.register")
            if self.store.get(NS_MODELS, rec.key) is not None:
                raise ValidationFailed(f"model {rec.key} already registered; register a new version instead")
            if not seed:
                rec.status, rec.approved_by, rec.approved_at = "draft", None, None
            elif rec.status == "approved":
                rec.approved_by, rec.approved_at = rec.approved_by or "config", rec.approved_at or _now()
            rec.registered_by = _actor(principal) if not seed else (rec.registered_by or "config")
            self._history(rec, principal, "seeded" if seed else "registered", status=rec.status)
            self.store.put(NS_MODELS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.model.register",
                resource=f"model:{rec.key}",
                decision=rec.status,
                seed=seed,
                availability=rec.availability,
                licence=rec.licence,
                checksum=rec.weights_checksum,
            )
            return rec

    def seed_models(self, specs: list[dict[str, Any]]) -> int:
        """Load models declared in configuration that are not registered yet."""
        n = 0
        for spec in specs:
            rec = ModelRecord.model_validate(spec)
            if self.store.get(NS_MODELS, rec.key) is None:
                self.register_model(rec, None, seed=True)
                n += 1
        return n

    # ------------------------------------------------------------------ release-gate evidence
    def record_evaluation(self, *, release: str, passed: bool, report: dict[str, Any],
                          artifact: dict[str, Any] | None = None, principal: Principal | None = None) -> dict[str, Any]:
        """Keep a release-gate verdict as registry evidence. When the artifact names a model version
        (``{"kind": "model", "name", "version"}``), its suite scores land in the model's ``eval_scores`` and
        history, so approvers see the evaluation that the approval is based on."""
        artifact = artifact or {}
        entry = {"release": release, "passed": passed, "artifact": artifact, "ts": _now(), "by": _actor(principal),
                 "checks": report.get("checks", []), "baselines": report.get("baselines", {})}
        self.store.put(NS_EVALUATIONS, f"{release}:{int(entry['ts'] * 1000)}", entry)
        if artifact.get("kind") == "model" and artifact.get("name") and artifact.get("version"):
            rec = self._load_model(f"{artifact['name']}@{artifact['version']}")
            for c in report.get("checks", []):
                if c.get("actual") is not None and c.get("suite") and c.get("name"):
                    rec.eval_scores[f"{c['suite']}.{c['name']}"] = float(c["actual"])
            self._history(rec, principal, "evaluated", release=release, passed=passed)
            self.store.put(NS_MODELS, rec.key, rec.model_dump(mode="json"))
        _audit(principal, "registry.evaluation", resource=f"release:{release}",
               decision="passed" if passed else "blocked", artifact=artifact)
        return entry

    def evaluations(self, release: str | None = None) -> list[dict[str, Any]]:
        return [v for _, v in self.store.list(NS_EVALUATIONS, prefix=f"{release}:" if release else "")]

    def _load_model(self, key: str) -> ModelRecord:
        raw = self.store.get(NS_MODELS, key)
        if raw is None:
            raise NotFound(f"model {key} not found")
        return ModelRecord.model_validate(raw)

    def _versions(self, name: str) -> list[ModelRecord]:
        return sorted(
            (
                ModelRecord.model_validate(v)
                for v in self.store.values(NS_MODELS, prefix=f"{name}@")
                if v.get("name") == name
            ),
            key=lambda r: r.created_at,
        )

    def find_model(self, model: str, version: str | None = None) -> ModelRecord | None:
        """``name``, ``name@version`` or (name, version). Without a version: newest approved, else newest."""
        if "@" in model and version is None:
            model, version = model.split("@", 1)
        if version is not None:
            raw = self.store.get(NS_MODELS, f"{model}@{version}")
            return ModelRecord.model_validate(raw) if raw else None
        versions = self._versions(model)
        approved = [r for r in versions if r.status == "approved"]
        return (approved or versions or [None])[-1]

    def get_model(self, model: str, version: str | None = None) -> ModelRecord:
        rec = self.find_model(model, version)
        if rec is None:
            raise NotFound(f"model {model}{'@' + version if version else ''} not found")
        return rec

    def list_models(self, *, status: str | None = None, slot: str | None = None) -> list[ModelRecord]:
        out = [ModelRecord.model_validate(v) for v in self.store.values(NS_MODELS)]
        return [r for r in out if (status is None or r.status == status) and (slot is None or r.slot == slot)]

    def update_model(
        self, model: str, version: str, changes: dict[str, Any], principal: Principal | None = None
    ) -> ModelRecord:
        with span("registry.model.update"):
            self._require(principal, self.writer_roles, "registry.model.update")
            rec = self._load_model(f"{model}@{version}")
            if rec.status != "draft":
                raise ValidationFailed(
                    f"{rec.key} is {rec.status}: approved records are immutable, register a new version"
                )
            bad = set(changes) - _MODEL_MUTABLE
            if bad:
                raise ValidationFailed(f"fields not editable: {sorted(bad)}")
            data = rec.model_dump()
            data.update(changes)
            if "parameters_b" in changes and "parameter_tier" not in changes:
                data["parameter_tier"] = None
            new = ModelRecord.model_validate(data)
            self._history(new, principal, "updated", fields=sorted(changes))
            self.store.put(NS_MODELS, new.key, new.model_dump(mode="json"))
            _audit(
                principal,
                "registry.model.update",
                resource=f"model:{new.key}",
                decision="updated",
                fields=sorted(changes),
            )
            return new

    def approve_model(
        self, model: str, version: str, principal: Principal | None, note: str = ""
    ) -> ModelRecord:
        """Release gate: required evidence present, four eyes, then ``approved``."""
        with span("registry.model.approve"):
            self._require(principal, self.approver_roles, "registry.model.approve")
            rec = self._load_model(f"{model}@{version}")
            self._transition(rec, "approved")
            missing = [f for f in self.cfg.approval_requires if not getattr(rec, f, None)]
            if missing:
                _audit(
                    principal,
                    "registry.model.approve",
                    resource=f"model:{rec.key}",
                    decision="rejected",
                    missing=missing,
                )
                raise ValidationFailed(f"cannot approve {rec.key}: missing {missing}")
            if self.cfg.four_eyes and principal is not None and rec.registered_by == principal.sub:
                _audit(
                    principal,
                    "registry.model.approve",
                    resource=f"model:{rec.key}",
                    decision="rejected",
                    reason="four_eyes",
                )
                raise PermissionDenied("the person who registered a model cannot approve it (four eyes)")
            rec.status, rec.approved_by, rec.approved_at = "approved", _actor(principal), _now()
            self._history(rec, principal, "approved", note=note)
            self.store.put(NS_MODELS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.model.approve",
                resource=f"model:{rec.key}",
                decision="approved",
                note=note,
                checksum=rec.weights_checksum,
                eval_scores=rec.eval_scores,
            )
            return rec

    def deprecate_model(
        self, model: str, version: str, principal: Principal | None, reason: str = ""
    ) -> ModelRecord:
        with span("registry.model.deprecate"):
            self._require(principal, self.approver_roles, "registry.model.deprecate")
            rec = self._load_model(f"{model}@{version}")
            self._transition(rec, "deprecated")
            rec.status, rec.deprecated_reason = "deprecated", reason or None
            self._history(rec, principal, "deprecated", reason=reason)
            self.store.put(NS_MODELS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.model.deprecate",
                resource=f"model:{rec.key}",
                decision="deprecated",
                reason=reason,
            )
            return rec

    def set_disabled(
        self,
        model: str,
        disabled: bool,
        reason: str,
        principal: Principal | None = None,
        version: str | None = None,
    ) -> list[ModelRecord]:
        """Incident switch: disable (or re-enable) every version, or one version, of a model."""
        recs = [self._load_model(f"{model}@{version}")] if version else self._versions(model)
        if not recs:
            raise NotFound(f"model {model} not found")
        for rec in recs:
            rec.disabled, rec.disabled_reason = disabled, reason if disabled else None
            self._history(rec, principal, "disabled" if disabled else "enabled", reason=reason)
            self.store.put(NS_MODELS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.model.disable" if disabled else "registry.model.enable",
                resource=f"model:{rec.key}",
                decision="disabled" if disabled else "enabled",
                reason=reason,
            )
        return recs

    def is_routable(self, model: str, data_class: DataClass | str | None = None) -> bool:
        """Only approved, non-disabled models are routable; optionally also for this data class."""
        rec = self.find_model(model)
        if rec is None or not rec.routable:
            return False
        if data_class is not None:
            return DataClass(data_class) in rec.data_classes_allowed
        return True

    def model_facts(self, model: str) -> dict[str, Any] | None:
        rec = self.find_model(model)
        if rec is None:
            return {"registry_status": "unregistered"} if self.cfg.require_registration else None
        return rec.facts()

    # ------------------------------------------------------------------ prompts
    def _head(self, name: str) -> PromptHead | None:
        raw = self.store.get(NS_PROMPTS, name)
        return PromptHead.model_validate(raw) if raw else None

    @staticmethod
    def _pv_key(name: str, version: int) -> str:
        return f"{name}@{version:06d}"

    def register_prompt(
        self,
        name: str,
        template: str,
        principal: Principal | None = None,
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        variables: list[str] | None = None,
        model: str | None = None,
        note: str = "",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Store a new version (identical content to the latest version is not duplicated)."""
        with span("registry.prompt.register", prompt=name):
            self._require(principal, self.writer_roles, "registry.prompt.register")
            if not name or "@" in name:
                raise ValidationFailed("prompt name must be non-empty and must not contain '@'")
            head = self._head(name) or PromptHead(name=name, description=description)
            pv = PromptVersion(
                name=name,
                version=head.latest + 1,
                template=template,
                tool_schemas=tool_schemas or [],
                variables=variables or [],
                model=model,
                note=note,
                metadata=metadata or {},
                created_by=_actor(principal),
            )
            pv.hash = pv.content_hash()
            if head.latest:
                latest = self.get_prompt(name, version=head.latest)
                if latest.hash == pv.hash:
                    return latest
            head.latest = pv.version
            head.description = description or head.description
            head.updated_at = _now()
            self.store.put(NS_PROMPT_VERSIONS, self._pv_key(name, pv.version), pv.model_dump(mode="json"))
            self.store.put(NS_PROMPTS, name, head.model_dump(mode="json"))
            _audit(
                principal,
                "registry.prompt.register",
                resource=f"prompt:{name}@{pv.version}",
                decision="registered",
                hash=pv.hash,
                note=note,
            )
            return pv

    def get_prompt(
        self, name: str, version: int | None = None, environment: str | None = None
    ) -> PromptVersion:
        head = self._head(name)
        if head is None:
            raise NotFound(f"prompt {name} not found")
        if version is None:
            if environment is not None:
                if environment not in head.active:
                    raise NotFound(f"prompt {name} has no active version in {environment}")
                version = head.active[environment]
            else:
                version = head.latest
        raw = self.store.get(NS_PROMPT_VERSIONS, self._pv_key(name, version))
        if raw is None:
            raise NotFound(f"prompt {name}@{version} not found")
        return PromptVersion.model_validate(raw)

    def prompt_versions(self, name: str) -> list[PromptVersion]:
        return [
            PromptVersion.model_validate(v)
            for v in self.store.values(NS_PROMPT_VERSIONS, prefix=f"{name}@")
            if v.get("name") == name
        ]

    def list_prompts(self) -> list[PromptHead]:
        return [PromptHead.model_validate(v) for v in self.store.values(NS_PROMPTS)]

    def activate_prompt(
        self, name: str, version: int, environment: str, principal: Principal | None = None
    ) -> PromptHead:
        """Point an environment at a version (deploy or rollback - every version is kept)."""
        with span("registry.prompt.activate", prompt=name, environment=environment):
            self._require(principal, self.writer_roles, "registry.prompt.activate")
            self.get_prompt(name, version=version)
            head = self._head(name)
            assert head is not None
            previous = head.active.get(environment)
            head.active[environment] = version
            head.updated_at = _now()
            self.store.put(NS_PROMPTS, name, head.model_dump(mode="json"))
            _audit(
                principal,
                "registry.prompt.activate",
                resource=f"prompt:{name}@{version}",
                decision="activated",
                environment=environment,
                previous=previous,
            )
            return head

    def diff_prompt(self, name: str, from_version: int, to_version: int) -> str:
        a, b = self.get_prompt(name, from_version), self.get_prompt(name, to_version)

        def render(p: PromptVersion) -> list[str]:
            body = p.template.splitlines()
            if p.tool_schemas:
                body += ["", "# tool schemas"] + json.dumps(
                    p.tool_schemas, indent=2, sort_keys=True
                ).splitlines()
            if p.variables:
                body += ["", "# variables: " + ", ".join(p.variables)]
            if p.model:
                body += ["# model: " + p.model]
            return body

        return "\n".join(
            difflib.unified_diff(
                render(a),
                render(b),
                fromfile=f"{name}@{a.version}",
                tofile=f"{name}@{b.version}",
                lineterm="",
            )
        )

    # ------------------------------------------------------------------ adapters
    def _load_adapter(self, key: str) -> AdapterRecord:
        raw = self.store.get(NS_ADAPTERS, key)
        if raw is None:
            raise NotFound(f"adapter {key} not found")
        return AdapterRecord.model_validate(raw)

    def register_adapter(
        self, record: AdapterRecord | dict[str, Any], principal: Principal | None = None
    ) -> AdapterRecord:
        with span("registry.adapter.register"):
            rec = record if isinstance(record, AdapterRecord) else AdapterRecord.model_validate(record)
            self._require(principal, self.writer_roles, "registry.adapter.register")
            if self.store.get(NS_ADAPTERS, rec.key) is not None:
                raise ValidationFailed(f"adapter {rec.key} already registered")
            if self.find_model(rec.base_model, rec.base_model_version) is None:
                raise ValidationFailed(
                    f"base model {rec.base_model}@{rec.base_model_version} is not registered"
                )
            rec.status, rec.approved_by, rec.approved_at = "draft", None, None
            rec.registered_by = _actor(principal)
            self._history(rec, principal, "registered")
            self.store.put(NS_ADAPTERS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.adapter.register",
                resource=f"adapter:{rec.key}",
                decision="draft",
                base=f"{rec.base_model}@{rec.base_model_version}",
                dataset_version=rec.dataset_version,
            )
            return rec

    def get_adapter(self, name: str, version: str | None = None) -> AdapterRecord:
        if version is not None:
            return self._load_adapter(f"{name}@{version}")
        recs = sorted(
            (
                AdapterRecord.model_validate(v)
                for v in self.store.values(NS_ADAPTERS, prefix=f"{name}@")
                if v.get("name") == name
            ),
            key=lambda r: r.created_at,
        )
        if not recs:
            raise NotFound(f"adapter {name} not found")
        approved = [r for r in recs if r.status == "approved"]
        return (approved or recs)[-1]

    def list_adapters(
        self, *, base_model: str | None = None, status: str | None = None
    ) -> list[AdapterRecord]:
        out = [AdapterRecord.model_validate(v) for v in self.store.values(NS_ADAPTERS)]
        return [
            r
            for r in out
            if (base_model is None or r.base_model == base_model) and (status is None or r.status == status)
        ]

    def approve_adapter(
        self, name: str, version: str, principal: Principal | None, note: str = ""
    ) -> AdapterRecord:
        """An adapter is approved only on an approved base and only when it beats the base."""
        with span("registry.adapter.approve"):
            self._require(principal, self.approver_roles, "registry.adapter.approve")
            rec = self._load_adapter(f"{name}@{version}")
            self._transition(rec, "approved")
            base = self.find_model(rec.base_model, rec.base_model_version)
            problems: list[str] = []
            if base is None or base.status != "approved":
                problems.append(f"base model {rec.base_model}@{rec.base_model_version} is not approved")
            cmp = rec.comparison()
            if self.cfg.adapter_must_beat_base and not cmp["beats_base"]:
                problems.append(
                    f"adapter does not beat its base (better={cmp['better']}, worse={cmp['worse']})"
                )
            if self.cfg.four_eyes and principal is not None and rec.registered_by == principal.sub:
                problems.append("the person who registered an adapter cannot approve it (four eyes)")
            if problems:
                _audit(
                    principal,
                    "registry.adapter.approve",
                    resource=f"adapter:{rec.key}",
                    decision="rejected",
                    problems=problems,
                )
                raise ValidationFailed(
                    f"cannot approve adapter {rec.key}: {'; '.join(problems)}", details={"comparison": cmp}
                )
            rec.status, rec.approved_by, rec.approved_at = "approved", _actor(principal), _now()
            self._history(rec, principal, "approved", note=note)
            self.store.put(NS_ADAPTERS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.adapter.approve",
                resource=f"adapter:{rec.key}",
                decision="approved",
                comparison=cmp["metrics"],
                note=note,
            )
            return rec

    def deprecate_adapter(
        self, name: str, version: str, principal: Principal | None, reason: str = ""
    ) -> AdapterRecord:
        with span("registry.adapter.deprecate"):
            self._require(principal, self.approver_roles, "registry.adapter.deprecate")
            rec = self._load_adapter(f"{name}@{version}")
            self._transition(rec, "deprecated")
            rec.status, rec.deprecated_reason = "deprecated", reason or None
            self._history(rec, principal, "deprecated", reason=reason)
            self.store.put(NS_ADAPTERS, rec.key, rec.model_dump(mode="json"))
            _audit(
                principal,
                "registry.adapter.deprecate",
                resource=f"adapter:{rec.key}",
                decision="deprecated",
                reason=reason,
            )
            return rec
