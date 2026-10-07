"""Governance service: policy decisions enriched with registry, egress and incident facts.

``decide()`` is the single policy decision point every service calls (``connect("policy")``). Before rules
run, the request is enriched with facts the caller must not be able to assert about itself:

* the deny-list (a revoked user or token is denied before any rule is evaluated);
* the model registry (``resource.registry_status``, ``boundary``, ``data_classes_allowed``...) - the registry,
  not the caller, says whether a model is approved and whether it is private or public;
* the egress policy (``context.egress_allowed``) for public models and ``egress.send``.

Caller-supplied values for these keys are overwritten, so a compromised or buggy caller cannot talk its
way past the registry or the egress policy.
"""

from __future__ import annotations

import logging
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import PolicyViolation
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.stores import DocStore, KVStore, get_docstore, make_kv
from aiready.core.telemetry import metrics, span
from aiready.core.types import DataClass
from aiready.governance.access_review import AccessReviewService
from aiready.governance.config import EgressConfig, GovernanceConfig
from aiready.governance.egress import EgressDecision, EgressPolicy
from aiready.governance.incident import IncidentService
from aiready.governance.policy import BundleInfo, build_engine
from aiready.governance.registry import (
    PUBLIC_AVAILABILITY,
    AdapterRecord,
    ModelRecord,
    PromptVersion,
    RegistryService,
)

log = logging.getLogger(__name__)
NS_BUNDLES = "policy.bundles"
_TRUSTED_RESOURCE_KEYS = ("registry_status", "registry_version", "data_classes_allowed", "disabled")
_TRUSTED_CONTEXT_KEYS = ("egress_allowed", "egress_reason", "egress_destination", "model_allows_data_class")


def derive_boundary(resource: dict[str, Any]) -> str | None:
    """private / public from whatever the caller described (registry facts take precedence)."""
    availability = resource.get("availability")
    if isinstance(availability, str):
        return "public" if availability in PUBLIC_AVAILABILITY or availability == "public" else "private"
    if resource.get("public") is True or resource.get("private") is False:
        return "public"
    if resource.get("private") is True or resource.get("public") is False:
        return "private"
    for key in ("location", "category", "boundary"):
        v = resource.get(key)
        if isinstance(v, str):
            v = v.lower()
            if v.startswith(("public", "vendor", "external")):
                return "public"
            if v.startswith(("private", "internal")):
                return "private"
    return None


def _data_class(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return DataClass(str(getattr(value, "value", value)).lower()).value
    except ValueError:
        return str(value).lower()


class GovernanceService:
    """Implements ``core.interfaces.PolicyEngine`` and owns the registries, egress, reviews and incidents."""

    def __init__(
        self,
        cfg: GovernanceConfig | None = None,
        egress_cfg: EgressConfig | None = None,
        *,
        engine: Any = None,
        store: DocStore | None = None,
        kv: KVStore | None = None,
    ):
        self.cfg = cfg or GovernanceConfig()
        self._store = store
        self.engine = engine if engine is not None else build_engine(self.cfg.policy)
        roles = self.cfg.api_roles
        self.registry = RegistryService(
            store, self.cfg.registry, approver_roles=roles.registry_approve, writer_roles=roles.registry_write
        )
        self.egress = EgressPolicy(egress_cfg or EgressConfig(), store)
        self.kv = kv or make_kv(self.cfg.kv)
        self.incidents = IncidentService(
            self.kv,
            store=store,
            registry=self.registry,
            egress=self.egress,
            revocation_ttl=self.cfg.revocation_ttl_seconds,
        )
        self.access_reviews = AccessReviewService(
            self._evaluate, self.registry, self.cfg.access_review, store, policy_version=self._policy_version
        )
        if self.cfg.registry.models:
            self.registry.seed_models(self.cfg.registry.models)
        self.engine.on_change(self._bundle_changed)
        self._record_bundle(None)

    @classmethod
    def from_config(cls) -> GovernanceService:
        conf = get_config()
        return cls(
            conf.section("governance", GovernanceConfig), conf.section("security.egress", EgressConfig)
        )

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    # ------------------------------------------------------------------ bundles
    def _policy_version(self) -> str:
        return str((self.engine.bundles().get("active") or {}).get("version", ""))

    def _record_bundle(self, old: BundleInfo | None) -> None:
        active = self.engine.bundles().get("active") or {}
        h = active.get("hash") or active.get("version")
        if h and self.store.get(NS_BUNDLES, h) is None:
            self.store.put(NS_BUNDLES, h, {**active, "previous": old.version if old else None})

    def _bundle_changed(self, old: BundleInfo | None, new: BundleInfo) -> None:
        self._record_bundle(old)
        get_audit().record(
            "config.change",
            actor="svc:governance",
            action="policy.bundle.load",
            resource=f"policy_bundle:{new.version}",
            decision="loaded",
            previous=old.version if old else None,
            rules=new.rules,
            domains=new.domains,
        )

    def bundles(self) -> dict[str, Any]:
        info = self.engine.bundles()
        info["history"] = sorted(
            self.store.values(NS_BUNDLES), key=lambda b: b.get("loaded_at", 0), reverse=True
        )
        return info

    def reload(self) -> bool:
        return bool(self.engine.reload(force=True))

    # ------------------------------------------------------------------ enrichment
    def enrich(self, inp: PolicyInput) -> PolicyInput:
        resource = dict(inp.resource or {})
        context = dict(inp.context or {})
        for k in _TRUSTED_RESOURCE_KEYS:
            resource.pop(k, None)
        for k in _TRUSTED_CONTEXT_KEYS:
            context.pop(k, None)
        dc = _data_class(context.get("data_class"))
        if dc:
            context["data_class"] = dc

        record = None
        model = resource.get("model")
        if inp.action.startswith("model.") and isinstance(model, str) and model:
            facts = self.registry.model_facts(model)
            if facts:
                resource.update(facts)
            record = self.registry.find_model(model)
        boundary = resource.get("boundary") if record is not None else derive_boundary(resource)
        if boundary:
            resource["boundary"] = boundary

        if inp.action.startswith("egress.") or resource.get("boundary") == "public":
            candidates = [
                str(c)
                for c in (
                    resource.get("destination"),
                    resource.get("egress_destination"),
                    record.egress_destination if record else None,
                    resource.get("endpoint"),
                    resource.get("url"),
                    model,
                    resource.get("category"),
                )
                if c
            ]
            # The first candidate the egress policy knows decides; otherwise the first one is judged (closed).
            dest = next(
                (c for c in candidates if self.egress.resolve(c, include_closed=True) is not None),
                candidates[0] if candidates else None,
            )
            if dest:
                eg = self.egress.check_egress(dest, dc, inp.principal, audit=False)
                context["egress_allowed"], context["egress_reason"] = eg.allowed, eg.reason
                context["egress_destination"] = eg.destination
            else:
                context["egress_allowed"], context["egress_reason"] = False, "no egress destination named"
        allowed = resource.get("data_classes_allowed")
        if dc and isinstance(allowed, list):
            context["model_allows_data_class"] = dc in allowed
        return PolicyInput(
            action=inp.action, principal=dict(inp.principal or {}), resource=resource, context=context
        )

    # ------------------------------------------------------------------ decisions
    async def _evaluate(self, inp: PolicyInput) -> PolicyDecision:
        """Decision without audit/metrics (access reviews evaluate thousands of synthetic inputs)."""
        revoked = self.incidents.is_revoked(inp.principal)
        if revoked:
            return PolicyDecision(
                allow=False,
                effect="deny",
                reasons=[revoked],
                matched_rules=["incident.revoked"],
                policy_version=self._policy_version(),
            )
        return await self.engine.decide(self.enrich(inp))

    async def decide(self, inp: PolicyInput) -> PolicyDecision:
        with span("governance.decide", action=inp.action) as s, metrics.time_stage("governance", "decide"):
            decision = await self._evaluate(inp)
            s.set(
                effect=decision.effect, policy_version=decision.policy_version, rules=decision.matched_rules
            )
            metrics.policy.labels("governance", inp.action, decision.effect).inc()
            if decision.effect != "allow" or self.cfg.policy.audit_allows:
                self._audit_decision(inp, decision)
            return decision

    async def explain(self, inp: PolicyInput) -> dict[str, Any]:
        with span("governance.explain", action=inp.action):
            revoked = self.incidents.is_revoked(inp.principal)
            enriched = self.enrich(inp)
            out = await self.engine.explain(enriched)
            if revoked:
                out["decision"] = PolicyDecision(
                    allow=False,
                    effect="deny",
                    reasons=[revoked],
                    matched_rules=["incident.revoked"],
                    policy_version=out.get("policy_version", ""),
                ).model_dump()
                out["revoked"] = revoked
            return out

    @staticmethod
    def _audit_decision(inp: PolicyInput, d: PolicyDecision) -> None:
        p = inp.principal or {}
        res = inp.resource or {}
        label = next(
            (
                f"{k}:{res[k]}"
                for k in ("model", "tool", "collection", "table", "destination", "name")
                if res.get(k)
            ),
            None,
        )
        get_audit().record(
            "policy.decision",
            actor=str(p.get("sub") or "anonymous"),
            tenant=str(p.get("tenant") or "default"),
            app=str(p.get("app") or "default"),
            on_behalf_of=p.get("actor") or p.get("service"),
            action=inp.action,
            resource=label,
            decision=d.effect,
            rules=d.matched_rules,
            reasons=d.reasons,
            policy_version=d.policy_version,
            data_class=(inp.context or {}).get("data_class"),
        )

    async def authorize(
        self,
        principal: Principal,
        action: str,
        resource: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Decide for a signed-in principal and raise ``PolicyViolation`` unless allowed."""
        d = await self.decide(
            PolicyInput(
                action=action,
                principal=principal.model_dump(),
                resource=resource or {},
                context=context or {},
            )
        )
        if not d.allow:
            raise PolicyViolation(
                f"{action} not allowed: {'; '.join(d.reasons)}",
                details={"effect": d.effect, "rules": d.matched_rules},
            )
        return d

    # ------------------------------------------------------------------ convenience (async, like the client)
    async def check_egress(
        self,
        destination: str,
        data_class: DataClass | str | None,
        principal: Principal | dict[str, Any] | None,
    ) -> EgressDecision:
        return self.egress.check_egress(destination, data_class, principal)

    async def is_routable(self, model: str, data_class: DataClass | str | None = None) -> bool:
        if self.incidents.route_disabled(model):
            return False
        return self.registry.is_routable(model, data_class)


class AsyncRegistry:
    """Async facade over the registry: what ``connect("registry")`` returns in-process.

    Same method signatures as ``aiready.governance.client.RegistryClient`` so callers (gateway, model
    serving, fine-tuning) work unchanged whether governance runs in-process or as its own pod.
    """

    def __init__(self, gov: GovernanceService):
        self.gov = gov
        self.registry = gov.registry

    async def is_routable(self, model: str, data_class: DataClass | str | None = None) -> bool:
        return await self.gov.is_routable(model, data_class)

    async def model_facts(self, model: str) -> dict[str, Any] | None:
        return self.registry.model_facts(model)

    async def get_model(self, model: str, version: str | None = None) -> ModelRecord:
        return self.registry.get_model(model, version)

    async def list_models(self, *, status: str | None = None, slot: str | None = None) -> list[ModelRecord]:
        return self.registry.list_models(status=status, slot=slot)

    async def register_model(self, record: ModelRecord | dict[str, Any], principal: Principal) -> ModelRecord:
        return self.registry.register_model(record, principal)

    async def approve_model(
        self, model: str, version: str, principal: Principal, note: str = ""
    ) -> ModelRecord:
        return self.registry.approve_model(model, version, principal, note)

    async def deprecate_model(
        self, model: str, version: str, principal: Principal, reason: str = ""
    ) -> ModelRecord:
        return self.registry.deprecate_model(model, version, principal, reason)

    async def get_prompt(
        self, name: str, version: int | None = None, environment: str | None = None
    ) -> PromptVersion:
        return self.registry.get_prompt(name, version, environment)

    async def register_prompt(
        self, name: str, template: str, principal: Principal, **kw: Any
    ) -> PromptVersion:
        return self.registry.register_prompt(name, template, principal, **kw)

    async def get_adapter(self, name: str, version: str | None = None) -> AdapterRecord:
        return self.registry.get_adapter(name, version)

    async def list_adapters(
        self, *, base_model: str | None = None, status: str | None = None
    ) -> list[AdapterRecord]:
        return self.registry.list_adapters(base_model=base_model, status=status)

    async def register_adapter(
        self, record: AdapterRecord | dict[str, Any], principal: Principal
    ) -> AdapterRecord:
        return self.registry.register_adapter(record, principal)
