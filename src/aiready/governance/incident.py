"""Incident runbook actions: revoke keys, close routes, trace impact.

* **Revoke** puts a token id (``jti``) or a whole subject on a deny-list. The list lives in the KV store
  (redis in production, so every replica sees it within milliseconds) and is mirrored to the DocStore so a
  restart of an in-memory deployment does not silently lift a revocation. The policy engine checks the
  deny-list before evaluating any rule, so a revoked user is denied at every governed decision point.
* **Close routes** disables a model in the registry (no longer routable), marks a gateway route closed in
  the KV store (``route_disabled:<name>``, read by the gateway) and/or closes an egress destination.
* **Trace impact** reads the tamper-evident audit log for a time window and lists the affected requests,
  users and traces, so the incident report says exactly who was exposed.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.errors import ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, KVStore, get_docstore
from aiready.core.telemetry import span
from aiready.governance.egress import EgressPolicy
from aiready.governance.registry import RegistryService

NS_REVOKED = "incident.revocations"
NS_ROUTES = "incident.closed_routes"
NS_INCIDENTS = "incident.log"


class ImpactReport(BaseModel):
    start: float
    end: float
    filters: dict[str, Any]
    records: int
    traces: list[str] = Field(default_factory=list)
    users: dict[str, int] = Field(default_factory=dict)
    events: dict[str, int] = Field(default_factory=dict)
    samples: list[dict[str, Any]] = Field(default_factory=list)


def _sub_and_jti(principal: Principal | dict[str, Any]) -> tuple[str | None, str | None]:
    if isinstance(principal, Principal):
        return principal.sub, (principal.claims or {}).get("jti")
    claims = principal.get("claims") or {}
    return principal.get("sub"), principal.get("jti") or (
        claims.get("jti") if isinstance(claims, dict) else None
    )


class IncidentService:
    def __init__(
        self,
        kv: KVStore,
        *,
        store: DocStore | None = None,
        registry: RegistryService | None = None,
        egress: EgressPolicy | None = None,
        revocation_ttl: int = 86400,
    ):
        self.kv = kv
        self._store = store
        self.registry = registry
        self.egress = egress
        self.revocation_ttl = revocation_ttl
        self._warm()

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def _warm(self) -> None:
        """Re-populate the KV deny-list from the durable mirror (process restart, new redis)."""
        now = time.time()
        for key, entry in self.store.list(NS_REVOKED):
            exp = entry.get("expires_at")
            if exp is None or exp > now:
                self.kv.set(f"revoked:{key}", entry, ttl=(exp - now) if exp else None)
        for name, entry in self.store.list(NS_ROUTES):
            self.kv.set(f"route_disabled:{name}", entry)

    def _log(
        self, principal: Principal, event: str, resource: str, incident_id: str | None, **details: Any
    ) -> None:
        get_audit().for_principal(
            principal,
            event,
            action=event,
            resource=resource,
            decision="done",
            incident_id=incident_id,
            **details,
        )
        self.store.put(
            NS_INCIDENTS,
            f"{time.time():.6f}:{event}",
            {
                "event": event,
                "resource": resource,
                "by": principal.sub,
                "incident_id": incident_id,
                "ts": time.time(),
                **details,
            },
        )

    # ------------------------------------------------------------------ revocation
    def revoke(
        self,
        principal: Principal,
        *,
        jti: str | None = None,
        subject: str | None = None,
        reason: str = "",
        ttl: int | None = None,
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        if not jti and not subject:
            raise ValidationFailed("revoke needs a token id (jti) or a subject")
        with span("governance.incident.revoke"):
            out: dict[str, Any] = {"revoked": []}
            now = time.time()
            for kind, value in (("jti", jti), ("sub", subject)):
                if not value:
                    continue
                # A jti only needs to stay listed for the token lifetime; a subject stays until restored.
                life = (ttl or self.revocation_ttl) if kind == "jti" else ttl
                entry = {
                    "kind": kind,
                    "value": value,
                    "reason": reason,
                    "by": principal.sub,
                    "at": now,
                    "expires_at": now + life if life else None,
                    "incident_id": incident_id,
                }
                key = f"{kind}:{value}"
                self.kv.set(f"revoked:{key}", entry, ttl=life)
                self.store.put(NS_REVOKED, key, entry)
                self._log(principal, "incident.revoke", f"{kind}:{value}", incident_id, reason=reason)
                out["revoked"].append(key)
            return out

    def restore(
        self,
        principal: Principal,
        *,
        jti: str | None = None,
        subject: str | None = None,
        reason: str = "",
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        restored = []
        for kind, value in (("jti", jti), ("sub", subject)):
            if value:
                key = f"{kind}:{value}"
                self.kv.delete(f"revoked:{key}")
                self.store.delete(NS_REVOKED, key)
                self._log(principal, "incident.restore", key, incident_id, reason=reason)
                restored.append(key)
        return {"restored": restored}

    def is_revoked(self, principal: Principal | dict[str, Any] | None) -> str | None:
        """Returns the revocation reason when the principal's subject or token id is on the deny-list."""
        if not principal:
            return None
        sub, jti = _sub_and_jti(principal)
        for key in ([f"sub:{sub}"] if sub else []) + ([f"jti:{jti}"] if jti else []):
            entry = self.kv.get(f"revoked:{key}")
            if entry:
                return f"{key} revoked: {entry.get('reason') or 'incident'}"
        return None

    def revocations(self) -> list[dict[str, Any]]:
        now = time.time()
        return [e for _, e in self.store.list(NS_REVOKED) if not e.get("expires_at") or e["expires_at"] > now]

    # ------------------------------------------------------------------ routes
    def close_route(
        self,
        principal: Principal,
        *,
        model: str | None = None,
        route: str | None = None,
        egress_destination: str | None = None,
        reason: str = "",
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        if not (model or route or egress_destination):
            raise ValidationFailed("close_route needs a model, a route or an egress destination")
        with span("governance.incident.close_route"):
            out: dict[str, Any] = {}
            entry = {"reason": reason, "by": principal.sub, "at": time.time(), "incident_id": incident_id}
            if model:
                if self.registry is None:
                    raise ValidationFailed("no registry configured")
                recs = self.registry.set_disabled(model, True, reason or "incident", principal)
                out["models_disabled"] = [r.key for r in recs]
                self.kv.set(f"route_disabled:{model}", {**entry, "kind": "model"})
                self.store.put(NS_ROUTES, model, {**entry, "kind": "model"})
            if route:
                self.kv.set(f"route_disabled:{route}", {**entry, "kind": "route"})
                self.store.put(NS_ROUTES, route, {**entry, "kind": "route"})
                out["route_closed"] = route
            if egress_destination:
                if self.egress is None:
                    raise ValidationFailed("no egress policy configured")
                self.egress.close_route(egress_destination, principal, reason)
                out["egress_closed"] = egress_destination
            self._log(
                principal,
                "incident.close_route",
                ",".join(filter(None, [model, route, egress_destination])),
                incident_id,
                reason=reason,
                **out,
            )
            return out

    def reopen_route(
        self,
        principal: Principal,
        *,
        model: str | None = None,
        route: str | None = None,
        egress_destination: str | None = None,
        reason: str = "",
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if model and self.registry is not None:
            out["models_enabled"] = [
                r.key for r in self.registry.set_disabled(model, False, reason, principal)
            ]
        for name in filter(None, [model, route]):
            self.kv.delete(f"route_disabled:{name}")
            self.store.delete(NS_ROUTES, name)
        if egress_destination and self.egress is not None:
            self.egress.reopen_route(egress_destination, principal)
            out["egress_reopened"] = egress_destination
        self._log(
            principal,
            "incident.reopen_route",
            ",".join(filter(None, [model, route, egress_destination])),
            incident_id,
            reason=reason,
        )
        return out

    def route_disabled(self, name: str) -> bool:
        return self.kv.get(f"route_disabled:{name}") is not None

    def closed_routes(self) -> list[dict[str, Any]]:
        return [{"name": k, **v} for k, v in self.store.list(NS_ROUTES)]

    # ------------------------------------------------------------------ impact
    def trace_impact(
        self,
        principal: Principal,
        *,
        start: float,
        end: float | None = None,
        model: str | None = None,
        user: str | None = None,
        event_prefix: str | None = None,
        limit: int = 200,
        incident_id: str | None = None,
    ) -> ImpactReport:
        """Who and what was affected between ``start`` and ``end`` (filtered by model / user / event)."""
        end = end or time.time()
        with span("governance.incident.trace_impact"):
            traces: list[str] = []
            users: Counter[str] = Counter()
            events: Counter[str] = Counter()
            samples: list[dict[str, Any]] = []
            n = 0
            for rec in get_audit().read(since=start):
                if rec.ts > end or rec.event.startswith("incident."):
                    continue
                if user and user not in (rec.actor, rec.on_behalf_of):
                    continue
                if event_prefix and not rec.event.startswith(event_prefix):
                    continue
                if model and not _mentions_model(rec.resource, rec.details, model):
                    continue
                n += 1
                users[rec.actor] += 1
                events[rec.event] += 1
                if rec.trace_id and rec.trace_id not in traces:
                    traces.append(rec.trace_id)
                if len(samples) < limit:
                    samples.append(
                        {
                            "id": rec.id,
                            "ts": rec.ts,
                            "event": rec.event,
                            "actor": rec.actor,
                            "trace_id": rec.trace_id,
                            "resource": rec.resource,
                            "decision": rec.decision,
                        }
                    )
            report = ImpactReport(
                start=start,
                end=end,
                filters={"model": model, "user": user, "event_prefix": event_prefix},
                records=n,
                traces=traces,
                users=dict(users),
                events=dict(events),
                samples=samples,
            )
            self._log(
                principal,
                "incident.trace_impact",
                model or user or "*",
                incident_id,
                records=n,
                users=len(users),
                start=start,
                end=end,
            )
            return report


def _mentions_model(resource: str | None, details: dict[str, Any], model: str) -> bool:
    if resource and (
        resource == model
        or resource.startswith(f"{model}@")
        or resource in (f"model:{model}",)
        or resource.startswith(f"model:{model}@")
    ):
        return True
    for key in ("model", "route", "resolved_model", "upstream_model"):
        v = details.get(key)
        if isinstance(v, str) and (v == model or v.startswith(f"{model}@")):
            return True
    return False
