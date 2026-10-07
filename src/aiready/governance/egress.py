"""Egress policy: the only route out of the client's boundary, closed by default.

A public model or outside API is reachable only through a *destination* that names its hosts and the data
classes allowed to leave to it ("public route needs a policy entry naming the classes of data allowed to
leave"). Destinations come from ``security.egress.destinations`` in configuration, or are opened at runtime
by a security admin (persisted, audited). An incident can close any destination instantly.

The same allow-list is exported for the deployment's egress proxy (Squid ACL, Envoy virtual hosts or plain
JSON), so the network control and the application control can never disagree.
"""

from __future__ import annotations

import fnmatch
import json
import time
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics, span
from aiready.core.types import DataClass
from aiready.governance.config import EgressConfig, EgressDestination

NS_ROUTES = "egress.routes"  # destinations opened at runtime
NS_CLOSED = "egress.closed"  # destinations closed by an incident (overrides configuration)


class EgressDecision(BaseModel):
    allowed: bool
    destination: str | None = None
    data_class: str | None = None
    reason: str


def _host_of(destination: str) -> str:
    if "://" in destination:
        return (urlparse(destination).hostname or "").lower()
    return destination.split("/", 1)[0].split(":", 1)[0].lower()


def _principal_fields(principal: Principal | dict[str, Any] | None) -> tuple[str, list[str], list[str]]:
    if principal is None:
        return "", [], []
    if isinstance(principal, Principal):
        return principal.sub, principal.roles, principal.groups
    return (
        str(principal.get("sub", "")),
        list(principal.get("roles") or []),
        list(principal.get("groups") or []),
    )


class EgressPolicy:
    def __init__(self, cfg: EgressConfig | None = None, store: DocStore | None = None):
        self.cfg = cfg or EgressConfig()
        self._store = store

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    # ------------------------------------------------------------------ destinations
    def closed(self) -> dict[str, Any]:
        return dict(self.store.list(NS_CLOSED))

    def destinations(self, *, include_closed: bool = False) -> list[EgressDestination]:
        """Configured plus runtime-opened destinations; runtime entries replace configured ones by name."""
        by_name = {d.name: d for d in self.cfg.destinations}
        for _, raw in self.store.list(NS_ROUTES):
            d = EgressDestination.model_validate(raw)
            by_name[d.name] = d
        closed = self.closed()
        now = time.time()
        out = []
        for d in by_name.values():
            if not include_closed and (
                d.name in closed or not d.enabled or (d.expires_at and d.expires_at < now)
            ):
                continue
            out.append(d)
        return sorted(out, key=lambda d: d.name)

    def resolve(self, destination: str, *, include_closed: bool = False) -> EgressDestination | None:
        """Match a destination name, URL or host against the open destinations."""
        dests = self.destinations(include_closed=include_closed)
        for d in dests:
            if d.name == destination or destination in d.aliases:
                return d
        host = _host_of(destination)
        if not host:
            return None
        for d in dests:
            if any(fnmatch.fnmatchcase(host, h.lower()) for h in d.hosts):
                return d
        return None

    # ------------------------------------------------------------------ decision
    def check_egress(
        self,
        destination: str,
        data_class: DataClass | str | None,
        principal: Principal | dict[str, Any] | None = None,
        *,
        audit: bool = True,
    ) -> EgressDecision:
        """May data of ``data_class`` leave to ``destination`` for this principal? Closed unless opened."""
        with span("governance.egress.check", destination=destination):
            dc = DataClass(data_class).value if data_class else None
            decision = self._decide(destination, dc, principal)
            metrics.policy.labels("governance", "egress.send", "allow" if decision.allowed else "deny").inc()
            if audit and not decision.allowed:
                sub, _, _ = _principal_fields(principal)
                get_audit().record(
                    "governance.egress.denied",
                    actor=sub or "anonymous",
                    action="egress.send",
                    resource=destination,
                    decision="deny",
                    data_class=dc,
                    reason=decision.reason,
                )
            return decision

    def _decide(
        self, destination: str, dc: str | None, principal: Principal | dict[str, Any] | None
    ) -> EgressDecision:
        dest = self.resolve(destination)
        if dest is None:
            if self.resolve(destination, include_closed=True) is not None:
                return EgressDecision(
                    allowed=False,
                    destination=destination,
                    data_class=dc,
                    reason=f"egress route '{destination}' is closed",
                )
            return EgressDecision(
                allowed=False,
                destination=destination,
                data_class=dc,
                reason="egress is closed by default: no destination opens this route",
            )
        if dc is None:
            return EgressDecision(
                allowed=False,
                destination=dest.name,
                data_class=dc,
                reason="data class of the payload is unknown; egress requires a classification",
            )
        if dc not in [DataClass(c).value for c in dest.data_classes]:
            return EgressDecision(
                allowed=False,
                destination=dest.name,
                data_class=dc,
                reason=f"data class '{dc}' may not leave to '{dest.name}' (allowed: {dest.data_classes})",
            )
        sub, roles, groups = _principal_fields(principal)
        if dest.roles or dest.groups:
            if not (set(dest.roles) & set(roles) or set(dest.groups) & set(groups)):
                return EgressDecision(
                    allowed=False,
                    destination=dest.name,
                    data_class=dc,
                    reason=f"principal lacks a role/group required by '{dest.name}'",
                )
        return EgressDecision(
            allowed=True,
            destination=dest.name,
            data_class=dc,
            reason=f"egress opened to '{dest.name}' for data class '{dc}'",
        )

    # ------------------------------------------------------------------ changes (security admin)
    def open_route(
        self, destination: EgressDestination | dict[str, Any], principal: Principal | None
    ) -> EgressDestination:
        dest = (
            destination
            if isinstance(destination, EgressDestination)
            else EgressDestination.model_validate(destination)
        )
        if not dest.data_classes:
            raise ValidationFailed("an egress route must name the data classes allowed to leave")
        for c in dest.data_classes:
            DataClass(c)
        dest.owner = dest.owner or (principal.sub if principal else None)
        self.store.put(NS_ROUTES, dest.name, dest.model_dump(mode="json"))
        self.store.delete(NS_CLOSED, dest.name)
        self._audit(
            principal,
            "governance.egress.open",
            dest.name,
            "opened",
            hosts=dest.hosts,
            data_classes=dest.data_classes,
        )
        return dest

    def close_route(self, name: str, principal: Principal | None, reason: str = "") -> dict[str, Any]:
        if self.resolve(name, include_closed=True) is None:
            raise NotFound(f"egress destination '{name}' not found")
        entry = {
            "name": name,
            "closed_at": time.time(),
            "closed_by": principal.sub if principal else "svc:governance",
            "reason": reason,
        }
        self.store.put(NS_CLOSED, name, entry)
        self._audit(principal, "governance.egress.close", name, "closed", reason=reason)
        return entry

    def reopen_route(self, name: str, principal: Principal | None) -> None:
        self.store.delete(NS_CLOSED, name)
        self._audit(principal, "governance.egress.reopen", name, "reopened")

    @staticmethod
    def _audit(principal: Principal | None, event: str, resource: str, decision: str, **details: Any) -> None:
        if principal is not None:
            get_audit().for_principal(
                principal, event, action=event, resource=f"egress:{resource}", decision=decision, **details
            )
        else:
            get_audit().record(
                event,
                actor="svc:governance",
                action=event,
                resource=f"egress:{resource}",
                decision=decision,
                **details,
            )

    # ------------------------------------------------------------------ proxy export
    def allowlist(self, fmt: Literal["json", "squid", "envoy"] = "json") -> str:
        """The allow-list for the egress proxy. Hosts of closed destinations are left out."""
        dests = self.destinations()
        if fmt == "json":
            return json.dumps(
                {
                    "default": "deny",
                    "generated_at": int(time.time()),
                    "proxy": self.cfg.proxy,
                    "destinations": [
                        {
                            "name": d.name,
                            "hosts": d.hosts,
                            "ports": d.ports,
                            "data_classes": [DataClass(c).value for c in d.data_classes],
                        }
                        for d in dests
                    ],
                },
                indent=2,
            )
        if fmt == "squid":
            lines = ["# AI-Ready egress allow-list (generated). Everything not listed is denied."]
            for d in dests:
                acl = "aiready_" + "".join(ch if ch.isalnum() else "_" for ch in d.name)
                domains = " ".join(("." + h[2:]) if h.startswith("*.") else h for h in d.hosts)
                lines.append(f"acl {acl} dstdomain {domains}")
                lines.append(f"acl {acl}_ports port {' '.join(str(p) for p in d.ports)}")
                lines.append(f"http_access allow {acl} {acl}_ports")
            lines.append("http_access deny all")
            return "\n".join(lines) + "\n"
        if fmt == "envoy":
            return json.dumps(
                {
                    "virtual_hosts": [
                        {
                            "name": d.name,
                            "domains": [f"{h}:{p}" for h in d.hosts for p in d.ports] + d.hosts,
                            "routes": [{"match": {"prefix": "/"}, "route": {"cluster": f"egress_{d.name}"}}],
                        }
                        for d in dests
                    ],
                    "clusters": [
                        {
                            "name": f"egress_{d.name}",
                            "type": "LOGICAL_DNS",
                            "endpoints": [
                                {"host": h, "port": p} for h in d.hosts if "*" not in h for p in d.ports
                            ],
                        }
                        for d in dests
                    ],
                    "default_action": "deny",
                },
                indent=2,
            )
        raise ValidationFailed(f"unknown allow-list format '{fmt}' (json, squid, envoy)")
