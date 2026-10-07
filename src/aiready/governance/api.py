"""Governance HTTP API (port 8091): policy decisions, registries, egress, access reviews, incidents.

    uvicorn aiready.governance.api:create_app --factory --port 8091

Every endpoint authenticates the caller. Administrative endpoints are gated twice: by the role lists in
``governance.api_roles`` and by the policy engine itself (``admin.*`` actions), so policy-as-code can add
conditions (time windows, tenants...) without a code change. Revoked callers are refused outright.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from aiready.core.app import create_service_app
from aiready.core.errors import AuthenticationError, PermissionDenied
from aiready.core.identity import Principal, current_principal, current_principal_or_service
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.wiring import local as _local
from aiready.governance.config import EgressDestination
from aiready.governance.policy import PolicyTestReport, run_policy_cases
from aiready.governance.registry import AdapterRecord, ModelRecord
from aiready.governance.service import GovernanceService


class NoteBody(BaseModel):
    note: str = ""


class ReasonBody(BaseModel):
    reason: str = ""


class PromptBody(BaseModel):
    name: str
    template: str
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    model: str | None = None
    note: str = ""
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActivateBody(BaseModel):
    version: int
    environment: str


class EgressCheckBody(BaseModel):
    destination: str
    data_class: str | None = None


class SignOffBody(BaseModel):
    decisions: dict[str, Literal["keep", "revoke"]] = Field(default_factory=dict)
    comment: str = ""


class RevokeBody(BaseModel):
    jti: str | None = None
    subject: str | None = None
    reason: str = ""
    ttl_seconds: int | None = None
    incident_id: str | None = None


class RouteBody(BaseModel):
    model: str | None = None
    route: str | None = None
    egress_destination: str | None = None
    reason: str = ""
    incident_id: str | None = None


class ImpactBody(BaseModel):
    start: float
    end: float | None = None
    model: str | None = None
    user: str | None = None
    event_prefix: str | None = None
    limit: int = 200
    incident_id: str | None = None


class PolicyTestBody(BaseModel):
    cases: list[dict[str, Any]]


def create_app(service: GovernanceService | None = None) -> FastAPI:
    svc = service or _local("policy")
    if service is None:
        from aiready.core.identity import set_revocation_checker

        set_revocation_checker(svc.incidents.is_revoked)
    roles = svc.cfg.api_roles

    async def ready() -> dict[str, Any]:
        b = svc.engine.bundles()
        return {"policy_version": (b.get("active") or {}).get("version"), "policy_error": b.get("last_error")}

    app = create_service_app("governance", readiness=ready)
    app.state.governance = svc

    def _not_revoked(p: Principal) -> Principal:
        reason = svc.incidents.is_revoked(p)
        if reason:
            raise AuthenticationError("credentials revoked", details={"reason": reason})
        return p

    async def user(request: Request) -> Principal:
        return _not_revoked(await current_principal(request))

    async def caller(request: Request) -> Principal:
        return _not_revoked(await current_principal_or_service(request))

    async def admin(
        p: Principal, allowed: list[str], action: str, resource: dict[str, Any] | None = None
    ) -> None:
        if not p.has_role(*allowed):
            raise PermissionDenied(f"requires one of roles {allowed}")
        await svc.authorize(p, action, resource)

    def bind(inp: PolicyInput, p: Principal) -> PolicyInput:
        """A user-authenticated call decides for that user; only a service may decide for someone else."""
        if p.is_service_only:
            principal = dict(inp.principal or {})
            principal.setdefault("service", p.service)
            return inp.model_copy(update={"principal": principal})
        return inp.model_copy(update={"principal": p.model_dump()})

    # ------------------------------------------------------------------ policy
    @app.post("/v1/policy/decide", response_model=PolicyDecision)
    async def decide(inp: PolicyInput, p: Principal = Depends(caller)) -> PolicyDecision:
        return await svc.decide(bind(inp, p))

    @app.post("/v1/policy/explain")
    async def explain(inp: PolicyInput, p: Principal = Depends(caller)) -> dict[str, Any]:
        if not p.is_service_only:
            await admin(p, roles.policy_admin, "admin.policy.explain")
        return await svc.explain(
            inp
            if p.is_service_only
            else inp.model_copy(update={"principal": inp.principal or p.model_dump()})
        )

    @app.get("/v1/policy/bundles")
    async def bundles(p: Principal = Depends(caller)) -> dict[str, Any]:
        return svc.bundles()

    @app.post("/v1/policy/reload")
    async def reload(p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.policy_admin, "admin.policy.reload")
        changed = svc.reload()
        return {"reloaded": changed, **svc.engine.bundles()}

    @app.post("/v1/policy/test", response_model=PolicyTestReport)
    async def policy_test(body: PolicyTestBody, p: Principal = Depends(user)) -> PolicyTestReport:
        await admin(p, roles.policy_admin, "admin.policy.test")
        return await run_policy_cases(body.cases, svc.engine)

    # ------------------------------------------------------------------ registry: models
    @app.get("/v1/registry/models")
    async def list_models(
        status: str | None = None, slot: str | None = None, p: Principal = Depends(caller)
    ) -> list[dict[str, Any]]:
        return [m.public() for m in svc.registry.list_models(status=status, slot=slot)]

    @app.post("/v1/registry/models")
    async def register_model(rec: ModelRecord, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.registry_write, "admin.registry.register", {"model": rec.name})
        return svc.registry.register_model(rec, p).public()

    @app.get("/v1/registry/models/{name}")
    async def get_model(
        name: str, version: str | None = None, p: Principal = Depends(caller)
    ) -> dict[str, Any]:
        return svc.registry.get_model(name, version).public()

    @app.patch("/v1/registry/models/{name}/{version}")
    async def update_model(
        name: str, version: str, changes: dict[str, Any], p: Principal = Depends(user)
    ) -> dict[str, Any]:
        await admin(p, roles.registry_write, "admin.registry.update", {"model": name})
        return svc.registry.update_model(name, version, changes, p).public()

    @app.post("/v1/registry/models/{name}/{version}/approve")
    async def approve_model(
        name: str, version: str, body: NoteBody | None = None, p: Principal = Depends(user)
    ) -> dict[str, Any]:
        await admin(p, roles.registry_approve, "admin.registry.approve", {"model": name})
        return svc.registry.approve_model(name, version, p, (body or NoteBody()).note).public()

    @app.post("/v1/registry/models/{name}/{version}/deprecate")
    async def deprecate_model(
        name: str, version: str, body: ReasonBody | None = None, p: Principal = Depends(user)
    ) -> dict[str, Any]:
        await admin(p, roles.registry_approve, "admin.registry.deprecate", {"model": name})
        return svc.registry.deprecate_model(name, version, p, (body or ReasonBody()).reason).public()

    @app.get("/v1/registry/routable/{name}")
    async def routable(
        name: str, data_class: str | None = None, p: Principal = Depends(caller)
    ) -> dict[str, Any]:
        return {"model": name, "routable": await svc.is_routable(name, data_class)}

    @app.get("/v1/registry/facts/{name}")
    async def facts(name: str, p: Principal = Depends(caller)) -> dict[str, Any]:
        return {"model": name, "facts": svc.registry.model_facts(name)}

    # ------------------------------------------------------------------ registry: prompts
    @app.get("/v1/registry/prompts")
    async def list_prompts(p: Principal = Depends(caller)) -> list[dict[str, Any]]:
        return [h.model_dump() for h in svc.registry.list_prompts()]

    @app.post("/v1/registry/prompts")
    async def register_prompt(body: PromptBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.registry_write, "admin.registry.prompt", {"prompt": body.name})
        data = body.model_dump()
        name, template = data.pop("name"), data.pop("template")
        return svc.registry.register_prompt(name, template, p, **data).model_dump()

    @app.get("/v1/registry/prompts/{name}")
    async def get_prompt(
        name: str, version: int | None = None, environment: str | None = None, p: Principal = Depends(caller)
    ) -> dict[str, Any]:
        return svc.registry.get_prompt(name, version, environment).model_dump()

    @app.get("/v1/registry/prompts/{name}/versions")
    async def prompt_versions(name: str, p: Principal = Depends(caller)) -> list[dict[str, Any]]:
        return [v.model_dump() for v in svc.registry.prompt_versions(name)]

    @app.post("/v1/registry/prompts/{name}/activate")
    async def activate_prompt(name: str, body: ActivateBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(
            p,
            roles.registry_write,
            "admin.registry.prompt",
            {"prompt": name, "environment": body.environment},
        )
        return svc.registry.activate_prompt(name, body.version, body.environment, p).model_dump()

    @app.get("/v1/registry/prompts/{name}/diff", response_class=PlainTextResponse)
    async def diff_prompt(
        name: str,
        from_version: int = Query(..., alias="from"),
        to_version: int = Query(..., alias="to"),
        p: Principal = Depends(caller),
    ) -> str:
        return svc.registry.diff_prompt(name, from_version, to_version)

    # ------------------------------------------------------------------ registry: adapters
    @app.get("/v1/registry/adapters")
    async def list_adapters(
        base_model: str | None = None, status: str | None = None, p: Principal = Depends(caller)
    ) -> list[dict[str, Any]]:
        return [a.model_dump() for a in svc.registry.list_adapters(base_model=base_model, status=status)]

    @app.post("/v1/registry/adapters")
    async def register_adapter(rec: AdapterRecord, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.registry_write, "admin.registry.register", {"adapter": rec.name})
        return svc.registry.register_adapter(rec, p).model_dump()

    @app.get("/v1/registry/adapters/{name}")
    async def get_adapter(
        name: str, version: str | None = None, p: Principal = Depends(caller)
    ) -> dict[str, Any]:
        rec = svc.registry.get_adapter(name, version)
        return {**rec.model_dump(), "comparison": rec.comparison()}

    @app.post("/v1/registry/adapters/{name}/{version}/approve")
    async def approve_adapter(
        name: str, version: str, body: NoteBody | None = None, p: Principal = Depends(user)
    ) -> dict[str, Any]:
        await admin(p, roles.registry_approve, "admin.registry.approve", {"adapter": name})
        return svc.registry.approve_adapter(name, version, p, (body or NoteBody()).note).model_dump()

    @app.post("/v1/registry/adapters/{name}/{version}/deprecate")
    async def deprecate_adapter(
        name: str, version: str, body: ReasonBody | None = None, p: Principal = Depends(user)
    ) -> dict[str, Any]:
        await admin(p, roles.registry_approve, "admin.registry.deprecate", {"adapter": name})
        return svc.registry.deprecate_adapter(name, version, p, (body or ReasonBody()).reason).model_dump()

    # ------------------------------------------------------------------ egress
    @app.post("/v1/egress/check")
    async def egress_check(body: EgressCheckBody, p: Principal = Depends(user)) -> dict[str, Any]:
        return svc.egress.check_egress(body.destination, body.data_class, p).model_dump()

    @app.get("/v1/egress/routes")
    async def egress_routes(p: Principal = Depends(caller)) -> dict[str, Any]:
        return {
            "default": svc.egress.cfg.default,
            "open": [d.model_dump() for d in svc.egress.destinations()],
            "closed": svc.egress.closed(),
        }

    @app.post("/v1/egress/routes")
    async def egress_open(dest: EgressDestination, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.egress.open", {"destination": dest.name})
        return svc.egress.open_route(dest, p).model_dump()

    @app.delete("/v1/egress/routes/{name}")
    async def egress_close(name: str, reason: str = "", p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.egress.close", {"destination": name})
        return svc.egress.close_route(name, p, reason)

    @app.get("/v1/egress/allowlist", response_class=PlainTextResponse)
    async def egress_allowlist(
        format: Literal["json", "squid", "envoy"] = "json", p: Principal = Depends(caller)
    ) -> str:
        if not p.is_service_only and not p.has_role(*roles.security):
            raise PermissionDenied("the egress allow-list is for the proxy and security admins")
        return svc.egress.allowlist(format)

    # ------------------------------------------------------------------ access reviews
    @app.post("/v1/access-reviews")
    async def review_generate(p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.access_review, "admin.access_review.generate")
        return (await svc.access_reviews.generate(p)).model_dump()

    @app.get("/v1/access-reviews")
    async def review_list(p: Principal = Depends(user)) -> list[dict[str, Any]]:
        await admin(p, roles.access_review, "admin.access_review.read")
        return [
            {k: v for k, v in r.model_dump().items() if k != "entries"} for r in svc.access_reviews.list()
        ]

    @app.get("/v1/access-reviews/{report_id}")
    async def review_get(report_id: str, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.access_review, "admin.access_review.read")
        return svc.access_reviews.get(report_id).model_dump()

    @app.post("/v1/access-reviews/{report_id}/signoff")
    async def review_signoff(
        report_id: str, body: SignOffBody, p: Principal = Depends(user)
    ) -> dict[str, Any]:
        await admin(p, roles.access_review, "admin.access_review.sign")
        return svc.access_reviews.sign_off(report_id, p, dict(body.decisions), body.comment).model_dump()

    # ------------------------------------------------------------------ incident runbook
    @app.post("/v1/incident/revoke")
    async def incident_revoke(body: RevokeBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.incident.revoke")
        return svc.incidents.revoke(
            p,
            jti=body.jti,
            subject=body.subject,
            reason=body.reason,
            ttl=body.ttl_seconds,
            incident_id=body.incident_id,
        )

    @app.post("/v1/incident/restore")
    async def incident_restore(body: RevokeBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.incident.restore")
        return svc.incidents.restore(
            p, jti=body.jti, subject=body.subject, reason=body.reason, incident_id=body.incident_id
        )

    @app.get("/v1/incident/revocations")
    async def incident_revocations(p: Principal = Depends(user)) -> list[dict[str, Any]]:
        await admin(p, roles.security, "admin.incident.read")
        return svc.incidents.revocations()

    @app.post("/v1/incident/close-route")
    async def incident_close(body: RouteBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.incident.close_route")
        return svc.incidents.close_route(
            p,
            model=body.model,
            route=body.route,
            egress_destination=body.egress_destination,
            reason=body.reason,
            incident_id=body.incident_id,
        )

    @app.post("/v1/incident/reopen-route")
    async def incident_reopen(body: RouteBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.incident.reopen_route")
        return svc.incidents.reopen_route(
            p,
            model=body.model,
            route=body.route,
            egress_destination=body.egress_destination,
            reason=body.reason,
            incident_id=body.incident_id,
        )

    @app.get("/v1/incident/closed-routes")
    async def incident_closed(p: Principal = Depends(caller)) -> list[dict[str, Any]]:
        return svc.incidents.closed_routes()

    @app.post("/v1/incident/trace-impact")
    async def incident_impact(body: ImpactBody, p: Principal = Depends(user)) -> dict[str, Any]:
        await admin(p, roles.security, "admin.incident.trace_impact")
        return svc.incidents.trace_impact(
            p,
            start=body.start,
            end=body.end,
            model=body.model,
            user=body.user,
            event_prefix=body.event_prefix,
            limit=body.limit,
            incident_id=body.incident_id,
        ).model_dump()

    return app
