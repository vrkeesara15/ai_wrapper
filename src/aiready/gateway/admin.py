"""Gateway control plane: Admin API (port 8081), separate from the data plane.

    uvicorn aiready.gateway.admin:create_admin_app --factory --port 8081

Every endpoint requires the ``platform-admin`` role (``gateway.admin.required_role``). Changes to routes,
models, pools, budgets and rate limits apply live through ``ControlPlane`` (no restart) and are audited as
``config.change`` with before/after values. Service API keys are created here (shown once, stored hashed).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, Field

from aiready.core.app import create_service_app
from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import ValidationFailed
from aiready.core.identity import Principal, current_principal
from aiready.core.wiring import connect
from aiready.gateway.config import GatewayConfig
from aiready.gateway.control import KINDS, ControlPlane
from aiready.gateway.keys import KeyService
from aiready.gateway.usage import UsageStore

log = logging.getLogger(__name__)


class KeyCreate(BaseModel):
    name: str
    app: str | None = None
    team: str | None = None
    scopes: list[str] = Field(default_factory=list)
    ttl_days: float | None = None


def _ts(value: str | None) -> float | None:
    """Accept epoch seconds or an ISO date / datetime (UTC when no zone is given)."""
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        raise ValidationFailed(f"invalid date '{value}': use epoch seconds or ISO 8601") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


async def _policy_bundles() -> dict[str, Any]:
    """Versioned policy bundles, from the governance service when it exposes them, else from configuration."""
    for name in ("registry", "policy"):
        try:
            svc = connect(name)
        except Exception as exc:  # governance not deployed in this process: fall back to configuration
            log.debug("no %s service for policy bundles: %s", name, exc)
            continue
        for attr in ("list_policy_bundles", "list_bundles", "bundles"):
            fn = getattr(svc, attr, None)
            if callable(fn):
                res = fn()
                if hasattr(res, "__await__"):
                    res = await res
                return {"source": name, "bundles": res}
    cfg = get_config()
    return {"source": "config", "config_version": cfg.version,
            "bundles": (cfg.get("policy") or {}).get("bundles") or (cfg.get("governance") or {}).get("bundles") or []}


def create_admin_app(gateway: Any = None, control: ControlPlane | None = None,
                     usage: UsageStore | None = None, keys: KeyService | None = None) -> FastAPI:
    app = create_service_app("gateway-admin", title="AI-Ready gateway admin API")
    ctl = control or ControlPlane()
    key_service = keys or KeyService()
    state: dict[str, Any] = {"usage": usage}

    def gw() -> Any:
        return gateway or connect("gateway")

    def usage_store() -> UsageStore:
        if state["usage"] is None:
            state["usage"] = UsageStore(get_config().section("gateway", GatewayConfig).usage.url)
        return state["usage"]

    async def admin(principal: Principal = Depends(current_principal)) -> Principal:
        role = get_config().section("gateway", GatewayConfig).admin.required_role
        principal.require_role(role)
        return principal

    # ---------------------------------------------------------------- routes, models, pools, budgets, limits
    @app.get("/admin/config/{kind}")
    async def list_kind(kind: str, p: Principal = Depends(admin)) -> dict[str, Any]:
        return {"kind": kind, "config_version": get_config().version, "items": ctl.list(kind)}

    @app.get("/admin/config/{kind}/{name}")
    async def get_item(kind: str, name: str, p: Principal = Depends(admin)) -> dict[str, Any]:
        return ctl.get(kind, name)

    @app.put("/admin/config/{kind}/{name}")
    async def put_item(kind: str, name: str, body: dict[str, Any], p: Principal = Depends(admin)) -> dict[str, Any]:
        return ctl.upsert(kind, name, body, p)

    @app.delete("/admin/config/{kind}/{name}")
    async def delete_item(kind: str, name: str, p: Principal = Depends(admin)) -> dict[str, Any]:
        return ctl.delete(kind, name, p)

    @app.get("/admin/config")
    async def overview(p: Principal = Depends(admin)) -> dict[str, Any]:
        return {"config_version": get_config().version, "kinds": sorted(KINDS),
                "counts": {k: len(ctl.list(k)) for k in KINDS}}

    @app.get("/admin/changes")
    async def changes(limit: int = Query(100, ge=1, le=1000), p: Principal = Depends(admin)) -> dict[str, Any]:
        return {"changes": ctl.history(limit)}

    @app.get("/admin/policy-bundles")
    async def policy_bundles(p: Principal = Depends(admin)) -> dict[str, Any]:
        return await _policy_bundles()

    # ---------------------------------------------------------------- keys
    @app.post("/admin/keys")
    async def create_key(body: KeyCreate, p: Principal = Depends(admin)) -> dict[str, Any]:
        plaintext, rec = key_service.create(body.name, app=body.app, team=body.team, scopes=body.scopes,
                                            ttl_days=body.ttl_days, created_by=p.sub)
        get_audit().for_principal(p, "config.change", action="gateway.keys.create", resource=f"keys/{rec.id}",
                                  decision="applied", before=None, after=rec.public())
        return {**rec.public(), "key": plaintext, "note": "store this key now: it cannot be shown again"}

    @app.get("/admin/keys")
    async def list_keys(p: Principal = Depends(admin)) -> dict[str, Any]:
        return {"keys": key_service.list()}

    @app.delete("/admin/keys/{key_id}")
    async def revoke_key(key_id: str, p: Principal = Depends(admin)) -> dict[str, Any]:
        before = key_service.get(key_id).public()
        rec = key_service.revoke(key_id)
        get_audit().for_principal(p, "config.change", action="gateway.keys.revoke", resource=f"keys/{key_id}",
                                  decision="applied", before=before, after=rec.public())
        return rec.public()

    # ---------------------------------------------------------------- usage, cache, pools
    @app.get("/admin/usage")
    async def usage_report(group_by: list[str] = Query(default=["team", "feature", "model"]),
                           start: str | None = None, end: str | None = None,
                           team: str | None = None, feature: str | None = None, user: str | None = None,
                           model: str | None = None, app_name: str | None = Query(None, alias="app"),
                           p: Principal = Depends(admin)) -> dict[str, Any]:
        filters = {k: v for k, v in {"team": team, "feature": feature, "user": user, "model": model,
                                     "app": app_name}.items() if v}
        s, e = _ts(start), _ts(end)
        rows = usage_store().report(group_by=group_by, start=s, end=e, filters=filters)
        cfg = get_config().section("gateway", GatewayConfig)
        return {"group_by": group_by, "start": s, "end": e, "filters": filters, "currency": cfg.currency,
                "rows": rows, "totals": {"requests": sum(r["requests"] for r in rows),
                                         "total_tokens": sum(r["total_tokens"] for r in rows),
                                         "cost": round(sum(r["cost"] for r in rows), 6)}}

    @app.post("/admin/cache/flush")
    async def flush(p: Principal = Depends(admin)) -> dict[str, Any]:
        g = gw()
        res = g.flush_cache() if not hasattr(g, "http") else await g.flush_cache(p)
        get_audit().for_principal(p, "config.change", action="gateway.cache.flush", resource="cache",
                                  decision="applied", before=None, after={"flushed": res})
        return {"flushed": res, "at": time.time()}

    @app.get("/admin/pools")
    async def pools(p: Principal = Depends(admin)) -> dict[str, Any]:
        g = gw()
        return g.pool_stats() if not hasattr(g, "http") else await g.pool_stats(p)

    return app
