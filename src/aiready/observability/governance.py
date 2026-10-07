"""Governance helpers shared by observability, evals and fine-tuning.

One place that turns "check policy, write audit, count the decision" into a single call so no code path in
these packages can forget a step. Policy is *fail closed*: if the policy engine cannot be reached the action is
refused, because the actions guarded here (replaying a request, using request logs for training, releasing an
adapter) are exactly the ones a security review asks about.
"""

from __future__ import annotations

import logging
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.errors import PolicyViolation
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.telemetry import metrics

log = logging.getLogger(__name__)

SERVICE_PRINCIPAL = Principal(sub="svc:observability", roles=["service"], service="observability")


def service_principal(name: str) -> Principal:
    """Identity used for audit records written by background jobs (scorers, alert evaluator, CI runner)."""
    return Principal(sub=f"svc:{name}", roles=["service"], service=name)


async def authorize(principal: Principal, action: str, *, service: str, resource: dict[str, Any] | None = None,
                    context: dict[str, Any] | None = None, fail_open: bool = False) -> PolicyDecision:
    """Ask the policy engine; raise ``PolicyViolation`` on deny (after auditing the denial)."""
    from aiready.core.wiring import connect

    inp = PolicyInput(action=action, principal=principal.model_dump(exclude={"token", "claims"}),
                      resource=resource or {}, context={"service": service, **(context or {})})
    try:
        decision: PolicyDecision = await connect("policy").decide(inp)
    except PolicyViolation:
        raise
    except Exception as exc:
        if fail_open:
            log.warning("policy engine unavailable, fail-open configured for %s: %s", action, exc)
            decision = PolicyDecision(allow=True, reasons=[f"policy unavailable (fail-open): {exc}"])
        else:
            decision = PolicyDecision(allow=False, effect="deny", reasons=[f"policy engine unavailable: {exc}"])
    metrics.policy.labels(service, action, decision.effect if decision.allow else "deny").inc()
    if not decision.allow or decision.effect == "deny":
        get_audit().for_principal(principal, f"{service}.policy_denied", action=action,
                                  resource=str((resource or {}).get("id", "")), decision="deny",
                                  reasons=decision.reasons, matched_rules=decision.matched_rules)
        raise PolicyViolation(f"{action} denied: {'; '.join(decision.reasons) or 'policy'}",
                              details={"reasons": decision.reasons})
    return decision


def require_any_role(principal: Principal, roles: list[str], *, what: str) -> None:
    """Role gate with a readable message; an empty role list means "any signed-in user"."""
    if roles and not principal.has_role(*roles):
        from aiready.core.errors import PermissionDenied

        raise PermissionDenied(f"{what} requires one of roles {roles}")
