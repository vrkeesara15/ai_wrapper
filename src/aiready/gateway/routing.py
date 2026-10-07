"""Step 5 - Route: pick the model by task, cost and policy.

1. Find the route: the highest-priority enabled route whose ``match`` fits the request (model alias, task,
   feature, app). A request naming a catalog model directly is pinned to it (when allowed).
2. Candidates: the route's models, restricted to those that passed the evaluation for this task
   (``eval_pass``), with the capabilities the request needs (tools, JSON).
3. Governance per candidate, evaluated up front so a fallback can never land on a model the request may not
   use:
     * the model must be enabled and approved in the registry;
     * the request's data class must be allowed on the model (public models allow nothing by default);
     * a public (outside) model additionally needs ``egress.enabled`` and policy ``egress.send`` = allow;
     * policy ``model.invoke`` must allow it.
4. Order: smallest tier first, then cheapest ("simple requests go to a small model; only the hard ones reach
   the large model"). Large prompts skip the small tier; models whose window cannot hold the prompt, or whose
   pool has no healthy replica, move to the back. Explicit ``fallback`` models follow the primary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiready.core.errors import PolicyViolation
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.telemetry import metrics
from aiready.core.types import ChatRequest, DataClass
from aiready.core.wiring import connect
from aiready.gateway.config import GatewayConfig, RouteConfig
from aiready.models.catalog import ModelCatalog, ModelSpec, Tier


@dataclass
class RoutePlan:
    route: str | None
    chain: list[ModelSpec]
    rejected: dict[str, str] = field(default_factory=dict)
    obligations: dict[str, Any] = field(default_factory=dict)
    policy_versions: list[str] = field(default_factory=list)
    route_config: RouteConfig | None = None


def find_route(cfg: GatewayConfig, catalog: ModelCatalog, req: ChatRequest,
               app: str | None = None) -> tuple[str | None, RouteConfig | None]:
    """Return (route name, route) or (None, None) for a request pinned to a catalog model."""
    meta = req.metadata or {}
    task, feature = meta.get("task"), meta.get("feature")
    if cfg.allow_direct_models and req.model in catalog and req.model not in cfg.routes:
        return None, None
    default = cfg.routes.get(cfg.default_route)
    # generic aliases ("auto", plus the default route's aliases) reach task/feature routes without a model list
    generic = {"auto", ""} | set(default.match.model if default else [])
    ordered = sorted(((n, r) for n, r in cfg.routes.items() if r.enabled), key=lambda x: -x[1].priority)
    for name, r in ordered:
        m = r.match
        if not (req.model == name or req.model in m.model or (not m.model and req.model in generic)):
            continue
        if m.task and task not in m.task:
            continue
        if m.feature and feature not in m.feature:
            continue
        if m.app and app not in m.app:
            continue
        return name, r
    if default is not None and default.enabled and req.model in generic:
        return cfg.default_route, default
    from aiready.core.errors import NotFound

    raise NotFound(f"no route or model named '{req.model}'")


def endpoint_of(catalog: ModelCatalog, spec: ModelSpec) -> str | None:
    """The first replica's address: what an egress rule (and the egress proxy) names as the destination."""
    pool = catalog.pools.get(spec.pool)
    for r in (pool.replicas if pool else []):
        url = r.get("base_url") or r.get("url")
        if url:
            return str(url)
    return None


def required_capabilities(req: ChatRequest) -> list[str]:
    caps = ["chat"]
    if req.tools:
        caps.append("tools")
    if req.response_format and req.response_format.get("type") in ("json_object", "json_schema"):
        caps.append("json")
    return caps


async def policy_decide(action: str, principal: Principal, resource: dict[str, Any],
                        context: dict[str, Any]) -> PolicyDecision:
    """Ask the policy engine; any failure is a deny (fail closed)."""
    try:
        decision = await connect("policy").decide(PolicyInput(action=action, principal=principal.model_dump(),
                                                              resource=resource, context=context))
    except Exception as exc:  # policy engine unreachable: never route around governance
        decision = PolicyDecision(allow=False, effect="deny", reasons=[f"policy engine error: {exc}"])
    metrics.policy.labels("gateway", action, decision.effect if decision.allow else "deny").inc()
    return decision


async def check_model(spec: ModelSpec, *, principal: Principal, data_class: DataClass, egress_enabled: bool,
                      context: dict[str, Any], plan: RoutePlan, action: str = "model.invoke",
                      endpoint: str | None = None) -> bool:
    """Governance for one model. Records the reason in ``plan.rejected`` when it may not be used."""
    if not spec.enabled or not spec.approved:
        plan.rejected[spec.name] = "disabled or not approved in the model registry"
        return False
    if not spec.allows(data_class):
        plan.rejected[spec.name] = f"data class '{data_class.value}' not allowed on this model"
        return False
    resource = {"model": spec.name, "tier": spec.tier.value, "category": spec.category.value,
                "public": spec.is_public, "licence": spec.licence, "pool": spec.pool}
    ctx = {**context, "data_class": data_class.value}
    if spec.is_public:
        if not egress_enabled:
            plan.rejected[spec.name] = "egress to public models is closed"
            return False
        # the governance service resolves the egress destination from the endpoint / model name
        eg = await policy_decide("egress.send", principal, {**resource, "endpoint": endpoint, "url": endpoint}, ctx)
        if eg.policy_version:
            plan.policy_versions.append(eg.policy_version)
        if not eg.allow or eg.effect != "allow":
            plan.rejected[spec.name] = "egress policy: " + ("; ".join(eg.reasons) or "denied")
            return False
    dec = await policy_decide(action, principal, {**resource, "endpoint": endpoint} if endpoint else resource, ctx)
    if dec.policy_version:
        plan.policy_versions.append(dec.policy_version)
    if not dec.allow or dec.effect != "allow":
        plan.rejected[spec.name] = "policy: " + ("; ".join(dec.reasons) or dec.effect)
        return False
    for k, v in (dec.obligations or {}).items():
        if k == "max_tokens" and k in plan.obligations:
            plan.obligations[k] = min(plan.obligations[k], v)
        else:
            plan.obligations[k] = v
    return True


async def plan_route(*, cfg: GatewayConfig, catalog: ModelCatalog, req: ChatRequest, principal: Principal,
                     data_class: DataClass, context: dict[str, Any], prompt_tokens: int,
                     healthy: Any) -> RoutePlan:
    name, route = find_route(cfg, catalog, req, context.get("app"))
    plan = RoutePlan(route=name, chain=[], route_config=route)
    caps = required_capabilities(req)
    if route is None:
        names, fallback = [req.model], []
    else:
        names = list(route.candidates)
        if route.eval_pass is not None:
            passed = set(route.eval_pass)
            for n in names:
                if n not in passed:
                    plan.rejected[n] = "did not pass the evaluation for this route"
            names = [n for n in names if n in passed]
        fallback = [n for n in route.fallback if n not in names]
    eligible: list[ModelSpec] = []
    for n in names + fallback:
        spec = catalog.find(n)
        if spec is None:
            plan.rejected[n] = "not in the model catalog"
            continue
        if not spec.has(*caps):
            plan.rejected[n] = f"lacks capability {[c for c in caps if c not in spec.capabilities]}"
            continue
        if await check_model(spec, principal=principal, data_class=data_class, egress_enabled=cfg.egress.enabled,
                             context={**context, "route": name}, plan=plan, endpoint=endpoint_of(catalog, spec)):
            eligible.append(spec)
            plan.rejected.pop(n, None)  # e.g. not eval-passed as primary but allowed as an explicit fallback
    if not eligible:
        raise PolicyViolation(f"no permitted model for route '{name or req.model}'",
                              details={"route": name, "data_class": data_class.value, "rejected": plan.rejected})

    reserve = cfg.call.reserve_output_tokens
    min_rank = 0
    if route is not None and route.escalate_prompt_tokens and prompt_tokens > route.escalate_prompt_tokens:
        min_rank = Tier.MID.rank

    def key(s: ModelSpec) -> tuple:
        return (not healthy(s.pool),                       # healthy pools first
                prompt_tokens + reserve > s.context_length,  # models that fit the prompt first
                s.tier.rank < min_rank,                     # escalation: small tier last for big prompts
                s.tier.rank, s.cost_per_1k_input + s.cost_per_1k_output, s.name)

    primary_pool = [s for s in eligible if s.name in names]
    ordered = sorted(primary_pool, key=key)
    if ordered:
        head, rest = ordered[0], ordered[1:]
    else:  # only explicit fallbacks are permitted
        head, rest = sorted(eligible, key=key)[0], []
    explicit = [s for s in eligible if s.name in fallback and s is not head]
    chain = [head] + explicit + [s for s in rest if s not in explicit]
    plan.chain = chain[: 1 + cfg.call.max_fallbacks]
    return plan
