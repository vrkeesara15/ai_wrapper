"""Governance: policy as code, model / prompt / adapter registries, egress, access reviews, incidents.

Wiring names: ``policy`` (``core.interfaces.PolicyEngine`` + explain/egress helpers) and ``registry``
(async registry facade). In-process by default; set ``services.policy.url`` / ``services.registry.url``
to use the governance pod (port 8091). Imports are kept light: heavy modules load inside the providers.
"""

from __future__ import annotations

from typing import Any

from aiready.core.wiring import provider


@provider("policy", "local")
def _policy_local(cfg: dict[str, Any]) -> Any:
    from aiready.core.identity import set_revocation_checker
    from aiready.governance.service import GovernanceService

    gov = GovernanceService.from_config()
    # every service in this process now refuses revoked subjects / token ids at authentication time
    set_revocation_checker(gov.incidents.is_revoked)
    return gov


@provider("policy", "http")
def _policy_http(cfg: dict[str, Any]) -> Any:
    from aiready.governance.client import GovernanceClient

    return GovernanceClient(cfg["url"], timeout=float(cfg.get("timeout", 5.0)))


@provider("registry", "local")
def _registry_local(cfg: dict[str, Any]) -> Any:
    from aiready.core.wiring import connect
    from aiready.governance.service import AsyncRegistry, GovernanceService

    policy = connect("policy")
    # Share the in-process governance service when there is one; otherwise build a local one.
    gov = policy if isinstance(policy, GovernanceService) else GovernanceService.from_config()
    return AsyncRegistry(gov)


@provider("registry", "http")
def _registry_http(cfg: dict[str, Any]) -> Any:
    from aiready.governance.client import RegistryClient

    return RegistryClient(cfg["url"], timeout=float(cfg.get("timeout", 5.0)))
