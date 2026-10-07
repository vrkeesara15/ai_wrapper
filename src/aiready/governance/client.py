"""HTTP clients for the governance service (same signatures as the in-process objects).

``GovernanceClient`` is what ``connect("policy")`` returns when ``services.policy.url`` is set;
``RegistryClient`` is ``connect("registry")``. Policy decisions are requested with this service's own
token: the user identity travels inside ``PolicyInput.principal`` as the calling service saw it.
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.types import DataClass
from aiready.core.wiring import service_token
from aiready.governance.egress import EgressDecision
from aiready.governance.registry import AdapterRecord, ModelRecord, PromptVersion


def _client(url: str, timeout: float = 5.0, transport: Any = None) -> ServiceClient:
    return ServiceClient(
        url,
        service_token=service_token(os.environ.get("AIREADY_SERVICE", "aiready")),
        timeout=timeout,
        transport=transport,
    )


class GovernanceClient:
    def __init__(self, url: str, *, timeout: float = 5.0, transport: Any = None):
        self.http = _client(url, timeout, transport)

    async def decide(self, inp: PolicyInput) -> PolicyDecision:
        return PolicyDecision.model_validate(
            await self.http.json("POST", "/v1/policy/decide", json=inp.model_dump(mode="json"), retry=True)
        )

    async def explain(self, inp: PolicyInput) -> dict[str, Any]:
        return await self.http.json(
            "POST", "/v1/policy/explain", json=inp.model_dump(mode="json"), retry=True
        )

    async def bundles(self) -> dict[str, Any]:
        return await self.http.json("GET", "/v1/policy/bundles")

    async def check_egress(
        self, destination: str, data_class: DataClass | str | None, principal: Principal | None
    ) -> EgressDecision:
        dc = getattr(data_class, "value", data_class)
        return EgressDecision.model_validate(
            await self.http.json(
                "POST",
                "/v1/egress/check",
                principal=principal,
                json={"destination": destination, "data_class": dc},
                retry=True,
            )
        )

    async def is_routable(self, model: str, data_class: DataClass | str | None = None) -> bool:
        params = {"data_class": getattr(data_class, "value", data_class)} if data_class else None
        return bool(
            (await self.http.json("GET", f"/v1/registry/routable/{model}", params=params))["routable"]
        )

    async def aclose(self) -> None:
        await self.http.aclose()


class RegistryClient:
    def __init__(self, url: str, *, timeout: float = 5.0, transport: Any = None):
        self.http = _client(url, timeout, transport)

    async def is_routable(self, model: str, data_class: DataClass | str | None = None) -> bool:
        params = {"data_class": getattr(data_class, "value", data_class)} if data_class else None
        return bool(
            (await self.http.json("GET", f"/v1/registry/routable/{model}", params=params))["routable"]
        )

    async def model_facts(self, model: str) -> dict[str, Any] | None:
        return (await self.http.json("GET", f"/v1/registry/facts/{model}"))["facts"]

    async def record_evaluation(self, *, release: str, passed: bool, report: dict[str, Any],
                                artifact: dict[str, Any] | None = None,
                                principal: Principal | None = None) -> dict[str, Any]:
        return await self.http.json("POST", "/v1/registry/evaluations", principal=principal,
                                    json={"release": release, "passed": passed, "report": report,
                                          "artifact": artifact or {}})

    async def get_model(self, model: str, version: str | None = None) -> ModelRecord:
        params = {"version": version} if version else None
        return ModelRecord.model_validate(
            await self.http.json("GET", f"/v1/registry/models/{model}", params=params)
        )

    async def list_models(self, *, status: str | None = None, slot: str | None = None) -> list[ModelRecord]:
        params = {k: v for k, v in {"status": status, "slot": slot}.items() if v}
        return [
            ModelRecord.model_validate(m)
            for m in await self.http.json("GET", "/v1/registry/models", params=params)
        ]

    async def register_model(self, record: ModelRecord | dict[str, Any], principal: Principal) -> ModelRecord:
        body = record.model_dump(mode="json") if isinstance(record, ModelRecord) else record
        return ModelRecord.model_validate(
            await self.http.json("POST", "/v1/registry/models", json=body, principal=principal)
        )

    async def approve_model(
        self, model: str, version: str, principal: Principal, note: str = ""
    ) -> ModelRecord:
        return ModelRecord.model_validate(
            await self.http.json(
                "POST",
                f"/v1/registry/models/{model}/{version}/approve",
                json={"note": note},
                principal=principal,
            )
        )

    async def deprecate_model(
        self, model: str, version: str, principal: Principal, reason: str = ""
    ) -> ModelRecord:
        return ModelRecord.model_validate(
            await self.http.json(
                "POST",
                f"/v1/registry/models/{model}/{version}/deprecate",
                json={"reason": reason},
                principal=principal,
            )
        )

    async def get_prompt(
        self, name: str, version: int | None = None, environment: str | None = None
    ) -> PromptVersion:
        params = {k: v for k, v in {"version": version, "environment": environment}.items() if v is not None}
        return PromptVersion.model_validate(
            await self.http.json("GET", f"/v1/registry/prompts/{name}", params=params)
        )

    async def register_prompt(
        self, name: str, template: str, principal: Principal, **kw: Any
    ) -> PromptVersion:
        return PromptVersion.model_validate(
            await self.http.json(
                "POST",
                "/v1/registry/prompts",
                json={"name": name, "template": template, **kw},
                principal=principal,
            )
        )

    async def get_adapter(self, name: str, version: str | None = None) -> AdapterRecord:
        params = {"version": version} if version else None
        return AdapterRecord.model_validate(
            await self.http.json("GET", f"/v1/registry/adapters/{name}", params=params)
        )

    async def list_adapters(
        self, *, base_model: str | None = None, status: str | None = None
    ) -> list[AdapterRecord]:
        params = {k: v for k, v in {"base_model": base_model, "status": status}.items() if v}
        return [
            AdapterRecord.model_validate(a)
            for a in await self.http.json("GET", "/v1/registry/adapters", params=params)
        ]

    async def register_adapter(
        self, record: AdapterRecord | dict[str, Any], principal: Principal
    ) -> AdapterRecord:
        body = record.model_dump(mode="json") if isinstance(record, AdapterRecord) else record
        return AdapterRecord.model_validate(
            await self.http.json("POST", "/v1/registry/adapters", json=body, principal=principal)
        )

    async def aclose(self) -> None:
        await self.http.aclose()
