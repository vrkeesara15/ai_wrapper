"""Guardrails HTTP API (port 8090).

    uvicorn aiready.guardrails.api:create_app --factory --port 8090

Screening endpoints accept a user token (relayed by a service) or a service token alone (ingestion
pipelines screening documents). Unmasking requires a signed-in user with an unmask role.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from aiready.core.app import create_service_app
from aiready.core.identity import Principal, current_principal, current_principal_or_service
from aiready.core.interfaces import ScreenResult, ValidationResult
from aiready.core.wiring import local as _local
from aiready.guardrails.service import GuardrailService


class ScreenBody(BaseModel):
    text: str
    context: dict[str, Any] = Field(default_factory=dict)


class ValidateBody(BaseModel):
    text: str
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class DocumentBody(BaseModel):
    text: str
    source: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class MaskBody(BaseModel):
    text: str


class UnmaskBody(BaseModel):
    text: str
    mapping: dict[str, str]
    purpose: str = ""


def create_app(service: GuardrailService | None = None) -> FastAPI:
    svc = service or _local("guardrails")
    app = create_service_app("guardrails", readiness=lambda: {"detectors": sorted(svc.cfg.detectors)})
    app.state.guardrails = svc

    def _ctx(body_ctx: dict[str, Any], p: Principal) -> dict[str, Any]:
        ctx = dict(body_ctx)
        ctx.pop("principal", None)  # the verified caller, never a body field
        ctx.setdefault("app", p.app)
        return ctx

    @app.post("/v1/guardrails/screen-input", response_model=ScreenResult)
    async def screen_input(
        body: ScreenBody, p: Principal = Depends(current_principal_or_service)
    ) -> ScreenResult:
        return await svc.screen_input(
            body.text, principal=None if p.is_service_only else p, context=_ctx(body.context, p)
        )

    @app.post("/v1/guardrails/screen-document", response_model=ScreenResult)
    async def screen_document(
        body: DocumentBody, p: Principal = Depends(current_principal_or_service)
    ) -> ScreenResult:
        return await svc.screen_document(
            body.text,
            source=body.source,
            principal=None if p.is_service_only else p,
            context=_ctx(body.context, p),
        )

    @app.post("/v1/guardrails/validate-output", response_model=ValidationResult)
    async def validate_output(
        body: ValidateBody, p: Principal = Depends(current_principal_or_service)
    ) -> ValidationResult:
        ctx = _ctx(body.context, p)
        if not p.is_service_only:
            ctx["principal"] = p
        return await svc.validate_output(body.text, schema=body.json_schema, context=ctx)

    @app.post("/v1/guardrails/mask", response_model=ScreenResult)
    async def mask(body: MaskBody, p: Principal = Depends(current_principal_or_service)) -> ScreenResult:
        return await svc.mask(body.text)

    @app.post("/v1/guardrails/unmask")
    async def unmask(body: UnmaskBody, p: Principal = Depends(current_principal)) -> dict[str, str]:
        return {"text": svc.unmask(body.text, body.mapping, p, purpose=body.purpose)}

    @app.get("/v1/guardrails/detectors")
    async def detectors(p: Principal = Depends(current_principal_or_service)) -> dict[str, Any]:
        return svc.detectors_info()

    return app
