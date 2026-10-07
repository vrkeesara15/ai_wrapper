"""HTTP client for the guardrails service: same signatures as ``GuardrailService``.

``connect("guardrails")`` returns this when ``services.guardrails.url`` is set. The user's token is forwarded
when a principal is given (or present as ``context["principal"]``); otherwise the call is made with this
service's own token (e.g. ingestion screening documents).
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.interfaces import ScreenResult, ValidationResult
from aiready.core.wiring import service_token


def _split(context: dict[str, Any] | None) -> tuple[dict[str, Any], Principal | None]:
    ctx = dict(context or {})
    p = ctx.pop("principal", None)
    ctx = {k: v for k, v in ctx.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    return ctx, p if isinstance(p, Principal) else None


class GuardrailsClient:
    def __init__(self, url: str, *, timeout: float = 5.0, transport: Any = None):
        self.http = ServiceClient(
            url,
            service_token=service_token(os.environ.get("AIREADY_SERVICE", "aiready")),
            timeout=timeout,
            transport=transport,
        )

    async def screen_input(
        self, text: str, *, principal: Principal | None = None, context: dict[str, Any] | None = None
    ) -> ScreenResult:
        ctx, p = _split(context)
        data = await self.http.json(
            "POST",
            "/v1/guardrails/screen-input",
            principal=principal or p,
            json={"text": text, "context": ctx},
            retry=True,
        )
        return ScreenResult.model_validate(data)

    async def screen_document(
        self,
        text: str,
        *,
        source: str | None = None,
        principal: Principal | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScreenResult:
        ctx, p = _split(context)
        data = await self.http.json(
            "POST",
            "/v1/guardrails/screen-document",
            principal=principal or p,
            json={"text": text, "source": source, "context": ctx},
            retry=True,
        )
        return ScreenResult.model_validate(data)

    async def validate_output(
        self, text: str, *, schema: dict[str, Any] | None = None, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        ctx, p = _split(context)
        data = await self.http.json(
            "POST",
            "/v1/guardrails/validate-output",
            principal=p,
            json={"text": text, "schema": schema, "context": ctx},
            retry=True,
        )
        return ValidationResult.model_validate(data)

    async def mask(self, text: str) -> ScreenResult:
        return ScreenResult.model_validate(
            await self.http.json("POST", "/v1/guardrails/mask", json={"text": text}, retry=True)
        )

    async def unmask(
        self, text: str, mapping: dict[str, str], principal: Principal, *, purpose: str = ""
    ) -> str:
        data = await self.http.json(
            "POST",
            "/v1/guardrails/unmask",
            principal=principal,
            json={"text": text, "mapping": mapping, "purpose": purpose},
        )
        return str(data["text"])

    async def aclose(self) -> None:
        await self.http.aclose()
