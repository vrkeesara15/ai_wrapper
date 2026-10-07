"""HTTP client for service-to-service calls.

Forwards the user's token, adds this service's own token and the trace id, applies a timeout and retries
idempotent failures with backoff. Every internal hop therefore keeps the caller's identity and trace.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from aiready.core.errors import UpstreamError
from aiready.core.identity import Principal
from aiready.core.telemetry import TRACE_HEADER, current_trace_id

RETRY_STATUS = {429, 502, 503, 504}


class ServiceClient:
    def __init__(self, base_url: str, *, service_token: str | None = None, timeout: float = 30.0,
                 retries: int = 2, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.service_token = service_token
        self.retries = retries
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport)

    def _headers(self, principal: Principal | None, extra: dict[str, str] | None) -> dict[str, str]:
        h = principal.forward_headers(self.service_token) if principal else {}
        if not principal and self.service_token:
            h["X-Service-Token"] = self.service_token
        tid = current_trace_id()
        if tid:
            h[TRACE_HEADER] = tid
        if extra:
            h.update(extra)
        return h

    async def request(self, method: str, path: str, *, principal: Principal | None = None,
                      json: Any = None, params: dict[str, Any] | None = None,
                      headers: dict[str, str] | None = None, retry: bool | None = None) -> httpx.Response:
        retry = method.upper() in {"GET", "HEAD", "PUT", "DELETE"} if retry is None else retry
        attempts = self.retries + 1 if retry else 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                r = await self._client.request(method, path, json=json, params=params,
                                               headers=self._headers(principal, headers))
                if r.status_code in RETRY_STATUS and i < attempts - 1:
                    await asyncio.sleep(0.2 * 2 ** i + random.random() * 0.1)
                    continue
                return r
            except httpx.TransportError as exc:
                last = exc
                if i < attempts - 1:
                    await asyncio.sleep(0.2 * 2 ** i + random.random() * 0.1)
        raise UpstreamError(f"{method} {self.base_url}{path} failed: {last}")

    async def json(self, method: str, path: str, **kw: Any) -> Any:
        r = await self.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {"text": r.text[:500]}
            err = UpstreamError(f"{method} {path} returned {r.status_code}", details=body)
            err.status_code = r.status_code if r.status_code < 500 else 502
            raise err
        return r.json() if r.content else None

    def stream(self, method: str, path: str, *, principal: Principal | None = None, json: Any = None):
        return self._client.stream(method, path, json=json, headers=self._headers(principal, None))

    async def aclose(self) -> None:
        await self._client.aclose()
