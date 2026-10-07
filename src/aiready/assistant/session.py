"""Session exchange for the widget: app session -> short-lived AI-Ready token.

The widget never sees the app's credentials. Either the host page already holds an AI-Ready (or trusted IdP)
bearer token and sends it, or the browser's own app session cookie travels to this endpoint (same site, or
``credentials: include`` with an allow-listed origin) and is relayed to the identity bridge, which validates it
against the app's login system and issues a signed token. The token is verified here before it is returned,
so a misconfigured bridge cannot hand the widget an unusable or foreign token.
"""

from __future__ import annotations

from typing import Any

import httpx

from aiready.assistant.config import IdentityBridgeConfig
from aiready.core.errors import AuthenticationError, ConfigError, UpstreamError
from aiready.core.identity import Principal, get_token_service, principal_from_headers
from aiready.core.telemetry import TRACE_HEADER, current_trace_id, span


def _session_view(principal: Principal, token: str) -> dict[str, Any]:
    exp = principal.claims.get("exp")
    return {"token": token, "token_type": "Bearer", "expires_at": exp, "user": {
        "sub": principal.sub, "name": principal.name, "email": principal.email, "app": principal.app,
        "tenant": principal.tenant, "roles": principal.roles}}


class SessionExchanger:
    def __init__(self, cfg: IdentityBridgeConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._transport = transport

    async def exchange(self, headers: Any, cookies: dict[str, str], app: str | None) -> dict[str, Any]:
        if headers.get("authorization"):
            p = principal_from_headers(headers)
            return _session_view(p, p.token or "")
        if not self.cfg.url:
            raise AuthenticationError("no bearer token and no identity bridge configured (assistant.identity_bridge.url)")
        wanted = self.cfg.forward_cookies
        jar = {k: v for k, v in cookies.items() if not wanted or k in wanted}
        if not jar:
            raise AuthenticationError("no app session found")
        fwd = {h: headers[h] for h in self.cfg.forward_headers if headers.get(h)}
        tid = current_trace_id()
        if tid:
            fwd[TRACE_HEADER] = tid
        fwd["Cookie"] = "; ".join(f"{k}={v}" for k, v in jar.items())
        url = self.cfg.url.rstrip("/") + self.cfg.exchange_path
        with span("assistant.session.exchange", app=app):
            try:
                async with httpx.AsyncClient(transport=self._transport, timeout=self.cfg.timeout_seconds) as client:
                    body = {"validator": self.cfg.validator} if self.cfg.validator else {}
                    r = await client.post(url, headers=fwd, json=body)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"identity bridge unreachable: {exc}") from exc
        if r.status_code in (401, 403):
            raise AuthenticationError("the app session was not accepted")
        if r.status_code >= 400:
            raise UpstreamError(f"identity bridge returned {r.status_code}")
        try:
            payload = r.json()
            token = payload.get("access_token") or payload["token"]
        except (ValueError, KeyError, AttributeError) as exc:
            raise ConfigError("identity bridge response has no 'access_token'") from exc
        principal = get_token_service().verify(token)
        if principal.is_service_only:
            raise AuthenticationError("identity bridge returned a service token, not a user token")
        return _session_view(principal, token)
