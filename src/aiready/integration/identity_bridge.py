"""Identity bridge: turns an existing application session into a short-lived, signed AI-Ready token.

Most legacy applications have their own login (session cookie, reverse-proxy SSO, a sessions table) and no
single sign-on. The bridge *trusts* that login instead of replacing it: a pluggable **session validator**
confirms the session with the system that owns it, the result is mapped to AI-Ready roles and groups by
configured rules, and the token service signs a token that every AI-Ready service verifies. The user's own
rights therefore flow through every call; there is no shared service account.

Validators (plugin kind ``session_validator``):

* ``introspect_http`` - calls the app's existing "who am I" endpoint with the user's cookie / header.
* ``oidc`` - verifies a token from the client's IdP via JWKS (token exchange).
* ``shared_session_store`` - reads the session the app keeps in Redis or a database table.
* ``header_trust`` - behind an authenticated reverse proxy; headers are trusted only from allowed CIDRs.

A client adds its own with ``class: their_pkg.module:Validator`` (any object with ``name`` and
``async validate(creds) -> SessionIdentity``).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import time
from typing import Any, Protocol

import httpx
import jwt
from pydantic import BaseModel, Field

from aiready.core import plugins
from aiready.core.audit import get_audit
from aiready.core.errors import AuthenticationError, ConfigError, PermissionDenied, PolicyViolation
from aiready.core.identity import TokenService, get_token_service
from aiready.core.telemetry import metrics, span
from aiready.integration.config import IdentityBridgeConfig, RoleMapping
from aiready.integration.jsonpath import extract

log = logging.getLogger(__name__)
SERVICE = "identity-bridge"


# =========================================================================== models


class SessionCredentials(BaseModel):
    """Everything the caller presented. Validators pick what they understand."""

    cookies: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)  # lower-cased names
    client_ip: str | None = None
    session: str | None = None  # explicit session id (body / refresh)
    id_token: str | None = None  # IdP token for token exchange

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


class SessionIdentity(BaseModel):
    """What a validator learned from the app (before role mapping)."""

    sub: str
    tenant: str | None = None
    email: str | None = None
    name: str | None = None
    app_roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    session: str | None = Field(default=None, repr=False)  # credential to forward to the app later
    validator: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    sub: str
    tenant: str
    roles: list[str]
    groups: list[str]
    validator: str


class SessionValidator(Protocol):
    name: str

    async def validate(self, creds: SessionCredentials) -> SessionIdentity: ...


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _identity_from_mapping(data: dict[str, Any], mappings: dict[str, Any], validator: str,
                           session: str | None) -> SessionIdentity:
    """Apply ``{sub: "$.user.login", roles: "$.roles", attributes: {region: "$.region"}, ...}``."""
    sub = extract(data, mappings.get("sub", "$.sub"))
    if sub in (None, ""):
        raise AuthenticationError(f"{validator}: no subject in session data")
    attrs = {k: extract(data, p) for k, p in (mappings.get("attributes") or {}).items()}
    return SessionIdentity(
        sub=str(sub),
        tenant=_opt_str(extract(data, mappings["tenant"])) if mappings.get("tenant") else None,
        email=_opt_str(extract(data, mappings.get("email", "$.email"))),
        name=_opt_str(extract(data, mappings.get("name", "$.name"))),
        app_roles=_as_list(extract(data, mappings.get("roles", "$.roles"))),
        groups=_as_list(extract(data, mappings.get("groups", "$.groups"))),
        attributes={k: v for k, v in attrs.items() if v is not None},
        session=session, validator=validator,
    )


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)


# =========================================================================== validators


@plugins.register("session_validator", "introspect_http")
class IntrospectHttpValidator:
    """Ask the application itself who owns the session (its "who am I" endpoint).

    Why: the app remains the authority on its sessions (expiry, logout, lockout) and nothing about its
    session format has to be reimplemented. Positive answers are cached briefly (keyed by a hash of the
    credential, never the credential itself) so token refreshes do not hammer the app.
    """

    def __init__(self, url: str, *, name: str = "introspect_http", method: str = "GET",
                 session_cookie: str | None = "LEGACYSESSID", forward_cookies: list[str] | None = None,
                 forward_headers: list[str] | None = None, mappings: dict[str, Any] | None = None,
                 timeout: float = 5.0, cache_ttl: float = 30.0, verify_tls: bool = True,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.name, self.url, self.method = name, url, method.upper()
        self.session_cookie = session_cookie
        self.forward_cookies = forward_cookies  # None -> only the session cookie
        self.forward_headers = [h.lower() for h in (forward_headers or [])]
        self.mappings = mappings or {"sub": "$.sub"}
        self.timeout, self.cache_ttl = timeout, cache_ttl
        self._client = httpx.AsyncClient(timeout=timeout, verify=verify_tls, transport=transport)
        self._cache: dict[str, tuple[float, SessionIdentity]] = {}

    def _request_parts(self, creds: SessionCredentials) -> tuple[dict[str, str], dict[str, str], str | None]:
        cookies: dict[str, str] = {}
        session = creds.session
        if self.session_cookie:
            session = creds.cookies.get(self.session_cookie) or session
            if session:
                cookies[self.session_cookie] = session
        for name in self.forward_cookies or []:
            if name in creds.cookies:
                cookies[name] = creds.cookies[name]
        headers = {h: creds.headers[h] for h in self.forward_headers if h in creds.headers}
        return cookies, headers, session

    async def validate(self, creds: SessionCredentials) -> SessionIdentity:
        cookies, headers, session = self._request_parts(creds)
        if not cookies and not headers:
            raise AuthenticationError(f"{self.name}: no session cookie or header presented")
        key = hashlib.sha256(json.dumps([cookies, headers], sort_keys=True).encode()).hexdigest()
        hit = self._cache.get(key)
        if hit and hit[0] > time.time():
            metrics.cache.labels(SERVICE, "introspection", "hit").inc()
            return hit[1]
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        if cookie_header:
            headers = {**headers, "cookie": cookie_header}
        try:
            r = await self._client.request(self.method, self.url, headers=headers)
        except httpx.HTTPError as exc:
            raise AuthenticationError(f"{self.name}: application unreachable: {exc}") from exc
        if r.status_code in (401, 403):
            raise AuthenticationError(f"{self.name}: session rejected by the application")
        if r.status_code >= 400:
            raise AuthenticationError(f"{self.name}: introspection failed with HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError as exc:
            raise AuthenticationError(f"{self.name}: introspection did not return JSON") from exc
        ident = _identity_from_mapping(data, self.mappings, self.name, session)
        if self.cache_ttl > 0:
            self._cache[key] = (time.time() + self.cache_ttl, ident)
            metrics.cache.labels(SERVICE, "introspection", "miss").inc()
        return ident


@plugins.register("session_validator", "oidc")
class OidcValidator:
    """Token exchange: verify a token from the client's IdP (signature via JWKS, issuer, audience, expiry)."""

    def __init__(self, issuer: str, *, name: str = "oidc", jwks_url: str | None = None, audience: str | None = None,
                 algorithms: list[str] | None = None, public_key: str | Any | None = None,
                 mappings: dict[str, Any] | None = None, leeway: int = 30, header: str = "authorization"):
        if not jwks_url and public_key is None:
            raise ConfigError("oidc validator needs jwks_url (or public_key)")
        self.name, self.issuer, self.audience = name, issuer, audience
        self.algorithms = algorithms or ["RS256"]
        self.public_key = public_key
        self.mappings = mappings or {"sub": "$.sub", "email": "$.email", "name": "$.name",
                                     "roles": "$.roles", "groups": "$.groups"}
        self.leeway, self.header = leeway, header.lower()
        self._jwks = jwt.PyJWKClient(jwks_url, cache_keys=True) if jwks_url else None

    def _token(self, creds: SessionCredentials) -> str | None:
        if creds.id_token:
            return creds.id_token
        raw = creds.header(self.header)
        if raw and raw.lower().startswith("bearer "):
            return raw[7:].strip()
        return raw

    async def validate(self, creds: SessionCredentials) -> SessionIdentity:
        token = self._token(creds)
        if not token:
            raise AuthenticationError(f"{self.name}: no IdP token presented")
        try:
            key = self.public_key if self.public_key is not None else self._jwks.get_signing_key_from_jwt(token).key  # type: ignore[union-attr]
            claims = jwt.decode(token, key, algorithms=self.algorithms, audience=self.audience, issuer=self.issuer,
                                leeway=self.leeway, options={"verify_aud": self.audience is not None})
        except (jwt.PyJWTError, jwt.PyJWKClientError) as exc:
            raise AuthenticationError(f"{self.name}: invalid IdP token: {exc}") from exc
        # The IdP token is not an app session; nothing to forward to the app.
        return _identity_from_mapping(claims, self.mappings, self.name, None)


@plugins.register("session_validator", "shared_session_store")
class SharedSessionStoreValidator:
    """Read the session directly from where the app keeps it (Redis key or a sessions table).

    Use when the app has no "who am I" endpoint. The lookup is read-only and goes to a replica / read
    connection where possible.
    """

    def __init__(self, url: str, *, name: str = "shared_session_store", backend: str | None = None,
                 cookie_name: str = "SESSIONID", key_template: str = "session:{id}",
                 query: str = "SELECT * FROM sessions WHERE id = :id", data_column: str | None = None,
                 expires_at_field: str | None = None, mappings: dict[str, Any] | None = None):
        self.name, self.url = name, url
        self.backend = backend or ("redis" if url.startswith(("redis://", "rediss://")) else "sql")
        self.cookie_name, self.key_template = cookie_name, key_template
        self.query, self.data_column, self.expires_at_field = query, data_column, expires_at_field
        self.mappings = mappings or {"sub": "$.user_id"}
        self._engine: Any = None
        self._redis: Any = None

    async def _load(self, session_id: str) -> dict[str, Any] | None:
        if self.backend == "redis":
            if self._redis is None:
                import redis.asyncio as aioredis  # optional dependency

                self._redis = aioredis.Redis.from_url(self.url, decode_responses=True)
            raw = await self._redis.get(self.key_template.format(id=session_id))
            return json.loads(raw) if raw else None
        from sqlalchemy import create_engine, text

        if self._engine is None:
            self._engine = create_engine(self.url, future=True)
        with self._engine.connect() as c:
            row = c.execute(text(self.query), {"id": session_id}).mappings().first()
        if row is None:
            return None
        data = dict(row)
        if self.data_column:
            raw = data.get(self.data_column)
            nested = json.loads(raw) if isinstance(raw, (str, bytes)) else (raw or {})
            data = {**{k: v for k, v in data.items() if k != self.data_column}, **nested}
        return data

    async def validate(self, creds: SessionCredentials) -> SessionIdentity:
        sid = creds.cookies.get(self.cookie_name) or creds.session
        if not sid:
            raise AuthenticationError(f"{self.name}: no session cookie '{self.cookie_name}'")
        data = await self._load(sid)
        if not data:
            raise AuthenticationError(f"{self.name}: unknown session")
        if self.expires_at_field:
            exp = extract(data, self.expires_at_field)
            if exp is not None and float(exp) < time.time():
                raise AuthenticationError(f"{self.name}: session expired")
        return _identity_from_mapping(data, self.mappings, self.name, sid)


@plugins.register("session_validator", "header_trust")
class HeaderTrustValidator:
    """Behind an authenticating reverse proxy (e.g. SiteMinder, oauth2-proxy) that sets identity headers.

    Headers are trivially forgeable, so they are accepted only when the *immediate peer* is one of the
    configured proxy networks. Anything else is rejected and audited.
    """

    def __init__(self, trusted_cidrs: list[str], *, name: str = "header_trust", user_header: str = "X-Remote-User",
                 roles_header: str | None = "X-Remote-Roles", groups_header: str | None = "X-Remote-Groups",
                 email_header: str | None = "X-Remote-Email", tenant_header: str | None = None,
                 session_header: str | None = None):
        if not trusted_cidrs:
            raise ConfigError("header_trust needs at least one trusted CIDR")
        self.name = name
        self.networks = [ipaddress.ip_network(c, strict=False) for c in trusted_cidrs]
        self.user_header, self.roles_header, self.groups_header = user_header, roles_header, groups_header
        self.email_header, self.tenant_header, self.session_header = email_header, tenant_header, session_header

    def _trusted(self, ip: str | None) -> bool:
        if not ip:
            return False
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in n for n in self.networks)

    async def validate(self, creds: SessionCredentials) -> SessionIdentity:
        user = creds.header(self.user_header)
        if not user:
            raise AuthenticationError(f"{self.name}: no {self.user_header} header")
        if not self._trusted(creds.client_ip):
            raise AuthenticationError(f"{self.name}: identity headers from untrusted address {creds.client_ip}")
        return SessionIdentity(
            sub=user,
            tenant=creds.header(self.tenant_header) if self.tenant_header else None,
            email=creds.header(self.email_header) if self.email_header else None,
            app_roles=_as_list(creds.header(self.roles_header)) if self.roles_header else [],
            groups=_as_list(creds.header(self.groups_header)) if self.groups_header else [],
            session=creds.header(self.session_header) if self.session_header else None,
            validator=self.name,
        )


# =========================================================================== mapping


def map_roles(ident: SessionIdentity, mapping: RoleMapping) -> tuple[list[str], list[str]]:
    """Apply role/group rules. Returns (roles, groups) for the AI-Ready token."""
    roles: set[str] = set(mapping.default_roles)
    groups: set[str] = set(mapping.default_groups) | set(ident.groups)
    mapped_app_roles: set[str] = set()
    for rule in mapping.rules:
        if rule.app_role is not None and rule.app_role not in ident.app_roles:
            continue
        if any(str(ident.attributes.get(k)) != str(v) for k, v in rule.attributes.items()):
            continue
        roles.update(rule.roles)
        groups.update(rule.groups)
        if rule.app_role:
            mapped_app_roles.add(rule.app_role)
    if mapping.keep_unmapped_roles:
        roles.update(f"{mapping.unmapped_prefix}{r}" for r in ident.app_roles if r not in mapped_app_roles)
    for ag in mapping.attribute_groups:
        val = ident.attributes.get(ag.attribute)
        for v in val if isinstance(val, list) else [val]:
            if v not in (None, ""):
                groups.add(ag.template.format(value=v))
    return sorted(roles), sorted(groups)


# =========================================================================== bridge


class IdentityBridge:
    """Validates sessions, maps identities, applies policy, issues and refreshes tokens, audits everything."""

    def __init__(self, config: IdentityBridgeConfig | None = None, *,
                 validators: list[SessionValidator] | None = None, token_service: TokenService | None = None,
                 validator_extra: dict[str, Any] | None = None):
        self.config = config or IdentityBridgeConfig()
        self._token_service = token_service
        if validators is None:
            validators = []
            for spec in self.config.validators:
                spec = dict(spec)
                spec.setdefault("name", spec.get("type") or spec.get("class", "custom"))
                validators.append(plugins.create("session_validator", spec, **(validator_extra or {})))
        self.validators: dict[str, SessionValidator] = {v.name: v for v in validators}
        if not self.validators:
            raise ConfigError("identity bridge needs at least one session validator")

    @property
    def token_service(self) -> TokenService:
        return self._token_service or get_token_service()

    # ------------------------------------------------------------------ exchange
    async def exchange(self, creds: SessionCredentials, *, validator: str | None = None) -> TokenResponse:
        with span("identity_bridge.exchange", validator=validator or "auto") as sp, \
                metrics.time_stage(SERVICE, "exchange"):
            ident = await self._validate(creds, validator)
            sp.set(sub=ident.sub, via=ident.validator)
            return await self._issue(ident, event="identity.token_issue")

    async def _validate(self, creds: SessionCredentials, validator: str | None) -> SessionIdentity:
        if validator is not None and validator not in self.validators:
            self._deny("anonymous", f"unknown validator '{validator}'", validator)
            raise AuthenticationError(f"unknown validator '{validator}'")
        candidates = [self.validators[validator]] if validator else list(self.validators.values())
        errors: list[str] = []
        for v in candidates:
            try:
                return await v.validate(creds)
            except AuthenticationError as exc:
                errors.append(exc.message)
        reason = "; ".join(errors) or "no credentials"
        self._deny("anonymous", reason, validator, client_ip=creds.client_ip)
        raise AuthenticationError("session could not be validated", details={"reasons": errors})

    def _deny(self, actor: str, reason: str, validator: str | None, **details: Any) -> None:
        metrics.policy.labels(SERVICE, "identity.token_issue", "deny").inc()
        get_audit().record("identity.token_denied", actor=actor, action="identity.token_issue",
                           resource=validator or "auto", decision="deny", reason=reason, **details)

    async def _issue(self, ident: SessionIdentity, *, event: str, extra_details: dict[str, Any] | None = None) -> TokenResponse:
        cfg = self.config
        roles, groups = map_roles(ident, cfg.role_mapping)
        tenant = ident.tenant or cfg.default_tenant
        principal_view = {"sub": ident.sub, "tenant": tenant, "app": cfg.app, "roles": roles, "groups": groups}
        if cfg.policy_check:
            from aiready.core.interfaces import PolicyInput
            from aiready.core.wiring import connect

            decision = await connect("policy").decide(PolicyInput(
                action="identity.token_issue", principal=principal_view,
                resource={"validator": ident.validator, "app": cfg.app},
                context={"attributes": ident.attributes}))
            if not decision.allow:
                self._deny(ident.sub, "; ".join(decision.reasons) or "policy denied", ident.validator,
                           policy_version=decision.policy_version, matched_rules=decision.matched_rules)
                raise PolicyViolation("token issuance denied by policy", details={"reasons": decision.reasons})
        extra: dict[str, Any] = {"idv": ident.validator}
        if cfg.embed_app_session and ident.session:
            extra[cfg.app_session_claim] = ident.session
        if ident.attributes:
            extra["attrs"] = {k: v for k, v in ident.attributes.items() if isinstance(v, (str, int, float, bool))}
        token = self.token_service.issue(sub=ident.sub, tenant=tenant, app=cfg.app, roles=roles, groups=groups,
                                         email=ident.email, name=ident.name, ttl=cfg.token_ttl_seconds, extra=extra)
        claims = jwt.decode(token, options={"verify_signature": False})
        metrics.policy.labels(SERVICE, "identity.token_issue", "allow").inc()
        get_audit().record(event, actor=ident.sub, tenant=tenant, app=cfg.app, action="identity.token_issue",
                           resource=ident.validator, decision="allow", jti=claims.get("jti"), roles=roles,
                           groups=groups, ttl=cfg.token_ttl_seconds, **(extra_details or {}))
        return TokenResponse(access_token=token, expires_in=cfg.token_ttl_seconds, sub=ident.sub, tenant=tenant,
                             roles=roles, groups=groups, validator=ident.validator)

    # ------------------------------------------------------------------ refresh
    async def refresh(self, token: str, creds: SessionCredentials | None = None) -> TokenResponse:
        """Re-issue a token for the same user after re-validating the app session.

        The old token may have expired up to ``refresh_grace_seconds`` ago. The session is checked again
        with the app (a logged-out session yields no new token), and the subject must not change.
        """
        with span("identity_bridge.refresh") as sp, metrics.time_stage(SERVICE, "refresh"):
            claims = self._decode_for_refresh(token)
            sp.set(sub=claims.get("sub"))
            creds = creds or SessionCredentials()
            session = claims.get(self.config.app_session_claim)
            if session and not creds.session:
                creds = creds.model_copy(update={"session": session})
            validator = claims.get("idv") if claims.get("idv") in self.validators else None
            ident = await self._validate(creds, validator)
            if ident.sub != claims.get("sub"):
                self._deny(str(claims.get("sub")), "session belongs to a different user", ident.validator)
                raise PermissionDenied("session does not belong to the token subject")
            return await self._issue(ident, event="identity.token_refresh",
                                     extra_details={"previous_jti": claims.get("jti")})

    def _decode_for_refresh(self, token: str) -> dict[str, Any]:
        ts = self.token_service
        key = ts.secret if ts.algorithm.startswith("HS") else ts.public_key
        try:
            claims = jwt.decode(token, key, algorithms=[ts.algorithm], audience=ts.audience, issuer=ts.issuer,
                                options={"verify_exp": False})
        except jwt.PyJWTError as exc:
            self._deny("anonymous", f"refresh with invalid token: {exc}", None)
            raise AuthenticationError(f"invalid token: {exc}") from exc
        if float(claims.get("exp", 0)) + self.config.refresh_grace_seconds < time.time():
            self._deny(str(claims.get("sub")), "token too old to refresh", claims.get("idv"))
            raise AuthenticationError("token expired beyond the refresh grace period")
        return claims

    # ------------------------------------------------------------------ jwks
    def jwks(self) -> dict[str, Any] | None:
        """Public keys for verifiers when tokens are signed asymmetrically; None for HS* (shared secret)."""
        ts = self.token_service
        if ts.algorithm.startswith("HS") or not ts.public_key:
            return None
        algo = jwt.get_algorithm_by_name(ts.algorithm)
        key = algo.prepare_key(ts.public_key)
        jwk = json.loads(algo.to_jwk(key))
        jwk.update({"use": "sig", "alg": ts.algorithm,
                    "kid": hashlib.sha256(json.dumps(jwk, sort_keys=True).encode()).hexdigest()[:16]})
        return {"keys": [jwk]}


def build_bridge(config: IdentityBridgeConfig | None = None, **kw: Any) -> IdentityBridge:
    if config is None:
        from aiready.integration.config import load_integration_config

        config = load_integration_config().identity_bridge
    return IdentityBridge(config, **kw)
