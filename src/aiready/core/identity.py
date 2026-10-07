"""Identity: every call carries the signed-in user's identity.

* The **identity bridge** turns an existing app session (or an IdP token) into a short-lived, signed
  AI-Ready token holding the user, tenant, roles and groups (``TokenService.issue``).
* Every service verifies that token (``TokenService.verify``) and builds a ``Principal``.
* Internal services authenticate themselves with a *service* token in ``X-Service-Token`` while still
  forwarding the *user* token in ``Authorization``. An agent therefore always acts as the user - there are no
  shared service accounts with their own data rights.
* External IdP tokens (OIDC) can be verified directly against a JWKS URL.
"""

from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Request
from pydantic import BaseModel, Field

from aiready.core.errors import AuthenticationError, ConfigError, PermissionDenied

USER_HEADER = "Authorization"
SERVICE_HEADER = "X-Service-Token"


class Principal(BaseModel):
    """Who is calling. ``service`` is set when an internal service relays the call."""

    sub: str
    tenant: str = "default"
    app: str = "default"
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    email: str | None = None
    name: str | None = None
    service: str | None = None
    actor: str | None = None  # agent / client acting for the user (e.g. "agent:claims-assistant")
    jti: str | None = None  # token id, so a single token can be revoked during an incident
    token: str | None = Field(default=None, exclude=True, repr=False)
    claims: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @property
    def is_service_only(self) -> bool:
        return self.sub.startswith("svc:")

    @property
    def access_groups(self) -> list[str]:
        """Groups used to filter retrieval: the user's groups, roles and personal tag."""
        out = {f"user:{self.sub}", "public"}
        out.update(self.groups)
        out.update(f"role:{r}" for r in self.roles)
        return sorted(out)

    def has_role(self, *roles: str) -> bool:
        return bool(set(roles) & set(self.roles))

    def require_role(self, *roles: str) -> None:
        if not self.has_role(*roles):
            raise PermissionDenied(f"requires one of roles {list(roles)}")

    def forward_headers(self, service_token: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.token:
            h[USER_HEADER] = f"Bearer {self.token}"
        if service_token:
            h[SERVICE_HEADER] = service_token
        return h


class TokenService:
    """Issues and verifies AI-Ready tokens.

    Key material: HS256 with ``AIREADY_SECRET_TOKEN_SIGNING_KEY`` for small installs, RS256/ES256 with a
    private key (issuer only) and public key (every verifier) for production. Keys are held by the client
    (vault / HSM) and mounted as secrets.
    """

    def __init__(self, *, issuer: str = "aiready", audience: str = "aiready", algorithm: str = "HS256",
                 secret: str | None = None, private_key: str | None = None, public_key: str | None = None,
                 ttl_seconds: int = 900, leeway: int = 30, trusted_issuers: list[dict[str, Any]] | None = None):
        self.issuer, self.audience, self.algorithm = issuer, audience, algorithm
        self.ttl, self.leeway = ttl_seconds, leeway
        self.secret = secret
        self.private_key, self.public_key = private_key, public_key
        self.trusted_issuers = trusted_issuers or []
        if algorithm.startswith("HS") and not secret:
            raise ConfigError("HS* token signing requires a secret (secret:token_signing_key)")
        if not algorithm.startswith("HS") and not public_key:
            raise ConfigError(f"{algorithm} token verification requires public_key")
        self._jwks_clients: dict[str, jwt.PyJWKClient] = {}

    @property
    def _signing_key(self) -> str:
        key = self.secret if self.algorithm.startswith("HS") else self.private_key
        if not key:
            raise ConfigError("this service is not allowed to issue tokens (no signing key)")
        return key

    @property
    def _verify_key(self) -> str:
        return self.secret if self.algorithm.startswith("HS") else self.public_key  # type: ignore[return-value]

    def issue(self, *, sub: str, tenant: str = "default", app: str = "default", roles: list[str] | None = None,
              groups: list[str] | None = None, scopes: list[str] | None = None, email: str | None = None,
              name: str | None = None, actor: str | None = None, ttl: int | None = None,
              extra: dict[str, Any] | None = None) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": self.issuer, "aud": self.audience, "sub": sub, "iat": now, "nbf": now,
            "exp": now + (ttl or self.ttl), "jti": uuid.uuid4().hex,
            "tenant": tenant, "app": app, "roles": roles or [], "groups": groups or [], "scopes": scopes or [],
        }
        if email:
            claims["email"] = email
        if name:
            claims["name"] = name
        if actor:
            claims["act"] = {"sub": actor}
        if extra:
            claims.update(extra)
        return jwt.encode(claims, self._signing_key, algorithm=self.algorithm)

    def issue_service_token(self, service: str, ttl: int = 3600) -> str:
        return self.issue(sub=f"svc:{service}", roles=["service"], ttl=ttl)

    def verify(self, token: str) -> Principal:
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"malformed token: {exc}") from exc
        iss = unverified.get("iss")
        if iss != self.issuer:
            return self._verify_external(token, iss)
        try:
            claims = jwt.decode(token, self._verify_key, algorithms=[self.algorithm], audience=self.audience,
                                issuer=self.issuer, leeway=self.leeway)
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token expired") from exc
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"invalid token: {exc}") from exc
        return self._principal(claims, token)

    def _verify_external(self, token: str, iss: str | None) -> Principal:
        """Accept tokens from the client's own IdP when configured as trusted (OIDC + JWKS)."""
        for t in self.trusted_issuers:
            if t.get("issuer") != iss:
                continue
            url = t["jwks_url"]
            client = self._jwks_clients.setdefault(url, jwt.PyJWKClient(url, cache_keys=True))
            try:
                key = client.get_signing_key_from_jwt(token).key
                claims = jwt.decode(token, key, algorithms=t.get("algorithms", ["RS256"]),
                                    audience=t.get("audience"), issuer=iss, leeway=self.leeway)
            except jwt.PyJWTError as exc:
                raise AuthenticationError(f"invalid IdP token: {exc}") from exc
            mapping = t.get("claims", {})
            claims["roles"] = _claim(claims, mapping.get("roles", "roles"), [])
            claims["groups"] = _claim(claims, mapping.get("groups", "groups"), [])
            claims["tenant"] = _claim(claims, mapping.get("tenant", "tenant"), t.get("tenant", "default"))
            claims["sub"] = _claim(claims, mapping.get("sub", "sub"), claims.get("sub"))
            return self._principal(claims, token)
        raise AuthenticationError(f"untrusted token issuer '{iss}'")

    @staticmethod
    def _principal(claims: dict[str, Any], token: str) -> Principal:
        act = claims.get("act")
        return Principal(
            sub=str(claims["sub"]), tenant=claims.get("tenant", "default"), app=claims.get("app", "default"),
            roles=list(claims.get("roles") or []), groups=list(claims.get("groups") or []),
            scopes=list(claims.get("scopes") or []), email=claims.get("email"), name=claims.get("name"),
            actor=act.get("sub") if isinstance(act, dict) else None, jti=claims.get("jti"), token=token,
            claims=claims,
        )


def _claim(claims: dict[str, Any], path: str, default: Any) -> Any:
    node: Any = claims
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@lru_cache(maxsize=1)
def _env_token_service() -> TokenService:
    from aiready.core.config import get_config, resolve_secret

    cfg = get_config().get("security.tokens", {}) or {}
    algorithm = cfg.get("algorithm", "HS256")
    secret = cfg.get("secret") or resolve_secret("token_signing_key") or os.environ.get("AIREADY_DEV_SIGNING_KEY")
    return TokenService(
        issuer=cfg.get("issuer", "aiready"), audience=cfg.get("audience", "aiready"), algorithm=algorithm,
        secret=secret, private_key=cfg.get("private_key") or resolve_secret("token_private_key"),
        public_key=cfg.get("public_key") or resolve_secret("token_public_key"),
        ttl_seconds=int(cfg.get("ttl_seconds", 900)), trusted_issuers=cfg.get("trusted_issuers", []),
    )


_override: TokenService | None = None


def get_token_service() -> TokenService:
    return _override or _env_token_service()


def set_token_service(svc: TokenService | None) -> None:
    global _override
    _override = svc
    _env_token_service.cache_clear()


# --------------------------------------------------------------------------- revocation
# During an incident, governance puts a subject or token id on a deny-list (KV keys revoked:sub:<sub> and
# revoked:jti:<jti>). Every service checks it on every request, so a revoked user is cut off everywhere, not
# only at policy decision points. The governance service registers its own checker when it runs in-process;
# otherwise the shared KV from `governance.kv` is read directly (must be redis when services are split).

_revocation_checker: Any = None
_revocation_kv: Any = None


def set_revocation_checker(fn: Any) -> None:
    """Install ``fn(principal) -> reason | None``. ``None`` restores the KV-based default."""
    global _revocation_checker, _revocation_kv
    _revocation_checker = fn
    _revocation_kv = None


def check_revoked(p: Principal) -> str | None:
    global _revocation_kv
    if _revocation_checker is not None:
        return _revocation_checker(p)
    if _revocation_kv is None:
        from aiready.core.config import get_config
        from aiready.core.stores import make_kv

        spec = get_config().get("governance.kv") or {"type": "memory"}
        if (spec.get("type") if isinstance(spec, dict) else spec) == "memory":
            return None  # nothing shared to read; the in-process governance service registers its checker
        _revocation_kv = make_kv(spec)
    for key in (f"sub:{p.sub}", f"jti:{p.jti}" if p.jti else None):
        if key and _revocation_kv.get(f"revoked:{key}"):
            return f"{key} revoked"
    return None


def bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else value.strip()


def principal_from_headers(headers: Any, *, allow_service_only: bool = False) -> Principal:
    ts = get_token_service()
    user_token = bearer_token(headers.get(USER_HEADER) or headers.get(USER_HEADER.lower()))
    svc_token = headers.get(SERVICE_HEADER) or headers.get(SERVICE_HEADER.lower())
    service: str | None = None
    if svc_token:
        svc = ts.verify(svc_token)
        if not svc.is_service_only:
            raise AuthenticationError("X-Service-Token must be a service token")
        service = svc.sub[4:]
    if not user_token:
        if allow_service_only and service:
            return Principal(sub=f"svc:{service}", roles=["service"], service=service, token=svc_token)
        raise AuthenticationError("missing user token")
    p = ts.verify(user_token)
    reason = check_revoked(p)
    if reason:
        raise AuthenticationError(f"token rejected: {reason}")
    if p.is_service_only and not allow_service_only:
        raise AuthenticationError("a user token is required: agents act as the signed-in user")
    p.service = service
    return p


async def current_principal(request: Request) -> Principal:
    """FastAPI dependency: the verified caller."""
    p = principal_from_headers(request.headers)
    request.state.principal = p
    return p


async def current_principal_or_service(request: Request) -> Principal:
    p = principal_from_headers(request.headers, allow_service_only=True)
    request.state.principal = p
    return p


_bearer = bearer_token  # backwards-compatible alias
