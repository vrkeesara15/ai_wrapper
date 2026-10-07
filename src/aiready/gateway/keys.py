"""Key service: API keys that identify a *calling application or service* (not a user).

A key proves which client relays a call - it never carries data rights of its own; the user token still
decides what the user may see. Keys are shown once at creation; only a salted SHA-256 hash is stored, and
verification is constant-time.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.config import resolve_secret
from aiready.core.errors import AuthenticationError, NotFound
from aiready.core.stores import DocStore, get_docstore

NAMESPACE = "gateway.keys"
PREFIX = "ark"


class ClientIdentity(BaseModel):
    """The verified calling client."""

    key_id: str
    name: str
    app: str | None = None
    team: str | None = None
    scopes: list[str] = Field(default_factory=list)


class ApiKeyRecord(BaseModel):
    id: str
    name: str
    app: str | None = None
    team: str | None = None
    scopes: list[str] = Field(default_factory=list)
    hash: str
    salt: str
    created_at: float = Field(default_factory=time.time)
    created_by: str | None = None
    expires_at: float | None = None
    revoked: bool = False
    last_used_at: float | None = None

    def public(self) -> dict[str, Any]:
        return self.model_dump(exclude={"hash", "salt"})


class KeyService:
    def __init__(self, store: DocStore | None = None, pepper: str | None = None):
        self._store = store
        self.pepper = pepper if pepper is not None else (resolve_secret("gateway_key_pepper") or "")

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def _hash(self, secret: str, salt: str) -> str:
        return hashlib.sha256(f"{self.pepper}:{salt}:{secret}".encode()).hexdigest()

    def create(self, name: str, *, app: str | None = None, team: str | None = None, scopes: list[str] | None = None,
               ttl_days: float | None = None, created_by: str | None = None) -> tuple[str, ApiKeyRecord]:
        """Returns (plaintext key, record). The plaintext is never stored and cannot be shown again."""
        key_id = uuid.uuid4().hex[:16]
        secret = secrets.token_urlsafe(32)
        salt = secrets.token_hex(8)
        rec = ApiKeyRecord(id=key_id, name=name, app=app, team=team, scopes=scopes or [], hash=self._hash(secret, salt),
                           salt=salt, created_by=created_by,
                           expires_at=time.time() + ttl_days * 86400 if ttl_days else None)
        self.store.put(NAMESPACE, key_id, rec.model_dump())
        return f"{PREFIX}_{key_id}_{secret}", rec

    def verify(self, key: str) -> ClientIdentity:
        try:
            prefix, key_id, secret = key.split("_", 2)
        except ValueError:
            raise AuthenticationError("malformed API key") from None
        raw = self.store.get(NAMESPACE, key_id) if prefix == PREFIX else None
        if raw is None:
            raise AuthenticationError("unknown API key")
        rec = ApiKeyRecord.model_validate(raw)
        if not hmac.compare_digest(rec.hash, self._hash(secret, rec.salt)):
            raise AuthenticationError("invalid API key")
        if rec.revoked:
            raise AuthenticationError("API key revoked")
        if rec.expires_at and rec.expires_at < time.time():
            raise AuthenticationError("API key expired")
        return ClientIdentity(key_id=rec.id, name=rec.name, app=rec.app, team=rec.team, scopes=rec.scopes)

    def list(self) -> list[dict[str, Any]]:
        return [ApiKeyRecord.model_validate(v).public() for v in self.store.values(NAMESPACE)]

    def get(self, key_id: str) -> ApiKeyRecord:
        raw = self.store.get(NAMESPACE, key_id)
        if raw is None:
            raise NotFound(f"API key '{key_id}' not found")
        return ApiKeyRecord.model_validate(raw)

    def revoke(self, key_id: str) -> ApiKeyRecord:
        rec = self.get(key_id)
        rec.revoked = True
        self.store.put(NAMESPACE, key_id, rec.model_dump())
        return rec
