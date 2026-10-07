"""Tamper-evident audit log.

Every prompt, retrieved record, tool call, approval, policy decision, config change and answer is written
as an append-only record. Records form a hash chain (each record's hash covers the previous hash), and are
optionally HMAC-signed with a key held by the client, so deleting or editing any record breaks
``verify()``. Sinks are pluggable: JSONL file, SQL table, stdout (for log shippers), HTTP (client SIEM).
Several sinks can run at once; the first is the system of record used for verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import httpx
from pydantic import BaseModel, Field

from aiready.core.plugins import create, register

log = logging.getLogger("aiready.audit")


def _lock_file(f: Any) -> None:
    try:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except ImportError:  # pragma: no cover - Windows: msvcrt byte-range lock on the first byte
        import msvcrt

        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)


def _unlock_file(f: Any) -> None:
    try:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover
        import msvcrt

        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
GENESIS = "0" * 64


class AuditRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    event: str  # e.g. gateway.request, rag.retrieve, tool.call, approval.decision, config.change
    actor: str  # user sub (or svc:<name>)
    on_behalf_of: str | None = None
    tenant: str = "default"
    app: str = "default"
    trace_id: str | None = None
    action: str | None = None
    resource: str | None = None
    decision: str | None = None  # allow / deny / approved / rejected / blocked ...
    details: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""
    signature: str | None = None

    def canonical(self) -> bytes:
        body = self.model_dump(exclude={"hash", "signature"})
        return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()


class AuditSink(Protocol):
    def write(self, record: AuditRecord) -> None: ...
    def last_hash(self) -> str: ...
    def read(self, since: float | None = None, limit: int | None = None) -> Iterable[AuditRecord]: ...


@register("audit_sink", "jsonl")
class JsonlSink:
    def __init__(self, path: str = "data/audit/audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def append_chained(self, build: Callable[[str], AuditRecord]) -> AuditRecord:
        """Append under an exclusive file lock, chaining to the hash actually at the end of the file, so
        several processes (CLI, workers, a pilot running every service) can share one chain without forking it."""
        with self.path.open("a+", encoding="utf-8") as f:
            _lock_file(f)
            try:
                rec = build(self.last_hash())
                f.write(rec.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock_file(f)
        return rec

    def last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with self.path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 65536)
            f.seek(size - block)
            lines = f.read().splitlines()
        for line in reversed(lines):
            if line.strip():
                return json.loads(line)["hash"]
        return GENESIS

    def read(self, since: float | None = None, limit: int | None = None) -> Iterable[AuditRecord]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = AuditRecord.model_validate_json(line)
                if since is None or rec.ts >= since:
                    out.append(rec)
        return out[-limit:] if limit else out


@register("audit_sink", "sqlite")
class SqliteSink:
    def __init__(self, path: str = "data/audit/audit.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, ts REAL, "
            "event TEXT, actor TEXT, tenant TEXT, trace_id TEXT, hash TEXT, body TEXT)"
        )
        # Append-only at the database level: updates and deletes are refused.
        self._db.execute("CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit "
                         "BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END")
        self._db.execute("CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit "
                         "BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END")
        self._db.commit()

    def write(self, record: AuditRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO audit (id, ts, event, actor, tenant, trace_id, hash, body) VALUES (?,?,?,?,?,?,?,?)",
                (record.id, record.ts, record.event, record.actor, record.tenant, record.trace_id, record.hash,
                 record.model_dump_json()),
            )
            self._db.commit()

    def append_chained(self, build: Callable[[str], AuditRecord]) -> AuditRecord:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")  # serialises writers across processes sharing the file
            try:
                row = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
                rec = build(row[0] if row else GENESIS)
                self._db.execute(
                    "INSERT INTO audit (id, ts, event, actor, tenant, trace_id, hash, body) VALUES (?,?,?,?,?,?,?,?)",
                    (rec.id, rec.ts, rec.event, rec.actor, rec.tenant, rec.trace_id, rec.hash, rec.model_dump_json()),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return rec

    def last_hash(self) -> str:
        row = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def read(self, since: float | None = None, limit: int | None = None) -> Iterable[AuditRecord]:
        q, args = "SELECT body FROM audit", []
        if since is not None:
            q += " WHERE ts >= ?"
            args.append(since)
        q += " ORDER BY seq"
        rows = [AuditRecord.model_validate_json(r[0]) for r in self._db.execute(q, args).fetchall()]
        return rows[-limit:] if limit else rows


@register("audit_sink", "stdout")
class StdoutSink:
    """Writes one JSON line per record to the 'aiready.audit' logger for log shippers."""

    def __init__(self) -> None:
        self._last = GENESIS

    def write(self, record: AuditRecord) -> None:
        self._last = record.hash
        log.info(record.model_dump_json())

    def last_hash(self) -> str:
        return self._last

    def read(self, since: float | None = None, limit: int | None = None) -> Iterable[AuditRecord]:
        return []


@register("audit_sink", "http")
class HttpSink:
    """Forwards records to the client's security monitoring (SIEM) as they happen."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0):
        self.url, self.headers, self.timeout = url, headers or {}, timeout
        self._last = GENESIS

    def write(self, record: AuditRecord) -> None:
        self._last = record.hash
        try:
            httpx.post(self.url, content=record.model_dump_json(), headers={"content-type": "application/json",
                       **self.headers}, timeout=self.timeout)
        except httpx.HTTPError as exc:  # never fail the request because the SIEM is down; the primary sink holds it
            log.warning("SIEM forward failed: %s", exc)

    def last_hash(self) -> str:
        return self._last

    def read(self, since: float | None = None, limit: int | None = None) -> Iterable[AuditRecord]:
        return []


@register("audit_sink", "memory")
class MemorySink:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def write(self, record: AuditRecord) -> None:
        self.records.append(record)

    def last_hash(self) -> str:
        return self.records[-1].hash if self.records else GENESIS

    def read(self, since: float | None = None, limit: int | None = None) -> Iterable[AuditRecord]:
        out = [r for r in self.records if since is None or r.ts >= since]
        return out[-limit:] if limit else out


class AuditLog:
    def __init__(self, sinks: list[AuditSink] | None = None, hmac_key: str | None = None,
                 redactor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 previous_hmac_keys: list[str] | None = None):
        self.sinks = sinks or [MemorySink()]
        self.hmac_key = hmac_key.encode() if hmac_key else None
        # keys used before the last `aiready audit rotate-key`; still accepted when verifying older records
        self.previous_hmac_keys = [k.encode() for k in previous_hmac_keys or [] if k]
        self.redactor = redactor
        self._lock = threading.Lock()
        self._prev = self.sinks[0].last_hash()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "AuditLog":
        from aiready.core.config import resolve_secret

        cfg = cfg or {}
        specs = cfg.get("sinks") or [{"type": "jsonl", "path": "data/audit/audit.jsonl"}]
        sinks = [create("audit_sink", s) for s in specs]
        key = cfg.get("hmac_key") or resolve_secret("audit_hmac_key")
        prev = cfg.get("previous_hmac_keys") or resolve_secret("audit_hmac_keys_previous") or []
        if isinstance(prev, str):
            prev = [p.strip() for p in prev.split(",") if p.strip()]
        return cls(sinks=sinks, hmac_key=key, previous_hmac_keys=prev)

    def record(self, event: str, *, actor: str, tenant: str = "default", app: str = "default",
               on_behalf_of: str | None = None, trace_id: str | None = None, action: str | None = None,
               resource: str | None = None, decision: str | None = None, **details: Any) -> AuditRecord:
        if trace_id is None:
            from aiready.core.telemetry import current_trace_id

            trace_id = current_trace_id()
        if self.redactor:
            details = self.redactor(details)
        def build(prev: str) -> AuditRecord:
            rec = AuditRecord(event=event, actor=actor, on_behalf_of=on_behalf_of, tenant=tenant, app=app,
                              trace_id=trace_id, action=action, resource=resource, decision=decision,
                              details=details, prev_hash=prev)
            rec.hash = hashlib.sha256(rec.prev_hash.encode() + rec.canonical()).hexdigest()
            if self.hmac_key:
                rec.signature = hmac.new(self.hmac_key, rec.hash.encode(), hashlib.sha256).hexdigest()
            return rec

        with self._lock:
            primary = self.sinks[0]
            chained = getattr(primary, "append_chained", None)
            # the system of record must not silently drop records: its errors propagate
            rec = chained(build) if chained else build(self._prev)
            if not chained:
                primary.write(rec)
            for sink in self.sinks[1:]:
                try:
                    sink.write(rec)
                except Exception:
                    log.exception("audit sink %s failed", type(sink).__name__)
            self._prev = rec.hash
        return rec

    def for_principal(self, principal: Any, event: str, **kw: Any) -> AuditRecord:
        return self.record(event, actor=principal.sub, tenant=principal.tenant, app=principal.app,
                           on_behalf_of=principal.actor or principal.service, **kw)

    def read(self, since: float | None = None, limit: int | None = None) -> list[AuditRecord]:
        return list(self.sinks[0].read(since=since, limit=limit))

    def set_hmac_key(self, key: str) -> None:
        """Switch signing to a new key after an ``audit.key_rotated`` record was written with the old one."""
        with self._lock:
            if self.hmac_key:
                self.previous_hmac_keys.insert(0, self.hmac_key)
            self.hmac_key = key.encode()

    def verify(self) -> tuple[bool, str]:
        """Walk the chain; returns (ok, message)."""
        prev = GENESIS
        count = 0
        for rec in self.sinks[0].read():
            if rec.prev_hash != prev:
                return False, f"chain broken before record {rec.id} (#{count})"
            expected = hashlib.sha256(rec.prev_hash.encode() + rec.canonical()).hexdigest()
            if expected != rec.hash:
                return False, f"record {rec.id} (#{count}) was modified"
            if self.hmac_key:
                keys = [self.hmac_key, *self.previous_hmac_keys]
                if not rec.signature or not any(
                    hmac.compare_digest(hmac.new(k, rec.hash.encode(), hashlib.sha256).hexdigest(), rec.signature)
                    for k in keys
                ):
                    return False, f"record {rec.id} (#{count}) has an invalid signature"
            prev = rec.hash
            count += 1
        return True, f"{count} records verified"


_audit: AuditLog | None = None


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        from aiready.core.config import get_config

        _audit = AuditLog.from_config(get_config().get("audit"))
    return _audit


def set_audit(audit: AuditLog | None) -> None:
    global _audit
    _audit = audit
