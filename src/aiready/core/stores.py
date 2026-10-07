"""State backends shared by all services.

* ``KVStore`` - counters, rate limits, caches, locks. ``memory`` for a single process / tests,
  ``redis`` for anything with more than one replica (the gateway data plane is stateless; its state lives here).
* ``DocStore`` - durable JSON documents by namespace (model registry, run state, budgets, document
  registry, checkpoints, approvals, lineage...). SQLite for the pilot stage, PostgreSQL in production; the
  same table layout works on both via SQLAlchemy.
"""

from __future__ import annotations

import fnmatch
import json
import threading
import time
from typing import Any, Iterable, Protocol

from sqlalchemy import Column, Float, MetaData, String, Table, Text, create_engine, delete, select
from sqlalchemy.engine import Engine

from aiready.core.plugins import create, register


class KVStore(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def incr(self, key: str, amount: float = 1, ttl: float | None = None) -> float: ...
    def keys(self, pattern: str = "*") -> list[str]: ...
    def set_if_absent(self, key: str, value: Any, ttl: float | None = None) -> bool: ...


@register("kv_store", "memory")
class MemoryKV:
    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}
        self._lock = threading.RLock()

    def _alive(self, key: str) -> bool:
        item = self._data.get(key)
        if item is None:
            return False
        if item[1] is not None and item[1] < time.time():
            del self._data[key]
            return False
        return True

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data[key][0] if self._alive(key) else None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + ttl if ttl else None)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def incr(self, key: str, amount: float = 1, ttl: float | None = None) -> float:
        with self._lock:
            if self._alive(key):
                val, exp = self._data[key]
                val = float(val) + amount
                self._data[key] = (val, exp)
            else:
                val = float(amount)
                self._data[key] = (val, time.time() + ttl if ttl else None)
            return val

    def keys(self, pattern: str = "*") -> list[str]:
        with self._lock:
            return [k for k in list(self._data) if self._alive(k) and fnmatch.fnmatch(k, pattern)]

    def set_if_absent(self, key: str, value: Any, ttl: float | None = None) -> bool:
        with self._lock:
            if self._alive(key):
                return False
            self.set(key, value, ttl)
            return True


@register("kv_store", "redis")
class RedisKV:
    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "aiready:"):
        import redis  # optional dependency

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix

    def _k(self, key: str) -> str:
        return self.prefix + key

    def get(self, key: str) -> Any | None:
        raw = self._r.get(self._k(key))
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._r.set(self._k(key), json.dumps(value, default=str), px=int(ttl * 1000) if ttl else None)

    def delete(self, key: str) -> None:
        self._r.delete(self._k(key))

    def incr(self, key: str, amount: float = 1, ttl: float | None = None) -> float:
        pipe = self._r.pipeline()
        pipe.incrbyfloat(self._k(key), amount)
        if ttl:
            pipe.expire(self._k(key), int(ttl), nx=True)
        return float(pipe.execute()[0])

    def keys(self, pattern: str = "*") -> list[str]:
        n = len(self.prefix)
        return [k[n:] for k in self._r.scan_iter(self._k(pattern))]

    def set_if_absent(self, key: str, value: Any, ttl: float | None = None) -> bool:
        return bool(self._r.set(self._k(key), json.dumps(value, default=str), nx=True,
                                px=int(ttl * 1000) if ttl else None))


def make_kv(spec: dict[str, Any] | str | None) -> KVStore:
    return create("kv_store", spec or {"type": "memory"})


# --------------------------------------------------------------------------- documents


class DocStore:
    """Durable JSON documents keyed by (namespace, key)."""

    def __init__(self, url: str = "sqlite:///data/state.db", engine: Engine | None = None):
        if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
            from pathlib import Path

            Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                from sqlalchemy.pool import StaticPool

                kwargs["poolclass"] = StaticPool
        self.engine = engine or create_engine(url, **kwargs)
        self.meta = MetaData()
        self.table = Table(
            "aiready_docs", self.meta,
            Column("namespace", String(128), primary_key=True),
            Column("key", String(512), primary_key=True),
            Column("value", Text, nullable=False),
            Column("updated_at", Float, nullable=False, index=True),
        )
        self.meta.create_all(self.engine)
        self._lock = threading.RLock()

    def put(self, namespace: str, key: str, value: Any) -> None:
        t = self.table
        payload = json.dumps(value, default=str)
        with self._lock, self.engine.begin() as c:
            c.execute(delete(t).where(t.c.namespace == namespace, t.c.key == key))
            c.execute(t.insert().values(namespace=namespace, key=key, value=payload, updated_at=time.time()))

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        t = self.table
        with self.engine.connect() as c:
            row = c.execute(select(t.c.value).where(t.c.namespace == namespace, t.c.key == key)).first()
        return json.loads(row[0]) if row else default

    def delete(self, namespace: str, key: str) -> None:
        t = self.table
        with self._lock, self.engine.begin() as c:
            c.execute(delete(t).where(t.c.namespace == namespace, t.c.key == key))

    def list(self, namespace: str, prefix: str = "", limit: int | None = None) -> list[tuple[str, Any]]:
        t = self.table
        q = select(t.c.key, t.c.value).where(t.c.namespace == namespace)
        if prefix:
            q = q.where(t.c.key.like(prefix.replace("%", r"\%") + "%", escape="\\"))
        q = q.order_by(t.c.key)
        if limit:
            q = q.limit(limit)
        with self.engine.connect() as c:
            return [(k, json.loads(v)) for k, v in c.execute(q).all()]

    def values(self, namespace: str, prefix: str = "") -> list[Any]:
        return [v for _, v in self.list(namespace, prefix)]

    def find(self, namespace: str, **match: Any) -> Iterable[Any]:
        for _, v in self.list(namespace):
            if isinstance(v, dict) and all(v.get(k) == val for k, val in match.items()):
                yield v


_docstore: DocStore | None = None


def get_docstore() -> DocStore:
    global _docstore
    if _docstore is None:
        from aiready.core.config import get_config

        _docstore = DocStore(get_config().get("state.url", "sqlite:///data/state.db"))
    return _docstore


def set_docstore(store: DocStore | None) -> None:
    global _docstore
    _docstore = store
