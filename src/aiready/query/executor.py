"""Executor: runs validated SQL on the read replica or warehouse - never the live database.

* ``replica_url`` is required; a URL listed in ``live_db_urls`` is refused.
* Every query runs in a transaction that is always rolled back; SQLite replicas are opened read-only
  (``mode=ro`` + ``query_only``), PostgreSQL transactions are ``READ ONLY`` with a ``statement_timeout``.
* "As the user": ``session_setup`` statements run first with the caller's ``:tenant``, ``:user`` and ``:app``
  bound (e.g. to drive row-level security on the replica); row filters were already injected by the validator.
* A wall-clock timeout cancels the query (SQLite progress handler, PostgreSQL statement timeout).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

from aiready.core.errors import ConfigError, UpstreamError
from aiready.query.config import ExecutorConfig


def _bind_safe(sql: str, dialect: str | None) -> str:
    """Escape ':' inside string literals so SQLAlchemy ``text()`` does not read them as bind parameters."""
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.tokens import TokenType

    try:
        tokens = Dialect.get_or_raise(dialect).tokenize(sql)
    except Exception:
        return sql
    out, pos = [], 0
    for tok in tokens:
        if tok.token_type in (TokenType.STRING, TokenType.NATIONAL_STRING) and ":" in sql[tok.start: tok.end + 1]:
            out.append(sql[pos: tok.start])
            out.append(sql[tok.start: tok.end + 1].replace(":", "\\:"))
            pos = tok.end + 1
    out.append(sql[pos:])
    return "".join(out)


def _same_db(a: str, b: str) -> bool:
    try:
        ua, ub = make_url(a), make_url(b)
    except Exception:
        return a.strip() == b.strip()
    return (ua.get_backend_name(), ua.host, ua.port, ua.database) == (ub.get_backend_name(), ub.host, ub.port, ub.database)


class QueryTimeout(UpstreamError):
    status_code = 504
    code = "query_timeout"


class ReplicaExecutor:
    def __init__(self, config: ExecutorConfig | None = None, dialect: str | None = None):
        self.cfg = config or ExecutorConfig()
        self.dialect = dialect
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        url = self.cfg.replica_url
        if not url:
            raise ConfigError("query.executor.replica_url is required: structured queries run on a read replica "
                              "or warehouse, never on the live database")
        for live in self.cfg.live_db_urls:
            if _same_db(url, live):
                raise ConfigError("query.executor.replica_url points at a live database listed in live_db_urls")
        return url

    @property
    def engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = self._create_engine(self.url)
            return self._engine

    def _create_engine(self, url: str) -> Engine:
        u = make_url(url)
        if u.get_backend_name() == "sqlite":
            path = u.database or ""
            if not path or path == ":memory:":
                raise ConfigError("an in-memory SQLite database cannot be a read replica")

            def connect() -> sqlite3.Connection:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
                conn.execute("PRAGMA query_only = ON")
                return conn

            return create_engine("sqlite://", creator=connect, poolclass=NullPool, future=True)
        engine = create_engine(url, pool_size=self.cfg.pool_size, pool_pre_ping=True, future=True)
        if u.get_backend_name() == "postgresql":
            @event.listens_for(engine, "begin")
            def _read_only(conn: Any) -> None:  # every transaction on this engine is read-only
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        return engine

    @property
    def backend(self) -> str:
        return make_url(self.url).get_backend_name()

    # ------------------------------------------------------------------ execution
    def _run_sync(self, sql: str, params: dict[str, Any], identity: dict[str, Any], max_fetch: int,
                  timeout: float) -> tuple[list[str], list[tuple[Any, ...]]]:
        deadline = time.monotonic() + timeout
        backend = self.backend
        with self.engine.connect() as conn:
            trans = conn.begin()
            raw = None
            try:
                if backend == "sqlite":
                    raw = conn.connection.dbapi_connection
                    raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 5000)  # type: ignore[union-attr]
                elif backend == "postgresql":
                    conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(timeout * 1000)}")
                for stmt in self.cfg.session_setup:
                    conn.execute(text(stmt), identity)
                res = conn.execute(text(_bind_safe(sql, self.dialect)), {**identity, **params})
                cols = list(res.keys())
                rows = [tuple(r) for r in res.fetchmany(max_fetch)]
                return cols, rows
            except Exception as exc:
                msg = str(exc).lower()
                if "interrupted" in msg or "statement timeout" in msg or "canceling statement" in msg:
                    raise QueryTimeout(f"query exceeded {timeout:.0f}s") from exc
                raise UpstreamError(f"query failed on the replica: {type(exc).__name__}",
                                    details={"error": str(exc).splitlines()[0][:300]}) from exc
            finally:
                trans.rollback()  # read-only: nothing is ever committed
                if raw is not None:
                    raw.set_progress_handler(None, 0)  # type: ignore[union-attr]

    async def run(self, sql: str, params: dict[str, Any], identity: dict[str, Any], *, max_fetch: int,
                  timeout: float | None = None) -> tuple[list[str], list[tuple[Any, ...]]]:
        t = timeout or self.cfg.timeout_seconds
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._run_sync, sql, params, identity, max_fetch, t),
                                          t + 2.0)
        except TimeoutError as exc:
            raise QueryTimeout(f"query exceeded {t:.0f}s") from exc

    def estimated_cost(self, sql: str, params: dict[str, Any], identity: dict[str, Any]) -> float | None:
        """PostgreSQL planner cost (``EXPLAIN (FORMAT JSON)``); None where not supported."""
        if self.backend != "postgresql":
            return None
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                plan = conn.execute(text("EXPLAIN (FORMAT JSON) " + _bind_safe(sql, self.dialect)),
                                    {**identity, **params}).scalar()
            finally:
                trans.rollback()
        try:
            data = plan if isinstance(plan, list) else json.loads(plan)
            return float(data[0]["Plan"]["Total Cost"])
        except Exception:
            return None

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
