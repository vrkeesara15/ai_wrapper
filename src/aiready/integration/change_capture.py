"""Replica reader / change capture: rows and changes of the system of record become ingestion work.

Data leaves the existing database *with no load on the live system*: every reader here connects to a **read
replica** (``replica_url`` is mandatory, and equal to ``primary_url`` is refused) or consumes change events
the database already publishes. Each changed row becomes a ``core.types.Document`` (via the table's
``RowMapping``: content, access tags, owner, updated_at, checksum) and each deleted row a ``deleted=True``
document; both go onto the ingestion work queue, one message per document.

Readers (plugin kind ``change_capture``):

* ``polling`` - incremental query on an ``updated_at`` / rowversion column with a stored watermark; deletes
  come from a tombstone table or a periodic key diff.
* ``outbox`` - reads the application's transactional outbox table with an id watermark.
* ``debezium`` - consumes Debezium change events from Kafka (``aiokafka`` imported lazily).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from aiready.core import plugins
from aiready.core.audit import get_audit
from aiready.core.errors import ConfigError, PolicyViolation
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics, span
from aiready.core.types import Document, new_id
from aiready.ingestion.documents import DocumentSink, QueueSink, map_row
from aiready.integration.config import CaptureTable, ChangeCaptureConfig

log = logging.getLogger(__name__)
SERVICE = "change-capture"
ACTOR = "svc:change-capture"
WATERMARKS = "integration.cdc.watermarks"
KEYSETS = "integration.cdc.keys"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _ident(name: str) -> str:
    """Identifiers come from configuration but are still validated: they are interpolated into SQL."""
    if not _IDENT.match(name):
        raise ConfigError(f"invalid SQL identifier '{name}'")
    return name


def _redact_url(url: str) -> str:
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


class _Base:
    """Shared plumbing: replica engine, mappings, sink, state, governance."""

    kind = "base"

    def __init__(self, config: ChangeCaptureConfig, *, sink: DocumentSink | None = None,
                 docstore: DocStore | None = None, require_replica: bool = True):
        self.config = config
        if require_replica:
            if not config.replica_url:
                raise ConfigError("change capture requires 'replica_url' (a read replica); it never reads the live primary")
            if config.primary_url and config.replica_url.strip() == config.primary_url.strip():
                raise ConfigError("replica_url points at the primary database; configure a read replica")
        self.source = config.source
        self.tables: dict[str, CaptureTable] = {t.table: t for t in config.tables}
        for t in self.tables.values():
            _ident(t.table)
            _ident(t.watermark_column)
            for c in t.key_columns:
                _ident(c)
        self.sink: DocumentSink = sink or QueueSink(_make_queue(config.queue), origin="change_capture")
        self.docstore = docstore or get_docstore()
        self._engine: Any = None
        self.runs = 0

    @property
    def engine(self) -> Any:
        if self._engine is None:
            from sqlalchemy import create_engine

            kwargs: dict[str, Any] = {"future": True}
            if self.config.replica_url and self.config.replica_url.startswith("sqlite"):
                kwargs["connect_args"] = {"check_same_thread": False}
            self._engine = create_engine(self.config.replica_url, **kwargs)
        return self._engine

    def _select(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from sqlalchemy import text

        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(text(sql), params or {}).mappings().fetchall()]

    # ------------------------------------------------------------------ state
    def _wm_key(self, name: str) -> str:
        return f"{self.source}:{name}"

    def watermark(self, name: str) -> dict[str, Any]:
        return self.docstore.get(WATERMARKS, self._wm_key(name), {}) or {}

    def _set_watermark(self, name: str, value: dict[str, Any]) -> None:
        self.docstore.put(WATERMARKS, self._wm_key(name), {**value, "updated": time.time()})

    # ------------------------------------------------------------------ emission
    async def _emit(self, doc: Document, run_id: str, stats: dict[str, int]) -> None:
        await self.sink.emit(doc, source=self.source, run_id=run_id)
        outcome = "deleted" if doc.deleted else "changed"
        stats[outcome] = stats.get(outcome, 0) + 1
        metrics.pipeline.labels(self.source, "capture", outcome).inc()

    def _deleted(self, table: CaptureTable, key: Any) -> Document:
        return Document(id=table.doc_id_from_key(self.source, key), source=self.source, deleted=True,
                        uri=f"sql://{table.table}/{key}",
                        metadata={"table": table.table, "key": {table.key_columns[0]: key}})

    async def _row_doc(self, table: CaptureTable, key: Any) -> Document | None:
        """Re-read one row (outbox / events without a full image)."""
        kc = _ident(table.key_columns[0])
        rows = self._select(f"SELECT * FROM {table.table} WHERE {kc} = :k", {"k": key})
        return map_row(rows[0], table, self.source) if rows else None

    # ------------------------------------------------------------------ governance
    async def _authorize(self) -> None:
        if not self.config.policy_check:
            return
        from aiready.core.interfaces import PolicyInput
        from aiready.core.wiring import connect

        decision = await connect("policy").decide(PolicyInput(
            action="data.read", principal={"sub": ACTOR, "roles": ["service"]},
            resource={"source": self.source, "tables": sorted(self.tables),
                      "replica": _redact_url(self.config.replica_url or "")},
            context={"purpose": "ingestion", "reader": self.kind}))
        metrics.policy.labels(SERVICE, "data.read", "allow" if decision.allow else "deny").inc()
        if not decision.allow:
            get_audit().record("integration.change_capture", actor=ACTOR, action="data.read", resource=self.source,
                               decision="deny", reasons=decision.reasons)
            raise PolicyViolation(f"policy denied change capture for '{self.source}'",
                                  details={"reasons": decision.reasons})

    async def run_once(self) -> dict[str, Any]:
        run_id = new_id("cdc_")
        with span("change_capture.run", source=self.source, reader=self.kind, run_id=run_id) as sp, \
                metrics.time_stage(SERVICE, self.kind):
            await self._authorize()
            stats: dict[str, Any] = {"changed": 0, "deleted": 0}
            try:
                await self._capture(run_id, stats)
            except Exception as exc:
                metrics.pipeline.labels(self.source, "capture", "error").inc()
                get_audit().record("integration.change_capture", actor=ACTOR, action="data.read",
                                   resource=self.source, decision="error", run_id=run_id, error=str(exc)[:300])
                raise
            self.runs += 1
            sp.set(**{k: v for k, v in stats.items() if isinstance(v, int)})
            get_audit().record("integration.change_capture", actor=ACTOR, action="data.read", resource=self.source,
                               decision="allow", run_id=run_id, reader=self.kind, **stats)
            return {"run_id": run_id, "source": self.source, "reader": self.kind, **stats}

    async def _capture(self, run_id: str, stats: dict[str, Any]) -> None:
        raise NotImplementedError

    async def run_forever(self, interval: float | None = None, stop: asyncio.Event | None = None) -> None:
        """Run at a fixed interval until ``stop`` is set; failures are logged and retried next tick."""
        stop = stop or asyncio.Event()
        every = interval or self.config.interval_seconds
        while not stop.is_set():
            try:
                await self.run_once()
            except Exception:
                log.exception("change capture run failed for %s", self.source)
            try:
                await asyncio.wait_for(stop.wait(), timeout=every)
            except TimeoutError:
                pass

    async def aclose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()


def _make_queue(spec: dict[str, Any]) -> Any:
    from aiready.core.queues import make_queue

    return make_queue(spec)


# =========================================================================== polling


@plugins.register("change_capture", "polling")
class PollingCapture(_Base):
    """Incremental reads with a watermark; rows sharing the watermark value are de-duplicated by key so
    that rows committed in the same instant as the previous high-water mark are not lost (``>=`` + skip)."""

    kind = "polling"

    async def _capture(self, run_id: str, stats: dict[str, Any]) -> None:
        for table in self.tables.values():
            await self._capture_table(table, run_id, stats)
        if any(t.delete_detection == "tombstone" for t in self.tables.values()):
            await self._capture_tombstones(run_id, stats)
        for table in self.tables.values():
            if table.delete_detection == "key_diff" and self.runs % max(1, table.key_diff_every) == 0:
                await self._key_diff(table, run_id, stats)

    async def _capture_table(self, table: CaptureTable, run_id: str, stats: dict[str, Any]) -> None:
        wm = self.watermark(table.table)
        col = table.watermark_column
        base = f"({table.query}) AS q" if table.query else table.table
        params: dict[str, Any] = {"lim": self.config.batch_size}
        where = ""
        if wm.get("value") is not None:
            where = f"WHERE {col} >= :wm"
            params["wm"] = wm["value"]
        seen_at_wm = set(wm.get("keys_at_value", []))
        while True:
            rows = self._select(f"SELECT * FROM {base} {where} ORDER BY {col} LIMIT :lim", params)
            fresh = [r for r in rows
                     if not (wm.get("value") is not None and r.get(col) == wm["value"] and self._key(table, r) in seen_at_wm)]
            for r in fresh:
                await self._emit(map_row(r, table, self.source), run_id, stats)
            if not rows:
                break
            top = rows[-1].get(col)
            keys_at_top = {self._key(table, r) for r in rows if r.get(col) == top}
            if wm.get("value") == top:
                keys_at_top |= seen_at_wm
            wm = {"value": top, "keys_at_value": sorted(keys_at_top)}
            self._set_watermark(table.table, wm)
            if len(rows) < self.config.batch_size or not fresh:
                break
            where, params["wm"], seen_at_wm = f"WHERE {col} >= :wm", top, keys_at_top

    @staticmethod
    def _key(table: CaptureTable, row: dict[str, Any]) -> str:
        return ":".join(str(row.get(c)) for c in table.key_columns)

    async def _capture_tombstones(self, run_id: str, stats: dict[str, Any]) -> None:
        tc = self.config.tombstone
        id_col, tcol, kcol = _ident(tc.id_column), _ident(tc.table_column), _ident(tc.key_column)
        wm = self.watermark(f"__tombstones__{tc.table}")
        rows = self._select(
            f"SELECT {id_col} AS tid, {tcol} AS tname, {kcol} AS tkey FROM {_ident(tc.table)} "
            f"WHERE {id_col} > :wm ORDER BY {id_col}", {"wm": wm.get("value", 0)})
        for r in rows:
            table = self.tables.get(str(r["tname"]))
            if table is not None and table.delete_detection == "tombstone":
                await self._emit(self._deleted(table, r["tkey"]), run_id, stats)
        if rows:
            self._set_watermark(f"__tombstones__{tc.table}", {"value": rows[-1]["tid"]})

    async def _key_diff(self, table: CaptureTable, run_id: str, stats: dict[str, Any]) -> None:
        """Compare the replica's current keys with the keys seen last time; missing ones were deleted."""
        cols = ", ".join(table.key_columns)
        base = f"({table.query}) AS q" if table.query else table.table
        current = {self._key(table, r) for r in self._select(f"SELECT {cols} FROM {base}")}
        key = f"{self.source}:{table.table}"
        previous = set(self.docstore.get(KEYSETS, key, []) or [])
        for gone in sorted(previous - current):
            parts = gone.split(":")
            await self._emit(self._deleted(table, parts if len(parts) > 1 else parts[0]), run_id, stats)
        self.docstore.put(KEYSETS, key, sorted(current))
        stats["key_diff_checked"] = stats.get("key_diff_checked", 0) + len(current)


# =========================================================================== outbox


@plugins.register("change_capture", "outbox")
class OutboxCapture(_Base):
    """The app already records business events in an outbox table: follow it by id."""

    kind = "outbox"

    async def _capture(self, run_id: str, stats: dict[str, Any]) -> None:
        ob = self.config.outbox
        cols = [_ident(c) for c in (ob.id_column, ob.aggregate_column, ob.aggregate_id_column, ob.event_type_column)]
        wm = self.watermark(f"__outbox__{ob.table}")
        rows = self._select(
            f"SELECT {cols[0]} AS oid, {cols[1]} AS agg, {cols[2]} AS agg_id, {cols[3]} AS evt "
            f"FROM {_ident(ob.table)} WHERE {cols[0]} > :wm ORDER BY {cols[0]} LIMIT :lim",
            {"wm": wm.get("value", 0), "lim": self.config.batch_size})
        latest: dict[tuple[str, Any], str] = {}  # last event per aggregate wins within a batch
        for r in rows:
            table_name = ob.aggregate_tables.get(str(r["agg"]), str(r["agg"]))
            if table_name in self.tables:
                latest[(table_name, r["agg_id"])] = str(r["evt"])
        for (table_name, agg_id), evt in latest.items():
            table = self.tables[table_name]
            doc = None if evt in ob.delete_event_types else await self._row_doc(table, agg_id)
            await self._emit(doc or self._deleted(table, agg_id), run_id, stats)
        if rows:
            self._set_watermark(f"__outbox__{ob.table}", {"value": rows[-1]["oid"]})
        stats["events"] = len(rows)


# =========================================================================== debezium


@plugins.register("change_capture", "debezium")
class DebeziumCapture(_Base):
    """Consumes Debezium change events (Kafka). The database's own log is the source: zero query load.

    Events without a row image (e.g. ``REPLICA IDENTITY DEFAULT`` deletes carry only the key) are handled;
    full images are mapped directly without touching any database.
    """

    kind = "debezium"

    def __init__(self, config: ChangeCaptureConfig, **kw: Any):
        # Debezium reads the log, not a replica; a replica URL is optional (only used to re-read rows).
        super().__init__(config, require_replica=False, **kw)
        self._consumer: Any = None

    def table_for(self, event: dict[str, Any]) -> CaptureTable | None:
        src = event.get("source") or {}
        return self.tables.get(str(src.get("table", ""))) or self.tables.get(str(event.get("__table", "")))

    def to_document(self, value: dict[str, Any]) -> Document | None:
        """Map one Debezium envelope (with or without the schema wrapper) to a Document."""
        event = value.get("payload", value) if isinstance(value, dict) else None
        if not isinstance(event, dict) or "op" not in event:
            return None
        table = self.table_for(event)
        if table is None:
            return None
        op = event["op"]
        if op == "d":
            before = event.get("before") or {}
            keys = [before.get(c) for c in table.key_columns]
            return self._deleted(table, keys if len(keys) > 1 else keys[0])
        if op in ("c", "u", "r") and event.get("after"):
            return map_row(event["after"], table, self.source)
        return None

    async def _consumer_started(self) -> Any:
        if self._consumer is None:
            from aiokafka import AIOKafkaConsumer  # optional dependency

            k = self.config.kafka
            if not k.topics:
                raise ConfigError("debezium change capture needs kafka.topics")
            self._consumer = AIOKafkaConsumer(*k.topics, bootstrap_servers=k.bootstrap_servers, group_id=k.group_id,
                                              enable_auto_commit=False, auto_offset_reset="earliest")
            await self._consumer.start()
        return self._consumer

    async def handle_events(self, values: list[Any], run_id: str, stats: dict[str, Any]) -> None:
        for raw in values:
            if raw is None:  # tombstone record after a delete (log compaction marker)
                continue
            value = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
            doc = self.to_document(value)
            if doc is not None:
                await self._emit(doc, run_id, stats)
            else:
                stats["skipped"] = stats.get("skipped", 0) + 1

    async def _capture(self, run_id: str, stats: dict[str, Any]) -> None:
        consumer = await self._consumer_started()
        k = self.config.kafka
        batches = await consumer.getmany(timeout_ms=k.poll_timeout_ms, max_records=k.max_records)
        for _tp, records in batches.items():
            await self.handle_events([r.value for r in records], run_id, stats)
        if batches:
            await consumer.commit()  # commit only after every document reached the work queue

    async def aclose(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
        await super().aclose()


# =========================================================================== facade


class ChangeCapture:
    """Builds the configured reader. ``ChangeCapture.from_config()`` reads ``integration.change_capture``."""

    def __init__(self, reader: _Base):
        self.reader = reader

    @classmethod
    def from_config(cls, config: ChangeCaptureConfig | None = None, *, sink: DocumentSink | None = None,
                    docstore: DocStore | None = None) -> ChangeCapture:
        if config is None:
            from aiready.integration.config import load_integration_config

            config = load_integration_config().change_capture
        spec: dict[str, Any] = {"class": config.type} if ":" in config.type else {"type": config.type}
        reader = plugins.create("change_capture", spec, config=config, sink=sink, docstore=docstore)
        return cls(reader)

    async def run_once(self) -> dict[str, Any]:
        return await self.reader.run_once()

    async def run_forever(self, interval: float | None = None, stop: asyncio.Event | None = None) -> None:
        await self.reader.run_forever(interval, stop)

    async def aclose(self) -> None:
        await self.reader.aclose()
