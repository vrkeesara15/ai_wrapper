"""Change capture against a SQLite replica of the sample legacy database."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from aiready.core import wiring
from aiready.core.errors import ConfigError, PolicyViolation
from aiready.core.queues import MemoryQueue
from aiready.core.stores import get_docstore
from aiready.ingestion.documents import WORK_TOPIC, ListSink, QueueSink, decode_document
from aiready.integration.change_capture import WATERMARKS, ChangeCapture, DebeziumCapture
from aiready.integration.config import CaptureTable, ChangeCaptureConfig
from aiready.testing import AllowAllPolicy, DenyPolicy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.legacy_app.db import init_db, write_outbox

CLAIMS = CaptureTable(table="claims", title_template="Claim {claim_number}", owner_column="handler",
                      content_template="Claim {claim_number} ({status}): {description}",
                      access_tags=["claims-team"], access_tag_columns=[{"column": "region", "template": "region:{value}"}])
POLICIES = CaptureTable(table="policies", title_template="Policy {number}", access_tags=["claims-team"])


@pytest.fixture(autouse=True)
def policy():
    wiring.override("policy", AllowAllPolicy())


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "replica.db"
    conn = init_db(path)
    yield path, conn
    conn.close()


def capture(path, sink, **kw) -> ChangeCapture:
    cfg = ChangeCaptureConfig(replica_url=f"sqlite:///{path}", tables=kw.pop("tables", [CLAIMS, POLICIES]), **kw)
    return ChangeCapture.from_config(cfg, sink=sink)


async def test_polling_watermark_and_incremental(db):
    path, conn = db
    sink = ListSink()
    cc = capture(path, sink)
    stats = await cc.run_once()
    assert stats["changed"] == 6 and stats["deleted"] == 0
    docs = {d.id: d for _, d in sink.items}
    claim = docs["legacy-db:claims:1"]
    assert claim.title == "Claim CLM-2001" and claim.owner == "alice"
    assert claim.access_tags == ["claims-team", "region:north"] and claim.checksum
    assert "burst pipe" in claim.content
    wm = get_docstore().get(WATERMARKS, "legacy-db:claims")
    assert wm["value"] is not None

    sink.items.clear()
    assert (await cc.run_once())["changed"] == 0  # nothing changed -> nothing re-emitted
    conn.execute("UPDATE claims SET status = 'approved', updated_at = ? WHERE id = 2", (time.time() + 5,))
    conn.commit()
    stats = await cc.run_once()
    assert stats["changed"] == 1 and sink.items[0][1].id == "legacy-db:claims:2"
    assert "approved" in sink.items[0][1].content
    assert get_docstore().get(WATERMARKS, "legacy-db:claims")["value"] > wm["value"]


async def test_rows_sharing_the_watermark_are_not_lost(db):
    path, conn = db
    sink = ListSink()
    cc = capture(path, sink, tables=[CLAIMS])
    await cc.run_once()
    top = conn.execute("SELECT MAX(updated_at) FROM claims").fetchone()[0]
    conn.execute("INSERT INTO claims (claim_number, policy_id, claimant, status, amount, description, region, "
                 "updated_at) VALUES ('CLM-2004', 1, 'Jane Smith', 'open', 10, 'Broken window.', 'north', ?)", (top,))
    conn.commit()
    sink.items.clear()
    await cc.run_once()
    assert [d.title for _, d in sink.items] == ["Claim CLM-2004"]


async def test_delete_via_tombstone(db):
    path, conn = db
    sink = ListSink()
    cc = capture(path, sink)
    await cc.run_once()
    conn.execute("DELETE FROM claims WHERE id = 3")
    conn.commit()
    sink.items.clear()
    stats = await cc.run_once()
    assert stats["deleted"] == 1
    gone = sink.items[0][1]
    assert gone.deleted and gone.id == "legacy-db:claims:3"
    sink.items.clear()
    assert (await cc.run_once())["deleted"] == 0  # tombstone watermark advanced


async def test_delete_via_key_diff(db):
    path, conn = db
    conn.execute("DROP TRIGGER claims_tombstone")
    sink = ListSink()
    table = CLAIMS.model_copy(update={"delete_detection": "key_diff"})
    cc = capture(path, sink, tables=[table])
    await cc.run_once()
    conn.execute("DELETE FROM claims WHERE id = 1")
    conn.commit()
    sink.items.clear()
    stats = await cc.run_once()
    assert stats["deleted"] == 1 and sink.items[0][1].id == "legacy-db:claims:1"


async def test_refuses_without_replica_or_on_primary(db):
    path, _ = db
    with pytest.raises(ConfigError, match="replica"):
        ChangeCapture.from_config(ChangeCaptureConfig(tables=[CLAIMS]), sink=ListSink())
    url = f"sqlite:///{path}"
    with pytest.raises(ConfigError, match="primary"):
        ChangeCapture.from_config(ChangeCaptureConfig(replica_url=url, primary_url=url, tables=[CLAIMS]),
                                  sink=ListSink())


async def test_policy_denial_stops_capture(db):
    path, _ = db
    wiring.override("policy", DenyPolicy({"data.read"}))
    sink = ListSink()
    with pytest.raises(PolicyViolation):
        await capture(path, sink).run_once()
    assert sink.items == []


async def test_outbox_mode(db):
    path, conn = db
    sink = ListSink()
    cfg = ChangeCaptureConfig(type="outbox", replica_url=f"sqlite:///{path}", tables=[CLAIMS])
    cfg.outbox.aggregate_tables = {"claim": "claims"}
    cc = ChangeCapture.from_config(cfg, sink=sink)
    stats = await cc.run_once()
    assert stats["changed"] == 3 and stats["events"] == 3
    sink.items.clear()
    write_outbox(conn, "claim", 1, "note_added", {})
    write_outbox(conn, "claim", 1, "status_changed", {})  # same aggregate twice -> one document
    conn.execute("DELETE FROM claims WHERE id = 3")
    write_outbox(conn, "claim", 3, "deleted", {})
    conn.commit()
    stats = await cc.run_once()
    assert stats["changed"] == 1 and stats["deleted"] == 1
    assert {(d.id, d.deleted) for _, d in sink.items} == {("legacy-db:claims:1", False), ("legacy-db:claims:3", True)}


async def test_emits_onto_work_queue(db):
    path, _ = db
    q = MemoryQueue()
    cc = capture(path, QueueSink(q, origin="change_capture"), tables=[CLAIMS])
    await cc.run_once()
    assert await q.depth(WORK_TOPIC) == 3
    msg = await q.get(WORK_TOPIC)
    assert msg.payload["origin"] == "change_capture" and msg.payload["source"] == "legacy-db"
    assert decode_document(msg.payload["document"]).id.startswith("legacy-db:claims:")


async def test_debezium_envelope_mapping():
    sink = ListSink()
    cdc = DebeziumCapture(ChangeCaptureConfig(type="debezium", tables=[CLAIMS]), sink=sink)
    row = {"id": 7, "claim_number": "CLM-7", "status": "open", "description": "Hail damage.", "region": "south",
           "handler": "bob", "updated_at": 1700000000.0}
    events = [
        json.dumps({"schema": {}, "payload": {"op": "c", "after": row, "source": {"table": "claims"}}}),
        {"op": "d", "before": {"id": 8}, "after": None, "source": {"table": "claims"}},
        {"op": "u", "after": row, "source": {"table": "unknown"}},
        None,
    ]
    stats: dict = {}
    await cdc.handle_events(events, "run1", stats)
    assert stats == {"changed": 1, "deleted": 1, "skipped": 1}
    created, deleted = sink.items[0][1], sink.items[1][1]
    assert created.id == "legacy-db:claims:7" and "region:south" in created.access_tags
    assert deleted.deleted and deleted.id == "legacy-db:claims:8"
