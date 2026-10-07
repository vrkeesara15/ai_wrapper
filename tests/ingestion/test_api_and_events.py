"""Ingestion API (port 8160) and the event-driven path from change capture to the index."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aiready.core.identity import get_token_service
from aiready.ingestion.api import create_app
from aiready.ingestion.config import IngestionConfig
from aiready.ingestion.documents import QueueSink
from aiready.ingestion.pipeline import IngestionPipeline
from aiready.ingestion.worker import Worker


def _cfg(docs_dir, tmp_path) -> IngestionConfig:
    return IngestionConfig.model_validate({
        "sources": [{"name": "policy-docs", "collection": "claims-kb",
                     "connector": {"type": "filesystem", "root": str(docs_dir),
                                   "acl": ["sidecar", {"type": "path_map", "map": {"public/**": ["public"]}}]},
                     "verify": [{"question": "excess for escape of water", "expect_doc_contains": "home-policy"}]},
                    {"name": "legacy-db", "collection": "claims-records", "connector": {"type": "external"}}],
        "landing": {"path": str(tmp_path / "landing")},
    })


def _headers(roles: list[str]) -> dict[str, str]:
    return {"Authorization": "Bearer " + get_token_service().issue(sub="ops", roles=roles)}


def test_api_run_status_registry_lineage_dlq_metrics(services, docs_dir, tmp_path):
    pipe = IngestionPipeline(_cfg(docs_dir, tmp_path))
    client = TestClient(create_app(pipe))
    admin = _headers(["ingestion-admin"])

    assert client.get("/v1/sources", headers=_headers(["user"])).status_code == 403
    assert client.get("/v1/sources").status_code == 401
    assert [s["name"] for s in client.get("/v1/sources", headers=admin).json()] == ["policy-docs", "legacy-db"]

    run = client.post("/v1/sources/policy-docs/runs", json={"wait": True}, headers=admin).json()
    assert run["status"] == "succeeded" and run["counts"]["indexed"] == 5
    assert client.get(f"/v1/runs/{run['run_id']}", headers=admin).json()["run_id"] == run["run_id"]
    assert client.get("/v1/runs?source=policy-docs", headers=admin).json()[0]["run_id"] == run["run_id"]

    doc = client.get("/v1/documents/policy-docs:claims/claims-handbook.md", headers=admin).json()
    assert doc["status"] == "indexed" and doc["access_tags"] == ["claims-team"] and doc["lineage"]
    chunk_id = doc["chunk_ids"][0]
    lin = client.get(f"/v1/lineage/{chunk_id}", headers=admin).json()
    assert lin["doc_id"] == "policy-docs:claims/claims-handbook.md" and lin["run_id"] == run["run_id"]
    assert client.get("/v1/lineage/nope", headers=admin).status_code == 404
    assert client.get("/v1/documents?status=indexed", headers=admin).json()

    assert client.get("/v1/dlq", headers=admin).json() == []
    assert client.post("/v1/dlq/replay", headers=admin).json() == {"replayed": 0, "processed": 0}
    summary = client.get("/v1/metrics/summary", headers=admin).json()
    assert summary["items_indexed_last_hour"] == 5 and summary["dead_letters"] == 0
    assert summary["sources"]["policy-docs"]["documents"] == 5 and summary["lag_seconds"]["max"] is not None

    # service principals (scheduler, other services) may call it too
    svc = {"X-Service-Token": get_token_service().issue_service_token("scheduler")}
    assert client.get("/v1/runs", headers=svc).status_code == 200
    assert client.post("/v1/sources/unknown/runs", headers=admin).status_code == 404


async def test_change_capture_events_flow_into_index(services, docs_dir, tmp_path):
    """Change capture puts one message per changed row on the same work queue; the worker indexes it."""
    from aiready.integration.change_capture import ChangeCapture
    from aiready.integration.config import ChangeCaptureConfig
    from examples.legacy_app.db import init_db

    db = tmp_path / "replica.db"
    conn = init_db(db)
    pipe = IngestionPipeline(_cfg(docs_dir, tmp_path))
    cc_cfg = ChangeCaptureConfig.model_validate({
        "replica_url": f"sqlite:///{db}", "source": "legacy-db",
        "tables": [{"table": "claims", "title_template": "Claim {claim_number}",
                    "content_template": "Claim {claim_number}: {description}", "access_tags": ["fraud"],
                    "access_tag_columns": [{"column": "region", "template": "region:{value}"}]}]})
    cc = ChangeCapture.from_config(cc_cfg, sink=QueueSink(pipe.queue, pipe.topic, origin="change_capture"))
    await cc.run_once()
    assert await Worker(pipe).drain() == 3
    chunks = services["search"].chunks("claims-records")
    assert {c.doc_id for c in chunks} == {"legacy-db:claims:1", "legacy-db:claims:2", "legacy-db:claims:3"}
    assert next(c for c in chunks if c.doc_id == "legacy-db:claims:2").access_tags == ["fraud", "region:south"]

    # a row deleted in the app -> tombstone -> removed from the index
    conn.execute("DELETE FROM claims WHERE id = 3")
    conn.commit()
    await cc.run_once()
    await Worker(pipe).drain()
    assert "legacy-db:claims:3" not in {c.doc_id for c in services["search"].chunks("claims-records")}
    assert pipe.registry.get("legacy-db:claims:3", "legacy-db").status == "deleted"

    # a duplicate event for an unchanged row costs nothing
    embedded = len(services["gateway"].embedded)
    conn.execute("UPDATE claims SET updated_at = updated_at + 1 WHERE id = 1")
    conn.commit()
    await cc.run_once()
    await Worker(pipe).drain()
    assert len(services["gateway"].embedded) == embedded
    conn.close()
    await cc.aclose()
