from __future__ import annotations

import json
import time

from starlette.testclient import TestClient

from aiready.agents.api import create_app
from aiready.testing import make_principal


def _h(p):
    return {"Authorization": f"Bearer {p.token}"}


def _sse(text: str) -> list[dict]:
    return [json.loads(line[5:].strip()) for line in text.splitlines() if line.startswith("data:")]


def _wait_status(client, run_id, headers, status, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/v1/runs/{run_id}", headers=headers).json()
        if run["status"] == status:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run never reached {status}: {run}")


def test_start_stream_and_inbox_over_http(service, gateway, facade):
    alice = make_principal("alice", roles=["analyst"])
    bob = make_principal("bob", roles=["approver"])
    gateway.script(tool="close_claim", arguments={"claim_id": "C3", "status": "closed"})
    gateway.script(content="Closed C3.")
    with TestClient(create_app(service)) as client:
        r = client.post("/v1/runs", headers=_h(alice), json={"workflow": "claims_agent", "input": {"message": "close C3"}})
        assert r.status_code == 202
        run_id = r.json()["id"]
        assert "token" not in r.json() and "definition" not in r.json()

        events = _sse(client.get(f"/v1/runs/{run_id}/events", headers=_h(alice)).text)
        assert events[-1]["type"] == "approval" and events[-1]["data"]["status"] == "waiting_approval"
        approval_id = events[-1]["data"]["approval_ids"][0]

        assert client.get("/v1/approvals", headers=_h(alice)).json()["approvals"] == []
        assert client.get("/v1/approvals/mine", headers=_h(alice)).json()["approvals"][0]["id"] == approval_id
        assert client.post(f"/v1/approvals/{approval_id}/approve", headers=_h(alice)).status_code == 403
        inbox = client.get("/v1/approvals", headers=_h(bob)).json()["approvals"]
        assert inbox[0]["arguments"] == {"claim_id": "C3", "status": "closed"}
        r = client.post(f"/v1/approvals/{approval_id}/approve", headers=_h(bob), json={"comment": "ok"})
        assert r.status_code == 200 and r.json()["status"] == "approved"

        run = _wait_status(client, run_id, _h(alice), "completed")
        assert run["result"]["answer"] == "Closed C3."
        history = client.get(f"/v1/runs/{run_id}/history", headers=_h(alice)).json()["events"]
        assert history[-1]["type"] == "done"
        assert client.get(f"/v1/runs/{run_id}", headers=_h(make_principal("eve"))).status_code == 404
        assert client.get("/v1/runs", headers=_h(alice)).json()["runs"][0]["id"] == run_id
        names = {w["name"] for w in client.get("/v1/workflows", headers=_h(alice)).json()["workflows"]}
        assert "claims_agent" in names
        assert client.post("/v1/runs", json={}).status_code == 401
