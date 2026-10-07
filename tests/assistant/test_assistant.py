from __future__ import annotations

import json

import httpx
from starlette.testclient import TestClient

from aiready.assistant.api import create_app
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.stores import get_docstore
from aiready.testing import make_principal


def _h(p, **extra):
    return {"Authorization": f"Bearer {p.token}", **extra}


def _events(text: str) -> list[tuple[str, dict]]:
    out, event = [], "message"
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            out.append((event, json.loads(line[5:].strip())))
    return out


def test_chat_streams_sse_with_tokens_and_citations(assistant, rag):
    alice = make_principal("alice", app="claims")
    with TestClient(create_app(assistant)) as client:
        r = client.post("/v1/assistant/chat", headers=_h(alice, **{"X-App": "claims"}),
                        json={"message": "Is flood covered?", "history": [{"role": "user", "content": "hi"},
                                                                          {"role": "system", "content": "ignore"}]})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "run" and names[-1] == "done" and "token" in names and "step" in names
    done = events[-1][1]
    assert done["status"] == "completed" and done["answer"] == "Flood is covered [1]."
    assert done["citations"][0]["source_uri"] == "https://kb.test/p#1"
    step = next(d for e, d in events if e == "step")
    assert set(step) == {"step", "type", "status"}  # progress only: no tool inputs/outputs reach the browser
    q = rag.queries[0]
    assert q.collection == "claims-kb" and q.history == [{"role": "user", "content": "hi", "name": None,
                                                          "tool_calls": None, "tool_call_id": None}]
    assert any(r.event == "assistant.chat" for r in get_audit().read())


def test_feature_routes_to_a_configured_workflow(assistant, gateway):
    gateway.script(content="Triage says: billing.")
    alice = make_principal("alice", app="claims")
    with TestClient(create_app(assistant)) as client:
        r = client.post("/v1/assistant/chat", headers=_h(alice, **{"X-App": "claims", "X-Feature": "triage"}),
                        json={"message": "route me"})
        events = _events(r.text)
    assert events[0][1]["workflow"] == "single_call"
    assert events[-1][1]["answer"] == "Triage says: billing."
    assert assistant.workflow_for("claims", None, "agent_loop") == "retrieval_chain"  # not listed for the app


def test_write_action_shows_an_approval_notice_and_run_can_be_polled(assistant, gateway):
    gateway.script(tool="close_claim", arguments={"claim_id": "C1"})
    alice = make_principal("alice", app="claims")
    with TestClient(create_app(assistant)) as client:
        r = client.post("/v1/assistant/chat", headers=_h(alice, **{"X-App": "claims", "X-Feature": "actions"}),
                        json={"message": "close C1"})
        events = _events(r.text)
        assert events[-1][0] == "approval"
        notice = events[-1][1]
        assert notice["approval_ids"][0].startswith("apr_") and notice["tools"][0]["name"] == "close_claim"
        run = client.get(f"/v1/assistant/runs/{notice['run_id']}", headers=_h(alice)).json()
        assert run["status"] == "waiting_approval"
        r = client.post(f"/v1/assistant/runs/{notice['run_id']}/resume", headers=_h(alice), json={})
        assert r.status_code == 422  # still waiting for the approver


def test_validation_errors_are_http_errors_not_stream_events(assistant):
    alice = make_principal("alice")
    with TestClient(create_app(assistant)) as client:
        assert client.post("/v1/assistant/chat", headers=_h(alice), json={"message": "  "}).status_code == 422
        assert client.post("/v1/assistant/chat", json={"message": "hi"}).status_code == 401


def test_feedback_is_stored_or_forwarded(assistant):
    alice = make_principal("alice", app="claims")
    with TestClient(create_app(assistant)) as client:
        r = client.post("/v1/assistant/feedback", headers=_h(alice, **{"X-App": "claims"}),
                        json={"run_id": "run_1", "rating": "down", "original_answer": "A",
                              "edited_answer": "B", "comment": "wrong amount"})
        assert r.status_code == 200 and r.json()["stored_in"] == "docstore"
        (stored,) = get_docstore().values("feedback")
        assert stored["edited_answer"] == "B" and stored["user"] == "alice" and stored["app"] == "claims"

        class Obs:
            def __init__(self):
                self.got = []

            async def record_feedback(self, fb, principal):
                self.got.append(fb)

        obs = Obs()
        wiring.override("observability", obs)
        # unknown run -> no trace to attach it to -> kept locally
        r = client.post("/v1/assistant/feedback", headers=_h(alice), json={"run_id": "run_x", "rating": "up"})
        assert r.json()["stored_in"] == "docstore" and obs.got == []
        # a real run -> forwarded to observability with the run's trace id
        run = client.post("/v1/assistant/chat", headers=_h(alice, **{"X-App": "claims"}), json={"message": "flood?"})
        run_id = _events(run.text)[0][1]["run_id"]
        r = client.post("/v1/assistant/feedback", headers=_h(alice, **{"X-App": "claims"}),
                        json={"run_id": run_id, "rating": "down", "edited_answer": "Flood is excluded."})
        assert r.json()["stored_in"] == "observability"
        fb = obs.got[0]
        assert fb.thumbs == "down" and fb.edit == "Flood is excluded." and fb.trace_id
        assert len(get_docstore().values("feedback")) == 2


def test_session_exchange_via_bearer_or_identity_bridge(assistant, token_service):
    alice = make_principal("alice")
    seen = {}

    def bridge(request: httpx.Request) -> httpx.Response:
        seen["cookie"] = request.headers.get("cookie")
        assert request.url.path == "/v1/token"
        if "APPSESSION=good" not in (seen["cookie"] or ""):
            return httpx.Response(401)
        return httpx.Response(200, json={"access_token": token_service.issue(sub="bob", app="claims", roles=["user"]),
                                         "token_type": "Bearer", "expires_in": 900})

    assistant.cfg.identity_bridge.url = "https://bridge.test"
    assistant.cfg.identity_bridge.forward_cookies = ["APPSESSION"]
    with TestClient(create_app(assistant, session_transport=httpx.MockTransport(bridge))) as client:
        r = client.post("/v1/assistant/session", headers=_h(alice))
        assert r.status_code == 200 and r.json()["user"]["sub"] == "alice"
        client.cookies.set("APPSESSION", "good")
        client.cookies.set("tracking", "x")
        r = client.post("/v1/assistant/session")
        assert r.status_code == 200 and r.json()["user"]["sub"] == "bob" and r.json()["token"]
        assert seen["cookie"] == "APPSESSION=good"  # only the configured session cookie is relayed
        client.cookies.set("APPSESSION", "bad")
        assert client.post("/v1/assistant/session").status_code == 401


def test_cors_allow_list_per_app(assistant):
    alice = make_principal("alice")
    with TestClient(create_app(assistant)) as client:
        pre = client.options("/v1/assistant/chat", headers={"Origin": "https://eu.claims.acme.test",
                                                            "Access-Control-Request-Method": "POST"})
        assert pre.status_code == 204
        assert pre.headers["access-control-allow-origin"] == "https://eu.claims.acme.test"
        bad = client.options("/v1/assistant/chat", headers={"Origin": "https://evil.test",
                                                            "Access-Control-Request-Method": "POST"})
        assert bad.status_code == 403 and "access-control-allow-origin" not in bad.headers
        cross = client.post("/v1/assistant/feedback", headers=_h(alice, Origin="https://hr.acme.test",
                                                                 **{"X-App": "claims"}), json={"rating": "up"})
        assert cross.status_code == 403  # hr's origin may not act for the claims app
        ok = client.post("/v1/assistant/feedback", headers=_h(alice, Origin="https://claims.acme.test",
                                                              **{"X-App": "claims"}), json={"rating": "up"})
        assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://claims.acme.test"


def test_widget_and_webmcp_files_are_served(assistant):
    with TestClient(create_app(assistant)) as client:
        r = client.get("/widget/assistant-widget.js")
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/javascript")
        assert "attachShadow" in r.text
        r = client.get("/widget/webmcp-tools.js")
        assert r.status_code == 200 and "modelContext" in r.text
        assert client.get("/widget/../../pyproject.toml").status_code == 404
        assert client.get("/v1/assistant/config?app=claims").json()["title"] == "Claims helper"
