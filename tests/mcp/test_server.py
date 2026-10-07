from __future__ import annotations

from starlette.testclient import TestClient

from aiready.agents.approvals import get_approval_store
from aiready.mcp.api import create_app
from aiready.testing import make_principal

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
               "mcp-protocol-version": "2025-06-18"}


def _rpc(client: TestClient, token: str, method: str, params: dict, rid: int = 1) -> dict:
    r = client.post("/mcp", headers={**MCP_HEADERS, "Authorization": f"Bearer {token}"},
                    json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
    assert r.status_code == 200, r.text
    return r.json()


def test_mcp_list_tools_is_filtered_by_caller_roles(tool_service, facade):
    analyst = make_principal("alice", roles=["analyst"])
    handler = make_principal("hank", roles=["claims-handler"])
    with TestClient(create_app(tool_service)) as client:
        init = _rpc(client, analyst.token, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}})
        assert init["result"]["serverInfo"]["name"] == "aiready"
        names = {t["name"] for t in _rpc(client, analyst.token, "tools/list", {})["result"]["tools"]}
        assert "getClaim" in names and "payClaim" not in names
        tools = {t["name"]: t for t in _rpc(client, handler.token, "tools/list", {})["result"]["tools"]}
        assert "payClaim" in tools
        assert tools["getClaim"]["annotations"]["readOnlyHint"] is True
        assert tools["deleteClaim"]["annotations"]["destructiveHint"] is True


def test_mcp_call_tool_acts_as_the_bearer_and_gates_writes(tool_service, facade):
    alice = make_principal("alice", roles=["analyst"])
    bob = make_principal("bob", roles=["approver"])
    with TestClient(create_app(tool_service)) as client:
        res = _rpc(client, alice.token, "tools/call", {"name": "getClaim", "arguments": {"claim_id": "C9"}})["result"]
        assert res["isError"] is False and res["structuredContent"]["path"] == "/api/claims/C9"
        assert facade.requests[-1].headers["authorization"] == f"Bearer {alice.token}"

        args = {"claim_id": "C9", "status": "closed"}
        res = _rpc(client, alice.token, "tools/call", {"name": "updateClaim", "arguments": args})["result"]
        approval_id = res["structuredContent"]["approval_id"]
        assert res["structuredContent"]["status"] == "pending_approval" and len(facade.requests) == 1
        get_approval_store().decide(approval_id, bob, approve=True)
        res = _rpc(client, alice.token, "tools/call", {"name": "updateClaim", "arguments": args,
                                                      "_meta": {"aiready/approval_id": approval_id}})["result"]
        assert res["isError"] is False and facade.requests[-1].method == "PATCH"

        res = _rpc(client, alice.token, "tools/call", {"name": "getClaim", "arguments": {}})["result"]
        assert res["isError"] is True


def test_mcp_requires_a_user_token(tool_service):
    with TestClient(create_app(tool_service)) as client:
        r = client.post("/mcp", headers=MCP_HEADERS, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert r.status_code == 401


def test_rest_surface_and_webmcp_manifest(tool_service, facade):
    alice = make_principal("alice", roles=["analyst"])
    h = {"Authorization": f"Bearer {alice.token}"}
    with TestClient(create_app(tool_service)) as client:
        manifest = client.get("/v1/tools?format=webmcp", headers=h).json()
        tools = {t["name"]: t for t in manifest["tools"]}
        assert tools["getClaim"]["annotations"]["readOnlyHint"] is True and "inputSchema" in tools["getClaim"]
        assert manifest["call_endpoint"] == "/v1/tools/{name}:call"
        r = client.post("/v1/tools/getClaim:call", headers=h, json={"arguments": {"claim_id": "A"}})
        assert r.status_code == 200 and r.json()["ok"] is True
        r = client.post("/v1/tools/deleteClaim:call", headers=h, json={"arguments": {"claim_id": "A"}})
        assert r.status_code == 202 and r.json()["approval_id"]
        assert client.get("/v1/tools").status_code == 401


def test_own_service_url_in_config_does_not_loop_back(tool_service, facade):
    """In Kubernetes services.tools.url is set even inside the MCP pod: the app must use its own executor."""
    from aiready.core import wiring
    from aiready.core.config import ConfigStore, set_config

    set_config(ConfigStore(tree={"state": {"url": "sqlite:///:memory:"},
                                 "services": {"tools": {"url": "http://mcp:8140"}}}))
    wiring.reset()
    from aiready.testing import AllowAllPolicy

    wiring.override("policy", AllowAllPolicy())
    alice = make_principal("alice", roles=["analyst"])
    with TestClient(create_app(tool_service)) as client:
        r = client.post("/v1/tools/getClaim:call", headers={"Authorization": f"Bearer {alice.token}"},
                        json={"arguments": {"claim_id": "K1"}})
        assert r.status_code == 200 and r.json()["ok"] is True
        tools = _rpc(client, alice.token, "tools/list", {})["result"]["tools"]
        assert "getClaim" in {t["name"] for t in tools}
