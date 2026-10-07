from __future__ import annotations

import asyncio
import json

import httpx

from aiready.agents.approvals import get_approval_store
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.mcp.config import ToolsConfig
from aiready.mcp.service import ToolService
from aiready.testing import DenyPolicy, make_principal


async def test_read_executes_and_forwards_the_users_own_token(tool_service, facade, policy):
    alice = make_principal("alice", roles=["analyst"])
    res = await tool_service.call("getClaim", {"claim_id": "C 1/2"}, alice)
    assert res.ok, res.error
    (req,) = facade.requests
    assert req.url.raw_path == b"/api/claims/C%201%2F2"  # path parameters are escaped, never injected
    assert req.headers["authorization"] == f"Bearer {alice.token}"  # the user's token, not a service account
    assert "x-service-token" in req.headers  # proves which service relayed it
    inp = policy.inputs[-1]
    assert inp.action == "tool.call" and inp.resource["effect"] == "read" and inp.resource["tool"] == "getClaim"
    rec = [r for r in get_audit().read() if r.event == "tool.call"][-1]
    assert rec.decision == "executed" and rec.details["args_hash"] and rec.actor == "alice"


async def test_query_params_and_no_token_is_refused(tool_service, facade):
    alice = make_principal("alice")
    res = await tool_service.call("searchClaims", {"q": "flood", "limit": 5}, alice)
    assert res.ok and res.content["query"] == {"q": "flood", "limit": "5"}
    anonymous = alice.model_copy(update={"token": None})
    res = await tool_service.call("searchClaims", {"q": "flood"}, anonymous)
    assert not res.ok and "token" in res.error
    assert len(facade.requests) == 1


async def test_write_waits_for_approval_then_executes_once(tool_service, facade):
    alice = make_principal("alice", roles=["analyst"])
    bob = make_principal("bob", roles=["approver"])
    args = {"claim_id": "C1", "status": "closed"}
    pending = await tool_service.call("updateClaim", args, alice)
    assert not pending.ok and pending.approval_id and facade.requests == []

    store = get_approval_store()
    assert [r.id for r in store.list_pending(bob)] == [pending.approval_id]
    store.decide(pending.approval_id, bob, approve=True, comment="ok")

    done = await tool_service.call("updateClaim", {"status": "closed", "claim_id": "C1"}, alice,
                                   approval_id=pending.approval_id)
    assert done.ok, done.error
    assert done.content["method"] == "PATCH" and done.content["body"] == {"status": "closed"}
    assert facade.requests[0].headers["authorization"] == f"Bearer {alice.token}"

    again = await tool_service.call("updateClaim", args, alice, approval_id=pending.approval_id)
    assert not again.ok and "already used" in again.error
    assert len(facade.requests) == 1


async def test_approval_is_bound_to_arguments_and_user(tool_service, facade):
    alice, carol = make_principal("alice"), make_principal("carol")
    bob = make_principal("bob", roles=["approver"])
    pending = await tool_service.call("deleteClaim", {"claim_id": "C1"}, alice)
    get_approval_store().decide(pending.approval_id, bob, approve=True)
    other_args = await tool_service.call("deleteClaim", {"claim_id": "C2"}, alice, approval_id=pending.approval_id)
    assert not other_args.ok and "arguments differ" in other_args.error
    other_user = await tool_service.call("deleteClaim", {"claim_id": "C1"}, carol, approval_id=pending.approval_id)
    assert not other_user.ok and "another user" in other_user.error
    assert facade.requests == []


async def test_requester_cannot_self_approve_and_rejected_never_runs(tool_service, facade):
    import pytest

    from aiready.core.errors import PermissionDenied

    alice = make_principal("alice", roles=["approver"])
    bob = make_principal("bob", roles=["approver"])
    pending = await tool_service.call("updateClaim", {"claim_id": "C1", "status": "open"}, alice)
    store = get_approval_store()
    assert store.list_pending(alice) == []  # four eyes: not in the requester's own inbox
    with pytest.raises(PermissionDenied):
        store.decide(pending.approval_id, alice, approve=True)
    assert any(r.decision == "denied_self_approval" for r in get_audit().read())
    store.decide(pending.approval_id, bob, approve=False, comment="no")
    res = await tool_service.call("updateClaim", {"claim_id": "C1", "status": "open"}, alice,
                                  approval_id=pending.approval_id)
    assert not res.ok and "rejected" in res.error and facade.requests == []


async def test_payment_requires_role_and_is_hidden_otherwise(tool_service):
    analyst = make_principal("alice", roles=["analyst"])
    handler = make_principal("hank", roles=["claims-handler"])
    names = {t.name for t in await tool_service.list_tools(analyst)}
    assert "payClaim" not in names and "adminReindex" not in names and "getClaim" in names
    assert {"search_knowledge", "remember", "recall"} <= names and "query_data" not in names
    denied = await tool_service.call("payClaim", {"claim_id": "C1", "amount": 10}, analyst)
    assert not denied.ok and "requires one of roles" in denied.error
    pending = await tool_service.call("payClaim", {"claim_id": "C1", "amount": 10}, handler)
    assert pending.approval_id
    assert get_approval_store().get(pending.approval_id).effect == "payment"


async def test_schema_validation_blocks_bad_arguments(tool_service, facade):
    alice = make_principal("alice")
    res = await tool_service.call("updateClaim", {"claim_id": "C1", "status": "bogus"}, alice)
    assert not res.ok and "invalid arguments" in res.error
    res = await tool_service.call("getClaim", {"claim_id": "C1", "extra": 1}, alice)
    assert not res.ok
    res = await tool_service.call("nope", {}, alice)
    assert not res.ok and "unknown tool" in res.error
    assert facade.requests == []


async def test_policy_denial_is_respected_and_audited(tools_cfg, facade):
    wiring.override("policy", DenyPolicy({"tool.call"}))
    svc = ToolService(tools_cfg, transport=facade.transport, watch_config=False)
    res = await svc.call("getClaim", {"claim_id": "C1"}, make_principal())
    assert not res.ok and "deny" in res.error
    assert facade.requests == []
    assert [r for r in get_audit().read() if r.event == "tool.call"][-1].decision == "denied_policy"


async def test_result_cap_timeout_and_upstream_errors(policy):
    async def slow(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/slow":
            await asyncio.sleep(1)
        if request.url.path == "/big":
            return httpx.Response(200, json={"data": "x" * 5000})
        if request.url.path == "/fail":
            return httpx.Response(404, json={"detail": "no such claim"})
        return httpx.Response(200, json={})

    cfg = ToolsConfig.model_validate({
        "max_result_bytes": 1000,
        "builtins": {n: {"enabled": False} for n in ("search_knowledge", "query_data", "remember", "recall")},
        "http_tools": [
            {"name": "slow", "description": "slow", "url": "http://f/slow", "timeout_seconds": 0.05},
            {"name": "big", "description": "big", "url": "http://f/big"},
            {"name": "fail", "description": "fail", "url": "http://f/fail"},
        ]})
    svc = ToolService(cfg, transport=httpx.MockTransport(slow), watch_config=False)
    p = make_principal()
    assert "timed out" in (await svc.call("slow", {}, p)).error
    big = await svc.call("big", {}, p)
    assert big.ok and big.content["truncated"] is True and len(json.dumps(big.content)) < 1300
    fail = await svc.call("fail", {}, p)
    assert not fail.ok and "404" in fail.error


async def test_custom_plugin_tool(policy):
    cfg = ToolsConfig.model_validate({
        "builtins": {n: {"enabled": False} for n in ("search_knowledge", "query_data", "remember", "recall")},
        "custom": [{"class": "mcp_fixtures:EchoTool", "greeting": "hi"}]})
    svc = ToolService(cfg, watch_config=False)
    res = await svc.call("echo", {"text": "there"}, make_principal())
    assert res.ok and res.content == "hi there"



async def test_approved_call_carries_the_approval_header_and_pii_is_masked(policy):
    from mcp_fixtures import FacadeRecorder
    from test_openapi import FACADE_STYLE

    from aiready.agents.approvals import get_approval_store
    from aiready.core.interfaces import PolicyDecision, ScreenResult

    class MaskingGuardrails:
        async def mask(self, text):
            return ScreenResult(allowed=True, text=text.replace("CLM-1", "[CLAIM]"))

    class Obligating:
        async def decide(self, inp):
            if inp.resource["effect"] != "read" and not inp.context.get("approval_id"):
                return PolicyDecision(allow=False, effect="require_approval",
                                      obligations={"approval_roles": ["claims-lead"]})
            return PolicyDecision(allow=True, obligations={"mask_pii": True})

    wiring.override("policy", Obligating())
    wiring.override("guardrails", MaskingGuardrails())
    facade = FacadeRecorder()
    cfg = ToolsConfig.model_validate({
        "sources": [{"name": "facade", "spec": FACADE_STYLE, "base_url": "http://facade:8151"}],
        "builtins": {n: {"enabled": False} for n in ("search_knowledge", "query_data", "remember", "recall")}})
    svc = ToolService(cfg, transport=facade.transport, watch_config=False)
    alice = make_principal("alice", roles=["claims-user"])
    lead = make_principal("lena", roles=["claims-lead"])
    args = {"claim_number": "CLM-1", "text": "call back"}
    pending = await svc.call("add_claim_note", args, alice)
    req = get_approval_store().get(pending.approval_id)
    assert req.approver_roles == ["claims-lead"]  # routed by the policy obligation
    get_approval_store().decide(pending.approval_id, lead, approve=True)
    done = await svc.call("add_claim_note", args, alice, approval_id=pending.approval_id)
    assert done.ok
    sent = facade.requests[0]
    assert sent.headers["x-aiready-approval"] == pending.approval_id
    assert sent.url.path == "/v1/claims/CLM-1/notes" and json.loads(sent.content) == {"text": "call back"}
    assert done.content["path"] == "/v1/claims/[CLAIM]/notes"  # personal data masked before any model sees it
