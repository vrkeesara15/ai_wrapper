from __future__ import annotations

import pytest
from agents_fixtures import WORKFLOWS

from aiready.agents.approvals import get_approval_store
from aiready.agents.runtime import AgentRuntime
from aiready.agents.workflows import WorkflowRegistry
from aiready.core.audit import get_audit
from aiready.core.errors import PermissionDenied, UpstreamError, ValidationFailed
from aiready.testing import make_principal


def _steps(events):
    return [e.data["step"] for e in events if e.type == "step" and e.data.get("status") == "ok"]


async def test_agent_loop_runs_scripted_tool_calls_to_completion(service, gateway, facade):
    alice = make_principal("alice", roles=["analyst"])
    gateway.script(tool="get_claim", arguments={"claim_id": "C1"})
    gateway.script(content="Claim C1 is open.")
    view = await service.start_run("claims_agent", {"message": "What is the status of C1?"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "completed", run.error
    assert run.result["answer"] == "Claim C1 is open."
    assert run.version == "2" and run.tool_calls == 1 and run.tokens_used > 0
    history = await service.run_history(view.id, alice)
    assert _steps(history) == ["plan", "decide", "act", "plan", "decide", "done"]
    act = next(e for e in history if e.type == "step" and e.data["step"] == "act")
    assert act.data["decision"] == "executed:1,failed:0" and act.data["output"]["results"][0]["status"] == "done"
    # the model saw the tool result, and the facade saw the user's own token
    second_call = gateway.calls[1]
    assert second_call.messages[-1].role == "tool" and "open" in second_call.messages[-1].content
    assert {t.function["name"] for t in gateway.calls[0].tools} == {"get_claim", "close_claim"}
    assert facade.requests[0].headers["authorization"] == f"Bearer {alice.token}"
    events = {r.event for r in get_audit().read()}
    assert {"agents.run.start", "agents.step", "tool.call", "agents.run.finish"} <= events


async def test_pinned_older_version_uses_its_own_graph(service, gateway):
    alice = make_principal("alice")
    gateway.script(content="hi")
    view = await service.start_run("claims_agent@1", {"message": "hello"}, alice)
    await service.runtime.wait_idle()
    assert view.version == "1"
    assert {t.function["name"] for t in gateway.calls[0].tools} == {"get_claim"}


async def test_write_pauses_for_approval_and_resumes_when_decided(service, gateway, facade):
    alice = make_principal("alice", roles=["analyst"])
    bob = make_principal("bob", roles=["approver"])
    gateway.script(tool="close_claim", arguments={"claim_id": "C1", "status": "closed"})
    gateway.script(content="Claim C1 is now closed.")
    view = await service.start_run("claims_agent", {"message": "Close C1"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "waiting_approval" and facade.requests == []
    (approval_id,) = run.pending["approval_ids"]
    assert run.pending["tools"][0]["name"] == "close_claim"

    with pytest.raises(PermissionDenied):  # four eyes
        await service.decide_approval(approval_id, alice, approve=True)
    with pytest.raises(ValidationFailed):
        await service.resume_run(view.id, alice)  # still pending

    inbox = await service.list_approvals(bob)
    assert [a.id for a in inbox] == [approval_id] and inbox[0].run_id == view.id
    await service.decide_approval(approval_id, bob, approve=True, comment="fine")
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "completed" and run.result["answer"] == "Claim C1 is now closed."
    (req,) = facade.requests
    assert req.method == "PATCH" and req.headers["authorization"] == f"Bearer {alice.token}"
    assert get_approval_store().get(approval_id).status == "used"


async def test_rejected_write_is_reported_to_the_model_and_never_runs(service, gateway, facade):
    alice = make_principal("alice")
    bob = make_principal("bob", roles=["approver"])
    gateway.script(tool="close_claim", arguments={"claim_id": "C1", "status": "closed"})
    gateway.script(content="I could not close C1: the approver said no.")
    view = await service.start_run("claims_agent", {"message": "Close C1"}, alice)
    await service.runtime.wait_idle()
    approval_id = (await service.get_run(view.id, alice)).pending["approval_ids"][0]
    await service.decide_approval(approval_id, bob, approve=False, comment="needs review")
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "completed" and facade.requests == []
    tool_msg = gateway.calls[-1].messages[-1]
    assert tool_msg.role == "tool" and "NOT DONE" in tool_msg.content and "needs review" in tool_msg.content


async def test_step_limit_stops_a_looping_agent(service, gateway):
    alice = make_principal("alice")
    for _ in range(10):
        gateway.script(tool="get_claim", arguments={"claim_id": "C1"})
    view = await service.start_run("tight_agent", {"message": "loop"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "stopped" and "step limit" in run.stop_reason and run.step_count == 4


async def test_token_limit_stops_the_run(service, gateway):
    alice = make_principal("alice")
    for _ in range(5):
        gateway.script(tool="get_claim", arguments={"claim_id": "C1"})
    view = await service.start_run("token_capped", {"message": "loop"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "stopped" and "token limit" in run.stop_reason


async def test_workflow_cannot_raise_global_limits(make_runtime):
    rt = make_runtime(limits={"max_steps": 3})
    wf = WorkflowRegistry({"w": {"pattern": "single_call", "limits": {"max_steps": 100}}}).get("w")
    assert rt._limits(wf)["max_steps"] == 3


async def test_checkpoint_resume_on_a_fresh_runtime_instance(make_runtime, gateway, facade):
    alice = make_principal("alice")
    gateway.script(tool="get_claim", arguments={"claim_id": "C7"})
    gateway.script(content="C7 is open.")
    rt1 = make_runtime(workers={"mode": "pool", "steps_per_slice": 1})
    run = await rt1.start("claims_agent", {"message": "C7?"}, alice)
    await rt1.process(run.id)  # one step, checkpoint, yield
    await rt1.process(run.id)
    mid = rt1.store.get(run.id)
    assert mid.status == "queued" and mid.step_count == 2 and mid.current_step == "act"
    await rt1.close()

    rt2 = make_runtime(workers={"mode": "pool", "steps_per_slice": 1})  # a different worker, same checkpoint store
    for _ in range(10):
        state = await rt2.process(run.id)
        if state.status not in ("queued", "running"):
            break
    assert state.status == "completed" and state.result["answer"] == "C7 is open."
    assert len(facade.requests) == 1  # the tool ran exactly once across the two workers
    assert state.token is None  # the user's token is dropped when the run ends


async def test_approval_resume_on_a_fresh_runtime(make_runtime, gateway, facade):
    alice = make_principal("alice")
    bob = make_principal("bob", roles=["approver"])
    gateway.script(tool="close_claim", arguments={"claim_id": "C2", "status": "closed"})
    gateway.script(content="closed")
    rt1 = make_runtime()
    run = await rt1.start("claims_agent", {"message": "close C2"}, alice)
    await rt1.wait_idle()
    assert rt1.store.get(run.id).status == "waiting_approval"
    await rt1.close()
    rt2 = make_runtime()
    approval_id = rt2.store.get(run.id).pending["approval_ids"][0]
    get_approval_store().decide(approval_id, bob, approve=True)  # the inbox decides; rt2 hears about it
    for _ in range(50):
        await rt2.wait_idle()
        if rt2.store.get(run.id).status == "completed":
            break
    assert rt2.store.get(run.id).status == "completed" and facade.requests[0].method == "PATCH"


async def test_expired_token_waits_for_the_user_instead_of_a_service_identity(make_runtime, gateway):
    alice = make_principal("alice")
    gateway.script(content="ok")
    rt = make_runtime(workers={"mode": "pool"})
    run = await rt.start("single_call", {"message": "hi"}, alice)
    state = rt.store.get(run.id)
    state.token = "expired-or-tampered"
    rt.store.save(state)
    assert (await rt.process(run.id)).status == "waiting_user"
    await rt.resume(run.id, alice)  # the user comes back with a fresh token
    await rt.drain()
    assert rt.store.get(run.id).status == "completed"
    from aiready.core.errors import NotFound

    with pytest.raises(NotFound):
        await rt.resume(run.id, make_principal("mallory"))


async def test_transient_errors_are_retried_then_the_run_fails_clearly(service, gateway, monkeypatch):
    alice = make_principal("alice")
    calls = {"n": 0}
    original = gateway.chat

    async def flaky(req, principal):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise UpstreamError("model pool unavailable")
        return await original(req, principal)

    monkeypatch.setattr(gateway, "chat", flaky)
    gateway.script(content="recovered")
    view = await service.start_run("claims_agent", {"message": "x"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "completed" and run.result["answer"] == "recovered"

    async def broken(req, principal):
        raise UpstreamError("model pool unavailable")

    monkeypatch.setattr(gateway, "chat", broken)
    view = await service.start_run("claims_agent", {"message": "x"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "failed" and "step 'plan' failed: model pool unavailable" == run.error
    errors = [e for e in await service.run_history(view.id, alice) if e.type == "step" and e.data["status"] == "error"]
    assert [e.data["attempt"] for e in errors] == [1, 2, 3]


async def test_router_dispatches_to_a_subagent(service, gateway):
    alice = make_principal("alice")
    gateway.script(content="billing")  # classification
    gateway.script(content="Your invoice was paid on 3 May.")  # the child single_call
    view = await service.start_run("triage", {"message": "Was my invoice paid?"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "completed", run.error
    assert run.result["answer"] == "Your invoice was paid on 3 May."
    (child_id,) = run.child_run_ids
    child = await service.get_run(child_id, alice)
    assert child.parent_run_id == view.id and child.workflow == "single_call" and child.status == "completed"


async def test_retrieval_chain_returns_citations(service, rag):
    alice = make_principal("alice")
    view = await service.start_run(None, {"message": "Is flood covered?"}, alice)  # default workflow
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.workflow == "retrieval_chain" and run.status == "completed"
    assert run.result["citations"][0]["source_uri"] == "https://docs.test/policy#flood"
    assert rag.queries[0].question == "Is flood covered?"


async def test_explicit_approval_step_and_rejection(service, facade):
    alice = make_principal("alice")
    bob = make_principal("bob", roles=["approver"])
    view = await service.start_run("gated_report", {"claim_id": "C5"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "waiting_approval"
    await service.decide_approval(run.pending["approval_ids"][0], bob, approve=True)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    assert run.status == "completed" and run.result["answer"] == "sent C5"

    view = await service.start_run("gated_report", {"claim_id": "C6"}, alice)
    await service.runtime.wait_idle()
    run = await service.get_run(view.id, alice)
    await service.decide_approval(run.pending["approval_ids"][0], bob, approve=False)
    await service.runtime.wait_idle()
    assert (await service.get_run(view.id, alice)).status == "rejected"
    assert len(facade.requests) == 1


async def test_event_stream_has_steps_tokens_and_done(service, gateway):
    alice = make_principal("alice")
    gateway.script(content="Hello there friend")
    view = await service.start_run("single_call", {"message": "hi"}, alice)
    events = [e async for e in service.stream_events(view.id, alice)]
    types = [e.type for e in events]
    assert types[0] == "run" and types[-1] == "done"
    assert "".join(e.data["text"] for e in events if e.type == "token").strip() == "Hello there friend"
    assert events[-1].data["result"]["answer"] == "Hello there friend"


async def test_runs_are_private_to_their_owner_and_cancellable(service, gateway):
    alice, mallory = make_principal("alice"), make_principal("mallory")
    auditor = make_principal("audrey", roles=["auditor"])
    rt = service.runtime
    rt.cfg.workers.mode = "pool"  # keep it queued
    view = await service.start_run("single_call", {"message": "hi"}, alice)
    from aiready.core.errors import NotFound

    with pytest.raises(NotFound):
        await service.get_run(view.id, mallory)
    assert (await service.get_run(view.id, auditor)).id == view.id
    assert [r.id for r in await service.list_runs(mallory)] == []
    with pytest.raises(PermissionDenied):
        await service.cancel_run(view.id, auditor)
    assert (await service.cancel_run(view.id, alice)).status == "cancelled"
    assert (await rt.process(view.id)).status == "cancelled"


async def test_policy_can_refuse_a_workflow(make_runtime):
    from aiready.core import wiring
    from aiready.core.errors import PolicyViolation
    from aiready.testing import DenyPolicy

    rt = make_runtime()
    wiring.override("policy", DenyPolicy({"agent.run"}))
    with pytest.raises(PolicyViolation):
        await rt.start("single_call", {"message": "x"}, make_principal())


def test_workflow_registry_validates_graphs():
    from aiready.core.errors import ConfigError, NotFound

    reg = WorkflowRegistry(WORKFLOWS)
    assert reg.get("claims_agent").version == "2" and reg.get("claims_agent", "1").version == "1"
    assert {w["name"] for w in reg.list()} >= {"single_call", "retrieval_chain", "agent_loop", "triage"}
    with pytest.raises(NotFound):
        reg.get("nope")
    with pytest.raises(ConfigError):
        WorkflowRegistry({"bad": {"steps": [{"id": "a", "type": "llm", "next": "missing"}]}})
    with pytest.raises(ConfigError):
        WorkflowRegistry({"bad": {"pattern": "router"}})
    assert isinstance(AgentRuntime, type)
