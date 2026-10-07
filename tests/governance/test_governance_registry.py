import pytest

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PermissionDenied, ValidationFailed
from aiready.core.interfaces import PolicyInput
from aiready.governance.registry import ModelRecord, RegistryService, parameter_tier

MODEL = {
    "name": "general",
    "version": "2025-01",
    "slot": "general",
    "availability": "private_gpu",
    "type": "chat",
    "parameters_b": 32,
    "base_model": "open-family",
    "licence": "apache-2.0",
    "context_length": 32768,
    "weights_checksum": "sha256:abc",
    "data_classes_allowed": ["public", "internal", "personal"],
}


@pytest.fixture
def reg():
    return RegistryService()


def events(prefix):
    return [r for r in get_audit().read() if r.event.startswith(prefix)]


def test_parameter_tiers():
    assert [parameter_tier(x) for x in (7, 10, 40, 70, None)] == ["small", "mid", "mid", "large", None]
    assert (
        ModelRecord(name="v", availability="public_api", originality="proprietary").parameter_tier
        == "unpublished"
    )


def test_approve_flow_and_routability(reg, people):
    eng, approver = people("erin", "ai-engineer"), people("carol", "model-approver")
    rec = reg.register_model({**MODEL, "status": "approved"}, eng)  # status from callers is ignored
    assert rec.status == "draft" and rec.parameter_tier == "mid"
    assert not reg.is_routable("general")

    with pytest.raises(PermissionDenied):
        reg.approve_model("general", "2025-01", eng)  # not an approver
    with pytest.raises(PermissionDenied):  # four eyes: an approver who registered cannot approve
        both = people("dan", "ai-engineer", "model-approver")
        reg.register_model({**MODEL, "version": "2025-02"}, both)
        reg.approve_model("general", "2025-02", both)

    approved = reg.approve_model("general", "2025-01", approver, note="eval suite 0.91")
    assert approved.status == "approved" and approved.approved_by == "carol"
    assert reg.is_routable("general") and reg.is_routable("general@2025-01")
    assert reg.is_routable("general", "personal") and not reg.is_routable("general", "restricted")
    assert reg.get_model("general").version == "2025-01"  # newest *approved* version wins

    with pytest.raises(ValidationFailed):
        reg.update_model(
            "general", "2025-01", {"weights_checksum": "sha256:other"}, eng
        )  # approved is immutable
    with pytest.raises(ValidationFailed):
        reg.approve_model("general", "2025-01", approver)  # approved -> approved is not a transition

    reg.deprecate_model("general", "2025-01", approver, reason="superseded")
    assert not reg.is_routable("general")
    with pytest.raises(ValidationFailed):
        reg.approve_model("general", "2025-01", approver)

    approvals = events("registry.model.approve")
    assert any(
        r.decision == "approved" and r.actor == "carol" and r.resource == "model:general@2025-01"
        for r in approvals
    )
    assert (
        events("registry.model.register") and events("registry.model.deprecate") and events("registry.denied")
    )
    assert get_audit().verify()[0]


def test_release_gate_requires_evidence(reg, people):
    eng, approver = people("erin", "ai-engineer"), people("carol", "model-approver")
    reg.register_model({**MODEL, "weights_checksum": None}, eng)
    with pytest.raises(ValidationFailed, match="weights_checksum"):
        reg.approve_model("general", "2025-01", approver)
    reg.update_model("general", "2025-01", {"weights_checksum": "sha256:def"}, eng)
    assert reg.approve_model("general", "2025-01", approver).routable
    with pytest.raises(ValidationFailed):
        reg.register_model(MODEL, eng)  # versions are immutable: same name@version twice


def test_unknown_model_not_routable(reg):
    assert not reg.is_routable("nope")
    with pytest.raises(NotFound):
        reg.get_model("nope")


def test_prompt_registry_versions_activation_diff(reg, people):
    eng = people("erin", "ai-engineer")
    v1 = reg.register_prompt("rag.answer", "Answer from sources.\nCite [n].", eng, variables=["question"])
    same = reg.register_prompt("rag.answer", "Answer from sources.\nCite [n].", eng, variables=["question"])
    assert same.version == v1.version == 1  # identical content is not duplicated
    v2 = reg.register_prompt(
        "rag.answer",
        "Answer only from sources.\nCite [n].",
        eng,
        variables=["question"],
        tool_schemas=[{"name": "lookup"}],
    )
    assert v2.version == 2 and len(reg.prompt_versions("rag.answer")) == 2

    reg.activate_prompt("rag.answer", 1, "prod", eng)
    reg.activate_prompt("rag.answer", 2, "staging", eng)
    assert reg.get_prompt("rag.answer", environment="prod").version == 1
    assert reg.get_prompt("rag.answer", environment="staging").version == 2
    assert reg.get_prompt("rag.answer").version == 2
    with pytest.raises(NotFound):
        reg.get_prompt("rag.answer", environment="dev")

    diff = reg.diff_prompt("rag.answer", 1, 2)
    assert "-Answer from sources." in diff and "+Answer only from sources." in diff and "lookup" in diff
    assert any(
        r.event == "registry.prompt.activate" and r.details["environment"] == "prod"
        for r in get_audit().read()
    )
    with pytest.raises(PermissionDenied):
        reg.register_prompt("x", "y", people("viewer", "user"))


def test_adapter_registry(reg, people):
    eng, approver = people("erin", "ai-engineer"), people("carol", "model-approver")
    adapter = {
        "name": "claims-extract",
        "version": "1",
        "base_model": "general",
        "base_model_version": "2025-01",
        "task": "claims_extraction",
        "dataset_version": "ds-2025-03",
        "eval_scores": {"f1": 0.88, "leak_rate": 0.0},
        "base_eval_scores": {"f1": 0.80, "leak_rate": 0.0},
    }
    with pytest.raises(ValidationFailed, match="not registered"):
        reg.register_adapter(adapter, eng)
    reg.register_model(MODEL, eng)
    reg.register_adapter(adapter, eng)
    with pytest.raises(ValidationFailed, match="base model"):
        reg.approve_adapter("claims-extract", "1", approver)  # base still draft
    reg.approve_model("general", "2025-01", approver)
    rec = reg.approve_adapter("claims-extract", "1", approver)
    assert rec.status == "approved" and rec.comparison()["beats_base"]

    worse = {**adapter, "version": "2", "eval_scores": {"f1": 0.90, "leak_rate": 0.02}}
    reg.register_adapter(worse, eng)
    with pytest.raises(ValidationFailed, match="does not beat"):
        reg.approve_adapter("claims-extract", "2", approver)  # leak_rate is lower-is-better
    assert [a.version for a in reg.list_adapters(base_model="general", status="approved")] == ["1"]


async def test_policy_uses_registry_facts_not_caller_claims(gov, people):
    eng, approver = people("erin", "ai-engineer"), people("carol", "model-approver")
    gov.registry.register_model(MODEL, eng)
    alice = people("alice", "user").model_dump()
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource={"model": "general", "registry_status": "approved"},
            context={"data_class": "internal"},
        )
    )
    assert d.effect == "deny" and "model.registry.not_approved" in d.matched_rules
    gov.registry.approve_model("general", "2025-01", approver)
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource={"model": "general"},
            context={"data_class": "internal"},
        )
    )
    assert d.allow and d.obligations["max_tokens"] == 4096
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource={"model": "general"},
            context={"data_class": "restricted"},
        )
    )
    assert d.effect == "deny" and "model.registry.data_class" in d.matched_rules
    denials = [r for r in get_audit().read() if r.event == "policy.decision"]
    assert denials and denials[-1].decision == "deny" and denials[-1].resource == "model:general"


def test_seed_models_from_config(repo_config):
    from aiready.governance.service import GovernanceService

    tree = dict(repo_config.tree)
    tree["governance"] = {**tree["governance"], "registry": {"models": [{**MODEL, "status": "approved"}]}}
    repo_config.replace(tree)
    gov = GovernanceService.from_config()
    assert gov.registry.is_routable("general")
    assert gov.registry.get_model("general").approved_by == "config"
