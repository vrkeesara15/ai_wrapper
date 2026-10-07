import json
import time

import pydantic
import pytest

from aiready.core.audit import get_audit
from aiready.core.errors import ValidationFailed
from aiready.core.interfaces import PolicyInput
from aiready.governance.config import EgressConfig, EgressDestination
from aiready.governance.egress import EgressPolicy

VENDOR = {
    "name": "vendor-llm",
    "hosts": ["api.vendor.example", "*.vendor-cdn.example"],
    "aliases": ["public-proprietary"],
    "data_classes": ["public", "internal"],
}


def test_egress_closed_by_default(people):
    eg = EgressPolicy(EgressConfig())
    d = eg.check_egress("https://api.vendor.example/v1/chat", "public", people("alice"))
    assert not d.allowed and "closed by default" in d.reason
    assert any(r.event == "governance.egress.denied" for r in get_audit().read())
    with pytest.raises(pydantic.ValidationError):
        EgressConfig(default="open")


def test_egress_opened_per_destination_and_data_class(people):
    eg = EgressPolicy(EgressConfig(destinations=[EgressDestination(**VENDOR)]))
    alice = people("alice")
    assert eg.check_egress("https://api.vendor.example/v1/chat", "internal", alice).allowed
    assert eg.check_egress("x.vendor-cdn.example", "public", alice).allowed
    assert eg.check_egress("public-proprietary", "public", alice).destination == "vendor-llm"
    assert not eg.check_egress("api.vendor.example", "personal", alice).allowed
    assert not eg.check_egress("api.vendor.example", None, alice).allowed
    assert not eg.check_egress("api.other.example", "public", alice).allowed


def test_egress_role_restriction_runtime_open_close_and_export(people):
    eg = EgressPolicy(EgressConfig())
    admin = people("sec", "security-admin")
    eg.open_route({**VENDOR, "roles": ["escalation-user"]}, admin)
    assert not eg.check_egress("vendor-llm", "public", people("alice")).allowed
    assert eg.check_egress("vendor-llm", "public", people("bob", "escalation-user")).allowed

    squid = eg.allowlist("squid")
    assert "dstdomain api.vendor.example .vendor-cdn.example" in squid and squid.strip().endswith(
        "http_access deny all"
    )
    data = json.loads(eg.allowlist("json"))
    assert data["default"] == "deny" and data["destinations"][0]["hosts"][0] == "api.vendor.example"
    envoy = json.loads(eg.allowlist("envoy"))
    assert envoy["virtual_hosts"][0]["name"] == "vendor-llm"

    eg.close_route("vendor-llm", admin, "incident 42")
    d = eg.check_egress("vendor-llm", "public", people("bob", "escalation-user"))
    assert not d.allowed and "closed" in d.reason
    assert json.loads(eg.allowlist("json"))["destinations"] == []
    events = {r.event for r in get_audit().read()}
    assert {"governance.egress.open", "governance.egress.close"} <= events


async def test_public_model_needs_egress_and_low_data_class(gov, people):
    alice = people("alice").model_dump()
    gateway_resource = {"model": "vendor-x", "category": "public-proprietary", "public": True}
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource=gateway_resource,
            context={"data_class": "internal", "egress_allowed": True},
        )
    )  # self-asserted
    assert d.effect == "deny" and "model.public.egress_closed" in d.matched_rules

    gov.egress.open_route(VENDOR, people("sec", "security-admin"))
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource=gateway_resource,
            context={"data_class": "internal"},
        )
    )
    assert d.allow and d.obligations["mask_pii"] is True
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource=gateway_resource,
            context={"data_class": "personal"},
        )
    )
    assert d.effect == "deny"
    d = await gov.decide(
        PolicyInput(
            action="egress.send",
            principal=alice,
            resource={**gateway_resource, "destination": "public-proprietary"},
            context={"data_class": "public"},
        )
    )
    assert d.allow
    # private models are unaffected by egress
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=alice,
            resource={"model": "embedding", "kind": "embedding"},
            context={},
        )
    )
    assert d.allow


async def test_incident_revoke_subject_and_jti(gov, people):
    sec = people("sec", "security-admin")
    alice = people("alice")
    inp = PolicyInput(action="rag.answer", principal=alice.model_dump())
    assert (await gov.decide(inp)).allow
    gov.incidents.revoke(sec, subject="alice", reason="credential leak", incident_id="INC-1")
    d = await gov.decide(inp)
    assert (
        d.effect == "deny" and d.matched_rules == ["incident.revoked"] and "credential leak" in d.reasons[0]
    )
    assert gov.incidents.revocations()[0]["incident_id"] == "INC-1"
    gov.incidents.restore(sec, subject="alice")
    assert (await gov.decide(inp)).allow

    jti = alice.claims["jti"]
    gov.incidents.revoke(sec, jti=jti, reason="token stolen")
    assert gov.incidents.is_revoked(alice)
    assert gov.incidents.is_revoked({"sub": "alice", "jti": jti})
    assert not gov.incidents.is_revoked(people("alice"))  # a new token for the same user is fine


def test_revocations_survive_restart(gov, people):
    from aiready.core.stores import MemoryKV
    from aiready.governance.incident import IncidentService

    gov.incidents.revoke(people("sec", "security-admin"), subject="mallory", reason="x")
    fresh = IncidentService(MemoryKV(), registry=gov.registry, egress=gov.egress)  # empty KV, same DocStore
    assert fresh.is_revoked({"sub": "mallory"})


async def test_close_route_and_trace_impact(gov, people):
    eng, approver, sec = (
        people("erin", "ai-engineer"),
        people("carol", "model-approver"),
        people("sec", "security-admin"),
    )
    gov.registry.register_model(
        {"name": "general", "version": "1", "licence": "x", "weights_checksum": "y"}, eng
    )
    gov.registry.approve_model("general", "1", approver)
    assert await gov.is_routable("general")
    t0 = time.time() - 1
    audit = get_audit()
    for user in ("alice", "bob", "alice"):
        audit.record(
            "gateway.request",
            actor=user,
            trace_id=f"t-{user}-{time.time_ns()}",
            resource="general",
            model="general",
        )
    audit.record("gateway.request", actor="carol", resource="other", model="other")

    out = gov.incidents.close_route(
        sec, model="general", route="chat", reason="jailbreak wave", incident_id="INC-2"
    )
    assert out["models_disabled"] == ["general@1"]
    assert not await gov.is_routable("general") and gov.incidents.route_disabled("chat")
    d = await gov.decide(
        PolicyInput(
            action="model.invoke",
            principal=people("alice").model_dump(),
            resource={"model": "general"},
            context={"data_class": "public"},
        )
    )
    assert d.effect == "deny" and "model.registry.disabled" in d.matched_rules

    report = gov.incidents.trace_impact(sec, start=t0, model="general", event_prefix="gateway.")
    assert report.records == 3 and report.users == {"alice": 2, "bob": 1} and len(report.traces) == 3
    assert gov.incidents.trace_impact(sec, start=t0, user="carol").records >= 1

    gov.incidents.reopen_route(sec, model="general", route="chat")
    assert await gov.is_routable("general")
    assert {"incident.close_route", "incident.trace_impact", "incident.reopen_route"} <= {
        r.event for r in audit.read()
    }


async def test_access_review_report_and_signoff(repo_config, people):
    from aiready.governance.service import GovernanceService

    tree = dict(repo_config.tree)
    tree["governance"] = {
        **tree["governance"],
        "access_review": {
            "group_members": {"claims-team": ["alice", "bob"], "empty-group": []},
            "role_members": {"analyst": ["alice"]},
            "collections": {"claims": ["claims-team"], "hr": ["hr-team"]},
            "stale_after_days": 30,
        },
    }
    repo_config.replace(tree)
    gov = GovernanceService.from_config()
    eng, approver, reviewer = (
        people("erin", "ai-engineer"),
        people("carol", "model-approver"),
        people("rita", "access-reviewer"),
    )
    gov.registry.register_model(
        {
            "name": "general",
            "version": "1",
            "licence": "x",
            "weights_checksum": "y",
            "data_classes_allowed": ["public", "internal", "personal"],
        },
        eng,
    )
    gov.registry.approve_model("general", "1", approver)
    get_audit().record("gateway.request", actor="alice", resource="general", model="general")

    report = await gov.access_reviews.generate(reviewer)
    model_entries = [e for e in report.entries if e.resource_type == "model" and e.subject == "claims-team"]
    assert model_entries and model_entries[0].data_classes == ["public", "internal", "personal"]
    entry = model_entries[0]
    assert (
        entry.last_used["alice"] is not None
        and entry.stale_members == ["bob"]
        and "stale_grants" in entry.flags
    )
    colls = {(e.subject, e.resource) for e in report.entries if e.resource_type == "collection"}
    assert ("claims-team", "claims") in colls
    assert any("empty_subject" in e.flags for e in report.entries)
    assert report.summary["previous_review_overdue"] is True

    signed = gov.access_reviews.sign_off(
        report.id, reviewer, {entry.id: "revoke"}, comment="bob left the team"
    )
    assert signed.status == "signed" and signed.revocations[0]["members"] == ["bob"]
    assert not gov.access_reviews.overdue()
    with pytest.raises(ValidationFailed):
        gov.access_reviews.sign_off(report.id, reviewer)
    assert {"governance.access_review.generate", "governance.access_review.sign_off"} <= {
        r.event for r in get_audit().read()
    }
