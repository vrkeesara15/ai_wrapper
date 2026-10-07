import json
from pathlib import Path

import httpx
import pytest

from aiready.core import config
from aiready.core.interfaces import PolicyInput
from aiready.governance.expressions import PolicyCompileError, compile_condition
from aiready.governance.policy import (
    DirectoryPolicySource,
    InlinePolicySource,
    LocalPolicyEngine,
    OpaPolicyEngine,
    PolicyBundle,
    run_policy_tests,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def engine(policies, defaults=None):
    return LocalPolicyEngine(
        [InlinePolicySource(policies)], defaults=defaults or {"": "deny"}, version="1.2.3"
    )


def inp(action="model.invoke", principal=None, resource=None, context=None):
    return PolicyInput(
        action=action,
        principal=principal if principal is not None else {"sub": "alice", "roles": ["user"]},
        resource=resource or {},
        context=context or {},
    )


POLICIES = {
    "demo": {
        "version": "1.0.0",
        "rules": [
            {
                "id": "allow-all-models",
                "action": "model.*",
                "effect": "allow",
                "obligations": {"max_tokens": 4000, "mask_pii": False},
            },
            {
                "id": "deny-restricted",
                "action": "model.invoke",
                "effect": "deny",
                "when": {"context.data_class": "restricted"},
                "reason": "restricted data never reaches a model",
            },
            {
                "id": "tight-tokens",
                "action": "model.invoke",
                "effect": "obligate",
                "when": {"principal.roles": {"contains": "intern"}},
                "obligations": {"max_tokens": 500, "mask_pii": True},
            },
            {
                "id": "approve-writes",
                "action": "tool.call",
                "effect": "require_approval",
                "when": {"resource.effect": ["write", "delete"]},
                "obligations": {"approvers": ["lead"]},
            },
            {"id": "allow-tools", "action": "tool.call", "effect": "allow"},
        ],
    }
}


async def test_deny_overrides_allow():
    e = engine(POLICIES)
    d = await e.decide(inp(context={"data_class": "restricted"}))
    assert d.effect == "deny" and not d.allow
    assert d.matched_rules == ["deny-restricted"]
    assert d.reasons == ["restricted data never reaches a model"]
    assert d.obligations == {}  # a deny carries no obligations
    assert (await e.decide(inp(context={"data_class": "internal"}))).allow


async def test_require_approval_beats_allow():
    e = engine(POLICIES)
    d = await e.decide(inp("tool.call", resource={"effect": "write"}))
    assert d.effect == "require_approval" and not d.allow
    assert d.obligations == {"approvers": ["lead"]}
    assert (await e.decide(inp("tool.call", resource={"effect": "read"}))).effect == "allow"


async def test_obligations_merge_most_restrictive():
    e = engine(POLICIES)
    d = await e.decide(inp(principal={"sub": "ivy", "roles": ["intern"]}))
    assert d.allow
    assert d.obligations == {"max_tokens": 500, "mask_pii": True}
    assert "tight-tokens" in d.matched_rules and "allow-all-models" in d.matched_rules
    plain = await e.decide(inp())
    assert plain.obligations == {"max_tokens": 4000, "mask_pii": False}


async def test_obligate_rule_never_grants_access():
    e = engine(
        {
            "d": {
                "rules": [
                    {"id": "o", "action": "*", "effect": "obligate", "obligations": {"log_full_prompt": True}}
                ]
            }
        }
    )
    d = await e.decide(inp("sql.execute"))
    assert d.effect == "deny"  # default for "" prefix


async def test_default_effect_by_longest_prefix():
    e = engine({}, defaults={"": "deny", "agent.": "allow", "agent.admin.": "require_approval"})
    assert (await e.decide(inp("agent.run"))).effect == "allow"
    assert (await e.decide(inp("agent.admin.delete"))).effect == "require_approval"
    d = await e.decide(inp("payments.transfer"))
    assert d.effect == "deny" and "default deny" in d.reasons[0]


async def test_policy_version_has_semver_and_hash():
    e = engine(POLICIES)
    d = await e.decide(inp())
    assert d.policy_version.startswith("1.2.3+") and len(d.policy_version.split("+")[1]) == 12
    assert e.bundles()["active"]["domains"]["demo"]["rules"] == 5


def test_condition_operators_and_refs():
    doc = {
        "principal": {"sub": "alice", "roles": ["analyst"], "groups": ["g1"], "tenant": "acme"},
        "resource": {"owner": "alice", "tags": ["g1", "g2"], "size": 10, "name": "claims-2024"},
        "context": {"ip": "10.1.2.3"},
        "action": "data.read",
    }
    def ok(spec):
        return compile_condition(spec).evaluate(doc)

    assert ok({"resource.owner": {"ref": "principal.sub"}})
    assert ok({"resource.tags": {"any_in": {"ref": "principal.groups"}}})
    assert not ok({"resource.tags": {"all_in": {"ref": "principal.groups"}}})
    assert ok({"resource.size": {"gt": 5, "lte": 10}})
    assert ok({"resource.name": {"glob": "claims-*", "regex": r"\d{4}$"}})
    assert ok({"resource.missing": {"exists": False}}) and ok({"resource.tags": {"empty": False}})
    assert ok({"any": [{"action": "x"}, {"action": "data.read"}]})
    assert ok({"not": {"principal.roles": {"contains": "admin"}}})
    assert not ok({"resource.size": {"gt": "not-a-number"}})  # type mismatch is False, never an exception
    assert not ok({"context.approved": True}) and not ok({"resource.size": {"eq": True}})


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os').system('echo pwned')",  # a string is not a condition
        {"principal.roles": {"__class__": "x"}},  # unknown operator
        {"__import__('os')": 1},  # path outside principal/resource/context
        {"os.system": {"eq": 1}},
        {"resource.x": {"regex": "(" * 600}},  # oversized regex
        {"resource.x": {"regex": "(unclosed"}},
        {"resource.x": {"exists": "yes"}},
        {"all": {"resource.x": 1}},
    ],
)
def test_no_eval_conditions_are_rejected(bad):
    with pytest.raises(PolicyCompileError):
        compile_condition(bad)


def test_code_like_values_are_plain_data():
    cond = compile_condition({"principal.sub": "__import__('os').system('rm -rf /')"})
    assert cond.evaluate({"principal": {"sub": "__import__('os').system('rm -rf /')"}})
    assert not cond.evaluate({"principal": {"sub": "alice"}})


def test_invalid_rules_rejected():
    with pytest.raises(PolicyCompileError):
        PolicyBundle({"d": {"rules": [{"id": "x", "action": "a", "effect": "permit"}]}})
    with pytest.raises(PolicyCompileError):
        PolicyBundle(
            {
                "d": {
                    "rules": [
                        {"id": "x", "action": "a", "effect": "allow"},
                        {"id": "x", "action": "b", "effect": "allow"},
                    ]
                }
            }
        )
    with pytest.raises(PolicyCompileError):
        PolicyBundle({"d": {"rules": [{"id": "x", "action": "a", "effect": "allow", "exec": "print(1)"}]}})


async def test_hot_reload_from_config_and_rejects_broken_bundle():
    store = config.ConfigStore(
        tree={"policies": {"p": {"rules": [{"id": "r1", "action": "x.y", "effect": "deny"}]}}}
    )
    config.set_config(store)
    e = LocalPolicyEngine()  # default source: the `policies` config section
    changes = []
    e.on_change(lambda old, new: changes.append((old.version, new.version)))
    assert (await e.decide(inp("x.y"))).effect == "deny"
    v1 = e.bundle.version

    store.replace({"policies": {"p": {"rules": [{"id": "r1", "action": "x.y", "effect": "allow"}]}}})
    d = await e.decide(inp("x.y"))
    assert d.effect == "allow" and d.policy_version != v1
    assert changes and changes[0][0] == v1

    store.replace(
        {"policies": {"p": {"rules": [{"id": "r1", "action": "x.y", "effect": "allow", "when": "1 == 1"}]}}}
    )
    d = await e.decide(inp("x.y"))
    assert d.effect == "allow" and e.last_error and "condition" in e.last_error  # previous bundle kept


async def test_directory_source_hot_reload(tmp_path):
    f = tmp_path / "tools.yaml"
    f.write_text("version: 1.0.0\nrules:\n  - {id: t, action: tool.call, effect: deny}\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "cases.yaml").write_text("policy_tests: []\n")
    e = LocalPolicyEngine([DirectoryPolicySource([str(tmp_path)], reload_interval=0)])
    assert (await e.decide(inp("tool.call"))).effect == "deny"
    assert list(e.bundle.documents) == ["tools"]
    f.write_text(
        "version: 1.0.1\nrules:\n  - {id: t, action: tool.call, effect: allow}\n  - {id: t2, action: x, effect: allow}\n"
    )
    assert (await e.decide(inp("tool.call"))).effect == "allow"
    assert e.bundles()["active"]["domains"]["tools"]["version"] == "1.0.1"


async def test_explain_returns_trace():
    e = engine(POLICIES)
    out = await e.explain(inp(context={"data_class": "restricted"}))
    assert out["decision"]["effect"] == "deny"
    rules = {r["rule"]: r for r in out["rules"]}
    assert set(rules) == {
        "allow-all-models",
        "deny-restricted",
        "tight-tokens",
    }  # only rules covering the action
    assert rules["deny-restricted"]["matched"] is True
    leaf = rules["deny-restricted"]["condition"]
    assert leaf["field"] == "context.data_class" and leaf["actual"] == "restricted"
    assert rules["tight-tokens"]["matched"] is False
    out = await e.explain(inp("unknown.action"))
    assert out["default_applied"] == {"prefix": "", "effect": "deny"}


def test_repository_policy_tests_pass():
    report = run_policy_tests(REPO_CONFIG / "policies" / "tests", config_dir=REPO_CONFIG)
    assert report.ok, report.summary()
    assert report.total >= 20


def test_policy_test_harness_reports_failures(tmp_path):
    (tmp_path / "t.yaml").write_text(
        "policy_tests:\n"
        "  - name: wrong expectation\n"
        "    input: {action: model.invoke, principal: {sub: a}}\n"
        "    expect: {effect: deny, obligations: {max_tokens: 1}}\n"
        "  - name: right\n"
        "    input: {action: model.invoke, principal: {sub: a}}\n"
        "    expect: allow\n"
    )
    report = run_policy_tests(tmp_path, engine=engine(POLICIES))
    assert report.total == 2 and report.passed == 1 and not report.ok
    assert "effect allow != expected deny" in report.failures[0].problems[0]
    assert "wrong expectation" in report.summary()


async def test_opa_adapter_and_fail_closed():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        body = json.loads(request.content)
        allow = body["input"]["principal"]["sub"] == "alice"
        return httpx.Response(
            200, json={"result": {"allow": allow, "obligations": {"mask_pii": True}, "version": "rego-7"}}
        )

    opa = OpaPolicyEngine("http://opa:8181", package="aiready.authz", transport=httpx.MockTransport(handler))
    d = await opa.decide(inp())
    assert d.allow and d.obligations == {"mask_pii": True} and d.policy_version == "rego-7"
    assert seen["path"] == "/v1/data/aiready/authz"
    assert not (await opa.decide(inp(principal={"sub": "bob"}))).allow

    down = OpaPolicyEngine("http://opa:8181", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    d = await down.decide(inp())
    assert d.effect == "deny" and "unavailable" in d.reasons[0]


def test_opa_engine_from_config():
    from aiready.governance.config import PolicyConfig
    from aiready.governance.policy import build_engine

    e = build_engine(PolicyConfig(engine={"type": "opa", "url": "http://opa:8181"}))
    assert isinstance(e, OpaPolicyEngine)
