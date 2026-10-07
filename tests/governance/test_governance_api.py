import pytest
from fastapi.testclient import TestClient

from aiready.governance.api import create_app


@pytest.fixture
def client(gov):
    return TestClient(create_app(gov))


@pytest.fixture
def auth(token_service):
    def headers(sub: str, *roles: str, groups: list[str] | None = None) -> dict[str, str]:
        tok = token_service.issue(
            sub=sub, tenant="acme", app="claims", roles=list(roles), groups=groups or []
        )
        return {"Authorization": f"Bearer {tok}"}

    return headers


def test_requires_authentication(client):
    assert client.get("/healthz").status_code == 200
    assert (
        client.post("/v1/policy/decide", json={"action": "model.invoke", "principal": {}}).status_code == 401
    )
    assert client.get("/v1/registry/models").status_code == 401
    r = client.post("/v1/incident/revoke", json={"subject": "x"}, headers={"Authorization": "Bearer junk"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated"


def test_decide_binds_the_verified_user(client, auth):
    body = {
        "action": "sql.execute",
        "principal": {"sub": "mallory", "roles": ["analyst"]},
        "resource": {"read_only": True},
    }
    r = client.post("/v1/policy/decide", json=body, headers=auth("bob", "user"))
    assert r.status_code == 200 and r.json()["effect"] == "deny"  # body roles ignored: bob is not an analyst
    r = client.post("/v1/policy/decide", json=body, headers=auth("alice", "analyst"))
    assert r.json()["effect"] == "allow" and r.json()["obligations"]["max_rows"] == 1000


def test_service_token_may_decide_for_a_user(client, token_service):
    svc = {"X-Service-Token": token_service.issue_service_token("gateway")}
    body = {"action": "memory.write", "principal": {"sub": "alice"}, "context": {"consent": True}}
    r = client.post("/v1/policy/decide", json=body, headers=svc)
    assert r.status_code == 200 and r.json()["allow"] is True


def test_explain_and_bundles(client, auth):
    body = {"action": "tool.call", "principal": {"sub": "alice"}, "resource": {"effect": "payment"}}
    assert client.post("/v1/policy/explain", json=body, headers=auth("alice", "user")).status_code == 403
    r = client.post("/v1/policy/explain", json=body, headers=auth("sec", "security-admin"))
    assert r.status_code == 200 and r.json()["decision"]["effect"] == "require_approval"
    assert any(x["rule"] == "tool.side_effect.require_approval" and x["matched"] for x in r.json()["rules"])
    b = client.get("/v1/policy/bundles", headers=auth("alice")).json()
    assert b["active"]["version"].startswith("1.0.0+") and b["history"]
    r = client.post(
        "/v1/policy/test",
        headers=auth("sec", "security-admin"),
        json={
            "cases": [
                {"name": "c", "input": {"action": "agent.run", "principal": {"sub": "a"}}, "expect": "allow"}
            ]
        },
    )
    assert r.json()["passed"] == 1


def test_registry_roles_and_approval(client, auth):
    model = {"name": "general", "version": "1", "licence": "apache-2.0", "weights_checksum": "sha256:1"}
    assert client.post("/v1/registry/models", json=model, headers=auth("alice", "user")).status_code == 403
    r = client.post("/v1/registry/models", json=model, headers=auth("erin", "ai-engineer"))
    assert r.status_code == 200 and r.json()["status"] == "draft" and r.json()["routable"] is False
    assert client.get("/v1/registry/routable/general", headers=auth("alice")).json()["routable"] is False
    r = client.post(
        "/v1/registry/models/general/1/approve", json={"note": "ok"}, headers=auth("erin", "ai-engineer")
    )
    assert r.status_code == 403
    r = client.post(
        "/v1/registry/models/general/1/approve", json={"note": "ok"}, headers=auth("carol", "model-approver")
    )
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert client.get("/v1/registry/routable/general", headers=auth("alice")).json()["routable"] is True
    assert client.get("/v1/registry/models/general", headers=auth("alice")).json()["approved_by"] == "carol"
    assert (
        client.get("/v1/registry/facts/general", headers=auth("alice")).json()["facts"]["registry_status"]
        == "approved"
    )

    r = client.post(
        "/v1/registry/prompts", json={"name": "p", "template": "v1"}, headers=auth("erin", "ai-engineer")
    )
    client.post(
        "/v1/registry/prompts", json={"name": "p", "template": "v2"}, headers=auth("erin", "ai-engineer")
    )
    client.post(
        "/v1/registry/prompts/p/activate",
        json={"version": 1, "environment": "prod"},
        headers=auth("erin", "ai-engineer"),
    )
    assert (
        client.get("/v1/registry/prompts/p?environment=prod", headers=auth("alice")).json()["template"]
        == "v1"
    )
    diff = client.get("/v1/registry/prompts/p/diff?from=1&to=2", headers=auth("alice")).text
    assert "-v1" in diff and "+v2" in diff


def test_security_admin_endpoints(client, auth, token_service):
    sec = auth("sec", "security-admin")
    dest = {"name": "vendor", "hosts": ["api.vendor.example"], "data_classes": ["public"]}
    assert client.post("/v1/egress/routes", json=dest, headers=auth("alice", "user")).status_code == 403
    assert client.post("/v1/egress/routes", json=dest, headers=sec).status_code == 200
    r = client.post(
        "/v1/egress/check",
        json={"destination": "api.vendor.example", "data_class": "public"},
        headers=auth("alice"),
    )
    assert r.json()["allowed"] is True
    squid = client.get("/v1/egress/allowlist?format=squid", headers=sec)
    assert squid.status_code == 200 and "api.vendor.example" in squid.text
    assert client.get("/v1/egress/allowlist", headers=auth("alice")).status_code == 403

    # revoke a user: their next call is refused outright
    alice = auth("alice", "user")
    assert client.post("/v1/incident/revoke", json={"subject": "alice"}, headers=alice).status_code == 403
    assert (
        client.post(
            "/v1/incident/revoke", json={"subject": "alice", "reason": "leak"}, headers=sec
        ).status_code
        == 200
    )
    r = client.post("/v1/policy/decide", json={"action": "agent.run", "principal": {}}, headers=alice)
    assert r.status_code == 401
    assert client.post("/v1/incident/restore", json={"subject": "alice"}, headers=sec).status_code == 200
    assert (
        client.post(
            "/v1/policy/decide", json={"action": "agent.run", "principal": {}}, headers=alice
        ).status_code
        == 200
    )

    r = client.post("/v1/incident/trace-impact", json={"start": 0, "user": "alice"}, headers=sec)
    assert r.status_code == 200 and r.json()["records"] >= 0

    r = client.post("/v1/access-reviews", headers=auth("rita", "access-reviewer"))
    assert r.status_code == 200
    rid = r.json()["id"]
    r = client.post(
        f"/v1/access-reviews/{rid}/signoff", json={"comment": "ok"}, headers=auth("rita", "access-reviewer")
    )
    assert r.json()["status"] == "signed"
    assert client.post("/v1/access-reviews", headers=alice).status_code == 403


async def test_http_clients_match_service(gov, people):
    import httpx

    from aiready.core.interfaces import PolicyInput
    from aiready.governance.client import GovernanceClient, RegistryClient

    app = create_app(gov)
    transport = httpx.ASGITransport(app=app)
    pol = GovernanceClient("http://governance", transport=transport)
    d = await pol.decide(
        PolicyInput(action="memory.write", principal={"sub": "alice"}, context={"consent": False})
    )
    assert d.effect == "deny" and "memory.write.no_consent" in d.matched_rules
    assert (await pol.bundles())["active"]["rules"] > 10

    reg = RegistryClient("http://governance", transport=transport)
    eng, approver = people("erin", "ai-engineer"), people("carol", "model-approver")
    await reg.register_model({"name": "m", "version": "1", "licence": "x", "weights_checksum": "y"}, eng)
    assert not await reg.is_routable("m")
    await reg.approve_model("m", "1", approver)
    assert await reg.is_routable("m") and (await reg.get_model("m")).status == "approved"
    assert (await reg.model_facts("m"))["registry_status"] == "approved"
    await pol.aclose()
    await reg.aclose()


def test_wiring_providers(repo_config):
    from aiready.core import wiring
    from aiready.governance.service import AsyncRegistry, GovernanceService

    wiring.reset()
    policy = wiring.connect("policy")
    registry = wiring.connect("registry")
    assert isinstance(policy, GovernanceService) and isinstance(registry, AsyncRegistry)
    assert registry.gov is policy
