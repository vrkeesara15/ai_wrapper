"""API facade over the sample legacy app: identity forwarding, OpenAPI, write gating, roles, policy, SQL guard."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import yaml

from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.config import _substitute
from aiready.core.errors import ConfigError
from aiready.core.identity import get_token_service
from aiready.integration.config import EndpointSpec, FacadeConfig, IntegrationConfig
from aiready.integration.facade import FacadeService, ensure_read_only_select
from aiready.integration.facade_api import create_app as create_facade_app
from aiready.testing import AllowAllPolicy, DenyPolicy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.legacy_app.app import create_app as create_legacy


@pytest.fixture
def legacy(tmp_path):
    return create_legacy(str(tmp_path / "legacy.db"))


@pytest.fixture
def facade_config(legacy) -> FacadeConfig:
    """The shipped config/integration.yaml facade, pointed at the in-process app and its database."""
    tree = _substitute(yaml.safe_load((ROOT / "config" / "integration.yaml").read_text()))
    cfg = IntegrationConfig.model_validate(tree["integration"]).facade
    return cfg.model_copy(update={"base_url": "http://legacy", "replica_url": f"sqlite:///{legacy.state.db_path}"})


@pytest.fixture(autouse=True)
def policy():
    p = AllowAllPolicy()
    wiring.override("policy", p)
    return p


@pytest.fixture
async def client(legacy, facade_config):
    app = create_facade_app(config=facade_config, transport=httpx.ASGITransport(app=legacy))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://facade") as c:
        yield c


async def token_for(legacy, user: str, roles: list[str]) -> str:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=legacy), base_url="http://legacy") as c:
        sid = (await c.post("/login", json={"login": user, "password": f"{user}-pw"})).cookies["LEGACYSESSID"]
    return get_token_service().issue(sub=user, tenant="acme", app="claims", roles=roles, extra={"app_session": sid})


def auth(tok: str, **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}", **extra}


async def test_forwards_user_identity(client, legacy):
    alice = await token_for(legacy, "alice", ["claims-user"])
    bob = await token_for(legacy, "bob", ["claims-user"])
    r = await client.get("/v1/list_claims", headers=auth(alice))
    assert r.status_code == 200, r.text
    assert sorted(c["claim_number"] for c in r.json()) == ["CLM-2001", "CLM-2003"]  # alice: region north only
    assert set(r.json()[0]) == {"claim_number", "status", "amount", "claimant", "region"}  # output mapping
    r = await client.get("/v1/list_claims", params={"status": "under_review"}, headers=auth(bob))
    assert [c["claim_number"] for c in r.json()] == ["CLM-2002"]
    # the app's own rule is enforced for the forwarded user
    r = await client.get("/v1/get_claim", params={"claim_number": "CLM-2002"}, headers=auth(alice))
    assert r.status_code == 403
    rec = [a for a in get_audit().read() if a.event == "facade.call" and a.decision == "allow"]
    assert rec and rec[0].actor == "alice" and rec[0].resource == "list_claims"


async def test_requires_token_and_app_session(client):
    assert (await client.get("/v1/list_claims")).status_code == 401
    no_session = get_token_service().issue(sub="alice", roles=["claims-user"])
    r = await client.get("/v1/list_claims", headers=auth(no_session))
    assert r.status_code == 401


async def test_openapi_describes_declared_endpoints(client):
    doc = (await client.get("/openapi.json")).json()
    ops = {op["operationId"]: op for path in doc["paths"].values() for op in path.values()}
    assert set(ops) == {"list_claims", "get_claim", "add_claim_note", "set_claim_status", "get_policy",
                        "claims_summary_sql"}
    note = doc["paths"]["/v1/claims/{claim_number}/notes"]["post"]
    assert note["x-aiready-effect"] == "write" and note["x-aiready-required-roles"] == ["claims-user"]
    assert note["requestBody"]["content"]["application/json"]["schema"]["required"] == ["text"]
    assert {p["name"] for p in note["parameters"]} == {"claim_number", "X-AIReady-Approval"}
    assert ops["list_claims"]["x-aiready-effect"] == "read"
    assert ops["list_claims"]["x-aiready-data-class"] == "personal"


async def test_write_requires_approval_header(client, legacy):
    alice = await token_for(legacy, "alice", ["claims-user"])
    path = "/v1/claims/CLM-2001/notes"
    r = await client.post(path, json={"text": "Called the customer"}, headers=auth(alice))
    assert r.status_code == 403 and r.json()["error"]["code"] == "approval_required"
    r = await client.post(path, json={"text": "Called the customer"}, headers=auth(alice, **{"X-AIReady-Approval": "apr_1"}))
    assert r.status_code == 200, r.text
    assert r.json()["author"] == "alice"  # written as the user, not a service account
    claim = (await client.get("/v1/get_claim", params={"claim_number": "CLM-2001"}, headers=auth(alice))).json()
    assert claim["notes"] == ["Called the customer"]
    decisions = [a.decision for a in get_audit().read() if a.event == "facade.call" and a.resource == "add_claim_note"]
    assert decisions == ["deny", "allow"]


async def test_role_and_policy_denials(client, legacy):
    carol = await token_for(legacy, "carol", ["service-agent"])
    r = await client.post("/v1/set_claim_status", json={"claim_number": "CLM-2001", "status": "closed"},
                          headers=auth(carol, **{"X-AIReady-Approval": "apr_2"}))
    assert r.status_code == 403 and r.json()["error"]["code"] == "permission_denied"
    alice = await token_for(legacy, "alice", ["claims-user"])
    wiring.override("policy", DenyPolicy({"facade.call"}))
    r = await client.get("/v1/list_claims", headers=auth(alice))
    assert r.status_code == 403 and r.json()["error"]["code"] == "policy_violation"


async def test_input_validation(client, legacy):
    alice = await token_for(legacy, "alice", ["claims-user"])
    r = await client.get("/v1/get_claim", params={"claim_number": "DROP TABLE"}, headers=auth(alice))
    assert r.status_code == 422
    r = await client.get("/v1/list_claims", params={"page": "0"}, headers=auth(alice))
    assert r.status_code == 422


async def test_sql_endpoint_on_replica(client, legacy, policy):
    alice = await token_for(legacy, "alice", ["claims-user"])
    r = await client.get("/v1/claims_summary_sql", headers=auth(alice))
    assert r.status_code == 200
    assert {row["status"] for row in r.json()} == {"open", "under_review", "closed"}
    assert policy.inputs[-1].action == "facade.call" and policy.inputs[-1].resource["upstream"] == "sql"


def test_sql_guard_rejects_non_select():
    ensure_read_only_select("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    for bad in ("DELETE FROM claims", "SELECT 1; DROP TABLE claims", "UPDATE claims SET amount = 0",
                "INSERT INTO claims SELECT * FROM claims"):
        with pytest.raises(ConfigError):
            ensure_read_only_select(bad)
    with pytest.raises(ConfigError):
        FacadeService(FacadeConfig(endpoints=[EndpointSpec(
            name="evil", upstream={"type": "sql", "query": "DELETE FROM claims"})]))


async def test_tool_definitions(facade_config):
    tools = {t.name: t for t in FacadeService(facade_config).to_tool_definitions()}
    assert tools["add_claim_note"].effect == "write" and tools["get_claim"].effect == "read"
    assert tools["get_claim"].input_schema["required"] == ["claim_number"]


async def test_tools_endpoint_filters_by_role(client, legacy):
    carol = await token_for(legacy, "carol", ["service-agent"])
    names = {t["name"] for t in (await client.get("/v1/tools", headers=auth(carol))).json()}
    assert names == {"list_claims", "get_claim", "get_policy"}
