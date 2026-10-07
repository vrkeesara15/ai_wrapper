"""Identity bridge: app session -> signed AI-Ready token, with mapping, policy, audit and refresh."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.errors import AuthenticationError, PolicyViolation
from aiready.core.identity import TokenService, get_token_service, set_token_service
from aiready.integration.bridge_api import create_app as create_bridge_app
from aiready.integration.config import IdentityBridgeConfig, RoleMapping, RoleRule
from aiready.integration.identity_bridge import (
    HeaderTrustValidator,
    IdentityBridge,
    OidcValidator,
    SessionCredentials,
    SessionIdentity,
    SharedSessionStoreValidator,
    map_roles,
)
from aiready.testing import AllowAllPolicy, DenyPolicy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.legacy_app.app import create_app as create_legacy

MAPPINGS = {"sub": "$.user.login", "email": "$.user.email", "name": "$.user.display_name", "tenant": "$.company",
            "roles": "$.roles", "attributes": {"region": "$.region"}}


def bridge_config(**kw) -> IdentityBridgeConfig:
    return IdentityBridgeConfig(
        app="claims",
        validators=[{"type": "introspect_http", "name": "legacy", "url": "http://legacy/api/me",
                     "session_cookie": "LEGACYSESSID", "mappings": MAPPINGS, "cache_ttl": 0}],
        role_mapping=RoleMapping(
            rules=[RoleRule(app_role="adjuster", roles=["claims-user"], groups=["claims-team"]),
                   RoleRule(app_role="fraud_investigator", roles=["fraud-investigator"], groups=["fraud"])],
            attribute_groups=[{"attribute": "region", "template": "region:{value}"}]),
        **kw)


@pytest.fixture
def legacy(tmp_path):
    return create_legacy(str(tmp_path / "legacy.db"))


@pytest.fixture
def transport(legacy):
    return httpx.ASGITransport(app=legacy)


@pytest.fixture(autouse=True)
def policy():
    p = AllowAllPolicy()
    wiring.override("policy", p)
    return p


async def login(transport, user: str) -> str:
    async with httpx.AsyncClient(transport=transport, base_url="http://legacy") as c:
        r = await c.post("/login", json={"login": user, "password": f"{user}-pw"})
        assert r.status_code == 200
        return r.cookies["LEGACYSESSID"]


async def test_introspection_maps_roles_and_token_verifies(transport, policy):
    sid = await login(transport, "bob")
    bridge = IdentityBridge(bridge_config(), validator_extra={"transport": transport})
    app = create_bridge_app(bridge)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bridge") as c:
        r = await c.post("/v1/token", headers={"Cookie": f"LEGACYSESSID={sid}"})
    assert r.status_code == 200, r.text
    body = r.json()
    p = get_token_service().verify(body["access_token"])
    assert p.sub == "bob" and p.tenant == "acme" and p.app == "claims"
    assert set(p.roles) == {"user", "claims-user", "fraud-investigator"}
    assert {"claims-team", "fraud", "region:south"} <= set(p.groups)
    assert p.claims["app_session"] == sid and p.claims["idv"] == "legacy"
    assert p.claims["exp"] - p.claims["iat"] == 900
    assert policy.inputs[0].action == "identity.token_issue"
    issued = [r for r in get_audit().read() if r.event == "identity.token_issue"]
    assert issued and issued[-1].actor == "bob" and issued[-1].decision == "allow"


async def test_bad_session_is_rejected_and_audited(transport):
    bridge = IdentityBridge(bridge_config(), validator_extra={"transport": transport})
    app = create_bridge_app(bridge)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bridge") as c:
        r = await c.post("/v1/token", headers={"Cookie": "LEGACYSESSID=forged"})
        r2 = await c.post("/v1/token")
    assert r.status_code == 401 and r2.status_code == 401
    denials = [a for a in get_audit().read() if a.event == "identity.token_denied"]
    assert len(denials) == 2 and all(d.decision == "deny" for d in denials)


async def test_policy_can_refuse_issuance(transport):
    wiring.override("policy", DenyPolicy({"identity.token_issue"}))
    sid = await login(transport, "alice")
    bridge = IdentityBridge(bridge_config(), validator_extra={"transport": transport})
    with pytest.raises(PolicyViolation):
        await bridge.exchange(SessionCredentials(cookies={"LEGACYSESSID": sid}))
    assert get_audit().read()[-1].event == "identity.token_denied"


async def test_refresh_revalidates_session(transport):
    sid = await login(transport, "alice")
    bridge = IdentityBridge(bridge_config(), validator_extra={"transport": transport})
    first = await bridge.exchange(SessionCredentials(cookies={"LEGACYSESSID": sid}))
    # refresh needs no cookie: the app session travels in the token
    second = await bridge.refresh(first.access_token)
    assert second.sub == "alice" and second.access_token != first.access_token
    # after logout in the app, refresh is refused
    async with httpx.AsyncClient(transport=transport, base_url="http://legacy") as c:
        await c.post("/logout", headers={"Cookie": f"LEGACYSESSID={sid}"})
    with pytest.raises(AuthenticationError):
        await bridge.refresh(second.access_token)


async def test_refresh_grace_window():
    ident = SessionIdentity(sub="x", validator="h")
    bridge = IdentityBridge(IdentityBridgeConfig(policy_check=False, refresh_grace_seconds=10),
                            validators=[_Static(ident)])
    ts = get_token_service()
    old = ts.issue(sub="x", ttl=1, extra={"idv": "h"})
    claims = jwt.decode(old, options={"verify_signature": False})
    expired = ts.issue(sub="x", extra={"idv": "h", "exp": int(time.time()) - 5, "iat": claims["iat"] - 10})
    assert (await bridge.refresh(expired)).sub == "x"
    too_old = ts.issue(sub="x", extra={"idv": "h", "exp": int(time.time()) - 60})
    with pytest.raises(AuthenticationError):
        await bridge.refresh(too_old)


class _Static:
    name = "h"

    def __init__(self, ident: SessionIdentity):
        self.ident = ident

    async def validate(self, creds: SessionCredentials) -> SessionIdentity:
        return self.ident


async def test_header_trust_only_from_allowed_cidrs():
    v = HeaderTrustValidator(["10.0.0.0/8"])
    headers = {"x-remote-user": "carol", "x-remote-roles": "customer_service", "x-remote-groups": "service-team"}
    with pytest.raises(AuthenticationError):
        await v.validate(SessionCredentials(headers=headers, client_ip="192.168.1.5"))
    with pytest.raises(AuthenticationError):
        await v.validate(SessionCredentials(headers=headers, client_ip=None))
    ident = await v.validate(SessionCredentials(headers=headers, client_ip="10.2.3.4"))
    assert ident.sub == "carol" and ident.app_roles == ["customer_service"] and ident.groups == ["service-team"]


def _rsa() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


async def test_oidc_token_exchange():
    priv, pub = _rsa()
    idp = OidcValidator("https://idp.example", public_key=pub, audience="aiready",
                        mappings={"sub": "$.preferred_username", "roles": "$.realm_access.roles", "groups": "$.groups"})
    bridge = IdentityBridge(IdentityBridgeConfig(policy_check=False, role_mapping=RoleMapping(
        rules=[RoleRule(app_role="adjuster", roles=["claims-user"])])), validators=[idp])
    now = int(time.time())
    good = jwt.encode({"iss": "https://idp.example", "aud": "aiready", "sub": "u1", "preferred_username": "dana",
                       "realm_access": {"roles": ["adjuster"]}, "groups": ["claims-team"], "iat": now,
                       "exp": now + 60}, priv, algorithm="RS256")
    resp = await bridge.exchange(SessionCredentials(id_token=good))
    p = get_token_service().verify(resp.access_token)
    assert p.sub == "dana" and "claims-user" in p.roles and "claims-team" in p.groups
    assert "app_session" not in p.claims  # nothing to forward for IdP tokens
    other_priv, _ = _rsa()
    forged = jwt.encode({"iss": "https://idp.example", "aud": "aiready", "sub": "u1", "exp": now + 60},
                        other_priv, algorithm="RS256")
    with pytest.raises(AuthenticationError):
        await bridge.exchange(SessionCredentials(headers={"authorization": f"Bearer {forged}"}))


async def test_shared_session_store_sql(tmp_path):
    import sqlite3

    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (id TEXT, data TEXT, expires_at REAL)")
    conn.execute("INSERT INTO sessions VALUES ('s1', '{\"username\": \"erin\", \"roles\": [\"adjuster\"]}', ?)",
                 (time.time() + 60,))
    conn.execute("INSERT INTO sessions VALUES ('s2', '{\"username\": \"old\"}', ?)", (time.time() - 60,))
    conn.commit()
    v = SharedSessionStoreValidator(f"sqlite:///{db}", cookie_name="SID", data_column="data",
                                    expires_at_field="$.expires_at", mappings={"sub": "$.username", "roles": "$.roles"})
    ident = await v.validate(SessionCredentials(cookies={"SID": "s1"}))
    assert ident.sub == "erin" and ident.app_roles == ["adjuster"] and ident.session == "s1"
    for sid in ("s2", "nope"):
        with pytest.raises(AuthenticationError):
            await v.validate(SessionCredentials(cookies={"SID": sid}))


def test_role_mapping_rules():
    from aiready.integration.config import AttributeGroup

    m = RoleMapping(rules=[RoleRule(app_role="adjuster", roles=["claims-user"]),
                           RoleRule(attributes={"department": "claims"}, groups=["claims-dept"])],
                    attribute_groups=[AttributeGroup(attribute="region")], keep_unmapped_roles=True)
    roles, groups = map_roles(SessionIdentity(sub="a", app_roles=["adjuster", "clerk"],
                                              attributes={"department": "claims", "region": "north"}), m)
    assert roles == ["app:clerk", "claims-user", "user"] and groups == ["claims-dept", "north"]
    m.keep_unmapped_roles = False
    assert map_roles(SessionIdentity(sub="a", app_roles=["clerk"]), m)[0] == ["user"]


async def test_jwks_published_for_asymmetric_keys_only():
    bridge = IdentityBridge(IdentityBridgeConfig(policy_check=False), validators=[_Static(SessionIdentity(sub="x"))])
    app = create_bridge_app(bridge)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bridge") as c:
        assert (await c.get("/.well-known/jwks.json")).status_code == 404
        priv, pub = _rsa()
        set_token_service(TokenService(algorithm="RS256", private_key=priv, public_key=pub))
        r = await c.get("/.well-known/jwks.json")
        assert r.status_code == 200
        key = r.json()["keys"][0]
        assert key["kty"] == "RSA" and key["alg"] == "RS256" and key["kid"]
        tok = (await c.post("/v1/token")).json()["access_token"]
        jwk = jwt.PyJWK(key)
        assert jwt.decode(tok, jwk.key, algorithms=["RS256"], audience="aiready")["sub"] == "x"
        assert (await c.get("/v1/validators")).json()["validators"][0]["name"] == "h"


async def test_one_bridge_serves_several_apps():
    """Each app's validator names its app and its own role mapping."""
    from aiready.integration.config import IdentityBridgeConfig
    from aiready.integration.identity_bridge import IdentityBridge, SessionIdentity
    from aiready.core import wiring
    from aiready.testing import AllowAllPolicy

    wiring.override("policy", AllowAllPolicy())
    cfg = IdentityBridgeConfig(app="claims", validators=[
        {"type": "header_trust", "name": "crm-proxy", "app": "crm", "trusted_cidrs": ["127.0.0.1/32"],
         "role_mapping": {"rules": [{"app_role": "rep", "roles": ["crm-user"], "groups": ["sales"]}]}},
    ])
    bridge = IdentityBridge(cfg)
    resp = await bridge._issue(SessionIdentity(sub="sam", app_roles=["rep"], validator="crm-proxy"), event="t")
    from aiready.core.identity import get_token_service

    p = get_token_service().verify(resp.access_token)
    assert p.app == "crm" and "crm-user" in p.roles and "sales" in p.groups
