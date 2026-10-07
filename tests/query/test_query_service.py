from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from aiready.core import config, wiring
from aiready.core.audit import get_audit
from aiready.core.config import deep_merge
from aiready.core.errors import ConfigError, NotFound, PermissionDenied, PolicyViolation, UpstreamError
from aiready.core.interfaces import PolicyDecision, QueryRequest
from aiready.query.config import ValidatorConfig
from aiready.query.executor import ReplicaExecutor
from aiready.query.service import QueryServiceImpl
from aiready.query.validator import SqlRejected, SqlValidator
from aiready.testing import AllowAllPolicy, DenyPolicy, FakeGateway, make_principal

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def make_replica(path: Path) -> str:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, tenant_id TEXT, name TEXT, email TEXT, region TEXT,
                                segment TEXT);
        CREATE TABLE claims (id INTEGER PRIMARY KEY, tenant_id TEXT, customer_id INTEGER REFERENCES customers(id),
                             region TEXT, status TEXT, amount REAL, created_at TEXT);
        CREATE TABLE salaries (id INTEGER PRIMARY KEY, employee TEXT, salary REAL);
    """)
    db.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", [
        (1, "acme", "Jane Roe", "jane@x.com", "north", "retail"),
        (2, "acme", "John Doe", "john@x.com", "south", "business"),
        (3, "globex", "Eve Other", "eve@x.com", "north", "retail")])
    db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?)", [
        (1, "acme", 1, "north", "open", 100.0, "2026-01-05"),
        (2, "acme", 1, "north", "open", 250.0, "2026-02-10"),
        (3, "acme", 2, "south", "paid", 400.0, "2026-02-11"),
        (4, "acme", 2, "south", "open", 50.0, "2026-03-01"),
        (5, "globex", 3, "north", "open", 999.0, "2026-01-01"),
        (6, "globex", 3, "north", "paid", 999.0, "2026-02-01")])
    db.executemany("INSERT INTO salaries VALUES (?,?,?)", [(1, "ceo", 1e6)])
    db.commit()
    db.close()
    return f"sqlite:///{path}"


def load_tree(**query_overrides) -> dict:
    tree: dict = {"state": {"url": "sqlite:///:memory:"}}
    for name in ("query.yaml", "semantic_layer.yaml", "prompts/query.yaml"):
        tree = deep_merge(tree, yaml.safe_load((REPO_CONFIG / name).read_text()))
    tree["query"]["prompts_source"] = "files"
    tree["query"] = deep_merge(tree["query"], query_overrides)
    return tree


@pytest.fixture
def replica(tmp_path):
    return make_replica(tmp_path / "replica.db")


@pytest.fixture
def env(replica):
    config.set_config(config.ConfigStore(tree=load_tree(executor={"replica_url": replica})))
    gw, pol = FakeGateway(), AllowAllPolicy()
    wiring.override("gateway", gw)
    wiring.override("policy", pol)
    return {"gateway": gw, "policy": pol, "replica": replica}


@pytest.fixture
def analyst():
    return make_principal("alice", roles=["analyst"], tenant="acme")


@pytest.fixture
def manager():
    return make_principal("mona", roles=["claims_manager"], tenant="acme")


def _audit(event):
    return [r for r in get_audit().read() if r.event == event]


# ------------------------------------------------------------------ routing / metrics


async def test_metric_preferred_over_generated_sql(env, analyst):
    svc = QueryServiceImpl()
    res = await svc.query(QueryRequest(question="How many open claims are there per region?"), analyst)
    assert res.source == "metric"
    assert env["gateway"].calls == []  # no model call: the approved metric answered
    assert res.columns == ["region", "open_claims"]
    assert res.rows == [["north", 2], ["south", 1]]  # globex's open claim is filtered out by tenant
    assert "tenant_id = :tenant" in res.sql and "LIMIT" in res.sql
    assert "Approved metric 'open_claims_by_region'" in res.summary
    rec = _audit("query.execute")[-1]
    assert rec.details["sql"] == res.sql and rec.details["row_count"] == 2 and rec.details["source"] == "metric"


async def test_metric_by_name_with_params_and_roles(env, analyst, manager):
    svc = QueryServiceImpl()
    req = QueryRequest(metric="paid_amount_between", params={"start_date": "2026-02-01", "end_date": "2026-03-01"})
    with pytest.raises(PermissionDenied):
        await svc.query(req, analyst)  # metric limited to claims_manager
    res = await svc.query(req, manager)
    assert res.rows == [[400, 1]] and res.source == "metric"
    with pytest.raises(PermissionDenied):  # tenant comes from identity, never from the request
        await svc.query(QueryRequest(metric="open_claims_by_region", params={"tenant": "globex"}), manager)
    with pytest.raises(NotFound):
        await svc.query(QueryRequest(metric="nope"), manager)


async def test_generated_sql_validated_and_executed(env, analyst):
    env["gateway"].script(content="```sql\nSELECT status, COUNT(*) AS n FROM claims GROUP BY status\n```")
    res = await QueryServiceImpl().query(QueryRequest(question="Break claims down by status"), analyst)
    assert res.source == "generated"
    assert dict((r[0], r[1]) for r in res.rows) == {"open": 3, "paid": 1}  # acme only
    assert "LIMIT 201" in res.sql  # default max_rows 200 + 1 to detect truncation
    call = env["gateway"].calls[0]
    assert "table claims" in call.messages[0].content and "salaries" not in call.messages[0].content
    assert "email" not in call.messages[0].content  # analyst cannot see customer e-mail
    assert env["policy"].inputs[-1].action == "sql.execute"
    assert env["policy"].inputs[-1].resource["tables"] == ["claims"]


async def test_generated_sql_retries_once_with_validator_errors(env, analyst):
    env["gateway"].script(content="DELETE FROM claims")
    env["gateway"].script(content="SELECT COUNT(*) AS total FROM claims")
    res = await QueryServiceImpl().query(QueryRequest(question="count everything"), analyst)
    assert res.rows == [[4]] and res.summary == "total = 4."
    retry = env["gateway"].calls[1].messages[-1].content
    assert "rejected" in retry and "DELETE" in retry
    assert _audit("query.rejected")[0].details["errors"]
    assert _audit("query.execute")[-1].details["attempts"] == 2


async def test_generated_sql_rejected_twice_raises(env, analyst):
    env["gateway"].script(content="SELECT * FROM salaries")
    env["gateway"].script(content="SELECT employee, salary FROM salaries")
    with pytest.raises(SqlRejected) as ei:
        await QueryServiceImpl().query(QueryRequest(question="show salaries"), analyst)
    assert "table salaries is not allowed" in ei.value.errors[0]
    assert len(_audit("query.rejected")) == 2 and not _audit("query.execute")


async def test_limit_injected_and_truncation(env, analyst):
    env["gateway"].script(content="SELECT id, amount FROM claims ORDER BY id LIMIT 500")
    res = await QueryServiceImpl().query(QueryRequest(question="list claims", max_rows=2), analyst)
    assert res.row_count == 2 and res.truncated
    assert "LIMIT 3" in res.sql
    assert any("truncated" in w for w in res.warnings)


async def test_policy_denial_and_obligations(env, analyst):
    svc = QueryServiceImpl()
    wiring.override("policy", DenyPolicy({"sql.execute"}))
    with pytest.raises(PolicyViolation):
        await svc.query(QueryRequest(metric="open_claims_by_region"), analyst)
    assert _audit("query.rejected")[-1].decision == "deny"

    class Obligations:
        async def decide(self, inp):
            return PolicyDecision(allow=True, obligations={"row_filters": {"claims": "region = 'south'"},
                                                           "max_rows": 1})

    wiring.override("policy", Obligations())
    res = await svc.query(QueryRequest(metric="open_claims_by_region"), analyst)
    assert res.rows == [["south", 1]] and "region = 'south'" in res.sql


async def test_empty_result_and_range_check(env, analyst):
    env["gateway"].script(content="SELECT id FROM claims WHERE status = 'rejected'")
    res = await QueryServiceImpl().query(QueryRequest(question="rejected claims list"), analyst)
    assert res.rows == [] and "the query returned no rows" in res.warnings and res.summary == "No rows matched."


# ------------------------------------------------------------------ replica only


async def test_executes_on_replica_only(env, analyst, tmp_path):
    cfg = load_tree()["query"]
    cfg["executor"]["replica_url"] = None
    config.set_config(config.ConfigStore(tree={"query": cfg, "semantic_layer": load_tree()["semantic_layer"]}))
    with pytest.raises(ConfigError, match="replica"):
        await QueryServiceImpl().query(QueryRequest(metric="open_claims_by_region"), analyst)

    from aiready.query.config import ExecutorConfig

    live = env["replica"]
    ex = ReplicaExecutor(ExecutorConfig(replica_url=live, live_db_urls=[live]), "sqlite")
    with pytest.raises(ConfigError, match="live database"):
        await ex.run("SELECT 1", {}, {}, max_fetch=1)

    # the replica connection itself is read-only, even if a write slipped past validation
    ro = ReplicaExecutor(ExecutorConfig(replica_url=env["replica"]), "sqlite")
    with pytest.raises(UpstreamError):
        await ro.run("DELETE FROM claims", {}, {}, max_fetch=1)
    cols, rows = await ro.run("SELECT COUNT(*) FROM claims", {}, {}, max_fetch=5)
    assert rows == [(6,)]


async def test_string_literal_with_colon_is_not_a_parameter(env, analyst):
    env["gateway"].script(content="SELECT id FROM claims WHERE status = 'x:open' OR status = 'open' ORDER BY id")
    res = await QueryServiceImpl().query(QueryRequest(question="open claim ids"), analyst)
    assert [r[0] for r in res.rows] == [1, 2, 4]


def test_api_and_tool(env, analyst):
    from aiready.query.api import create_app
    from aiready.query.tool import QUERY_TOOL

    client = TestClient(create_app(QueryServiceImpl()))
    h = {"Authorization": f"Bearer {analyst.token}"}
    r = client.post("/v1/query", json={"metric": "open_claims_by_region"}, headers=h)
    assert r.status_code == 200 and r.json()["source"] == "metric"
    names = [m["name"] for m in client.get("/v1/query/metrics", headers=h).json()]
    assert names == ["open_claims_by_region"]
    assert "salaries" not in client.get("/v1/query/catalog", headers=h).json()["catalog"]
    assert client.get("/v1/query/tool", headers=h).json()["name"] == QUERY_TOOL.name
    assert client.post("/v1/query", json={"metric": "paid_amount_between"}, headers=h).status_code == 403


async def test_tool_adapter(env, analyst):
    from aiready.query.tool import call_query_tool

    wiring.override("query", QueryServiceImpl())
    ok = await call_query_tool({"metric": "open_claims_by_region"}, analyst)
    assert ok.ok and ok.content["rows"]
    bad = await call_query_tool({"metric": "nope"}, analyst)
    assert not bad.ok and "not_found" in bad.error


# ------------------------------------------------------------------ validator unit tests

ACCESS = {"claims": None, "customers": {"id", "region", "segment"}}


@pytest.fixture
def v():
    return SqlValidator(ValidatorConfig(), "sqlite")


@pytest.mark.parametrize("sql,needle", [
    ("DELETE FROM claims", "only SELECT"),
    ("UPDATE claims SET amount = 0", "only SELECT"),
    ("DROP TABLE claims", "only SELECT"),
    ("SELECT 1; DELETE FROM claims", "exactly one statement"),
    ("SELECT id FROM claims; SELECT id FROM claims", "exactly one statement"),
    ("WITH x AS (DELETE FROM claims RETURNING id) SELECT * FROM x", ""),
    ("SELECT * FROM salaries", "table salaries is not allowed"),
    ("SELECT name FROM customers", "column name is not allowed"),
    ("SELECT c.email FROM customers c", "column customers.email is not allowed"),
    ("SELECT * FROM customers", "SELECT * is not allowed on customers"),
    ("SELECT email AS email FROM customers", "column email is not allowed"),
    ("SELECT id FROM claims, customers", "cartesian join"),
    ("SELECT claims.id FROM claims CROSS JOIN customers", "cartesian join"),
    ("SELECT id FROM claims WHERE id = :other", "unknown parameter :other"),
    ("SELECT load_extension('x') FROM claims", "load_extension"),
    ("SELECT x.id FROM (SELECT id, email FROM customers) AS x", "column email is not allowed"),
    ("PRAGMA table_info(claims)", "only SELECT"),
])
def test_validator_rejects(v, sql, needle):
    with pytest.raises(SqlRejected) as ei:
        v.validate(sql, access=ACCESS, row_filters={}, max_rows=100)
    assert needle in str(ei.value)


@pytest.mark.parametrize("sql", [
    "SELECT region, COUNT(*) AS n FROM claims GROUP BY region HAVING n > 1 ORDER BY n",
    "SELECT c.id, u.segment FROM claims c JOIN customers u ON u.id = c.customer_id",
    "SELECT c.id FROM claims c, customers u WHERE u.id = c.customer_id",
    "WITH o AS (SELECT region, amount FROM claims WHERE status = 'open') SELECT region, SUM(amount) FROM o GROUP BY region",
    "SELECT * FROM claims",
    "SELECT region FROM claims UNION SELECT region FROM customers",
    "SELECT id FROM main.claims",
])
def test_validator_accepts(v, sql):
    out = v.validate(sql, access=ACCESS, row_filters={"claims": ["tenant_id = :tenant"]}, max_rows=100)
    assert "LIMIT" in out.sql


def test_validator_row_filter_and_limit(v):
    out = v.validate("SELECT c.id FROM claims c JOIN customers u ON u.id = c.customer_id LIMIT 10",
                     access=ACCESS, row_filters={"claims": ["tenant_id = :tenant"],
                                                 "customers": ["tenant_id = :tenant"]}, max_rows=100)
    assert out.sql.count("tenant_id = :tenant") == 2
    assert "(SELECT * FROM claims WHERE tenant_id = :tenant) AS c" in out.sql
    assert out.sql.endswith("LIMIT 10") and out.limit == 10 and "tenant" in out.params
    capped = v.validate("SELECT id FROM claims LIMIT 5000", access=ACCESS, row_filters={}, max_rows=50)
    assert capped.sql.endswith("LIMIT 51") and capped.warnings


async def test_execution_timeout(env):
    from aiready.query.config import ExecutorConfig
    from aiready.query.executor import QueryTimeout

    ex = ReplicaExecutor(ExecutorConfig(replica_url=env["replica"]), "sqlite")
    with pytest.raises(QueryTimeout):
        await ex.run("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r",
                     {}, {}, max_fetch=1, timeout=0.2)
