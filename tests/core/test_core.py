import pytest

from aiready.core import plugins
from aiready.core.audit import AuditLog, MemorySink, SqliteSink
from aiready.core.audit import JsonlSink as JsonlSinkCls
from aiready.core.config import ConfigStore, deep_merge
from aiready.core.errors import AuthenticationError
from aiready.core.stores import DocStore, MemoryKV
from aiready.core.telemetry import current_trace_id, recorder, span


def test_token_roundtrip(token_service):
    tok = token_service.issue(sub="bob", roles=["agent"], groups=["g1"], actor="agent:x")
    p = token_service.verify(tok)
    assert p.sub == "bob" and p.actor == "agent:x"
    assert "role:agent" in p.access_groups and "user:bob" in p.access_groups


def test_bad_token_rejected(token_service):
    with pytest.raises(AuthenticationError):
        token_service.verify(token_service.issue(sub="x") + "tampered")


def test_audit_chain_detects_tampering(tmp_path):
    sink = SqliteSink(str(tmp_path / "a.db"))
    log = AuditLog([sink], hmac_key="k")
    for i in range(5):
        log.record("test", actor="alice", n=i)
    assert log.verify()[0]
    mem = MemorySink()
    log2 = AuditLog([mem])
    for i in range(3):
        log2.record("test", actor="alice", n=i)
    mem.records[1].details["n"] = 99
    ok, msg = log2.verify()
    assert not ok and "modified" in msg


def test_sqlite_audit_is_append_only(tmp_path):
    import sqlite3

    sink = SqliteSink(str(tmp_path / "a.db"))
    AuditLog([sink]).record("x", actor="a")
    with pytest.raises(sqlite3.DatabaseError):
        sink._db.execute("DELETE FROM audit")


def test_config_merge_and_env(tmp_path, monkeypatch):
    (tmp_path / "a.yaml").write_text("gateway:\n  routes: [{name: a}]\n  port: ${PORT:-8080}\n")
    (tmp_path / "b.yaml").write_text("gateway:\n  routes: [{name: b}]\n")
    monkeypatch.setenv("PORT", "9000")
    store = ConfigStore(tmp_path)
    assert [r["name"] for r in store.get("gateway.routes")] == ["a", "b"]
    assert store.get("gateway.port") == 9000
    assert deep_merge({"a": [1]}, {"=a": [2]}, overlay=True) == {"a": [2]}


def test_plugin_class_path():
    kv = plugins.create("kv_store", {"class": "aiready.core.stores:MemoryKV"})
    assert isinstance(kv, MemoryKV)
    assert isinstance(plugins.create("kv_store", {"type": "memory"}), MemoryKV)


def test_kv_and_docstore():
    kv = MemoryKV()
    assert kv.incr("c", 2) == 2 and kv.incr("c") == 3
    ds = DocStore("sqlite:///:memory:")
    ds.put("ns", "k1", {"a": 1})
    assert ds.get("ns", "k1") == {"a": 1}
    assert list(ds.find("ns", a=1))


def test_span_records():
    with span("outer", x=1):
        tid = current_trace_id()
        with span("inner"):
            pass
    spans = recorder.trace(tid)
    assert {s.name for s in spans} == {"outer", "inner"}


def test_revoked_subject_rejected_at_authentication(token_service):
    from aiready.core.identity import principal_from_headers, set_revocation_checker

    tok = token_service.issue(sub="mallory")
    set_revocation_checker(lambda p: "sub:mallory revoked" if p.sub == "mallory" else None)
    with pytest.raises(AuthenticationError):
        principal_from_headers({"Authorization": f"Bearer {tok}"})
    assert principal_from_headers({"Authorization": f"Bearer {token_service.issue(sub='bob')}"}).jti


def test_keyring_staged_rotation():
    from aiready.core.identity import TokenService

    old = {"kid": "k1", "secret": "a" * 32, "status": "active"}
    new = {"kid": "k2", "secret": "b" * 32, "status": "verify"}
    ts = TokenService(keyring=[old, new])
    t_old = ts.issue(sub="u")
    # stage activate: k2 signs, k1 still verifies tokens already issued
    ts2 = TokenService(keyring=[{**old, "status": "verify"}, {**new, "status": "active"}])
    assert ts2.verify(t_old).sub == "u"
    t_new = ts2.issue(sub="u")
    # stage retire: k1 tokens are rejected
    ts3 = TokenService(keyring=[{**old, "status": "retired"}, {**new, "status": "active"}])
    assert ts3.verify(t_new).sub == "u"
    with pytest.raises(AuthenticationError):
        ts3.verify(t_old)


def test_two_writers_share_one_chain(tmp_path):
    """Two AuditLog instances (e.g. the CLI and a server) appending to one file keep a single valid chain."""
    path = str(tmp_path / "shared.jsonl")
    a, b = AuditLog([JsonlSinkCls(path)], hmac_key="k"), AuditLog([JsonlSinkCls(path)], hmac_key="k")
    for i in range(5):
        a.record("x", actor="server", n=i)
        b.record("y", actor="cli", n=i)
    assert a.verify()[0], a.verify()[1]


def test_sqlite_two_writers(tmp_path):
    path = str(tmp_path / "a.db")
    a, b = AuditLog([SqliteSink(path)]), AuditLog([SqliteSink(path)])
    for i in range(3):
        a.record("x", actor="s", n=i)
        b.record("y", actor="c", n=i)
    assert a.verify()[0]
