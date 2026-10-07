"""Operator CLI commands referenced by the runbooks and CI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aiready.cli.main import app as cli

REPO = Path(__file__).resolve().parents[2]
run = CliRunner().invoke


def test_policy_test_passes_on_shipped_policies():
    res = run(cli, ["policy", "test", str(REPO / "config/policies/tests"), "--config-dir", str(REPO / "config")])
    assert res.exit_code == 0, res.output
    assert "passed" in res.output


def test_token_rotation_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("AIREADY_LOCAL_SECRET_DIR", str(tmp_path))
    monkeypatch.setattr("aiready.core.config.SECRET_DIRS", [tmp_path])
    for stage in ("add", "activate", "retire"):
        res = run(cli, ["token", "rotate", "--stage", stage, "--out-dir", str(tmp_path)])
        assert res.exit_code == 0, res.output
    ring = json.loads((tmp_path / "token_keyring").read_text())
    assert [k["status"] for k in ring] == ["retired", "active"]


def test_audit_verify_and_reset_chain(tmp_path, monkeypatch):
    from aiready.core import audit, config

    chain = tmp_path / "audit.jsonl"
    config.set_config(config.ConfigStore(tree={"audit": {"sinks": [{"type": "jsonl", "path": str(chain)}]},
                                               "state": {"url": "sqlite:///:memory:"}}))
    log = audit.AuditLog([audit.JsonlSink(str(chain))])
    for i in range(3):
        log.record("x", actor="a", n=i)
    assert run(cli, ["audit", "verify"]).exit_code == 0
    lines = chain.read_text().splitlines()
    lines[1] = lines[1].replace('"n":1', '"n":7')
    chain.write_text("\n".join(lines) + "\n")
    bad = run(cli, ["audit", "verify"])
    assert bad.exit_code == 1 and "modified" in bad.output
    assert run(cli, ["audit", "reset-chain", "--incident", "INC-9"]).exit_code == 0
    assert run(cli, ["audit", "verify", "--all-chains"]).exit_code == 0  # broken segment kept, quarantined


def test_sizing_and_cost(tmp_path):
    ws = tmp_path / "ws.yaml"
    ws.write_text("pools:\n  - pool: chat\n    peak_users: 200\n    requests_per_user_per_min: 2\n"
                  "    tokens_per_request: 1500\n    supply_tokens_per_sec_per_node: 2500\n"
                  "    supply_source: load test 2026-10-01 on client H100\n")
    res = run(cli, ["sizing", "worksheet", str(ws)])
    assert res.exit_code == 0 and "chat" in res.output, res.output
    res = run(cli, ["cost", "breakeven", "--gpu-platform-monthly", "20000", "--run-support-monthly", "10000",
                    "--api-price-per-request", "0.01"])
    assert res.exit_code == 0 and "3,000,000" in res.output.replace(" ", ""), res.output


@pytest.mark.parametrize("group", ["serve", "config", "policy", "token", "audit", "registry", "incident", "ingest",
                                   "sizing", "cost", "bundle", "eval", "app"])
def test_command_groups_have_help(group):
    assert run(cli, [group, "--help"]).exit_code == 0
