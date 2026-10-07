"""`aiready app init` produces configuration that every package accepts, and it is governed."""

import shutil
from pathlib import Path

from typer.testing import CliRunner

from aiready.cli.main import app as cli
from aiready.core.config import ConfigStore
from aiready.core.interfaces import PolicyInput

REPO = Path(__file__).resolve().parents[2]


def _init(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    shutil.copytree(REPO / "config", cfg)
    res = CliRunner().invoke(cli, ["app", "init", "crm", "--base-url", "http://crm:9000", "--session-cookie", "S",
                                   "--openapi", "/openapi.json", "--docs", str(tmp_path / "docs"),
                                   "--replica-url", "sqlite:///crm.db", "--user-role", "rep",
                                   "--config-dir", str(cfg)])
    assert res.exit_code == 0, res.output
    return cfg


def test_generated_config_validates(tmp_path):
    cfg = _init(tmp_path)
    res = CliRunner().invoke(cli, ["config", "validate", "--config-dir", str(cfg)])
    assert res.exit_code == 0, res.output


async def test_generated_app_is_wired_and_governed(tmp_path):
    from aiready.agents.workflows import WorkflowRegistry
    from aiready.governance.policy import engine_from_tree
    from aiready.ingestion.config import IngestionConfig
    from aiready.integration.config import IntegrationConfig
    from aiready.integration.identity_bridge import IdentityBridge

    tree = ConfigStore(_init(tmp_path)).tree
    # identity: one bridge, a validator per app, tokens carry the app
    bridge = IdentityBridge(IntegrationConfig.model_validate(tree["integration"]).identity_bridge)
    assert bridge._app_for["crm-session"] == "crm"
    # knowledge: physical collection matches what RAG queries for crm users
    sources = {s.name: s for s in IngestionConfig.model_validate(tree["ingestion"]).sources}
    assert sources["crm-docs"].collection == "crm-kb__crm__default"
    assert sources["crm-db"].enabled is False  # records stay off until the table mapping is filled in
    # orchestration
    reg = WorkflowRegistry()
    reg.load(tree["workflows"])
    assert reg.get("crm_assistant") and reg.get("crm_agent")
    # governance: app tools only for app users; deletes always wait for a person
    engine = engine_from_tree(tree)
    outsider = await engine.decide(PolicyInput(action="tool.call", principal={"sub": "x", "roles": ["user"]},
                                               resource={"tool": "crm_list_accounts", "effect": "read"}))
    assert not outsider.allow
    delete = await engine.decide(PolicyInput(action="tool.call", principal={"sub": "r", "roles": ["crm-user"]},
                                             resource={"tool": "crm_delete_account", "effect": "delete"}))
    assert delete.effect in ("require_approval", "deny")
