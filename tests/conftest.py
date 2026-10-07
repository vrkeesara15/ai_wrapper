import os

import pytest

os.environ.setdefault("AIREADY_SETUP_LOGGING", "0")
os.environ.setdefault("AIREADY_CONFIG_WATCH", "0")
os.environ.setdefault("AIREADY_DEV_SIGNING_KEY", "test-signing-key-please-change-0123456789")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh config, token service, audit log, doc store and wiring for every test."""
    from aiready.core import audit, config, identity, stores, wiring

    monkeypatch.chdir(tmp_path)
    config.set_config(config.ConfigStore(tree={"state": {"url": "sqlite:///:memory:"}}))
    identity.set_token_service(identity.TokenService(secret=os.environ["AIREADY_DEV_SIGNING_KEY"]))
    audit.set_audit(audit.AuditLog([audit.MemorySink()]))
    stores.set_docstore(stores.DocStore("sqlite:///:memory:"))
    wiring.reset()
    yield
    wiring.reset()


@pytest.fixture
def token_service():
    from aiready.core.identity import get_token_service

    return get_token_service()


@pytest.fixture
def user_token(token_service):
    return token_service.issue(sub="alice", tenant="acme", app="claims", roles=["analyst"], groups=["claims-team"])


@pytest.fixture
def principal(token_service, user_token):
    return token_service.verify(user_token)
