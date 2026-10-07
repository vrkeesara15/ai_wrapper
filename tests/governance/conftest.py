from pathlib import Path

import pytest

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture
def repo_config():
    """Install the repository's configuration (policies, governance, security) as the active config."""
    from aiready.core import config

    store = config.ConfigStore(REPO_CONFIG)
    config.set_config(store)
    return store


@pytest.fixture
def gov(repo_config):
    from aiready.governance.service import GovernanceService

    return GovernanceService.from_config()


@pytest.fixture
def people(token_service):
    def make(sub: str, *roles: str, groups: list[str] | None = None):
        return token_service.verify(
            token_service.issue(sub=sub, tenant="acme", app="claims", roles=list(roles), groups=groups or [])
        )

    return make
