from __future__ import annotations

import pytest
from mcp_fixtures import FACADE_SPEC, FacadeRecorder

from aiready.core import wiring
from aiready.mcp.config import ToolsConfig
from aiready.mcp.service import ToolService
from aiready.testing import AllowAllPolicy


@pytest.fixture
def facade() -> FacadeRecorder:
    return FacadeRecorder()


@pytest.fixture
def tools_cfg() -> ToolsConfig:
    return ToolsConfig.model_validate({
        "sources": [{"name": "facade", "spec": FACADE_SPEC, "exclude": ["tag:admin"]}],
        "overrides": {"payClaim": {"effect": "payment", "required_roles": ["claims-handler"]}},
        "builtins": {"query_data": {"enabled": False}},
    })


@pytest.fixture
def policy() -> AllowAllPolicy:
    p = AllowAllPolicy()
    wiring.override("policy", p)
    return p


@pytest.fixture
def tool_service(tools_cfg, facade, policy) -> ToolService:
    svc = ToolService(tools_cfg, transport=facade.transport, watch_config=False)
    wiring.override("tools", svc)
    return svc
