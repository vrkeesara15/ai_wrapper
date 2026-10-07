from __future__ import annotations

from typing import Any

import httpx
import pytest
from agents_fixtures import WORKFLOWS, Facade, FakeRag

from aiready.agents.config import AgentsConfig
from aiready.agents.runtime import AgentRuntime
from aiready.agents.service import AgentService
from aiready.agents.workflows import WorkflowRegistry
from aiready.core import wiring
from aiready.mcp.config import ToolsConfig
from aiready.mcp.service import ToolService
from aiready.testing import AllowAllPolicy, FakeGateway


@pytest.fixture
def gateway() -> FakeGateway:
    g = FakeGateway()
    wiring.override("gateway", g)
    return g


@pytest.fixture
def facade() -> Facade:
    return Facade()


@pytest.fixture
def rag() -> FakeRag:
    r = FakeRag()
    wiring.override("rag", r)
    return r


@pytest.fixture
def tools(facade) -> ToolService:
    cfg = ToolsConfig.model_validate({
        "builtins": {n: {"enabled": False} for n in ("search_knowledge", "query_data", "remember", "recall")},
        "http_tools": [
            {"name": "get_claim", "description": "Read a claim", "method": "GET",
             "url": "https://facade.test/claims/{claim_id}",
             "input_schema": {"type": "object", "properties": {"claim_id": {"type": "string"}},
                              "required": ["claim_id"]}},
            {"name": "close_claim", "description": "Close a claim", "method": "PATCH",
             "url": "https://facade.test/claims/{claim_id}",
             "input_schema": {"type": "object", "properties": {"claim_id": {"type": "string"},
                                                               "status": {"type": "string"}},
                              "required": ["claim_id", "status"]}},
        ]})
    svc = ToolService(cfg, transport=httpx.MockTransport(facade.handler), watch_config=False)
    wiring.override("tools", svc)
    return svc


@pytest.fixture
def make_runtime(gateway, tools):
    wiring.override("policy", AllowAllPolicy())
    made: list[AgentRuntime] = []

    def factory(**overrides: Any) -> AgentRuntime:
        cfg = AgentsConfig.model_validate({"retry": {"backoff_seconds": 0.01}, **overrides})
        rt = AgentRuntime(cfg, workflows=WorkflowRegistry(WORKFLOWS))
        made.append(rt)
        return rt

    yield factory
    from aiready.agents.approvals import get_approval_store

    for rt in made:  # stop listening to approvals: later tests must not see this runtime
        get_approval_store().unsubscribe(rt.on_approval_decided)


@pytest.fixture
def service(make_runtime) -> AgentService:
    svc = AgentService(make_runtime(), watch_config=False)
    wiring.override("agents", svc)
    return svc
