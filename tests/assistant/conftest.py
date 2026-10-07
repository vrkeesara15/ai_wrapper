from __future__ import annotations

from typing import Any

import pytest

from aiready.agents.config import AgentsConfig
from aiready.agents.runtime import AgentRuntime
from aiready.agents.service import AgentService
from aiready.agents.workflows import WorkflowRegistry
from aiready.assistant.config import AssistantConfig
from aiready.assistant.service import AssistantService
from aiready.core import wiring
from aiready.core.interfaces import RagAnswer, RagQuery, ToolDefinition, ToolResult
from aiready.core.types import Citation
from aiready.testing import AllowAllPolicy, FakeGateway, PassThroughGuardrails

ASSISTANT_CFG: dict[str, Any] = {
    "default_workflow": "retrieval_chain",
    "apps": {
        "claims": {"title": "Claims helper", "collection": "claims-kb",
                   "features": {"triage": "single_call", "actions": "ops_agent"},
                   "allowed_origins": ["https://claims.acme.test", "https://*.claims.acme.test"]},
        "hr": {"allowed_origins": ["https://hr.acme.test"]},
    },
}


class AssistantRag:
    def __init__(self) -> None:
        self.queries: list[RagQuery] = []

    async def answer(self, q: RagQuery, principal) -> RagAnswer:
        self.queries.append(q)
        return RagAnswer(answer="Flood is covered [1].", citations=[
            Citation(marker="[1]", chunk_id="c1", doc_id="d1", title="Policy", source_uri="https://kb.test/p#1")])

    async def retrieve(self, q, principal):
        return []


class GatedTools:
    """Every write waits for approval (stands in for the MCP executor)."""

    async def list_tools(self, principal):
        return [ToolDefinition(name="close_claim", description="Close a claim", effect="write",
                               input_schema={"type": "object", "properties": {"claim_id": {"type": "string"}}})]

    async def call(self, name, arguments, principal, *, approval_id=None, context=None):
        from aiready.agents.approvals import get_approval_store

        req = get_approval_store().create(tool=name, effect="write", arguments=arguments, principal=principal,
                                          run_id=(context or {}).get("run_id"))
        return ToolResult(ok=False, approval_id=req.id, error="waits for approval")


@pytest.fixture
def gateway() -> FakeGateway:
    g = FakeGateway()
    wiring.override("gateway", g)
    return g


@pytest.fixture
def rag() -> AssistantRag:
    r = AssistantRag()
    wiring.override("rag", r)
    return r


@pytest.fixture
def agents(gateway, rag):
    from aiready.agents.approvals import get_approval_store

    wiring.override("policy", AllowAllPolicy())
    wiring.override("tools", GatedTools())
    wiring.override("guardrails", PassThroughGuardrails())
    rt = AgentRuntime(AgentsConfig(), workflows=WorkflowRegistry(
        {"ops_agent": {"pattern": "agent_loop", "tools": ["close_claim"]}}))
    svc = AgentService(rt, watch_config=False)
    wiring.override("agents", svc)
    yield svc
    get_approval_store().unsubscribe(rt.on_approval_decided)


@pytest.fixture
def assistant(agents) -> AssistantService:
    return AssistantService(AssistantConfig.model_validate(ASSISTANT_CFG))
