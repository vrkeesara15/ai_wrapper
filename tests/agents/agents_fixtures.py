"""Shared test data for the agent tests (a module, not conftest, so tests can import it)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from aiready.core.interfaces import RagAnswer, RagQuery
from aiready.core.types import Chunk, Citation, SearchHit

WORKFLOWS: dict[str, Any] = {
    "claims_agent": {"pattern": "agent_loop", "tools": ["get_claim", "close_claim"], "version": "2"},
    "claims_agent@1": {"pattern": "agent_loop", "tools": ["get_claim"]},
    "tight_agent": {"pattern": "agent_loop", "tools": ["get_claim"], "limits": {"max_steps": 4}},
    "token_capped": {"pattern": "agent_loop", "tools": ["get_claim"], "limits": {"max_tokens": 15}},
    "triage": {"pattern": "router", "routes": {"billing": {"workflow": "single_call", "description": "invoices"},
                                               "policy": {"workflow": "retrieval_chain"}},
               "default": "policy"},
    "gated_report": {
        "description": "explicit approval step before a fixed write tool",
        "steps": [
            {"id": "ask", "type": "approval", "reason": "send report", "next": "send"},
            {"id": "send", "type": "tool", "name": "get_claim", "arguments": {"claim_id": "{{input.claim_id}}"},
             "next": "done"},
            {"id": "done", "type": "end", "output": "sent {{steps.send.content.claim_id}}"},
        ],
    },
}


class Facade:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        cid = request.url.path.rsplit("/", 1)[-1]
        if request.method == "GET":
            return httpx.Response(200, json={"claim_id": cid, "status": "open"})
        return httpx.Response(200, json={"claim_id": cid, "status": json.loads(request.content)["status"]})


class FakeRag:
    def __init__(self) -> None:
        self.queries: list[RagQuery] = []

    async def answer(self, q: RagQuery, principal) -> RagAnswer:
        self.queries.append(q)
        return RagAnswer(answer="Flood damage is covered up to 10k [1].", citations=[
            Citation(marker="[1]", chunk_id="c1", doc_id="policy", title="Policy wording",
                     source_uri="https://docs.test/policy#flood")])

    async def retrieve(self, q: RagQuery, principal) -> list[SearchHit]:
        return [SearchHit(chunk=Chunk(id="c1", doc_id="policy", text="Flood damage is covered.", title="Policy"),
                          score=1.0)]
