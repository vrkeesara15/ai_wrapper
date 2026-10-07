"""Fixtures for the RAG tests: fake search, fake gateway, allow-all policy, pass-through guardrails."""

from __future__ import annotations

import asyncio

import pytest

from aiready.core import config, wiring
from aiready.core.textutil import overlap_score
from aiready.core.types import Chunk, DataClass, SearchHit, SearchRequest
from aiready.testing import AllowAllPolicy, FakeGateway, PassThroughGuardrails, make_principal

CHUNKS = [
    Chunk(id="c1", doc_id="d1", title="Claims handbook", source="wiki", source_uri="https://wiki/claims#file",
          text="To file a claim, submit form C-12 within 30 days of the incident. "
               "Attach photos of the damage and the police report if there is one.",
          access_tags=["claims-team"]),
    Chunk(id="c2", doc_id="d2", title="Refund policy", source="wiki", source_uri="https://wiki/refunds",
          text="Refunds are paid within 14 days after the claim is approved by a claims manager.",
          access_tags=["claims-team"]),
    Chunk(id="c3", doc_id="d3", title="Executive bonus plan", source="hr", source_uri="https://hr/bonus",
          text="The executive bonus pool is approved by the board each March.",
          access_tags=["executives"]),
    Chunk(id="c4", doc_id="d4", title="Customer record", source="crm", source_uri="https://crm/42",
          text="Customer Jane Roe reported water damage in her kitchen on 3 May.",
          access_tags=["claims-team"], data_class=DataClass.PERSONAL),
]


class FakeSearch:
    """In-memory keyword search that enforces access tags like the real service."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = list(chunks or CHUNKS)
        self.requests: list[SearchRequest] = []

    async def search(self, req: SearchRequest, principal) -> list[SearchHit]:
        self.requests.append(req)
        out = []
        for c in self.chunks:
            if c.access_tags and not set(c.access_tags) & set(req.access_groups):
                continue
            s = overlap_score(req.query_text, c.text)
            if s > 0:
                out.append(SearchHit(chunk=c, score=s, dense_score=s, keyword_score=s))
        return sorted(out, key=lambda h: -h.score)[: req.top_k]

    async def upsert(self, collection, chunks):
        return 0

    async def delete(self, collection, *, doc_ids=None, chunk_ids=None):
        return 0

    async def stats(self, collection):
        return {}


class SlowGateway(FakeGateway):
    """The main model hangs; the fallback model answers."""

    def __init__(self, slow_model: str = "auto", delay: float = 1.0):
        super().__init__()
        self.slow_model, self.delay = slow_model, delay

    async def chat(self, req, principal):
        if req.model == self.slow_model:
            await asyncio.sleep(self.delay)
        return await super().chat(req, principal)

    async def chat_stream(self, req, principal):
        if req.model == self.slow_model:
            await asyncio.sleep(self.delay)
        async for c in super().chat_stream(req, principal):
            yield c


class BrokenReranker:
    async def rerank(self, *a, **kw):
        raise RuntimeError("reranker down")


def set_rag_config(**rag) -> None:
    tree = {"state": {"url": "sqlite:///:memory:"},
            "rag": {"prompts": {"source": "files"}, **rag}}
    config.set_config(config.ConfigStore(tree=tree))


@pytest.fixture
def rag_env():
    set_rag_config()
    gw, search, policy = FakeGateway(), FakeSearch(), AllowAllPolicy()
    wiring.override("gateway", gw)
    wiring.override("rerank", gw)
    wiring.override("search", search)
    wiring.override("policy", policy)
    wiring.override("guardrails", PassThroughGuardrails())
    return {"gateway": gw, "search": search, "policy": policy}


@pytest.fixture
def alice():
    return make_principal("alice", groups=["claims-team"], roles=["analyst"])


@pytest.fixture
def bob():
    return make_principal("bob", groups=["sales"], roles=["analyst"])
