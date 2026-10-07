"""Fixtures for the ingestion tests: fake gateway / policy / guardrails from ``aiready.testing`` and a small
in-memory search service that enforces access tags like the real one."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from aiready.core import wiring
from aiready.core.identity import Principal
from aiready.core.textutil import cosine, overlap_score
from aiready.core.types import Chunk, SearchHit, SearchRequest
from aiready.testing import AllowAllPolicy, FakeGateway, PassThroughGuardrails

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
LEGACY_DOCS = REPO / "examples" / "legacy_app" / "docs"


class FakeSearch:
    """In-memory hybrid search: access filter first (any tag matches), then cosine + keyword overlap."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Chunk]] = {}
        self.upserts = 0
        self.deletes: list[dict[str, Any]] = []

    async def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        col = self.collections.setdefault(collection, {})
        for c in chunks:
            assert c.vector is not None, "chunks must be embedded before upsert"
            col[c.id] = c.model_copy(deep=True)
        self.upserts += len(chunks)
        return len(chunks)

    async def delete(self, collection: str, *, doc_ids: list[str] | None = None,
                     chunk_ids: list[str] | None = None) -> int:
        self.deletes.append({"collection": collection, "doc_ids": doc_ids, "chunk_ids": chunk_ids})
        col = self.collections.get(collection, {})
        drop = [cid for cid, c in col.items() if (doc_ids and c.doc_id in doc_ids) or (chunk_ids and cid in chunk_ids)]
        for cid in drop:
            del col[cid]
        return len(drop)

    async def search(self, req: SearchRequest, principal: Principal) -> list[SearchHit]:
        groups = set(req.access_groups)
        hits = []
        for c in self.collections.get(req.collection, {}).values():
            if not groups & set(c.access_tags):
                continue
            dense = cosine(req.query_vector, c.vector) if req.query_vector and c.vector else 0.0
            kw = overlap_score(req.query_text, f"{c.title} {c.text}")
            hits.append(SearchHit(chunk=c, score=dense + kw, dense_score=dense, keyword_score=kw))
        hits.sort(key=lambda h: -h.score)
        return hits[: req.top_k]

    async def stats(self, collection: str) -> dict[str, Any]:
        return {"chunks": len(self.collections.get(collection, {}))}

    def chunks(self, collection: str) -> list[Chunk]:
        return list(self.collections.get(collection, {}).values())


class CountingGateway(FakeGateway):
    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []
        self.priorities: list[str] = []

    async def embed(self, texts: list[str], principal: Principal, *, model: str = "embedding",
                    priority: str = "live") -> list[list[float]]:
        self.embedded.extend(texts)
        self.priorities.append(priority)
        return await super().embed(texts, principal, model=model, priority=priority)


@pytest.fixture
def fake_search() -> FakeSearch:
    s = FakeSearch()
    wiring.override("search", s)
    return s


@pytest.fixture
def gateway() -> CountingGateway:
    g = CountingGateway()
    wiring.override("gateway", g)
    return g


@pytest.fixture
def policy() -> AllowAllPolicy:
    p = AllowAllPolicy()
    wiring.override("policy", p)
    return p


@pytest.fixture
def guardrails() -> PassThroughGuardrails:
    g = PassThroughGuardrails()
    wiring.override("guardrails", g)
    return g


@pytest.fixture
def services(fake_search, gateway, policy, guardrails):
    return {"search": fake_search, "gateway": gateway, "policy": policy, "guardrails": guardrails}


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """A private copy of the sample app's document share (tests modify it)."""
    target = tmp_path / "share"
    shutil.copytree(LEGACY_DOCS, target)
    return target
