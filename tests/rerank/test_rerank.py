import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aiready import testing
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.errors import PolicyViolation
from aiready.core.telemetry import metrics
from aiready.core.types import Chunk, SearchHit
from aiready.rerank.config import RerankConfig, SelectionConfig
from aiready.rerank.scorers import LexicalScorer, RemoteScorer, build_pairs
from aiready.rerank.service import FALLBACK_MODEL, RerankService

DOCS = [
    "The cafeteria opens at eight.",
    "To file a claim submit form C-12 within thirty days.",
    "Claims are reviewed by the claims team.",
    "Parking is free on weekends.",
    "File the claim form online.",
]


class SlowScorer:
    model_id, max_length = "slow", 512

    async def score(self, query, passages):
        await asyncio.sleep(1.0)
        return [1.0] * len(passages)


class BrokenScorer:
    model_id, max_length = "broken", 512

    async def score(self, query, passages):
        raise ConnectionError("reranker pod gone")


@pytest.fixture(autouse=True)
def _deps():
    wiring.override("policy", testing.AllowAllPolicy())


def _fallbacks(kind: str) -> float:
    return metrics.fallbacks.labels("rerank", kind)._value.get()


async def test_threshold_and_top_n():
    svc = RerankService(RerankConfig(defaults=SelectionConfig(top_n=2, threshold=0.1)))
    resp = await svc.rerank("how do I file a claim", DOCS)
    idx = [r.index for r in resp.results]
    assert len(idx) == 2 and set(idx) <= {1, 2, 4}
    assert resp.results[0].relevance_score >= resp.results[1].relevance_score
    # high threshold drops everything
    resp = await svc.rerank("how do I file a claim", DOCS, threshold=0.99)
    assert resp.results == []


async def test_max_candidates_and_per_app_config():
    cfg = RerankConfig(apps={"claims": SelectionConfig(max_candidates=2, top_n=1)})
    svc = RerankService(cfg)
    resp = await svc.rerank("file claim form online", DOCS, app="claims")
    assert [r.index for r in resp.results] == [1]  # only the first 2 candidates were considered


async def test_timeout_falls_back_to_search_order():
    svc = RerankService(RerankConfig(timeout_seconds=0.05), scorers={"reranker": SlowScorer()})
    before = _fallbacks("timeout")
    hits = [
        SearchHit(chunk=Chunk(id=f"c{i}", doc_id="d", text=t), score=1 - i / 10) for i, t in enumerate(DOCS)
    ]
    out = await svc.rerank_hits("claim", hits, top_n=3)
    assert out.fallback == "timeout" and out.model == FALLBACK_MODEL
    assert [h.chunk.id for h in out.hits] == ["c0", "c1", "c2"]  # search order kept
    assert all(h.rerank_score is None for h in out.hits)
    assert _fallbacks("timeout") == before + 1
    assert any(r.event == "rerank.score" and r.decision == "fallback" for r in get_audit().read())


async def test_unavailable_reranker_falls_back():
    svc = RerankService(RerankConfig(), scorers={"reranker": BrokenScorer()})
    before = _fallbacks("error")
    resp = await svc.rerank("q", ["a", "b"], top_n=5)
    assert resp.model == FALLBACK_MODEL and [r.index for r in resp.results] == [0, 1]
    assert _fallbacks("error") == before + 1


async def test_rerank_hits_scores_and_orders(principal):
    svc = RerankService(RerankConfig())
    hits = [SearchHit(chunk=Chunk(id=f"c{i}", doc_id="d", text=t), score=0.5) for i, t in enumerate(DOCS)]
    out = await svc.rerank_hits("file a claim form", hits, principal=principal)
    assert out.fallback is None and out.hits[0].chunk.id in {"c1", "c4"}
    scores = [h.rerank_score for h in out.hits]
    assert scores == sorted(scores, reverse=True)
    rec = [r for r in get_audit().read() if r.event == "rerank.score"][-1]
    assert rec.actor == "alice" and rec.details["kept"] == [h.chunk.id for h in out.hits]


async def test_policy_denied(principal):
    wiring.override("policy", testing.DenyPolicy({"model.invoke"}))
    with pytest.raises(PolicyViolation):
        await RerankService(RerankConfig()).rerank("q", ["a"], principal=principal)


def test_pairs_trimmed_to_model_length():
    pairs = build_pairs("short question", ["word " * 2000], max_length=128)
    q, p = pairs[0]
    assert q == "short question" and len(p.split()) < 128


async def test_lexical_batches():
    s = LexicalScorer(batch_size=2)
    assert len(await s.score("claim", DOCS)) == len(DOCS)


@pytest.mark.parametrize("style", ["tei", "cohere"])
async def test_remote_scorer_styles(style):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = body.get("texts") or body.get("documents")
        scores = [float(len(t)) for t in texts]
        if style == "tei":
            return httpx.Response(200, json=[{"index": i, "score": s} for i, s in enumerate(scores)])
        return httpx.Response(
            200, json={"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}
        )

    s = RemoteScorer("http://rr", style=style, batch_size=2, transport=httpx.MockTransport(handler))
    assert await s.score("q", ["a", "bbb", "cc"]) == [1.0, 3.0, 2.0]


def test_api(user_token, token_service):
    from aiready.rerank.api import create_app

    with TestClient(create_app(RerankService(RerankConfig()))) as client:
        assert client.post("/v1/rerank", json={"query": "q", "documents": ["a"]}).status_code == 401
        h = {"Authorization": f"Bearer {user_token}"}
        r = client.post(
            "/v1/rerank",
            headers=h,
            json={"query": "file a claim", "documents": DOCS, "top_n": 2, "return_documents": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 2 and "document" in body["results"][0]
        hits = [
            SearchHit(chunk=Chunk(id=f"c{i}", doc_id="d", text=t), score=0.1).model_dump(mode="json")
            for i, t in enumerate(DOCS)
        ]
        r = client.post(
            "/v1/rerank/hits",
            headers={"X-Service-Token": token_service.issue_service_token("rag")},
            json={"query": "claim form", "hits": hits, "top_n": 3},
        )
        assert r.status_code == 200 and len(r.json()["hits"]) == 3 and r.json()["fallback"] is None
