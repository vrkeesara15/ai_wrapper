import random

import pytest
from fastapi.testclient import TestClient

from aiready import testing
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.errors import PermissionDenied, PolicyViolation, ValidationFailed
from aiready.core.interfaces import PolicyDecision
from aiready.core.textutil import hash_embed
from aiready.core.types import Chunk, SearchRequest
from aiready.search.config import SearchConfig
from aiready.search.filters import SearchFilter, matches
from aiready.search.fusion import rrf, weighted
from aiready.search.keyword import Bm25Index
from aiready.search.service import SearchService, collection_name
from aiready.search.stores import MemoryVectorStore, pg_filter_sql

COL = collection_name("claims", "acme")


def chunk(cid, doc, text, tags, **kw):
    return Chunk(id=cid, doc_id=doc, text=text, access_tags=tags, vector=hash_embed(text), **kw)


CORPUS = [
    chunk("c1", "d1", "To file a claim submit form C-12 within thirty days.", ["claims-team"], source="wiki"),
    chunk("c2", "d1", "Claims over 10000 EUR need a manager approval.", ["claims-team"], source="wiki"),
    chunk("c3", "d2", "Executive salaries are reviewed every year.", ["hr-only"], source="hr"),
    chunk("c4", "d3", "The office cafeteria opens at eight.", ["public"], source="intranet"),
    chunk(
        "c5",
        "d4",
        "Error code ZX-9041 means the claim payment bounced.",
        ["claims-team"],
        source="jira",
        metadata={"region": "eu"},
    ),
]


@pytest.fixture(autouse=True)
def _deps():
    wiring.override("gateway", testing.FakeGateway())
    wiring.override("policy", testing.AllowAllPolicy())


@pytest.fixture
async def svc():
    s = SearchService(SearchConfig())
    await s.upsert(COL, CORPUS)
    return s


async def test_hybrid_search_returns_scored_hits(svc, principal):
    hits = await svc.search(SearchRequest(collection=COL, query_text="how do I file a claim"), principal)
    assert hits[0].chunk.id == "c1"
    assert hits[0].dense_score is not None and hits[0].keyword_score is not None
    assert all(h.chunk.vector is None for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    rec = [r for r in get_audit().read() if r.event == "search.query"][-1]
    assert rec.details["chunk_ids"] == [h.chunk.id for h in hits] and rec.resource == COL


async def test_keyword_finds_exact_codes(svc, principal):
    hits = await svc.search(SearchRequest(collection=COL, query_text="ZX-9041"), principal)
    assert hits[0].chunk.id == "c5" and hits[0].keyword_score > 0


async def test_access_filter_hides_other_groups(svc, principal):
    hits = await svc.search(
        SearchRequest(collection=COL, query_text="executive salaries reviewed"), principal
    )
    assert "c3" not in {h.chunk.id for h in hits}
    hr = testing.make_principal("hank", groups=["hr-only"])
    hits = await svc.search(SearchRequest(collection=COL, query_text="executive salaries reviewed"), hr)
    assert {h.chunk.id for h in hits} == {"c3", "c4"}  # own group + public


async def test_request_cannot_widen_access_groups(svc, principal):
    req = SearchRequest(collection=COL, query_text="executive salaries", access_groups=["hr-only"])
    assert await svc.search(req, principal) == []  # intersection of {hr-only} with alice's groups is empty


async def test_access_filter_never_leaks_property():
    """Random corpora, random users, random queries: no hit ever lies outside the user's groups."""
    rng = random.Random(1234)
    groups = [f"g{i}" for i in range(6)] + ["public"]
    words = "claim policy invoice salary refund contract code payment report travel".split()
    svc = SearchService(SearchConfig(), vector_store=MemoryVectorStore(), keyword_index=Bm25Index())
    chunks = []
    for i in range(200):
        text = " ".join(rng.choice(words) for _ in range(8))
        tags = rng.sample(groups, rng.randint(0, 2))  # some chunks carry no tags at all
        chunks.append(chunk(f"x{i}", f"doc{i % 40}", text, tags))
    await svc.upsert(COL, chunks)
    by_id = {c.id: c for c in chunks}
    for trial in range(60):
        user_groups = rng.sample(groups[:-1], rng.randint(0, 3))
        user = testing.make_principal(f"u{trial}", groups=user_groups)
        q = " ".join(rng.sample(words, 2))
        hits = await svc.search(SearchRequest(collection=COL, query_text=q, top_k=200), user)
        allowed = set(user.access_groups)
        for h in hits:
            assert set(by_id[h.chunk.id].access_tags) & allowed, (h.chunk.id, user_groups)
        # completeness for keyword matches: every visible chunk containing a query word is returned
        visible = {
            c.id for c in chunks if set(c.access_tags) & allowed and set(q.split()) & set(c.text.split())
        }
        assert visible <= {h.chunk.id for h in hits}


async def test_metadata_filters(svc, principal):
    hits = await svc.search(
        SearchRequest(collection=COL, query_text="claim", filters={"source": "jira"}), principal
    )
    assert [h.chunk.id for h in hits] == ["c5"]
    hits = await svc.search(
        SearchRequest(collection=COL, query_text="claim", filters={"region": "eu"}), principal
    )
    assert [h.chunk.id for h in hits] == ["c5"]
    hits = await svc.search(
        SearchRequest(collection=COL, query_text="claim", filters={"doc_id": {"in": ["d1"]}}), principal
    )
    assert {h.chunk.id for h in hits} == {"c1", "c2"}
    with pytest.raises(ValidationFailed):
        await svc.search(
            SearchRequest(collection=COL, query_text="claim", filters={"x": {"regex": "a"}}), principal
        )


async def test_delete_by_doc_and_chunk(svc, principal):
    assert await svc.delete(COL, doc_ids=["d1"]) == 2
    hits = await svc.search(SearchRequest(collection=COL, query_text="file a claim form"), principal)
    assert not {"c1", "c2"} & {h.chunk.id for h in hits}
    assert await svc.delete(COL, chunk_ids=["c5"]) == 1
    stats = await svc.stats(COL)
    assert stats["vectors"] == 2 and stats["keyword_docs"] == 2


async def test_snapshot_and_restore(svc, principal):
    snap = await svc.snapshot(COL)
    await svc.delete(COL, doc_ids=["d1", "d2", "d3", "d4"])
    assert (await svc.stats(COL))["vectors"] == 0
    stats = await svc.restore(COL, snap["id"])
    assert stats["vectors"] == 5 and stats["keyword_docs"] == 5
    hits = await svc.search(SearchRequest(collection=COL, query_text="file a claim"), principal)
    assert hits[0].chunk.id == "c1"
    assert [s["id"] for s in await svc.list_snapshots(COL)] == [snap["id"]]


async def test_memory_store_persistence(tmp_path):
    store = MemoryVectorStore(persist_dir=str(tmp_path / "vec"))
    store.upsert("col", CORPUS[:2])
    reloaded = MemoryVectorStore(persist_dir=str(tmp_path / "vec"))
    assert reloaded.count("col") == 2
    res = reloaded.search("col", hash_embed("file a claim"), SearchFilter(access_groups=["claims-team"]), 1)
    assert res[0][0].id == "c1"


async def test_query_embedded_via_gateway_and_missing_vectors(principal):
    svc = SearchService(SearchConfig())
    await svc.upsert(COL, [c.model_copy(update={"vector": None}) for c in CORPUS])  # embedded via gateway
    hits = await svc.search(
        SearchRequest(collection=COL, query_text="manager approval for claims"), principal
    )
    assert hits[0].chunk.id == "c2"


async def test_one_model_version_per_index():
    svc = SearchService(SearchConfig())
    a = chunk("a", "d", "alpha", ["public"], metadata={"embedding_fingerprint": "bge-m3@1/384"})
    await svc.upsert(COL, [a])
    b = chunk("b", "d", "beta", ["public"], metadata={"embedding_fingerprint": "e5@2/384"})
    with pytest.raises(ValidationFailed):
        await svc.upsert(COL, [b])
    c = Chunk(id="c", doc_id="d", text="gamma", access_tags=["public"], vector=[0.1] * 16)
    with pytest.raises(ValidationFailed):
        await svc.upsert(COL, [c])
    assert (await svc.stats(COL))["embedding_fingerprint"] == "bge-m3@1/384"


async def test_tenant_isolation(svc):
    other = testing.make_principal("eve", tenant="globex", groups=["claims-team"])
    with pytest.raises(PermissionDenied):
        await svc.search(SearchRequest(collection=COL, query_text="claim"), other)


async def test_policy_denial_and_obligations(svc, principal):
    wiring.override("policy", testing.DenyPolicy({"data.read"}))
    with pytest.raises(PolicyViolation):
        await svc.search(SearchRequest(collection=COL, query_text="claim"), principal)
    assert any(r.event == "search.denied" for r in get_audit().read())

    class Obliging:
        async def decide(self, inp):
            return PolicyDecision(allow=True, obligations={"max_results": 1, "filters": {"source": "wiki"}})

    wiring.override("policy", Obliging())
    hits = await svc.search(SearchRequest(collection=COL, query_text="claim"), principal)
    assert len(hits) == 1 and hits[0].chunk.source == "wiki"


def test_fusion():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    kw = [("c", 12.0), ("a", 3.0)]
    fused = rrf([dense, kw], k=60)
    assert fused[0][0] == "a" and fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert {x for x, _ in fused} == {"a", "b", "c"}
    w = weighted([dense, kw], weights=[0.5, 0.5])
    assert w[0][0] in {"a", "c"} and w[-1][0] == "b"


def test_filter_semantics_and_sql():
    p = {"id": "1", "doc_id": "d", "access_tags": ["g1"], "updated_at": 100.0, "metadata": {"lang": "en"}}
    assert matches(p, SearchFilter(access_groups=["g1"], metadata={"lang": "en", "updated_at": {"gte": 50}}))
    assert not matches(p, SearchFilter(access_groups=["g2"]))
    assert not matches({**p, "access_tags": []}, SearchFilter(access_groups=["g1", "public"]))
    params: dict = {}
    sql = pg_filter_sql(SearchFilter(access_groups=["g1"], metadata={"lang": "en'; DROP TABLE x;--"}), params)
    assert "DROP" not in sql and "access_tags && CAST(:groups AS text[])" in sql
    assert "DROP TABLE" in params["v0_0"]  # value is bound, never interpolated


def test_api_auth_for_writes(token_service, user_token):
    from aiready.search.api import create_app

    service = SearchService(SearchConfig())
    body = {"chunks": [c.model_dump(mode="json") for c in CORPUS]}
    with TestClient(create_app(service)) as client:
        user_h = {"Authorization": f"Bearer {user_token}"}
        svc_h = {"X-Service-Token": token_service.issue_service_token("ingestion")}
        assert client.post(f"/v1/collections/{COL}/upsert", json=body).status_code == 401
        assert client.post(f"/v1/collections/{COL}/upsert", json=body, headers=user_h).status_code == 403
        r = client.post(f"/v1/collections/{COL}/upsert", json=body, headers=svc_h)
        assert r.status_code == 200 and r.json()["upserted"] == 5
        ing = token_service.issue(sub="pipeline-bot", tenant="acme", roles=["ingestion"])
        assert (
            client.post(
                f"/v1/collections/{COL}/delete",
                json={"chunk_ids": ["c4"]},
                headers={"Authorization": f"Bearer {ing}"},
            ).json()["deleted"]
            == 1
        )
        assert (
            client.post(f"/v1/collections/{COL}/delete", json={"doc_ids": ["d1"]}, headers=user_h).status_code
            == 403
        )
        # search requires a user token: a bare service token is not enough (agents act as the user)
        assert (
            client.post(
                f"/v1/collections/{COL}/search", json={"query_text": "claim"}, headers=svc_h
            ).status_code
            == 401
        )
        r = client.post(f"/v1/collections/{COL}/search", json={"query_text": "file a claim"}, headers=user_h)
        assert r.status_code == 200 and r.json()["hits"][0]["chunk"]["id"] == "c1"
        assert client.get(f"/v1/collections/{COL}/stats", headers=user_h).json()["vectors"] == 4
        assert client.post(f"/v1/collections/{COL}/snapshot", headers=user_h).status_code == 403
        snap = client.post(f"/v1/collections/{COL}/snapshot", headers=svc_h).json()
        r = client.post(f"/v1/collections/{COL}/restore", json={"snapshot_id": snap["id"]}, headers=svc_h)
        assert r.status_code == 200 and r.json()["vectors"] == 4


async def test_wiring_local_provider():
    wiring.reset()
    wiring.override("gateway", testing.FakeGateway())
    from aiready.core.interfaces import SearchService as Proto

    assert isinstance(wiring.connect("search"), Proto)
    assert wiring.connect("embedding").index_model_fingerprint() == "hash-embed@1/384"
    assert wiring.connect("rerank").resolve(None)[0] == "reranker"
