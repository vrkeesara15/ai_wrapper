from __future__ import annotations

from pathlib import Path

import httpx
import yaml
from fastapi.testclient import TestClient

from aiready.core import wiring
from aiready.core.config import ConfigStore, deep_merge
from aiready.core.interfaces import RagQuery
from aiready.core.types import Chunk, SearchHit
from aiready.rag.citations import CitationMapper
from aiready.rag.client import RagClient
from aiready.rag.config import RagConfig
from aiready.rag.context import ContextBuilder, summarise_history
from aiready.rag.grounding import GroundingChecker
from aiready.rag.planner import QueryPlanner
from aiready.rag.prompts import PromptStore, render
from aiready.rag.retriever import merge_hits
from aiready.rag.service import RagServiceImpl

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def hit(cid, text, score=1.0, doc="d", title=None):
    return SearchHit(chunk=Chunk(id=cid, doc_id=doc, text=text, title=title), score=score)


def test_shipped_config_validates():
    from aiready.memory.config import MemoryConfig
    from aiready.query.config import QueryConfig, SemanticLayerConfig

    tree: dict = {}
    for f in ["rag.yaml", "memory.yaml", "query.yaml", "semantic_layer.yaml", *(f"prompts/{p.name}" for p in
                                                                                (REPO_CONFIG / "prompts").glob("*.yaml"))]:
        tree = deep_merge(tree, yaml.safe_load((REPO_CONFIG / f).read_text()))
    store = ConfigStore(tree=tree)
    assert store.section("rag", RagConfig).retrieval.top_k == 50
    assert store.section("memory", MemoryConfig).long_term.require_consent
    assert store.section("query", QueryConfig).executor.replica_url
    assert "open_claims_by_region" in store.section("semantic_layer", SemanticLayerConfig).metrics
    for name in ("rag.answer", "rag.answer_strict", "rag.plan", "rag.judge", "memory.summarize", "query.sql_writer"):
        assert name in tree["prompts"]


async def test_prompt_store_versions(rag_env):
    store = PromptStore("files")
    tpl = await store.get("rag.answer")
    assert tpl.ref == "rag.answer@v1" and "{sources}" in tpl.system
    assert render("a {x} {y} {\"k\": 1}", x=1) == "a 1 {y} {\"k\": 1}"
    from aiready.core import config

    config.set_config(ConfigStore(tree={"prompts": {"rag.answer": {"active": "v2", "versions": {
        "v1": {"system": "old"}, "v2": {"system": "new {sources}", "user": "{question}"}}}}}))
    assert (await store.get("rag.answer")).ref == "rag.answer@v2"
    assert (await store.get("rag.answer", "v1")).system == "old"


def test_merge_hits_dedupes():
    a = [hit("1", "Same text", 0.5), hit("2", "Other", 0.4)]
    b = [hit("1", "Same text", 0.9), hit("3", "same   TEXT", 0.3)]  # chunk 3 duplicates chunk 1's text
    out = merge_hits([a, b], top_k=10)
    assert [h.chunk.id for h in out] == ["1", "2"] and out[0].score == 0.9


def test_citation_mapper_markers():
    passages = [hit("a", "Form C-12 is required. It is due in 30 days.", title="T1"), hit("b", "Refunds take 14 days.")]
    res = CitationMapper().map("Use form C-12. [1] Refunds take 14 days [2, 7]. Nice!", passages)
    assert res.invalid_markers == [7]
    assert [c.markers for c in res.claims] == [[1], [2], []]
    assert [c.chunk_id for c in res.citations] == ["a", "b"]
    assert res.citations[0].quote == "Form C-12 is required."
    assert "[7]" not in res.answer


async def test_grounding_lexical(rag_env, alice):
    passages = [hit("a", "Refunds are paid within 14 days after approval.")]
    mapper, checker = CitationMapper(), GroundingChecker()
    good = mapper.map("Refunds are paid within 14 days [1].", passages)
    bad = mapper.map("Refunds are paid in bitcoin by courier [1].", passages)
    g1 = await checker.check(good.claims, passages, alice)
    g2 = await checker.check(bad.claims, passages, alice)
    assert g1.grounded and g1.score == 1.0
    assert not g2.grounded and g2.unsupported == ["Refunds are paid in bitcoin by courier."]


async def test_grounding_llm_judge_flags_claim(rag_env, alice):
    from aiready.rag.config import GroundingConfig

    rag_env["gateway"].script(content='{"unsupported": [1]}')
    passages = [hit("a", "Refunds are paid within 14 days after approval.")]
    claims = CitationMapper().map("Refunds are paid within 14 days [1].", passages).claims
    g = await GroundingChecker(GroundingConfig(use_llm_judge=True)).check(claims, passages, alice)
    assert g.judge_used and not g.grounded
    assert rag_env["gateway"].calls[-1].model == "judge"


async def test_planner_heuristics(rag_env, alice):
    p = QueryPlanner()
    plan = await p.plan("What is the claim deadline? And who approves refunds?", [], alice)
    assert plan.sub_queries == ["What is the claim deadline?", "And who approves refunds?"]
    rewritten = await p.rewrite("what about cars?", [{"role": "user", "content": "How do I file a home claim?"}], alice)
    assert "claim" in rewritten.standalone and rewritten.standalone.startswith("what about cars?")
    standalone = await p.rewrite("How are refunds approved?", [{"role": "user", "content": "hello"}], alice)
    assert standalone.standalone == "How are refunds approved?"


async def test_planner_uses_model_json(rag_env, alice):
    rag_env["gateway"].script(content='{"standalone": "Who approves refunds?", "sub_queries": []}')
    plan = await QueryPlanner().rewrite("who approves it?", [{"role": "user", "content": "Tell me about refunds"}], alice)
    assert plan.method == "llm" and plan.standalone == "Who approves refunds?"
    assert rag_env["gateway"].calls[-1].model == "small" and rag_env["gateway"].calls[-1].metadata["task"] == "rewrite"


async def test_context_builder_budget(rag_env):
    from aiready.rag.config import ContextConfig

    tpl = await PromptStore("files").get("rag.answer")
    passages = [hit(str(i), "word " * 200, title=f"Doc {i}") for i in range(10)]
    b = ContextBuilder(ContextConfig(max_context_tokens=500, max_passage_tokens=100)).build(
        tpl, question="q?", passages=passages, history=[], app="claims", not_found_message="nf")
    assert 1 <= len(b.passages) < 10
    assert "[1] Doc 0" in b.messages[0].content and b.messages[-1].content == "q?"
    hist = summarise_history([{"role": "user", "content": f"Turn {i}. More text here."} for i in range(10)],
                             recent_turns=2, max_tokens=1000)
    assert "user (earlier): Turn 0." in hist and hist.endswith("Turn 9. More text here.")


async def test_session_memory_integration(rag_env, alice):
    from aiready.memory.service import MemoryServiceImpl

    mem = MemoryServiceImpl()
    wiring.override("memory", mem)
    svc = RagServiceImpl.from_config()
    await svc.answer(RagQuery(question="How are refunds paid?", session_id="s1"), alice)
    ctx = await mem.session_context("s1", alice)
    assert [m["role"] for m in ctx] == ["user", "assistant"] and "14 days" in ctx[1]["content"]
    # follow-up resolved from session memory without the caller sending history
    await svc.answer(RagQuery(question="And when is it approved?", session_id="s1"), alice)
    assert "refunds" in rag_env["search"].requests[-1].query_text.lower()


def test_api_json_and_sse(rag_env, alice):
    from aiready.rag.api import create_app

    client = TestClient(create_app(RagServiceImpl.from_config()))
    h = {"Authorization": f"Bearer {alice.token}"}
    r = client.post("/v1/rag/answer", json={"question": "How do I file a claim?"}, headers=h)
    assert r.status_code == 200 and r.json()["citations"] and r.headers["X-Trace-Id"]
    assert client.post("/v1/rag/answer", json={"question": "x"}).status_code == 401
    r = client.post("/v1/rag/retrieve", json={"question": "How do I file a claim?"}, headers=h)
    assert r.json()[0]["chunk"]["id"] == "c1"
    with client.stream("POST", "/v1/rag/answer/stream", json={"question": "How do I file a claim?"}, headers=h) as s:
        body = "".join(s.iter_text())
    assert "event: start" in body and "event: verdict" in body and "event: done" in body


async def test_http_client_roundtrip(rag_env, alice):
    from aiready.rag.api import create_app

    app = create_app(RagServiceImpl.from_config())
    client = RagClient("http://rag", transport=httpx.ASGITransport(app=app))
    ans = await client.answer(RagQuery(question="How do I file a claim?"), alice)
    assert ans.grounded and ans.citations
    hits = await client.retrieve(RagQuery(question="How do I file a claim?"), alice)
    assert hits[0].chunk.id == "c1"
    events = [e async for e in client.answer_stream(RagQuery(question="How are refunds paid?"), alice)]
    assert events[0]["event"] == "start" and events[-1]["event"] == "done"
    await client.aclose()


def test_wiring_provider(rag_env):
    wiring._INSTANCES.pop("rag", None)
    svc = wiring.connect("rag")
    assert isinstance(svc, RagServiceImpl)


def test_collection_naming(rag_env, alice):
    q = "Tell me"
    svc = RagServiceImpl.from_config()
    assert svc._collection(RagQuery(question=q), alice) == "knowledge__claims__acme"
    assert svc._collection(RagQuery(question=q, collection="kb"), alice) == "kb__claims__acme"
    assert svc._collection(RagQuery(question=q, collection="x__y__z"), alice) == "x__y__z"
