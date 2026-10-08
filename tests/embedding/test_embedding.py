import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from aiready import testing
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.config import ConfigStore, set_config
from aiready.core.errors import NotFound, PolicyViolation, RateLimited
from aiready.core.textutil import hash_embed
from aiready.embedding.batcher import DynamicBatcher
from aiready.embedding.config import BatcherConfig, EmbeddingConfig
from aiready.embedding.encoders import HashEncoder, OpenAICompatibleEncoder
from aiready.embedding.service import EmbeddingService


class RecordingEncoder:
    """Slow fake encoder that records each batch it receives."""

    def __init__(self, delay: float = 0.0, dim: int = 8):
        self.model_id, self.version, self.max_batch_size, self._dim = "rec", "1", 1000, dim
        self.delay = delay
        self.batches: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    async def encode(self, texts):
        self.batches.append(list(texts))
        if self.delay:
            await asyncio.sleep(self.delay)
        return [hash_embed(t, self._dim) for t in texts]


@pytest.fixture(autouse=True)
def _deps():
    wiring.override("policy", testing.AllowAllPolicy())


async def test_batcher_fills_batches_from_concurrent_requests():
    enc = RecordingEncoder()
    b = DynamicBatcher(enc.encode, max_batch_size=4, max_wait_ms=50, live_max_wait_ms=50)
    out = await asyncio.gather(*(b.submit([f"text {i}"], "live") for i in range(10)))
    assert [len(x) for x in enc.batches] == [4, 4, 2]
    assert out[3][0] == hash_embed("text 3", 8)  # each caller gets its own vector back
    await b.aclose()


async def test_live_preempts_bulk():
    enc = RecordingEncoder(delay=0.03)
    b = DynamicBatcher(enc.encode, max_batch_size=4, max_wait_ms=0, live_max_wait_ms=0)
    bulk = asyncio.create_task(b.submit([f"doc {i}" for i in range(12)], "bulk"))
    await asyncio.sleep(0.01)  # first bulk batch is now on the "GPU"
    assert b.depth("bulk") == 8
    live = await b.submit(["user question"], "live")
    await bulk
    assert enc.batches[0] == ["doc 0", "doc 1", "doc 2", "doc 3"]
    assert enc.batches[1][0] == "user question"  # served before the remaining bulk texts
    assert live[0] == hash_embed("user question", 8)
    assert sum(len(x) for x in enc.batches) == 13
    await b.aclose()


async def test_bulk_back_pressure():
    enc = RecordingEncoder(delay=0.05)
    b = DynamicBatcher(enc.encode, max_batch_size=2, max_wait_ms=0, max_queue=5)
    with pytest.raises(RateLimited):
        await b.submit([f"t{i}" for i in range(6)], "bulk")
    await b.aclose()


async def test_encoder_failure_propagates_and_worker_survives():
    calls = {"n": 0}

    async def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("gpu fell over")
        return [[1.0] for _ in texts]

    b = DynamicBatcher(flaky, max_batch_size=8, max_wait_ms=0)
    with pytest.raises(RuntimeError):
        await b.submit(["a"], "live")
    assert await b.submit(["b"], "live") == [[1.0]]
    await b.aclose()


def _service(**kw) -> tuple[EmbeddingService, HashEncoder]:
    enc = HashEncoder(dim=64, model_id="hash-test", version="7")
    cfg = EmbeddingConfig(batcher=BatcherConfig(max_wait_ms=0, live_max_wait_ms=0), **kw)
    return EmbeddingService(cfg, encoders={"embedding": enc}), enc


async def test_cache_hit_avoids_encoder_call():
    svc, enc = _service()
    v1 = await svc.embed(["hello world", "hello world", "other"])
    assert enc.calls == 1 and v1[0] == v1[1]  # duplicate text encoded once
    v2 = await svc.embed(["hello world"], priority="bulk")
    assert enc.calls == 1 and v2[0] == v1[0]
    await svc.embed(["new text"])
    assert enc.calls == 2
    await svc.aclose()


async def test_cache_is_keyed_on_model_version():
    from aiready.embedding.cache import VectorCache

    cache = VectorCache()
    cache.set_many(["x"], [[1.0]], "m", "1")
    assert cache.get_many(["x"], "m", "1") == [[1.0]]
    assert cache.get_many(["x"], "m", "2") == [None]


async def test_fingerprint_and_resolution():
    svc, _ = _service()
    assert svc.index_model_fingerprint() == "hash-test@7/64"
    assert svc.resolve("hash-test")[0] == "embedding"  # concrete model id accepted too
    with pytest.raises(NotFound):
        svc.resolve("nope")
    info = await svc.info()
    assert info["models"][0]["fingerprint"] == "hash-test@7/64"
    await svc.aclose()


async def test_policy_denial_and_audit(principal):
    svc, enc = _service()
    wiring.override("policy", testing.DenyPolicy({"model.invoke"}))
    with pytest.raises(PolicyViolation):
        await svc.embed(["secret"], principal=principal)
    assert enc.calls == 0
    wiring.override("policy", testing.AllowAllPolicy())
    await svc.embed(["fine"], principal=principal)
    events = [r.event for r in get_audit().read()]
    assert "embedding.denied" in events and "embedding.embed" in events
    await svc.aclose()


async def test_openai_compatible_encoder():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen["url"], seen["model"] = str(request.url), body["model"]
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data, "model": body["model"]})

    enc = OpenAICompatibleEncoder("http://tei:80", "bge-m3", transport=httpx.MockTransport(handler))
    out = await enc.encode(["a", "b"])
    assert seen["url"] == "http://tei/v1/embeddings" and seen["model"] == "bge-m3"
    assert out[0] == pytest.approx([0.6, 0.8]) and enc.dim == 2


def test_default_config_file_parses():
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[2]
    from aiready.core.config import _substitute

    tree = _substitute(yaml.safe_load((root / "config" / "knowledge.yaml").read_text()))  # as the loader does
    set_config(ConfigStore(tree=tree))
    svc = EmbeddingService()
    assert svc.index_model_fingerprint() == "hash-embed@1/384"


def test_api_openai_compatible(token_service, user_token):
    svc, _ = _service()
    from aiready.embedding.api import create_app

    with TestClient(create_app(svc)) as client:
        assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 401
        r = client.post(
            "/v1/embeddings",
            json={"input": ["a b", "c"], "model": "embedding"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert (
            body["model"] == "hash-test"
            and len(body["data"]) == 2
            and len(body["data"][0]["embedding"]) == 64
        )
        # ingestion pipeline: service token only, bulk queue
        svc_tok = token_service.issue_service_token("ingestion")
        r = client.post(
            "/v1/embeddings",
            json={"input": ["doc"], "priority": "bulk"},
            headers={"X-Service-Token": svc_tok},
        )
        assert r.status_code == 200
        info = client.get("/v1/embeddings/info", headers={"X-Service-Token": svc_tok}).json()
        assert info["models"][0]["fingerprint"] == "hash-test@7/64"
