from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aiready.core import config, plugins
from aiready.core.errors import ConfigError
from aiready.core.types import ChatMessage, ChatRequest, DataClass, ToolSpec
from aiready.models.backends import (
    AnthropicBackend,
    BackendError,
    MockBackend,
    OpenAICompatibleBackend,
    anthropic_to_openai,
    openai_to_anthropic,
)
from aiready.models.catalog import LiveCatalog, ModelCatalog, PoolSpec
from aiready.models.pool import InferenceRouter, ModelPool


def req(text: str = "hello", **kw) -> ChatRequest:
    return ChatRequest(model="m", messages=[ChatMessage(role="user", content=text)], **kw)


# ----------------------------------------------------------------------------- mock backend


async def test_mock_backend_is_registered_and_deterministic():
    b = plugins.create("model_backend", {"type": "mock", "dim": 16})
    ctx = ChatRequest(model="m", messages=[
        ChatMessage(role="system", content="[1] Policy\nClaims must be filed within 30 days. Cats are nice."),
        ChatMessage(role="user", content="When must claims be filed?")])
    r1, r2 = await b.chat(ctx), await b.chat(ctx)
    assert r1.text == r2.text and "30 days" in r1.text and "[1]" in r1.text
    assert r1.usage.total_tokens > 0
    vec = (await b.embed(["claims"], "e"))[0]
    assert len(vec) == 16
    scores = await b.rerank("claims filed", ["claims filed late", "weather"], "r")
    assert scores[0] > scores[1]


async def test_mock_backend_tool_call_scripting_and_tool_result():
    b = MockBackend()
    tools = [ToolSpec(function={"name": "lookup", "parameters": {"type": "object"}})]
    r = await b.chat(req('please [[tool:lookup {"id": 7}]]', tools=tools))
    tc = r.choices[0].message.tool_calls[0]
    assert r.choices[0].finish_reason == "tool_calls" and json.loads(tc.function.arguments) == {"id": 7}
    follow = ChatRequest(model="m", tools=tools, messages=[
        ChatMessage(role="user", content="please [[tool:lookup]]"),
        ChatMessage(role="assistant", content="", tool_calls=[tc]),
        ChatMessage(role="tool", tool_call_id=tc.id, content='{"status": "open"}')])
    assert "open" in (await b.chat(follow)).text


async def test_mock_backend_json_schema_and_failure_mode():
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": ["a", "b"]},
                                               "score": {"type": "number"}}, "required": ["label", "score"]}
    r = await MockBackend().chat(req("x", response_format={"type": "json_schema", "json_schema": {"schema": schema}}))
    assert json.loads(r.text) == {"label": "a", "score": 0.0}
    with pytest.raises(BackendError):
        await MockBackend(fail=True).chat(req())


async def test_mock_backend_streams_openai_chunks():
    chunks = [c async for c in MockBackend().chat_stream(req())]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["usage"]["total_tokens"] > 0


# ----------------------------------------------------------------------------- OpenAI-compatible backend


async def test_openai_backend_strips_metadata_and_parses_responses():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append({"path": request.url.path, "body": body, "auth": request.headers.get("authorization")})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 1, "model": body["model"],
                                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                                          "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.0, 1.0]},
                                                      {"index": 0, "embedding": [1.0, 0.0]}]})
        if request.url.path == "/v1/rerank":
            return httpx.Response(404, json={"error": "nope"})
        if request.url.path == "/rerank":  # TEI style
            return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])
        return httpx.Response(500)

    b = OpenAICompatibleBackend(base_url="http://vllm:8000/v1", api_key="k", transport=httpx.MockTransport(handler))
    r = await b.chat(req(metadata={"feature": "secret-feature"}))
    assert r.text == "hi" and r.usage.total_tokens == 4
    assert "metadata" not in seen[0]["body"] and seen[0]["auth"] == "Bearer k" and seen[0]["body"]["stream"] is False
    assert await b.embed(["a", "b"], "e") == [[1.0, 0.0], [0.0, 1.0]]
    assert await b.rerank("q", ["d0", "d1"], "r") == [0.1, 0.9]  # fell back from /v1/rerank to TEI /rerank


async def test_openai_backend_streaming_and_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("messages")[0]["content"] == "boom":
            return httpx.Response(503, json={"error": "overloaded"})
        sse = "".join(f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': w}}]})}\n\n" for w in ("a", "b"))
        return httpx.Response(200, text=sse + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    b = OpenAICompatibleBackend(transport=httpx.MockTransport(handler))
    chunks = [c async for c in b.chat_stream(req())]
    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["a", "b"]
    with pytest.raises(BackendError) as exc:
        await b.chat(req("boom"))
    assert exc.value.retryable and exc.value.upstream_status == 503


# ----------------------------------------------------------------------------- Anthropic backend


def test_anthropic_mapping_round_trip_with_tools():
    tools = [ToolSpec(function={"name": "get", "description": "d", "parameters": {"type": "object"}})]
    from aiready.core.types import ToolCall, ToolCallFunction

    tc = ToolCall(id="t1", function=ToolCallFunction(name="get", arguments='{"a": 1}'))
    r = ChatRequest(model="claude-x", tools=tools, tool_choice="required", max_tokens=50, stop="END", messages=[
        ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="", tool_calls=[tc]),
        ChatMessage(role="tool", tool_call_id="t1", content="result")], metadata={"x": 1})
    body = openai_to_anthropic(r)
    assert body["system"] == "sys" and body["tool_choice"] == {"type": "any"} and body["stop_sequences"] == ["END"]
    assert body["messages"][1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "get", "input": {"a": 1}}
    assert body["messages"][2]["content"][0]["type"] == "tool_result" and "x" not in json.dumps(body.get("metadata", {}))
    resp = anthropic_to_openai({"id": "m", "content": [{"type": "text", "text": "ok"},
                                                       {"type": "tool_use", "id": "t2", "name": "get", "input": {"b": 2}}],
                                "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 2}}, "claude-x")
    assert resp.text == "ok" and resp.choices[0].finish_reason == "tool_calls" and resp.usage.total_tokens == 7
    assert json.loads(resp.choices[0].message.tool_calls[0].function.arguments) == {"b": 2}


async def test_anthropic_backend_http_and_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "k" and request.headers["anthropic-version"]
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(200, json={"id": "m1", "model": body["model"], "content": [{"type": "text", "text": "hey"}],
                                             "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
        events = [("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 4}}}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "text_delta", "text": "Hel"}}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "text_delta", "text": "lo"}}),
                  ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                     "usage": {"output_tokens": 2}}),
                  ("message_stop", {"type": "message_stop"})]
        text = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    b = AnthropicBackend(api_key="k", transport=httpx.MockTransport(handler))
    assert (await b.chat(req())).text == "hey"
    chunks = [c async for c in b.chat_stream(req())]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello" and chunks[-1]["usage"]["total_tokens"] == 6
    with pytest.raises(BackendError):
        await b.embed(["x"], "e")


# ----------------------------------------------------------------------------- pools and router


async def test_failing_replica_loses_traffic_and_is_ejected():
    good, bad = MockBackend(), MockBackend(fail=True)
    pool = ModelPool("p", PoolSpec(replicas=[{"type": "mock"}, {"type": "mock"}], eject_after_failures=2,
                                   cooldown_seconds=60), backends=[bad, good])
    r = await pool.run(lambda b: b.chat(req()))  # bad is tried first, fails, the call moves to good
    assert r.text and bad.calls == 1 and good.calls == 1
    assert pool.replicas[0].score() > pool.replicas[1].score()  # failure counted as slowness
    rep = pool.replicas[0]
    rep.in_flight += 1
    pool.release(rep, ok=False, latency=0.01)  # second consecutive failure opens the breaker
    stats = pool.stats()
    assert stats["healthy_replicas"] == 1 and stats["replicas"][0]["failures"] == 2
    pool.replicas[1].ewma_latency = 5.0  # even when the healthy replica is slower, the ejected one gets no traffic
    for _ in range(3):
        await pool.run(lambda b: b.chat(req()))
    assert bad.calls == 1


async def test_half_open_replica_recovers_after_cooldown():
    flaky = MockBackend(fail=True)
    pool = ModelPool("p", PoolSpec(replicas=[{"type": "mock"}], eject_after_failures=1, cooldown_seconds=0),
                     backends=[flaky])
    with pytest.raises(BackendError):
        await pool.run(lambda b: b.chat(req()))
    flaky.fail = False
    assert (await pool.run(lambda b: b.chat(req()))).text
    assert pool.stats()["replicas"][0]["consecutive_failures"] == 0


async def test_least_loaded_selection_uses_in_flight_and_latency():
    pool = ModelPool("p", PoolSpec(replicas=[{"type": "mock"}, {"type": "mock"}]))
    a, b = pool.replicas
    a.ewma_latency, b.ewma_latency = 0.5, 0.1
    assert (await pool.acquire()) is b
    b.in_flight = 10
    assert (await pool.acquire()) is a


async def test_router_sync_keeps_unchanged_pools_and_reports_stats():
    specs = {"a": PoolSpec(replicas=[{"type": "mock"}]), "b": PoolSpec(replicas=[{"type": "mock"}])}
    router = InferenceRouter(specs)
    a_before = router.pools["a"]
    await router.chat("a", req())
    router.sync({"a": specs["a"], "b": PoolSpec(replicas=[{"type": "mock"}, {"type": "mock"}])})
    assert router.pools["a"] is a_before and len(router.pools["b"].replicas) == 2
    st = router.stats()["a"]
    assert st["replicas"][0]["requests"] == 1 and st["tokens_per_second"] >= 0
    assert (await router.embed("a", ["x"], "e"))[0]


# ----------------------------------------------------------------------------- catalog


def _tree(**catalog):
    return {"models": {"pools": {"p": {"replicas": [{"type": "mock"}]}}, "catalog": catalog}}


def test_catalog_defaults_close_public_models():
    cat = ModelCatalog.from_tree(_tree(priv={"pool": "p"}, pub={"pool": "p", "category": "public-proprietary"}))
    assert cat.get("priv").allows(DataClass.RESTRICTED)
    assert not cat.get("pub").allows(DataClass.PUBLIC) and cat.get("pub").is_public
    with pytest.raises(ConfigError):
        ModelCatalog.from_tree(_tree(x={"pool": "missing"}))


def test_catalog_upstream_model_resolution():
    tree = _tree(base={"pool": "p"}, tuned={"pool": "p", "adapter": "claims-lora", "category": "private-tuned"})
    tree["models"]["pools"]["p"]["model"] = "Qwen/Qwen2.5-32B"
    cat = ModelCatalog.from_tree(tree)
    assert cat.upstream_model(cat.get("base")) == "Qwen/Qwen2.5-32B"
    assert cat.upstream_model(cat.get("tuned")) == "claims-lora"


def test_live_catalog_hot_reloads_and_rejects_invalid_change():
    store = config.ConfigStore(tree=_tree(a={"pool": "p"}))
    config.set_config(store)
    live = LiveCatalog()
    seen = []
    live.on_change(lambda c: seen.append(sorted(c.models)))
    assert "a" in live.current()
    store.replace(_tree(a={"pool": "p"}, b={"pool": "p", "tier": "large"}))
    assert "b" in live.current() and seen[-1] == ["a", "b"]
    store.replace(_tree(c={"pool": "nope"}))  # invalid: previous catalog stays in service
    assert "b" in live.current() and "c" not in live.current()


# ----------------------------------------------------------------------------- mock server


def test_mock_server_openai_endpoints():
    from aiready.models.mock_server import create_app

    c = TestClient(create_app())
    r = c.post("/v1/chat/completions", json={"model": "mock-mid", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["role"] == "assistant"
    with c.stream("POST", "/v1/chat/completions", json={"model": "mock-mid", "stream": True,
                                                         "messages": [{"role": "user", "content": "hi"}]}) as s:
        lines = [ln for ln in s.iter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]" and json.loads(lines[0][5:])["object"] == "chat.completion.chunk"
    e = c.post("/v1/embeddings", json={"model": "mock-embedding", "input": ["a", "b"]}).json()
    assert len(e["data"]) == 2 and len(e["data"][0]["embedding"]) == 384
    rr = c.post("/v1/rerank", json={"query": "claim", "documents": ["x", "claim form"], "top_n": 1}).json()
    assert rr["results"] == [{"index": 1, "relevance_score": rr["results"][0]["relevance_score"]}]
    assert any(m["id"] == "mock-mid" for m in c.get("/v1/models").json()["data"])


async def test_openai_backend_against_mock_server():
    """The real HTTP backend talking to the mock server: the air-gapped smoke test path."""
    from aiready.models.mock_server import create_app

    transport = httpx.ASGITransport(app=create_app())
    b = OpenAICompatibleBackend(base_url="http://mock/v1", transport=transport)
    r = await b.chat(req("hello", metadata={"feature": "x"}))
    assert r.choices[0].message.role == "assistant"
    chunks = [c async for c in b.chat_stream(req("hello"))]
    assert chunks and chunks[0]["object"] == "chat.completion.chunk"
    assert len((await b.embed(["a"], "mock-embedding"))[0]) == 384
    assert len(await b.rerank("claim", ["claim", "x"], "mock-reranker")) == 2
    assert (await b.health())["ok"]
