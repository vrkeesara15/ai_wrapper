"""Model backends: the adapters between the gateway and a model server.

Every backend speaks the gateway's OpenAI-shaped contract (``ChatRequest`` in, ``ChatResponse`` or
OpenAI-style stream chunks out). That is what makes "changing a model a routing change, not a code change":

* ``openai``    - any OpenAI-compatible server: vLLM, TGI, Ollama, llama.cpp, LiteLLM, a private vendor
                  endpoint in the client's tenancy, or a public API. Also embeddings and rerank (TEI / Cohere /
                  vLLM / Jina styles).
* ``anthropic`` - the Anthropic Messages API, mapped to and from the OpenAI shape (tools included). Used for
                  escalation / public routes, which stay closed until an egress policy opens them.
* ``mock``      - deterministic and offline, for tests, demos and air-gapped smoke tests.

Custom backends plug in with ``class: their_pkg.module:Backend`` in a pool replica spec.
AI-Ready-only fields (``metadata``) never leave the platform: they are stripped before forwarding.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

import httpx

from aiready.core.errors import UpstreamError
from aiready.core.plugins import create, register
from aiready.core.textutil import hash_embed, overlap_score
from aiready.core.tokens import count_messages, count_tokens
from aiready.core.types import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
    ToolCallFunction,
    Usage,
    new_id,
)

# Fields of ChatRequest that are AI-Ready extensions and must never reach an upstream server.
INTERNAL_FIELDS = {"metadata"}


class BackendError(UpstreamError):
    """An upstream model call failed. ``retryable`` tells the gateway whether a retry / fallback makes sense
    (timeouts, 429, 5xx) or whether the request itself is wrong (other 4xx)."""

    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None,
                 details: dict | None = None):
        super().__init__(message, details=details)
        self.retryable = retryable
        self.upstream_status = status


@runtime_checkable
class ModelBackend(Protocol):
    """What a pool replica must implement. ``model`` on the request is the *upstream* model id."""

    async def chat(self, req: ChatRequest) -> ChatResponse: ...

    def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]: ...

    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...

    async def rerank(self, query: str, docs: list[str], model: str) -> list[float]: ...

    async def health(self) -> dict[str, Any]: ...


def make_backend(spec: dict[str, Any] | str) -> ModelBackend:
    return create("model_backend", spec)


def upstream_payload(req: ChatRequest, *, stream: bool) -> dict[str, Any]:
    """OpenAI request body without AI-Ready extensions."""
    body = req.model_dump(exclude_none=True, exclude=INTERNAL_FIELDS)
    body["stream"] = stream
    # an empty tools list is rejected by several servers
    if not body.get("tools"):
        body.pop("tools", None)
        body.pop("tool_choice", None)
    return body


def _status_error(r: httpx.Response, what: str) -> BackendError:
    try:
        detail: Any = r.json()
    except ValueError:
        detail = r.text[:500]
    retryable = r.status_code in (408, 409, 425, 429) or r.status_code >= 500
    return BackendError(f"{what} returned HTTP {r.status_code}", retryable=retryable, status=r.status_code,
                        details={"upstream": detail})


async def _iter_sse(resp: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """Yield (event, data) pairs from a server-sent-event stream."""
    event: str | None = None
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)


# =========================================================================== OpenAI-compatible


@register("model_backend", "openai")
class OpenAICompatibleBackend:
    """Any server that speaks the OpenAI chat / embeddings API.

    ``base_url`` follows the OpenAI SDK convention and includes the version prefix, e.g.
    ``http://vllm-mid:8000/v1``. Rerank goes to ``{base_url}/rerank`` (vLLM, Jina, Cohere-v1 style); if that
    returns 404 the TEI style ``{server root}/rerank`` with ``texts`` is tried. ``rerank_url`` pins either one.
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 headers: dict[str, str] | None = None, timeout: float = 60.0, connect_timeout: float = 5.0,
                 model: str | None = None, rerank_url: str | None = None,
                 rerank_style: str = "auto", stream_usage: bool = False, verify_tls: bool | str = True,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.rerank_url = rerank_url
        self.rerank_style = rerank_style
        self.stream_usage = stream_usage
        h = {"content-type": "application/json", **(headers or {})}
        if api_key:
            h["authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=connect_timeout), headers=h,
                                         verify=verify_tls, transport=transport)

    def _payload(self, req: ChatRequest, stream: bool) -> dict[str, Any]:
        body = upstream_payload(req, stream=stream)
        if self.model and not req.model:
            body["model"] = self.model
        if stream and self.stream_usage:
            body["stream_options"] = {"include_usage": True}
        return body

    async def _post(self, url: str, body: dict[str, Any], what: str) -> Any:
        try:
            r = await self._client.post(url, json=body)
        except httpx.TimeoutException as exc:
            raise BackendError(f"{what} timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise BackendError(f"{what} unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise _status_error(r, what)
        return r.json()

    async def chat(self, req: ChatRequest) -> ChatResponse:
        data = await self._post(f"{self.base_url}/chat/completions", self._payload(req, False), "chat")
        data.setdefault("model", req.model)
        return ChatResponse.model_validate({k: v for k, v in data.items() if k in ChatResponse.model_fields})

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        try:
            async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                           json=self._payload(req, True)) as r:
                if r.status_code >= 400:
                    await r.aread()
                    raise _status_error(r, "chat stream")
                async for _, data in _iter_sse(r):
                    if data.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except ValueError:
                        continue
        except httpx.TimeoutException as exc:
            raise BackendError(f"chat stream timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise BackendError(f"chat stream failed: {exc}") from exc

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = await self._post(f"{self.base_url}/embeddings", {"model": model or self.model, "input": texts},
                                "embeddings")
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [list(d["embedding"]) for d in items]

    def _server_root(self) -> str:
        return re.sub(r"/v\d+$", "", self.base_url)

    async def rerank(self, query: str, docs: list[str], model: str) -> list[float]:
        model = model or self.model or ""
        if self.rerank_url:
            # versioned paths (/v1/rerank) are the Cohere/vLLM/Jina body; a bare /rerank is TEI
            style = self.rerank_style if self.rerank_style != "auto" else (
                "cohere" if re.search(r"/v\d+/", self.rerank_url) else "tei")
            return await self._rerank_at(self.rerank_url, style, query, docs, model)
        if self.rerank_style == "tei":
            return await self._rerank_at(f"{self._server_root()}/rerank", "tei", query, docs, model)
        try:
            return await self._rerank_at(f"{self.base_url}/rerank", "cohere", query, docs, model)
        except BackendError as exc:
            if self.rerank_style == "auto" and exc.upstream_status in (404, 405):
                return await self._rerank_at(f"{self._server_root()}/rerank", "tei", query, docs, model)
            raise

    async def _rerank_at(self, url: str, style: str, query: str, docs: list[str], model: str) -> list[float]:
        if style == "tei":
            data = await self._post(url, {"query": query, "texts": docs, "raw_scores": False}, "rerank")
        else:
            data = await self._post(url, {"model": model, "query": query, "documents": docs,
                                          "top_n": len(docs)}, "rerank")
        items = data.get("results", data) if isinstance(data, dict) else data
        scores = [0.0] * len(docs)
        for it in items or []:
            idx = int(it.get("index", 0))
            if 0 <= idx < len(docs):
                scores[idx] = float(it.get("relevance_score", it.get("score", 0.0)))
        return scores

    async def health(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            r = await self._client.get(f"{self.base_url}/models")
            return {"ok": r.status_code < 400, "status": r.status_code,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc)}

    async def aclose(self) -> None:
        await self._client.aclose()


# =========================================================================== Anthropic Messages API

_STOP_MAP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
             "pause_turn": "stop", "refusal": "content_filter"}


def openai_to_anthropic(req: ChatRequest, *, default_max_tokens: int = 1024) -> dict[str, Any]:
    """Map an OpenAI-shaped request to an Anthropic Messages API body."""
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        # the Messages API requires alternating roles: merge consecutive same-role turns
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in req.messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
        elif m.role == "user":
            push("user", [{"type": "text", "text": m.content or ""}])
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except ValueError:
                    args = {"_raw": tc.function.arguments}
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
            push("assistant", blocks or [{"type": "text", "text": ""}])
        elif m.role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.content or ""}])

    body: dict[str, Any] = {"model": req.model, "messages": messages,
                            "max_tokens": req.max_tokens or default_max_tokens}
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else list(req.stop)
    if req.tools:
        body["tools"] = [{"name": t.function["name"], "description": t.function.get("description", ""),
                          "input_schema": t.function.get("parameters") or {"type": "object", "properties": {}}}
                         for t in req.tools]
        tc = req.tool_choice
        if tc == "required":
            body["tool_choice"] = {"type": "any"}
        elif tc == "none":
            body["tool_choice"] = {"type": "none"}
        elif isinstance(tc, dict) and tc.get("function", {}).get("name"):
            body["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}
        elif tc == "auto":
            body["tool_choice"] = {"type": "auto"}
    if req.user:
        body["metadata"] = {"user_id": req.user}
    return body


def anthropic_to_openai(data: dict[str, Any], model: str) -> ChatResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(ToolCall(id=block.get("id") or new_id("call_"), function=ToolCallFunction(
                name=block["name"], arguments=json.dumps(block.get("input", {})))))
    usage = data.get("usage", {})
    pt, ct = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    msg = ChatMessage(role="assistant", content="".join(text_parts), tool_calls=tool_calls or None)
    return ChatResponse(id=data.get("id") or new_id("chatcmpl-"), model=data.get("model", model),
                        choices=[ChatChoice(message=msg, finish_reason=_STOP_MAP.get(data.get("stop_reason") or "",
                                                                                     "stop"))],
                        usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct))


@register("model_backend", "anthropic")
class AnthropicBackend:
    """Anthropic Messages API over httpx, exposed in the OpenAI shape.

    Used for escalation / public routes. Public routes are closed by default: the catalog gives public
    models no allowed data classes, and the gateway also asks the policy engine for ``egress.send``.
    """

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.anthropic.com",
                 version: str = "2023-06-01", timeout: float = 120.0, connect_timeout: float = 5.0,
                 model: str | None = None, default_max_tokens: int = 1024, headers: dict[str, str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_max_tokens = default_max_tokens
        h = {"content-type": "application/json", "anthropic-version": version, **(headers or {})}
        if api_key:
            h["x-api-key"] = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=connect_timeout), headers=h,
                                         transport=transport)

    def _body(self, req: ChatRequest, stream: bool) -> dict[str, Any]:
        r = req if req.model or not self.model else req.model_copy(update={"model": self.model})
        body = openai_to_anthropic(r, default_max_tokens=self.default_max_tokens)
        if stream:
            body["stream"] = True
        return body

    async def chat(self, req: ChatRequest) -> ChatResponse:
        try:
            r = await self._client.post(f"{self.base_url}/v1/messages", json=self._body(req, False))
        except httpx.TimeoutException as exc:
            raise BackendError(f"anthropic timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise BackendError(f"anthropic unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise _status_error(r, "anthropic messages")
        return anthropic_to_openai(r.json(), req.model)

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        cid, created = new_id("chatcmpl-"), int(time.time())
        tool_index: dict[int, int] = {}  # anthropic block index -> openai tool_calls index

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

        prompt_tokens = 0
        try:
            async with self._client.stream("POST", f"{self.base_url}/v1/messages", json=self._body(req, True)) as r:
                if r.status_code >= 400:
                    await r.aread()
                    raise _status_error(r, "anthropic stream")
                async for event, data in _iter_sse(r):
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    kind = ev.get("type") or event
                    if kind == "message_start":
                        prompt_tokens = int(ev.get("message", {}).get("usage", {}).get("input_tokens", 0))
                        yield chunk({"role": "assistant", "content": ""})
                    elif kind == "content_block_start":
                        block = ev.get("content_block", {})
                        if block.get("type") == "tool_use":
                            i = tool_index.setdefault(ev.get("index", 0), len(tool_index))
                            yield chunk({"tool_calls": [{"index": i, "id": block.get("id"), "type": "function",
                                                         "function": {"name": block.get("name"), "arguments": ""}}]})
                    elif kind == "content_block_delta":
                        d = ev.get("delta", {})
                        if d.get("type") == "text_delta":
                            yield chunk({"content": d.get("text", "")})
                        elif d.get("type") == "input_json_delta":
                            i = tool_index.get(ev.get("index", 0), 0)
                            yield chunk({"tool_calls": [{"index": i, "function": {
                                "arguments": d.get("partial_json", "")}}]})
                    elif kind == "message_delta":
                        out = int(ev.get("usage", {}).get("output_tokens", 0))
                        finish = _STOP_MAP.get(ev.get("delta", {}).get("stop_reason") or "", "stop")
                        yield chunk({}, finish, usage={"prompt_tokens": prompt_tokens, "completion_tokens": out,
                                                       "total_tokens": prompt_tokens + out})
                    elif kind == "error":
                        raise BackendError(f"anthropic stream error: {ev.get('error')}")
                    elif kind == "message_stop":
                        break
        except httpx.TimeoutException as exc:
            raise BackendError(f"anthropic stream timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise BackendError(f"anthropic stream failed: {exc}") from exc

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise BackendError("the Anthropic API has no embeddings endpoint; route embeddings to an embedding pool",
                           retryable=False)

    async def rerank(self, query: str, docs: list[str], model: str) -> list[float]:
        raise BackendError("the Anthropic API has no rerank endpoint; route rerank to a reranker pool",
                           retryable=False)

    async def health(self) -> dict[str, Any]:
        # a real probe would spend tokens; report configuration-level readiness only
        return {"ok": "x-api-key" in self._client.headers, "probe": "config"}

    async def aclose(self) -> None:
        await self._client.aclose()


# =========================================================================== mock

_TOOL_KEYWORD = re.compile(r"\[\[tool(?::(?P<name>[\w.\-]+))?\s*(?P<args>\{.*?\})?\s*\]\]", re.DOTALL)


def example_from_schema(schema: dict[str, Any] | None, answer: str) -> Any:
    """A minimal value satisfying a JSON schema, so the mock honours ``response_format``."""
    schema = schema or {"type": "object"}
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    t = schema.get("type", "object")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t == "object":
        props = schema.get("properties", {})
        out = {k: example_from_schema(v, answer) for k, v in props.items() if k in schema.get("required", props)}
        if not props:
            out = {"answer": answer}
        return out
    if t == "array":
        return [example_from_schema(schema.get("items", {"type": "string"}), answer)] * max(1, schema.get("minItems", 1))
    if t == "string":
        return answer
    if t == "integer":
        return int(schema.get("minimum", 0))
    if t == "number":
        return float(schema.get("minimum", 0))
    if t == "boolean":
        return True
    return None


@register("model_backend", "mock")
class MockBackend:
    """Deterministic, offline model.

    * chat: extractive answer from ``[n]``-marked passages in the prompt (the same logic as
      ``aiready.testing.FakeGateway``), JSON when ``response_format`` asks for it;
    * tool calls: a user message containing ``[[tool:NAME {"arg": 1}]]`` (or ``[[tool]]`` for the first tool
      offered) makes the model call that tool; after a tool result it answers with the result;
    * embeddings: feature-hashing vectors; rerank: word overlap;
    * ``latency_ms`` simulates model time, ``fail: true`` makes every call fail (fallback drills).
    """

    def __init__(self, latency_ms: float = 0.0, dim: int = 384, fail: bool = False, model: str = "mock",
                 fail_message: str = "mock backend configured to fail"):
        self.latency = latency_ms / 1000.0
        self.dim = dim
        self.fail = fail
        self.model = model
        self.fail_message = fail_message
        self.calls = 0

    async def _work(self) -> None:
        self.calls += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.fail:
            raise BackendError(self.fail_message)

    def _reply(self, req: ChatRequest) -> tuple[str, list[ToolCall] | None]:
        last = req.messages[-1] if req.messages else None
        if last is not None and last.role == "tool":
            return f"The tool returned: {last.content}", None
        text = req.last_user_text()
        m = _TOOL_KEYWORD.search(text)
        if m and req.tools:
            names = [t.function.get("name") for t in req.tools]
            name = m.group("name") or names[0]
            if name in names:
                args = m.group("args") or "{}"
                return "", [ToolCall(function=ToolCallFunction(name=name, arguments=args))]
        from aiready.testing import extractive_answer

        answer = extractive_answer(req)
        rf = req.response_format or {}
        if rf.get("type") in ("json_object", "json_schema"):
            schema = (rf.get("json_schema") or {}).get("schema") if rf.get("type") == "json_schema" else None
            return json.dumps(example_from_schema(schema, answer)), None
        return answer, None

    async def chat(self, req: ChatRequest) -> ChatResponse:
        await self._work()
        text, tool_calls = self._reply(req)
        pt = count_messages(req.messages)
        ct = count_tokens(text) if text else 8
        msg = ChatMessage(role="assistant", content=text, tool_calls=tool_calls)
        return ChatResponse(model=req.model or self.model,
                            choices=[ChatChoice(message=msg, finish_reason="tool_calls" if tool_calls else "stop")],
                            usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct))

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        resp = await self.chat(req)
        cid, model = resp.id, resp.model

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {"id": cid, "object": "chat.completion.chunk", "created": resp.created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        yield chunk({"role": "assistant", "content": ""})
        msg = resp.choices[0].message
        if msg.tool_calls:
            for i, tc in enumerate(msg.tool_calls):
                yield chunk({"tool_calls": [{"index": i, "id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
        else:
            words = (msg.content or "").split(" ")
            for i, w in enumerate(words):
                if self.latency:
                    await asyncio.sleep(self.latency / max(1, len(words)))
                yield chunk({"content": w if i == len(words) - 1 else w + " "})
        yield chunk({}, resp.choices[0].finish_reason)
        yield {"id": cid, "object": "chat.completion.chunk", "created": resp.created, "model": model,
               "choices": [], "usage": resp.usage.model_dump()}

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        await self._work()
        return [hash_embed(t, self.dim) for t in texts]

    async def rerank(self, query: str, docs: list[str], model: str) -> list[float]:
        await self._work()
        return [overlap_score(query, d) for d in docs]

    async def health(self) -> dict[str, Any]:
        return {"ok": not self.fail, "backend": "mock"}
