"""Test doubles for building and testing one service without the others.

    from aiready import testing, core
    core.wiring.override("gateway", testing.FakeGateway())
    core.wiring.override("policy", testing.AllowAllPolicy())
    core.wiring.override("guardrails", testing.PassThroughGuardrails())
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from aiready.core.identity import Principal, get_token_service
from aiready.core.interfaces import PolicyDecision, PolicyInput, ScreenResult, ValidationResult
from aiready.core.textutil import hash_embed, overlap_score, split_sentences
from aiready.core.types import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    RerankResponse,
    RerankResult,
    ToolCall,
    ToolCallFunction,
    Usage,
)


def make_principal(sub: str = "alice", *, roles: list[str] | None = None, groups: list[str] | None = None,
                   tenant: str = "acme", app: str = "claims") -> Principal:
    ts = get_token_service()
    return ts.verify(ts.issue(sub=sub, tenant=tenant, app=app, roles=roles or ["user"], groups=groups or []))


class FakeGateway:
    """Deterministic gateway. Chat answers extractively from context in the prompt; scripted replies and tool
    calls can be queued with ``script``. Embeddings are hash embeddings; rerank uses word overlap."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.calls: list[ChatRequest] = []
        self._script: list[dict[str, Any]] = []

    def script(self, *, content: str | None = None, tool: str | None = None, arguments: dict | None = None) -> None:
        self._script.append({"content": content, "tool": tool, "arguments": arguments or {}})

    async def chat(self, req: ChatRequest, principal: Principal) -> ChatResponse:
        self.calls.append(req)
        if self._script:
            s = self._script.pop(0)
            if s["tool"]:
                msg = ChatMessage(role="assistant", content="", tool_calls=[
                    ToolCall(function=ToolCallFunction(name=s["tool"], arguments=json.dumps(s["arguments"])))])
                return ChatResponse(model="fake", choices=[ChatChoice(message=msg, finish_reason="tool_calls")])
            text = s["content"] or ""
        else:
            text = extractive_answer(req)
        return ChatResponse(model="fake", choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
                            usage=Usage(prompt_tokens=10, completion_tokens=len(text) // 4, total_tokens=10 + len(text) // 4))

    async def chat_stream(self, req: ChatRequest, principal: Principal) -> AsyncIterator[dict[str, Any]]:
        resp = await self.chat(req, principal)
        for word in resp.text.split(" "):
            yield {"choices": [{"index": 0, "delta": {"content": word + " "}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    async def embed(self, texts: list[str], principal: Principal, *, model: str = "embedding",
                    priority: str = "live") -> list[list[float]]:
        return [hash_embed(t, self.dim) for t in texts]

    async def rerank(self, query: str, documents: list[str], principal: Principal, *, top_n: int | None = None,
                     model: str = "reranker") -> RerankResponse:
        scored = sorted(((i, overlap_score(query, d)) for i, d in enumerate(documents)), key=lambda x: -x[1])
        if top_n:
            scored = scored[:top_n]
        return RerankResponse(model="fake-reranker", results=[RerankResult(index=i, relevance_score=s) for i, s in scored])


def extractive_answer(req: ChatRequest) -> str:
    """Pick the context sentences that best match the question, keeping [n] citation markers."""
    question = req.last_user_text()
    last_user = max((i for i, m in enumerate(req.messages) if m.role == "user"), default=-1)
    context = "\n".join(m.content or "" for i, m in enumerate(req.messages)
                        if m.role in ("system", "user") and i != last_user)
    import re

    passages = re.findall(r"\[(\d+)\][^\n]*\n?(.*?)(?=\n\[\d+\]|\Z)", context, flags=re.S)
    best: list[tuple[float, str]] = []
    for marker, body in passages:
        for sent in split_sentences(body):
            s = overlap_score(question, sent)
            if s > 0:
                best.append((s, f"{sent} [{marker}]"))
    if not best:
        return "I could not find this in the available sources."
    best.sort(key=lambda x: -x[0])
    return " ".join(s for _, s in best[:2])


class AllowAllPolicy:
    def __init__(self) -> None:
        self.inputs: list[PolicyInput] = []

    async def decide(self, inp: PolicyInput) -> PolicyDecision:
        self.inputs.append(inp)
        return PolicyDecision(allow=True, effect="allow", reasons=["test: allow all"])


class DenyPolicy:
    def __init__(self, actions: set[str]):
        self.actions = actions

    async def decide(self, inp: PolicyInput) -> PolicyDecision:
        if inp.action in self.actions:
            return PolicyDecision(allow=False, effect="deny", reasons=[f"test: deny {inp.action}"])
        return PolicyDecision(allow=True)


class PassThroughGuardrails:
    async def screen_input(self, text: str, *, principal: Principal | None = None,
                           context: dict[str, Any] | None = None) -> ScreenResult:
        return ScreenResult(allowed=True, text=text)

    async def validate_output(self, text: str, *, schema: dict[str, Any] | None = None,
                              context: dict[str, Any] | None = None) -> ValidationResult:
        parsed = None
        if schema:
            try:
                parsed = json.loads(text)
            except ValueError:
                return ValidationResult(valid=False, text=text)
        return ValidationResult(valid=True, text=text, parsed=parsed)

    async def mask(self, text: str) -> ScreenResult:
        return ScreenResult(allowed=True, text=text)
