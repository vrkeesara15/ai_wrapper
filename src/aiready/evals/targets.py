"""What an evaluation runs against (plugin kind ``eval_target``).

``rag``      the RAG service (``connect("rag").answer``) - retrieval, rerank, generation and checks together
``gateway``  a chat model behind the gateway; case documents become numbered context passages
``agents``   an agent workflow (``connect("agents")``)
``http``     any HTTP endpoint (the client's own assistant API, a staging deployment...)
``callable`` a Python function ``module:fn`` (custom pipelines, tests)
``static``   recorded answers per case id (re-scoring an old run, tests)

Every target runs a case *as the case persona*: the principal carries the persona's roles and groups, so
access-control tests exercise the real filters instead of a privileged service account.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from aiready.core.errors import AIReadyError, GuardrailBlocked, PermissionDenied
from aiready.core.identity import Principal
from aiready.core.plugins import import_object, register
from aiready.evals.golden import EvalCase, EvalSuite
from aiready.evals.scorers import Passage

BLOCKED_ANSWER = "I cannot help with that request."


class TargetOutput(BaseModel):
    answer: str = ""
    passages: list[Passage] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    cited_chunks: list[str] = Field(default_factory=list)
    found: bool | None = None
    parsed: Any = None
    tool_calls: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    cost: float | None = None
    tokens: int | None = None
    model: str | None = None
    trace_id: str | None = None
    blocked: str | None = None  # error code when a guardrail / policy refused the request


class Target(Protocol):
    name: str

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput: ...


def blocked_output(exc: AIReadyError) -> TargetOutput:
    """A guardrail or policy refusal *is* an answer: for an attack it is the right one."""
    return TargetOutput(answer=f"{BLOCKED_ANSWER} ({exc.code})", found=False, blocked=exc.code)


async def timed(target: Target, case: EvalCase, principal: Principal) -> TargetOutput:
    t0 = time.perf_counter()
    try:
        out = await target.run(case, principal)
    except (GuardrailBlocked, PermissionDenied) as exc:
        out = blocked_output(exc)
    if out.latency_ms is None:
        out.latency_ms = (time.perf_counter() - t0) * 1000
    return out


def _passages_from_hits(hits: list[Any]) -> list[Passage]:
    out = []
    for i, h in enumerate(hits):
        ch = h.chunk if hasattr(h, "chunk") else (h.get("chunk", h) if isinstance(h, dict) else h)
        d = ch if isinstance(ch, dict) else {k: getattr(ch, k, None) for k in ("id", "doc_id", "text")}
        out.append(Passage(id=str(d.get("id") or ""), doc_id=str(d.get("doc_id") or ""), text=str(d.get("text") or ""),
                           marker=str(i + 1)))
    return out


@register("eval_target", "rag")
class RagTarget:
    name = "rag"

    def __init__(self, collection: str | None = None, mode: str = "check_then_send", feature: str = "evals",
                 model: str = "auto", top_k: int | None = None, top_n: int | None = None,
                 seed_collection: str | None = None):
        self.collection, self.mode, self.feature, self.model = collection, mode, feature, model
        self.top_k, self.top_n = top_k, top_n
        self.seed_collection = seed_collection

    async def prepare(self, suite: EvalSuite) -> None:
        """Index the suite's case documents (poisoned pages for indirect-injection tests) into a test collection."""
        if not self.seed_collection:
            return
        from aiready.core.types import Chunk
        from aiready.core.wiring import connect

        chunks = [Chunk(id=f"{d.id}#0", doc_id=d.id, text=d.text, title=d.title, source="evals",
                        access_tags=d.access_tags) for c in suite.cases for d in c.documents]
        if chunks:
            await connect("search").upsert(self.seed_collection, chunks)

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput:
        from aiready.core.interfaces import RagQuery
        from aiready.core.wiring import connect

        q = RagQuery(question=case.question, collection=self.collection or self.seed_collection, history=case.history,
                     mode=self.mode, feature=self.feature, model=self.model, top_k=self.top_k, top_n=self.top_n)
        ans = await connect("rag").answer(q, principal)
        return TargetOutput(
            answer=ans.answer, passages=_passages_from_hits(ans.passages),
            cited_sources=list(dict.fromkeys(c.doc_id for c in ans.citations)),
            cited_chunks=[c.chunk_id for c in ans.citations], found=ans.found, model=ans.model, trace_id=ans.trace_id,
        )


@register("eval_target", "gateway")
class GatewayTarget:
    name = "gateway"
    SYSTEM = ("Answer using only the numbered sources below and cite them like [1]. If the sources do not contain "
              "the answer, say you could not find it.")

    def __init__(self, model: str = "auto", system_prompt: str | None = None, temperature: float = 0.0,
                 max_tokens: int | None = 800, feature: str = "evals", tools: list[dict[str, Any]] | None = None):
        self.model, self.system = model, system_prompt or self.SYSTEM
        self.temperature, self.max_tokens, self.feature, self.tools = temperature, max_tokens, feature, tools

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput:
        from aiready.core.types import ChatMessage, ChatRequest, ToolSpec
        from aiready.core.wiring import connect

        msgs = [ChatMessage(role="system", content=self.system)]
        if case.documents:
            ctx = "\n".join(f"[{i + 1}] {d.title or d.id}\n{d.text}" for i, d in enumerate(case.documents))
            msgs.append(ChatMessage(role="system", content=f"Sources:\n{ctx}"))
        msgs += [ChatMessage(role=h.get("role", "user"), content=h.get("content", "")) for h in case.history]
        msgs.append(ChatMessage(role="user", content=case.question))
        req = ChatRequest(model=self.model, messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens,
                          tools=[ToolSpec(**t) for t in self.tools] if self.tools else None,
                          response_format={"type": "json_object"} if case.json_schema else None,
                          metadata={"feature": self.feature, "eval_case": case.id})
        resp = await connect("gateway").chat(req, principal)
        text = resp.text
        markers = set(re.findall(r"\[(\d+)\]", text))
        passages = [Passage(id=d.id, doc_id=d.id, text=d.text, marker=str(i + 1)) for i, d in enumerate(case.documents)]
        tool_calls = [tc.function.name for ch in resp.choices for tc in (ch.message.tool_calls or [])]
        cost = resp.aiready.get("cost") if isinstance(resp.aiready, dict) else None
        return TargetOutput(answer=text, passages=passages, cited_sources=[p.doc_id for p in passages if p.marker in markers],
                            tool_calls=tool_calls, cost=cost, tokens=resp.usage.total_tokens, model=resp.model,
                            trace_id=(resp.aiready or {}).get("trace_id"))


def _dig(data: Any, path: str | None) -> Any:
    if not path:
        return data
    for part in path.split("."):
        if isinstance(data, list) and part.isdigit():
            data = data[int(part)] if int(part) < len(data) else None
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            data = getattr(data, part, None)
        if data is None:
            return None
    return data


def _output_from_mapping(data: Any, *, answer_path: str = "answer", citations_path: str | None = "citations",
                         passages_path: str | None = "passages") -> TargetOutput:
    if isinstance(data, str):
        return TargetOutput(answer=data)
    answer = _dig(data, answer_path)
    if answer is None:
        for k in ("answer", "output", "text", "content", "result"):
            answer = _dig(data, k)
            if answer is not None:
                break
    cites = _dig(data, citations_path) or []
    cites = [c if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else {}) for c in cites]
    hits = _dig(data, passages_path) or []
    return TargetOutput(
        answer=answer if isinstance(answer, str) else json.dumps(answer, default=str),
        parsed=None if isinstance(answer, str) else answer, passages=_passages_from_hits(hits),
        cited_sources=list(dict.fromkeys(str(c["doc_id"]) for c in cites if c.get("doc_id"))),
        cited_chunks=[str(c["chunk_id"]) for c in cites if c.get("chunk_id")],
        found=_dig(data, "found"), tool_calls=[str(t) for t in (_dig(data, "tool_calls") or []) if isinstance(t, str)],
        cost=_dig(data, "cost"), trace_id=_dig(data, "trace_id"),
    )


@register("eval_target", "agents")
class AgentsTarget:
    """Runs an agent workflow: ``connect("agents").start_run(workflow, {"question": ...}, principal)`` and reads
    the answer from the returned run view (``result.answer``). Called duck-typed (``method``) so the runner does
    not depend on the runtime's internal types. A run that stops for a human approval is reported as such -
    the evaluation never approves a write on anyone's behalf."""

    name = "agents"

    def __init__(self, workflow: str | None = None, method: str = "start_run", input_key: str = "question",
                 answer_path: str = "result.answer"):
        self.workflow, self.method, self.input_key, self.answer_path = workflow, method, input_key, answer_path

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput:
        from aiready.core.wiring import connect

        svc = connect("agents")
        fn = getattr(svc, self.method, None)
        if fn is None:
            raise AIReadyError(f"agents service has no method '{self.method}'")
        res = fn(self.workflow, {self.input_key: case.question, "history": case.history}, principal)
        if inspect.isawaitable(res):
            res = await res
        data = res.model_dump() if hasattr(res, "model_dump") else res
        if isinstance(data, dict) and data.get("status") in ("waiting_approval", "waiting_child", "waiting_user"):
            return TargetOutput(answer=f"[run {data.get('id')} waiting for approval]", trace_id=data.get("trace_id"))
        out = _output_from_mapping(data, answer_path=self.answer_path)
        if not out.answer and isinstance(data, dict) and isinstance(data.get("result"), dict):
            out = _output_from_mapping(data["result"])
        out.trace_id = out.trace_id or (data.get("trace_id") if isinstance(data, dict) else None)
        return out


@register("eval_target", "http")
class HttpTarget:
    """POSTs a JSON body (``body_template`` with ``{question}`` placeholders) and reads the answer at
    ``answer_path``. The persona's token is forwarded, so the endpoint sees the persona as the signed-in user."""

    name = "http"

    def __init__(self, url: str, method: str = "POST", body_template: dict[str, Any] | None = None,
                 answer_path: str = "answer", citations_path: str | None = "citations",
                 passages_path: str | None = "passages", headers: dict[str, str] | None = None,
                 timeout: float = 60.0, forward_identity: bool = True, transport: Any = None):
        self.url, self.method = url, method
        self.body_template = body_template or {"question": "{question}"}
        self.answer_path, self.citations_path, self.passages_path = answer_path, citations_path, passages_path
        self.headers, self.timeout, self.forward_identity, self.transport = headers or {}, timeout, forward_identity, transport

    def _render(self, node: Any, case: EvalCase) -> Any:
        if isinstance(node, str):
            if node == "{history}":
                return case.history
            return node.replace("{question}", case.question).replace("{case_id}", case.id)
        if isinstance(node, dict):
            return {k: self._render(v, case) for k, v in node.items()}
        if isinstance(node, list):
            return [self._render(v, case) for v in node]
        return node

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput:
        import httpx

        headers = {**self.headers, **(principal.forward_headers() if self.forward_identity else {})}
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
            r = await c.request(self.method, self.url, json=self._render(self.body_template, case), headers=headers)
        if r.status_code in (403, 422):
            return TargetOutput(answer=f"{BLOCKED_ANSWER} (http {r.status_code})", found=False, blocked=str(r.status_code))
        r.raise_for_status()
        return _output_from_mapping(r.json(), answer_path=self.answer_path, citations_path=self.citations_path,
                                    passages_path=self.passages_path)


@register("eval_target", "callable")
class CallableTarget:
    name = "callable"

    def __init__(self, function: str | Callable[..., Any], name: str | None = None):
        self.fn = import_object(function) if isinstance(function, str) else function
        if name:
            self.name = name

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput:
        res = self.fn(case, principal)
        if inspect.isawaitable(res):
            res = await res
        if isinstance(res, TargetOutput):
            return res
        return _output_from_mapping(res.model_dump() if hasattr(res, "model_dump") else res)


@register("eval_target", "static")
class StaticTarget:
    name = "static"

    def __init__(self, answers: dict[str, Any]):
        self.answers = answers

    async def run(self, case: EvalCase, principal: Principal) -> TargetOutput:
        if case.id not in self.answers:
            raise AIReadyError(f"no recorded answer for case {case.id}")
        a = self.answers[case.id]
        return a if isinstance(a, TargetOutput) else _output_from_mapping(a)


def build_target(spec: dict[str, Any] | str | Target) -> Target:
    if not isinstance(spec, (dict, str)):
        return spec
    from aiready.core.plugins import create

    return create("eval_target", spec)
