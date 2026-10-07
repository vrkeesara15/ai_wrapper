"""Step implementations. Each step does one thing and returns a ``StepOutcome``; the runtime saves the
checkpoint, records history and picks the next step.

What a step may call ("Agent runtime architecture"): the model client (``connect("gateway")``), the retrieval
client (``connect("rag")``), the tool executor (``connect("tools")``), the query service and memory - all
through wiring, all with the user's principal. Custom steps register with ``plugins.register("agent_step", name)``.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from aiready.agents.approvals import get_approval_store
from aiready.agents.workflows import StepDef, WorkflowDef, evaluate, render
from aiready.core import plugins
from aiready.core.errors import AIReadyError, ConfigError
from aiready.core.identity import Principal
from aiready.core.interfaces import QueryRequest, RagQuery
from aiready.core.tokens import count_messages, count_tokens
from aiready.core.types import ChatMessage, ChatRequest, ToolCall, ToolCallFunction, ToolSpec
from aiready.core.wiring import connect

if TYPE_CHECKING:  # pragma: no cover
    from aiready.agents.models import RunState
    from aiready.agents.runtime import AgentRuntime

log = logging.getLogger(__name__)


class StepFailed(AIReadyError):
    """A step could not complete. ``retryable`` steps are retried with backoff before the run stops."""

    status_code = 500
    code = "step_failed"

    def __init__(self, message: str, *, retryable: bool = False, details: dict | None = None):
        super().__init__(message, details=details)
        self.retryable = retryable


class StepOutcome(BaseModel):
    output: Any = None
    next: str | None = None  # overrides the step's static `next`
    wait: Literal["approval", "child", "user"] | None = None
    pending: dict[str, Any] | None = None
    end: bool = False
    status: str | None = None  # final status when `end` (default completed)
    tokens: int = 0
    tool_calls: int = 0
    decision: str | None = None
    input: Any = None  # what the step acted on, for run history


@dataclass
class StepContext:
    runtime: "AgentRuntime"
    run: "RunState"
    step: StepDef
    workflow: WorkflowDef
    principal: Principal

    @property
    def params(self) -> dict[str, Any]:
        return self.step.params

    def scope(self) -> dict[str, Any]:
        r = self.run
        return {"input": r.input, "steps": r.outputs, "vars": r.vars,
                "last": r.outputs.get(r.last_step) if r.last_step else None,
                "run": {"id": r.id, "workflow": r.workflow, "session_id": r.session_id, "app": r.app}}

    def render(self, template: Any) -> Any:
        return render(template, self.scope())

    async def emit_token(self, text: str) -> None:
        await self.runtime.emit(self.run, "token", {"step": self.step.id, "text": text}, persist=False)

    def profile(self) -> Any:
        name = self.workflow.agent or self.run.input.get("agent")
        return self.runtime.cfg.profiles.get(name) if name else None


def optional_service(name: str) -> Any | None:
    """A service that improves a step but is not required for it (memory, guardrails)."""
    try:
        return connect(name)
    except (ConfigError, ImportError):
        return None
    except Exception:  # an optional dependency that cannot even start must not fail the run
        log.warning("optional service %s unavailable", name, exc_info=True)
        return None


def _history(input_: dict[str, Any]) -> list[ChatMessage]:
    out = []
    for m in input_.get("history") or []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content"):
            out.append(ChatMessage(role=m["role"], content=str(m["content"])))
    return out[-20:]


def _usage_tokens(resp: Any, req: ChatRequest) -> int:
    total = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
    return total or count_messages(req.messages) + count_tokens(getattr(resp, "text", "") or "")


# =========================================================================== llm


@plugins.register("agent_step", "llm")
class LlmStep:
    """Model call through the gateway. Modes: plain prompt, conversation with tools (agent loop),
    classification (router), with optional retrieved context."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        p = ctx.params
        profile = ctx.profile()
        cfg = ctx.runtime.cfg
        model = p.get("model") or (profile.model if profile and profile.model else None) or cfg.default_model
        system = ctx.render(p.get("system")) or (profile.system_prompt if profile and profile.system_prompt
                                                 else None) or cfg.system_prompt
        prompt = ctx.render(p.get("prompt", "{{input.message}}")) or ""
        gateway = connect("gateway")

        if p.get("classify"):
            return await self._classify(ctx, gateway, model, prompt, p)

        if p.get("context_from"):
            src = ctx.run.outputs.get(p["context_from"]) or {}
            context = src.get("context") if isinstance(src, dict) else None
            if context:
                system = (f"{system}\n\nAnswer only from these sources and cite them with their [n] markers. "
                          f"If they do not contain the answer, say so.\n\n{context}")

        conversation = bool(p.get("conversation"))
        if conversation:
            if not ctx.run.messages:
                msgs = [ChatMessage(role="system", content=system)]
                msgs += await self._memory_context(ctx)
                msgs += _history(ctx.run.input) + [ChatMessage(role="user", content=prompt)]
                ctx.run.messages = [m.model_dump(exclude_none=True) for m in msgs]
            messages = [ChatMessage.model_validate(m) for m in ctx.run.messages]
        else:
            messages = [ChatMessage(role="system", content=system), *await self._memory_context(ctx),
                        *_history(ctx.run.input), ChatMessage(role="user", content=prompt)]

        tools = await self._tool_specs(ctx, p.get("tools") if "tools" in p else None, profile)
        req = ChatRequest(model=model, messages=messages, tools=tools or None,
                          temperature=p.get("temperature"), max_tokens=p.get("max_tokens"),
                          response_format=p.get("response_format"),
                          metadata={"feature": f"agent.{ctx.run.workflow}", "run_id": ctx.run.id,
                                    "step": ctx.step.id, "app": ctx.run.app})

        if p.get("stream") and not tools:
            text = ""
            async for chunk in gateway.chat_stream(req, ctx.principal):
                for choice in chunk.get("choices", []) if isinstance(chunk, dict) else []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        text += piece
                        await ctx.emit_token(piece)
            text = text.strip()
            tokens = count_messages(req.messages) + count_tokens(text)
            out = {"text": text, "tool_calls": [], "model": model}
            if conversation:
                ctx.run.messages.append({"role": "assistant", "content": text})
            return StepOutcome(output=out, tokens=tokens, input={"model": model, "prompt": prompt})

        resp = await gateway.chat(req, ctx.principal)
        msg = resp.choices[0].message if resp.choices else ChatMessage(role="assistant", content="")
        calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in msg.tool_calls or []]
        text = msg.content or ""
        if text and not calls:
            await ctx.emit_token(text)
        if conversation:
            ctx.run.messages.append(msg.model_dump(exclude_none=True))
        out = {"text": text, "tool_calls": calls, "model": resp.model}
        decision = f"tool_calls:{','.join(c['name'] for c in calls)}" if calls else "answer"
        return StepOutcome(output=out, tokens=_usage_tokens(resp, req), decision=decision,
                           input={"model": model, "prompt": prompt, "tools": [t.function["name"] for t in tools]})

    async def _classify(self, ctx: StepContext, gateway: Any, model: str, prompt: str, p: dict[str, Any]) -> StepOutcome:
        labels = p["classify"]
        labels = labels if isinstance(labels, dict) else {k: "" for k in labels}
        listing = "\n".join(f"- {k}: {d}" if d else f"- {k}" for k, d in labels.items())
        req = ChatRequest(model=model, temperature=0, max_tokens=20, messages=[
            ChatMessage(role="system", content="Classify the user's request into exactly one category. "
                                               f"Reply with the category name only.\nCategories:\n{listing}"),
            ChatMessage(role="user", content=prompt)],
            metadata={"feature": f"agent.{ctx.run.workflow}.classify", "run_id": ctx.run.id})
        resp = await gateway.chat(req, ctx.principal)
        answer = (resp.text or "").strip().lower()
        label = next((k for k in labels if answer == k.lower()), None) \
            or next((k for k in labels if k.lower() in answer), None) or p.get("default_label") or next(iter(labels))
        return StepOutcome(output={"label": label, "raw": resp.text}, tokens=_usage_tokens(resp, req),
                           decision=f"route:{label}", input={"prompt": prompt, "labels": list(labels)})

    async def _tool_specs(self, ctx: StepContext, patterns: Any, profile: Any) -> list[ToolSpec]:
        if patterns is None and profile is None:
            return []
        globs = patterns if patterns is not None else profile.tools
        if isinstance(globs, str):
            globs = [globs]
        if profile is not None and patterns is not None:
            allowed = profile.tools  # the profile is an allow-list the workflow cannot widen
        else:
            allowed = ["*"]
        if not globs:
            return []
        defs = await connect("tools").list_tools(ctx.principal)
        return [ToolSpec(function={"name": d.name, "description": d.description, "parameters": d.input_schema})
                for d in defs
                if any(fnmatch.fnmatchcase(d.name, g) for g in globs)
                and any(fnmatch.fnmatchcase(d.name, g) for g in allowed)]

    async def _memory_context(self, ctx: StepContext) -> list[ChatMessage]:
        if not (ctx.params.get("memory") and ctx.run.session_id):
            return []
        mem = optional_service("memory")
        if mem is None:
            return []
        try:
            turns = await mem.session_context(ctx.run.session_id, ctx.principal)
        except AIReadyError as exc:
            log.warning("memory unavailable for run %s: %s", ctx.run.id, exc)
            return []
        return [ChatMessage(role=t["role"], content=t.get("content", "")) for t in turns
                if t.get("role") in ("system", "user", "assistant")]


# =========================================================================== tool


@plugins.register("agent_step", "tool")
class ToolStep:
    """Act: call tools through the executor (policy, approval gate, audit happen there).

    ``from: tool_calls`` executes the calls of the last model message (agent loop); ``name`` + ``arguments``
    calls one fixed tool. A write returns an approval id; the step then pauses the run, and on resume calls the
    tool again with the approved id - completed calls are never repeated.
    """

    async def run(self, ctx: StepContext) -> StepOutcome:
        if ctx.params.get("name"):
            return await self._fixed(ctx)
        return await self._from_model(ctx)

    async def _call(self, ctx: StepContext, call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run or resume one call. Returns {status: done|error|pending|rejected, ...}."""
        state = ctx.run.vars.setdefault("_tool_calls", {})
        prior = state.get(call_id) or {}
        approval_id = prior.get("approval_id")
        if prior.get("status") in ("done", "error", "rejected"):
            return prior
        if approval_id:
            req = get_approval_store().get(approval_id)
            if req.status == "pending":
                return prior
            if req.status != "approved":
                rec = {"status": "rejected", "approval_id": approval_id, "name": name,
                       "error": f"the approver {req.status} this action"
                                + (f" ({req.decided_by}: {req.comment})" if req.comment else "")}
                state[call_id] = rec
                return rec
        limit = int(ctx.run.limits.get("max_tool_calls", 25))
        if ctx.run.tool_calls >= limit:
            raise StepFailed(f"tool call limit reached ({limit})")
        ctx.run.tool_calls += 1
        res = await connect("tools").call(name, args, ctx.principal, approval_id=approval_id,
                                          context={"run_id": ctx.run.id, "step": ctx.step.id})
        if res.approval_id and not res.ok:
            rec = {"status": "pending", "approval_id": res.approval_id, "name": name, "arguments": args}
            ctx.runtime.store.index_approval(res.approval_id, ctx.run.id)
        elif res.ok:
            rec = {"status": "done", "name": name, "content": res.content, "approval_id": approval_id}
        else:
            rec = {"status": "error", "name": name, "error": res.error, "approval_id": approval_id}
        state[call_id] = rec
        return rec

    def _wait(self, pending: list[dict[str, Any]], output: Any, inp: Any) -> StepOutcome:
        return StepOutcome(output=output, wait="approval", decision="waiting_approval", input=inp,
                           pending={"kind": "approval", "approval_ids": [p["approval_id"] for p in pending],
                                    "tools": [{"name": p["name"], "arguments": p.get("arguments"),
                                               "approval_id": p["approval_id"]} for p in pending]})

    async def _fixed(self, ctx: StepContext) -> StepOutcome:
        name = ctx.params["name"]
        args = ctx.render(ctx.params.get("arguments") or {})
        key = f"{ctx.step.id}:fixed"
        rec = await self._call(ctx, key, name, args)
        inp = {"tool": name, "arguments": args}
        if rec["status"] == "pending":
            return self._wait([rec], None, inp)
        ctx.run.vars["_tool_calls"].pop(key, None)  # a later loop iteration may call it again
        if rec["status"] == "rejected":
            on_reject = ctx.params.get("on_reject")
            if on_reject:
                return StepOutcome(output=rec, next=on_reject, decision="rejected", input=inp)
            return StepOutcome(output=rec, end=True, status="rejected", decision="rejected", input=inp)
        if rec["status"] == "error":
            raise StepFailed(f"tool {name} failed: {rec['error']}")
        return StepOutcome(output={"content": rec["content"], "tool": name}, decision="executed", tool_calls=1,
                           input=inp)

    async def _from_model(self, ctx: StepContext) -> StepOutcome:
        msgs = ctx.run.messages
        idx = next((i for i in range(len(msgs) - 1, -1, -1)
                    if msgs[i].get("role") == "assistant" and msgs[i].get("tool_calls")), None)
        if idx is None:
            return StepOutcome(output={"results": []}, decision="no_tool_calls")
        answered = {m.get("tool_call_id") for m in msgs[idx + 1:] if m.get("role") == "tool"}
        calls = [ToolCall.model_validate(c) for c in msgs[idx]["tool_calls"]]
        pending: list[dict[str, Any]] = []
        results = []
        limit = ctx.runtime.cfg.tool_result_chars
        for c in calls:
            if c.id in answered:
                continue
            try:
                args = json.loads(c.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                rec = {"status": "error", "name": c.function.name, "error": f"arguments are not valid JSON: {exc}"}
            else:
                rec = await self._call(ctx, c.id, c.function.name, args)
            if rec["status"] == "pending":
                pending.append(rec)
                continue
            results.append({"tool_call_id": c.id, **{k: v for k, v in rec.items() if k != "arguments"}})
            msgs.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name,
                         "content": _tool_message(rec, limit)})
        out = {"results": results}
        inp = {"calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls]}
        if pending:
            return self._wait(pending, out, inp)
        errors = sum(1 for r in results if r["status"] != "done")
        return StepOutcome(output=out, tool_calls=len(results), input=inp,
                           decision=f"executed:{len(results) - errors},failed:{errors}")


def _tool_message(rec: dict[str, Any], limit: int) -> str:
    if rec["status"] == "done":
        content = rec.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
        return text if len(text) <= limit else text[:limit] + "...[truncated]"
    if rec["status"] == "rejected":
        return f"NOT DONE: {rec['error']}. Tell the user the action was not carried out."
    return f"ERROR: {rec.get('error')}"


# =========================================================================== retrieve / query


@plugins.register("agent_step", "retrieve")
class RetrieveStep:
    """Retrieval through the RAG service: a grounded answer with citations, or numbered passages for an llm step."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        p = ctx.params
        question = ctx.render(p.get("question", "{{input.message}}")) or ""
        q = RagQuery(question=question, collection=ctx.render(p.get("collection")) or ctx.run.input.get("collection"),
                     history=[m.model_dump() for m in _history(ctx.run.input)], top_n=p.get("top_n"),
                     filters=ctx.render(p.get("filters") or {}), feature=ctx.run.feature or "agents",
                     session_id=ctx.run.session_id)
        rag = connect("rag")
        if p.get("answer", True):
            ans = await rag.answer(q, ctx.principal)
            citations = [c.model_dump() for c in ans.citations]
            if ans.answer:
                await ctx.emit_token(ans.answer)
            return StepOutcome(output={"text": ans.answer, "answer": ans.answer, "citations": citations,
                                       "found": ans.found, "grounded": ans.grounded, "model": ans.model},
                               decision="found" if ans.found else "not_found", input={"question": question},
                               tokens=count_tokens(ans.answer))
        hits = await rag.retrieve(q, ctx.principal)
        lines, citations = [], []
        for i, h in enumerate(hits, 1):
            c = h.chunk
            lines.append(f"[{i}] {c.title or c.source or c.doc_id}\n{c.text}")
            citations.append({"marker": f"[{i}]", "chunk_id": c.id, "doc_id": c.doc_id, "title": c.title,
                              "source_uri": c.source_uri, "quote": c.text[:200]})
        return StepOutcome(output={"context": "\n\n".join(lines), "citations": citations, "count": len(hits)},
                           decision=f"passages:{len(hits)}", input={"question": question})


@plugins.register("agent_step", "query")
class QueryStep:
    """Governed structured-data question through the query service (approved metric or generated SQL)."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        p = ctx.params
        req = QueryRequest(question=ctx.render(p.get("question", "{{input.message}}")), metric=p.get("metric"),
                           params=ctx.render(p.get("params") or {}), max_rows=int(p.get("max_rows", 200)))
        res = await connect("query").query(req, ctx.principal)
        out = res.model_dump()
        out["text"] = res.summary or f"{res.row_count} rows"
        return StepOutcome(output=out, decision=res.source, input=req.model_dump())


# =========================================================================== approval / branch / subagent / end


@plugins.register("agent_step", "approval")
class ApprovalStep:
    """An explicit human gate in a workflow graph (beyond the automatic gate on write tools)."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        store = get_approval_store()
        key = f"_approval:{ctx.step.id}"
        approval_id = ctx.run.vars.get(key)
        if approval_id is None:
            details = ctx.render(ctx.params.get("details") or {"input": "{{input.message}}"})
            req = store.create(tool=f"workflow:{ctx.run.workflow}.{ctx.step.id}",
                               effect=ctx.params.get("effect", "write"), arguments=details if isinstance(details, dict)
                               else {"details": details}, principal=ctx.principal, run_id=ctx.run.id,
                               step=ctx.step.id, reason=ctx.render(ctx.params.get("reason")))
            ctx.run.vars[key] = req.id
            ctx.runtime.store.index_approval(req.id, ctx.run.id)
            return StepOutcome(wait="approval", decision="waiting_approval",
                               pending={"kind": "approval", "approval_ids": [req.id],
                                        "tools": [{"name": req.tool, "approval_id": req.id}]})
        req = store.get(approval_id)
        if req.status == "pending":
            return StepOutcome(wait="approval", pending={"kind": "approval", "approval_ids": [req.id]})
        ctx.run.vars.pop(key, None)
        out = {"status": req.status, "decided_by": req.decided_by, "comment": req.comment}
        if req.status == "approved":
            return StepOutcome(output=out, decision="approved")
        if ctx.params.get("on_reject"):
            return StepOutcome(output=out, next=ctx.params["on_reject"], decision=req.status)
        return StepOutcome(output=out, end=True, status="rejected", decision=req.status)


@plugins.register("agent_step", "branch")
class BranchStep:
    async def run(self, ctx: StepContext) -> StepOutcome:
        scope = ctx.scope()
        for case in ctx.params.get("cases", []):
            if evaluate(case.get("when", True), scope):
                return StepOutcome(next=case["next"], decision=case["next"], output={"taken": case["next"]})
        default = ctx.params.get("default") or ctx.step.next
        if not default:
            raise StepFailed(f"branch {ctx.step.id}: no case matched and no default")
        return StepOutcome(next=default, decision=default, output={"taken": default})


@plugins.register("agent_step", "subagent")
class SubagentStep:
    """Start a child run for one sub-task and wait for it (the parent is resumed when the child finishes)."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        key = f"_child:{ctx.step.id}"
        child_id = ctx.run.vars.get(key)
        if child_id is None:
            child_input = ctx.render(ctx.params.get("input") or {"message": "{{input.message}}"})
            child = await ctx.runtime.start(ctx.params["workflow"], child_input, ctx.principal, parent=ctx.run,
                                            session_id=ctx.run.session_id)
            ctx.run.vars[key] = child.id
            ctx.run.child_run_ids.append(child.id)
            return StepOutcome(wait="child", pending={"kind": "child", "run_id": child.id},
                               decision=f"started:{child.id}", input={"workflow": ctx.params["workflow"],
                                                                      "input": child_input})
        child = ctx.runtime.store.get(child_id)
        if child.status not in ("completed", "failed", "stopped", "cancelled", "rejected"):
            return StepOutcome(wait="child", pending={"kind": "child", "run_id": child.id})
        ctx.run.vars.pop(key, None)
        ctx.run.tokens_used += child.tokens_used
        if child.status != "completed":
            raise StepFailed(f"sub-agent {child.workflow} {child.status}: {child.error or child.stop_reason or ''}")
        return StepOutcome(output=child.result or {}, decision="child_completed")


@plugins.register("agent_step", "end")
class EndStep:
    """Assemble the answer, optionally validate it (guardrails) and save the turn to session memory."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        p = ctx.params
        answer = ctx.render(p["output"]) if "output" in p else _default_answer(ctx)
        citations = ctx.render(p["citations"]) if "citations" in p else None
        data = ctx.render(p.get("data")) if "data" in p else None
        if not isinstance(answer, str):
            answer = "" if answer is None else json.dumps(answer, default=str)
        if p.get("validate") and answer:
            guard = optional_service("guardrails")
            if guard is not None:
                res = await guard.validate_output(answer, schema=p.get("schema"),
                                                  context={"run_id": ctx.run.id, "workflow": ctx.run.workflow})
                if not res.valid:
                    raise StepFailed("the answer failed output validation",
                                     details={"findings": [f.model_dump() for f in res.findings]})
                answer = res.text
        if ctx.run.session_id and p.get("remember_turn", True):
            mem = optional_service("memory")
            if mem is not None:
                try:
                    if ctx.run.input.get("message"):
                        await mem.append_turn(ctx.run.session_id, ctx.principal, "user", ctx.run.input["message"])
                    await mem.append_turn(ctx.run.session_id, ctx.principal, "assistant", answer)
                except AIReadyError as exc:
                    log.warning("could not save turn for run %s: %s", ctx.run.id, exc)
        result = {"answer": answer, "citations": citations or [], "data": data}
        return StepOutcome(output=result, end=True, decision="completed")


def _default_answer(ctx: StepContext) -> Any:
    last = ctx.run.outputs.get(ctx.run.last_step) if ctx.run.last_step else None
    if isinstance(last, dict):
        return last.get("answer") or last.get("text") or last.get("content")
    return last


def step_handler(step: StepDef) -> Any:
    spec = step.type if isinstance(step.type, dict) else {"type": step.type}
    return plugins.create("agent_step", spec)
