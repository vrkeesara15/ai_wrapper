"""Assistant service: the chat endpoint behind the embedded widget.

A chat turn starts a workflow run (``connect("agents")``) chosen per app and feature - by default the
retrieval chain, which answers through the RAG service with citations - and translates the run's event stream
into the small, stable event vocabulary the widget understands:

    run       {run_id, workflow}
    token     {text}                          streamed words of the answer
    step      {step, type, status}            progress only - never tool inputs/outputs
    approval  {run_id, approval_ids, tools}   an action waits for a person; the widget shows a notice
    waiting   {run_id, reason}                the run needs the user (e.g. sign-in expired)
    done      {run_id, status, answer, citations, error}
    error     {message}

Feedback (thumbs, edited answers) goes to the observability service when one is wired, else to the DocStore.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from aiready.assistant.config import AssistantConfig
from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import AIReadyError, ConfigError, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import get_docstore
from aiready.core.telemetry import span
from aiready.core.wiring import connect

log = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    app: str | None = None
    feature: str | None = None
    session_id: str | None = None
    workflow: str | None = None  # honoured only if listed for the app (features) - see workflow_for
    context: dict[str, Any] = Field(default_factory=dict)  # page context from the host (screen, record id)


class Feedback(BaseModel):
    run_id: str | None = None
    rating: Literal["up", "down"] | int | None = None
    comment: str | None = None
    original_answer: str | None = None
    edited_answer: str | None = None
    app: str | None = None
    feature: str | None = None
    message: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)


class AssistantService:
    def __init__(self, cfg: AssistantConfig | None = None):
        self._cfg = cfg

    @property
    def cfg(self) -> AssistantConfig:
        return self._cfg or get_config().section("assistant", AssistantConfig)

    # ------------------------------------------------------------------ routing
    def workflow_for(self, app: str | None, feature: str | None, requested: str | None = None) -> str:
        """Feature mapping wins, then the app default, then the global default. A workflow named by the
        client is accepted only if the app's configuration already maps some feature to it."""
        app_cfg = self.cfg.apps.get(app or "")
        if app_cfg:
            if requested and requested in app_cfg.features.values():
                return requested
            if feature and feature in app_cfg.features:
                return app_cfg.features[feature]
            if app_cfg.workflow:
                return app_cfg.workflow
        return self.cfg.default_workflow

    # ------------------------------------------------------------------ chat
    async def chat(self, turn: ChatTurn, principal: Principal) -> AsyncIterator[dict[str, Any]]:
        message = (turn.message or "").strip()
        if not message:
            raise ValidationFailed("message is empty")
        if len(message) > self.cfg.max_message_chars:
            raise ValidationFailed(f"message longer than {self.cfg.max_message_chars} characters")
        app = turn.app or principal.app
        workflow = self.workflow_for(app, turn.feature, turn.workflow)
        app_cfg = self.cfg.apps.get(app)
        history = [{"role": h.get("role"), "content": str(h.get("content", ""))[: self.cfg.max_message_chars]}
                   for h in turn.history[-self.cfg.max_history_turns:]
                   if isinstance(h, dict) and h.get("role") in ("user", "assistant")]
        run_input = {"message": message, "history": history, "app": app, "feature": turn.feature,
                     "context": turn.context}
        if app_cfg and app_cfg.collection:
            run_input["collection"] = app_cfg.collection
        agents = connect("agents")
        with span("assistant.chat", app=app, feature=turn.feature, workflow=workflow):
            view = await agents.start_run(workflow, run_input, principal, session_id=turn.session_id,
                                          feature=f"assistant.{turn.feature or 'chat'}")
        get_audit().for_principal(principal, "assistant.chat", resource=workflow, decision="started",
                                  run_id=view.id, feature=turn.feature, session_id=turn.session_id)
        yield {"event": "run", "data": {"run_id": view.id, "workflow": workflow, "session_id": turn.session_id}}
        async for item in self._relay(view.id, principal, after=0):
            yield item

    async def resume(self, run_id: str, principal: Principal, *, input: dict[str, Any] | None = None
                     ) -> AsyncIterator[dict[str, Any]]:
        agents = connect("agents")
        before = await agents.get_run(run_id, principal)
        await agents.resume_run(run_id, principal, input=input)
        yield {"event": "run", "data": {"run_id": run_id, "workflow": before.workflow, "resumed": True}}
        async for item in self._relay(run_id, principal, after=before.event_seq):
            yield item

    async def follow(self, run_id: str, principal: Principal, *, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Re-attach to a run (e.g. after an approval resumed it in the background)."""
        async for item in self._relay(run_id, principal, after=after):
            yield item

    async def _relay(self, run_id: str, principal: Principal, *, after: int) -> AsyncIterator[dict[str, Any]]:
        agents = connect("agents")
        finished = False
        try:
            async for ev in agents.stream_events(run_id, principal, after=after):
                out = self._translate(ev)
                if out is not None:
                    finished = finished or out["event"] in ("done", "approval", "waiting")
                    yield out
        except AIReadyError as exc:
            yield {"event": "error", "data": {"message": exc.message, "code": exc.code}}
            return
        if not finished:  # the stream ended without a terminal event: report where the run stands
            run = await agents.get_run(run_id, principal)
            if run.status == "waiting_approval":
                yield {"event": "approval", "data": {"run_id": run_id, **(run.pending or {})}}
            elif run.status == "waiting_user":
                yield {"event": "waiting", "data": {"run_id": run_id, "reason": (run.pending or {}).get("reason")}}
            else:
                yield {"event": "done", "data": _done(run.model_dump())}

    @staticmethod
    def _translate(ev: Any) -> dict[str, Any] | None:
        d = ev.data
        if ev.type == "token":
            return {"event": "token", "data": {"text": d.get("text", "")}}
        if ev.type == "step":
            return {"event": "step", "data": {"step": d.get("step"), "type": d.get("type"), "status": d.get("status")}}
        if ev.type == "approval":
            if d.get("status") == "waiting_approval":
                tools = [{"name": t.get("name"), "approval_id": t.get("approval_id")} for t in d.get("tools", [])]
                return {"event": "approval", "data": {"run_id": ev.run_id, "approval_ids": d.get("approval_ids", []),
                                                      "tools": tools,
                                                      "message": "This action needs approval before it runs."}}
            return {"event": "approval_decided", "data": {"run_id": ev.run_id, "approval_id": d.get("approval_id"),
                                                          "status": d.get("status")}}
        if ev.type == "run" and d.get("status") == "waiting_user":
            return {"event": "waiting", "data": {"run_id": ev.run_id, "reason": d.get("reason")}}
        if ev.type == "done":
            return {"event": "done", "data": _done({"id": ev.run_id, **d})}
        return None

    # ------------------------------------------------------------------ feedback
    async def feedback(self, fb: Feedback, principal: Principal) -> dict[str, Any]:
        """Ratings and edits captured in the widget feed the evaluation set ("Each failure becomes a test").

        Forwarded to the observability service (keyed by the run's trace id, so reviewers see the whole
        request) when it is wired; otherwise - or if it refuses - kept in the DocStore namespace ``feedback``.
        """
        trace_id = None
        if fb.run_id:
            try:
                trace_id = (await connect("agents").get_run(fb.run_id, principal)).trace_id
            except AIReadyError:
                trace_id = None  # unknown or foreign run: keep the feedback, without a trace link
        record = {"id": "fb_" + uuid.uuid4().hex, "ts": time.time(), "user": principal.sub,
                  "tenant": principal.tenant, "app": fb.app or principal.app, "trace_id": trace_id,
                  **fb.model_dump(exclude={"app"})}
        destination = "docstore"
        obs = _optional("observability") if trace_id else None
        if obs is not None and hasattr(obs, "record_feedback"):
            try:
                await obs.record_feedback(_observability_feedback(record), principal)
                destination = "observability"
            except Exception as exc:  # never lose a user's correction because the collector is down
                log.warning("observability did not accept feedback, keeping it locally: %s", exc)
        if destination == "docstore":
            get_docstore().put(self.cfg.feedback_namespace, record["id"], record)
        get_audit().for_principal(principal, "assistant.feedback", resource=fb.run_id, decision=str(fb.rating),
                                  feedback_id=record["id"], edited=fb.edited_answer is not None,
                                  destination=destination, trace_id=trace_id)
        return {"id": record["id"], "stored_in": destination}

def _done(run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result") or {}
    return {"run_id": run.get("id"), "status": run.get("status"), "answer": result.get("answer", ""),
            "citations": result.get("citations") or [], "error": run.get("error"),
            "stop_reason": run.get("stop_reason")}


def _observability_feedback(record: dict[str, Any]) -> Any:
    """Shape the record as the observability contract (``FeedbackIn``) when that package is installed."""
    rating = record.get("rating")
    payload = {"trace_id": record["trace_id"], "thumbs": rating if rating in ("up", "down") else None,
               "rating": rating if isinstance(rating, int) else None, "edit": record.get("edited_answer"),
               "comment": record.get("comment"), "app": record.get("app"), "feature": record.get("feature")}
    try:
        from aiready.observability.feedback import FeedbackIn
    except ImportError:
        return payload
    return FeedbackIn.model_validate(payload)


def _optional(name: str) -> Any | None:
    try:
        return connect(name)
    except (ConfigError, ImportError):
        return None
    except Exception:
        log.warning("optional service %s unavailable", name, exc_info=True)
        return None
