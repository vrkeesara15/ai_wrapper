"""Run state and event models shared by the runtime, the API and the HTTP client."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

RunStatus = Literal["queued", "running", "waiting_approval", "waiting_child", "waiting_user", "completed",
                    "rejected", "failed", "cancelled", "stopped"]
TERMINAL: frozenset[str] = frozenset({"completed", "rejected", "failed", "cancelled", "stopped"})
WAITING: frozenset[str] = frozenset({"waiting_approval", "waiting_child", "waiting_user"})
# a stream stays open while the run is moving on its own (a child run counts: it will come back)
STREAMING: frozenset[str] = frozenset({"queued", "running", "waiting_child"})


class RunState(BaseModel):
    """The checkpoint. Saved after every step, so any worker can resume the run from here."""

    id: str = Field(default_factory=lambda: "run_" + uuid.uuid4().hex)
    workflow: str
    version: str
    definition: dict[str, Any]  # snapshot: a run finishes on the version it started with
    status: RunStatus = "queued"
    input: dict[str, Any] = Field(default_factory=dict)
    owner: str
    tenant: str = "default"
    app: str = "default"
    feature: str | None = None
    token: str | None = None  # the user's own short-lived token; never returned by the API
    session_id: str | None = None
    trace_id: str | None = None
    parent_run_id: str | None = None
    depth: int = 0
    child_run_ids: list[str] = Field(default_factory=list)
    current_step: str | None = None
    last_step: str | None = None
    step_count: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    active_seconds: float = 0.0
    limits: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)  # conversation for llm/tool steps
    outputs: dict[str, Any] = Field(default_factory=dict)  # step id -> output
    vars: dict[str, Any] = Field(default_factory=dict)
    pending: dict[str, Any] | None = None  # what the run waits for (approval ids, child run id, ...)
    attempts: dict[str, int] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    stop_reason: str | None = None
    event_seq: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    finished_at: float | None = None


class RunView(BaseModel):
    """What callers see of a run (no token, no workflow snapshot)."""

    id: str
    workflow: str
    version: str
    status: RunStatus
    owner: str
    app: str = "default"
    session_id: str | None = None
    parent_run_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)
    current_step: str | None = None
    step_count: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    pending: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    stop_reason: str | None = None
    trace_id: str | None = None
    event_seq: int = 0  # last persisted event: resume a stream after it without replaying history
    created_at: float
    updated_at: float
    finished_at: float | None = None

    @classmethod
    def of(cls, run: RunState) -> "RunView":
        return cls.model_validate(run.model_dump(include=set(cls.model_fields)))


class RunEvent(BaseModel):
    """One entry of the event stream. Step events are persisted (run history); token events are not."""

    run_id: str
    seq: int | None = None
    type: Literal["run", "step", "token", "approval", "done", "error"]
    after_seq: int | None = None  # live-only events: the persisted event they follow (ordering on replay)
    ts: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
