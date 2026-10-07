"""Checkpoint store, run history and the live event bus.

* checkpoints: DocStore namespace ``agent_runs`` - the full ``RunState`` after every step;
* run history: DocStore namespace ``agent_events`` - every persisted event (``<run_id>/<seq>``): step inputs,
  outputs and decisions, kept for audit and replayed to late SSE subscribers;
* approval index: ``agent_run_approvals`` - approval id -> run id, so a decision finds the paused run;
* live bus: in-process fan-out of step *and* token events to SSE subscribers. Token events are deliberately
  not persisted (volume); a subscriber in another pod still sees every step through the history poll.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any

from aiready.agents.models import RunEvent, RunState
from aiready.core.errors import NotFound
from aiready.core.stores import DocStore, get_docstore

RUNS = "agent_runs"
EVENTS = "agent_events"
APPROVAL_INDEX = "agent_run_approvals"


class RunStore:
    def __init__(self, docstore: DocStore | None = None):
        self._docs = docstore

    @property
    def docs(self) -> DocStore:
        return self._docs or get_docstore()

    def save(self, run: RunState) -> None:
        run.updated_at = time.time()
        self.docs.put(RUNS, run.id, run.model_dump(mode="json"))

    def get(self, run_id: str) -> RunState:
        raw = self.docs.get(RUNS, run_id)
        if raw is None:
            raise NotFound(f"run {run_id} not found")
        return RunState.model_validate(raw)

    def find(self, run_id: str) -> RunState | None:
        raw = self.docs.get(RUNS, run_id)
        return RunState.model_validate(raw) if raw else None

    def list(self, *, owner: str | None = None, tenant: str | None = None, status: str | None = None,
             limit: int = 50) -> list[RunState]:
        out = []
        for raw in self.docs.values(RUNS):
            if owner and raw.get("owner") != owner:
                continue
            if tenant and raw.get("tenant") != tenant:
                continue
            if status and raw.get("status") != status:
                continue
            out.append(RunState.model_validate(raw))
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out[:limit]

    # ------------------------------------------------------------------ history
    def append_event(self, run: RunState, event: RunEvent) -> RunEvent:
        run.event_seq += 1
        event.seq = run.event_seq
        self.docs.put(EVENTS, f"{run.id}/{event.seq:08d}", event.model_dump(mode="json"))
        return event

    def events(self, run_id: str, after: int = 0) -> list[RunEvent]:
        evs = [RunEvent.model_validate(v) for v in self.docs.values(EVENTS, prefix=f"{run_id}/")]
        return [e for e in evs if (e.seq or 0) > after]

    # ------------------------------------------------------------------ approvals
    def index_approval(self, approval_id: str, run_id: str) -> None:
        self.docs.put(APPROVAL_INDEX, approval_id, {"run_id": run_id})

    def run_for_approval(self, approval_id: str) -> str | None:
        raw = self.docs.get(APPROVAL_INDEX, approval_id)
        return raw["run_id"] if raw else None


class EventBus:
    """In-process publish/subscribe per run id.

    Live-only events (tokens) are also kept in a bounded per-run backlog, so a subscriber that attaches a
    moment after a fast run started (the usual case for an HTTP stream) still receives every word.
    """

    def __init__(self, backlog_runs: int = 256, backlog_events: int = 5000) -> None:
        self._subs: dict[str, set[asyncio.Queue[RunEvent]]] = defaultdict(set)
        self._backlog: OrderedDict[str, deque[RunEvent]] = OrderedDict()
        self._backlog_runs = backlog_runs
        self._backlog_events = backlog_events

    def backlog(self, run_id: str) -> list[RunEvent]:
        return list(self._backlog.get(run_id, ()))

    def subscribe(self, run_id: str) -> asyncio.Queue[RunEvent]:
        q: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=10000)
        self._subs[run_id].add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[RunEvent]) -> None:
        self._subs[run_id].discard(q)
        if not self._subs[run_id]:
            self._subs.pop(run_id, None)

    def publish(self, event: RunEvent) -> None:
        if event.seq is None:
            buf = self._backlog.get(event.run_id)
            if buf is None:
                buf = self._backlog[event.run_id] = deque(maxlen=self._backlog_events)
                while len(self._backlog) > self._backlog_runs:
                    self._backlog.popitem(last=False)
            buf.append(event)
        for q in list(self._subs.get(event.run_id, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a stalled subscriber loses tokens, never blocks the run
                pass

    def subscribers(self, run_id: str) -> int:
        return len(self._subs.get(run_id, ()))


def clip(value: Any, limit: int) -> Any:
    """Bound what goes into run history: long strings are cut, structures are kept when small."""
    import json

    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    try:
        raw = json.dumps(value, default=str)
    except (TypeError, ValueError):
        raw = str(value)
    if len(raw) <= limit:
        return value
    return {"truncated": True, "preview": raw[:limit]}
