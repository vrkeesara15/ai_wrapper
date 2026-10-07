"""Agent runtime: stateless workers executing versioned step graphs ("Agent runtime architecture").

    start -> run queue -> worker: [check limits -> run one step -> save checkpoint -> record history
                                   -> pick next step] ... -> completed | waiting_* | failed | stopped

* **Stateless workers.** All run state is in the checkpoint (DocStore). A worker holds a short lease on a run
  while it executes; if the worker dies the lease expires and ``recover_stale`` re-queues the run, which any
  worker resumes from its last checkpoint.
* **Limits.** Steps, active time and tokens per run (``agents.limits``; a workflow may only lower them).
  Crossing one stops the run with status ``stopped`` and a clear ``stop_reason``.
* **Errors.** Transient failures are retried with exponential backoff; then the run goes to the step's
  ``on_error`` target or stops with status ``failed`` and the reason.
* **Approval gate.** A tool step whose call returns an approval id pauses the run (``waiting_approval``); the
  approval inbox's decision re-queues it. Nothing that writes runs before a person says yes.
* **Identity.** A run acts as the user who started it, with their own short-lived token (kept with the
  checkpoint). If it has expired when the run resumes, the run waits for the user (``waiting_user``) instead
  of falling back to any service identity.
* **Audit.** Every step's input, output and decision go to run history; start/finish/approval waits go to the
  audit log; model and tool calls are audited by the gateway and the tool executor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from aiready.agents.approvals import ApprovalRequest, get_approval_store
from aiready.agents.config import AgentsConfig
from aiready.agents.models import STREAMING, TERMINAL, RunEvent, RunState
from aiready.agents.steps import StepContext, StepFailed, StepOutcome, step_handler
from aiready.agents.store import EventBus, RunStore, clip
from aiready.agents.workflows import WorkflowDef, WorkflowRegistry
from aiready.core.audit import get_audit
from aiready.core.errors import (
    AIReadyError,
    AuthenticationError,
    NotFound,
    PermissionDenied,
    PolicyViolation,
    UpstreamError,
    ValidationFailed,
)
from aiready.core.identity import Principal, get_token_service
from aiready.core.interfaces import PolicyInput
from aiready.core.queues import Queue, make_queue
from aiready.core.stores import KVStore, make_kv
from aiready.core.telemetry import current_trace_id, metrics, new_trace_id, set_trace_id, span
from aiready.core.wiring import connect

log = logging.getLogger(__name__)
SERVICE = "agents"
NON_RETRYABLE = (PermissionDenied, ValidationFailed, NotFound, AuthenticationError)


_PROCESS_LEASES: KVStore | None = None


def _lease_store(spec: dict[str, Any]) -> KVStore:
    """Leases must be shared by every runtime that can see the same runs. ``memory`` therefore means one
    store per *process* (not per runtime instance); use ``redis`` when workers run in several pods."""
    global _PROCESS_LEASES
    if (spec or {}).get("type", "memory") == "memory" and "class" not in (spec or {}):
        if _PROCESS_LEASES is None:
            _PROCESS_LEASES = make_kv({"type": "memory"})
        return _PROCESS_LEASES
    return make_kv(spec)


class AgentRuntime:
    def __init__(self, cfg: AgentsConfig | None = None, *, workflows: WorkflowRegistry | None = None,
                 store: RunStore | None = None, queue: Queue | None = None, leases: KVStore | None = None,
                 bus: EventBus | None = None):
        self.cfg = cfg or AgentsConfig()
        self.workflows = workflows or WorkflowRegistry(pins=self.cfg.workflow_pins)
        self.store = store or RunStore()
        self.queue = queue or make_queue(self.cfg.queue)
        self.leases = leases or _lease_store(self.cfg.lease_store)
        self.bus = bus or EventBus()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._workers: list[asyncio.Task[Any]] = []
        self._stop = asyncio.Event()
        get_approval_store().subscribe(self.on_approval_decided)

    async def close(self) -> None:
        """Stop workers, wait for inline tasks and stop listening to approval decisions."""
        get_approval_store().unsubscribe(self.on_approval_decided)
        await self.stop_workers()
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=5)

    # ================================================================== entry points
    async def start(self, workflow: str, input: dict[str, Any], principal: Principal, *,
                    session_id: str | None = None, version: str | None = None, parent: RunState | None = None,
                    feature: str | None = None) -> RunState:
        wf = self.workflows.get(workflow, version)
        depth = parent.depth + 1 if parent else 0
        limits = self._limits(wf)
        if depth > limits["max_subagent_depth"]:
            raise ValidationFailed(f"sub-agent depth limit reached ({limits['max_subagent_depth']})")
        decision = await connect("policy").decide(PolicyInput(
            action="agent.run", principal={"sub": principal.sub, "tenant": principal.tenant, "app": principal.app,
                                           "roles": principal.roles, "groups": principal.groups},
            resource={"workflow": wf.name, "version": wf.version}, context={"parent_run_id": parent.id if parent else None,
                                                                            "feature": feature}))
        metrics.policy.labels(SERVICE, "agent.run", "allow" if decision.allow else "deny").inc()
        if not decision.allow:
            get_audit().for_principal(principal, "agents.run.start", action="agent.run", resource=wf.name,
                                      decision="denied", reasons=decision.reasons)
            raise PolicyViolation("; ".join(decision.reasons) or f"not allowed to run workflow {wf.name}")
        run = RunState(workflow=wf.name, version=wf.version, definition=wf.model_dump(mode="json"), input=input,
                       owner=principal.sub, tenant=principal.tenant, app=principal.app, feature=feature,
                       token=principal.token if self.cfg.store_user_token else None, session_id=session_id,
                       trace_id=current_trace_id() or new_trace_id(), parent_run_id=parent.id if parent else None,
                       depth=depth, current_step=wf.first, limits=limits)
        self._event(run, "run", {"status": "queued", "workflow": wf.name, "version": wf.version})
        self.store.save(run)
        get_audit().for_principal(principal, "agents.run.start", action="agent.run", resource=f"{wf.name}@{wf.version}",
                                  decision="allow", run_id=run.id, parent_run_id=run.parent_run_id,
                                  policy_version=decision.policy_version)
        await self._enqueue(run.id)
        return run

    async def resume(self, run_id: str, principal: Principal, *, input: dict[str, Any] | None = None) -> RunState:
        """The user picks a paused run back up (fresh token, optional reply)."""
        run = self.get(run_id, principal, owner_only=True)
        if run.status in TERMINAL:
            raise ValidationFailed(f"run {run_id} is {run.status}")
        if run.status == "waiting_approval":
            store = get_approval_store()
            open_ids = [a for a in (run.pending or {}).get("approval_ids", []) if store.get(a).status == "pending"]
            if open_ids:
                raise ValidationFailed(f"run {run_id} is still waiting for approval {', '.join(open_ids)}")
        if run.status == "waiting_child":
            raise ValidationFailed(f"run {run_id} is waiting for a sub-agent; it resumes on its own")
        if self.cfg.store_user_token:
            run.token = principal.token
        if input:
            run.vars["user_reply"] = input
        if run.status in ("waiting_approval", "waiting_user"):
            run.status, run.pending = "queued", None
        self._event(run, "run", {"status": "queued", "resumed_by": principal.sub})
        self.store.save(run)
        get_audit().for_principal(principal, "agents.run.resume", resource=run.workflow, run_id=run.id)
        await self._enqueue(run.id)
        return run

    async def cancel(self, run_id: str, principal: Principal) -> RunState:
        run = self.get(run_id, principal)
        if run.owner != principal.sub and not principal.has_role("admin"):
            raise PermissionDenied("only the run owner or an admin can cancel a run")
        if run.status in TERMINAL:
            return run
        for child_id in run.child_run_ids:
            child = self.store.find(child_id)
            if child and child.status not in TERMINAL:
                await self.cancel(child_id, principal)
        if principal.sub == run.owner:  # withdraw what the run asked approvers for: nobody should act on it now
            store = get_approval_store()
            for aid in (run.pending or {}).get("approval_ids", []):
                try:
                    store.cancel(aid, principal)
                except AIReadyError:
                    log.debug("approval %s could not be cancelled", aid)
        self._finish(run, "cancelled", stop_reason=f"cancelled by {principal.sub}")
        self.store.save(run)
        get_audit().for_principal(principal, "agents.run.cancel", resource=run.workflow, run_id=run.id,
                                  decision="cancelled")
        return run

    def get(self, run_id: str, principal: Principal, *, owner_only: bool = False) -> RunState:
        run = self.store.get(run_id)
        if run.tenant != principal.tenant:
            raise NotFound(f"run {run_id} not found")
        if run.owner != principal.sub and (owner_only or not principal.has_role("admin", "auditor")):
            raise NotFound(f"run {run_id} not found")  # do not confirm that someone else's run exists
        return run

    def list(self, principal: Principal, *, status: str | None = None, limit: int = 50) -> list[RunState]:
        everyone = principal.has_role("admin", "auditor")
        return self.store.list(owner=None if everyone else principal.sub, tenant=principal.tenant, status=status,
                               limit=limit)

    # ================================================================== events
    async def events(self, run_id: str, principal: Principal, *, after: int = 0,
                     follow: bool = True) -> AsyncIterator[RunEvent]:
        """History replay then live events, until the run stops moving on its own."""
        self.get(run_id, principal)
        q = self.bus.subscribe(run_id)
        backlog = [e for e in self.bus.backlog(run_id) if (e.after_seq or 0) >= after]  # same tick: no overlap
        last = after
        carry: RunEvent | None = None

        def replay(ev: RunEvent | None, upto: int) -> list[RunEvent]:
            """The persisted event followed by the live-only events published right after it."""
            out = [ev] if ev is not None else []
            out += [b for b in backlog if (b.after_seq or 0) == upto]
            return out

        try:
            for ev in replay(None, after):
                yield ev
            while True:
                batch: list[RunEvent] = [carry] if carry else []
                carry = None
                while not q.empty():
                    batch.append(q.get_nowait())
                if batch:  # live events first, in publication order (tokens interleave with steps)
                    for ev in batch:
                        if ev.seq is None:
                            yield ev
                            continue
                        if ev.seq <= last:
                            continue
                        if ev.seq > last + 1:  # fill a gap from history (subscribed late / other pod)
                            for h in self.store.events(run_id, after=last):
                                if (h.seq or 0) < ev.seq:
                                    for item in replay(h, h.seq or 0):
                                        yield item
                        last = ev.seq
                        yield ev
                    continue
                for h in self.store.events(run_id, after=last):  # progress made before we subscribed / elsewhere
                    last = h.seq or last
                    for item in replay(h, last):
                        yield item
                run = self.store.get(run_id)
                if not follow or run.status not in STREAMING:
                    return
                try:
                    carry = await asyncio.wait_for(q.get(), timeout=self.cfg.event_poll_seconds)
                except TimeoutError:
                    continue
        finally:
            self.bus.unsubscribe(run_id, q)

    def _event(self, run: RunState, type_: str, data: dict[str, Any], *, persist: bool = True) -> RunEvent:
        ev = RunEvent(run_id=run.id, type=type_, data=data,  # type: ignore[arg-type]
                      after_seq=None if persist else run.event_seq)
        if persist:
            self.store.append_event(run, ev)
        self.bus.publish(ev)
        return ev

    async def emit(self, run: RunState, type_: str, data: dict[str, Any], *, persist: bool = True) -> None:
        self._event(run, type_, data, persist=persist)

    # ================================================================== queue and workers
    async def _enqueue(self, run_id: str) -> None:
        await self.queue.put(self.cfg.topic, {"run_id": run_id})
        if self.cfg.workers.mode == "inline":
            task = asyncio.ensure_future(self._take_one())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _take_one(self) -> None:
        msg = await self.queue.get(self.cfg.topic)
        if msg is None:
            return
        try:
            await self.process(msg.payload["run_id"])
            await self.queue.ack(msg)
        except Exception as exc:  # infrastructure failure: let the queue retry / dead-letter
            log.exception("run %s crashed the worker", msg.payload.get("run_id"))
            await self.queue.nack(msg, str(exc))

    async def run_workers(self, count: int | None = None) -> None:
        """Start the worker pool (``workers.mode: pool``)."""
        self._stop.clear()
        for i in range(count or self.cfg.workers.count):
            self._workers.append(asyncio.create_task(self._worker(i), name=f"agent-worker-{i}"))

    async def _worker(self, n: int) -> None:
        while not self._stop.is_set():
            try:
                msg = await self.queue.get(self.cfg.topic, timeout=self.cfg.workers.poll_interval_seconds)
                if msg is None:
                    continue
                try:
                    await self.process(msg.payload["run_id"])
                    await self.queue.ack(msg)
                except Exception as exc:
                    log.exception("worker %d: run %s failed", n, msg.payload.get("run_id"))
                    await self.queue.nack(msg, str(exc))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker %d loop error", n)
                await asyncio.sleep(self.cfg.workers.poll_interval_seconds)

    async def stop_workers(self) -> None:
        self._stop.set()
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def wait_idle(self, timeout: float = 10.0) -> None:
        """Wait until inline tasks are done and the queue is empty (tests, graceful shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._tasks:
                await asyncio.wait(list(self._tasks), timeout=max(0.0, deadline - time.monotonic()))
                continue
            if await self.queue.depth(self.cfg.topic) == 0:
                await asyncio.sleep(0)
                if not self._tasks:
                    return
            elif self.cfg.workers.mode == "inline":
                await self._take_one()
            else:
                await asyncio.sleep(0.02)
        raise TimeoutError("agent runtime did not become idle")

    async def drain(self) -> None:
        """Process every queued run in this task (pilot CLI, tests)."""
        while await self.queue.depth(self.cfg.topic):
            await self._take_one()

    async def recover_stale(self) -> list[str]:
        """Re-queue runs left ``running`` by a worker that disappeared (lease expired)."""
        cutoff = time.time() - self.cfg.workers.lease_seconds
        out = []
        for run in self.store.list(status="running", limit=10000):
            if run.updated_at < cutoff and self.leases.get(f"agents:lease:{run.id}") is None:
                run.status = "queued"
                self.store.save(run)
                await self._enqueue(run.id)
                out.append(run.id)
        return out

    async def on_approval_decided(self, req: ApprovalRequest) -> None:
        run_id = req.run_id or self.store.run_for_approval(req.id)
        run = self.store.find(run_id) if run_id else None
        if run is None or run.status != "waiting_approval":
            return
        store = get_approval_store()
        if any(store.get(a).status == "pending" for a in (run.pending or {}).get("approval_ids", [])):
            return  # wait for every pending approval of this step
        run.status, run.pending = "queued", None
        self._event(run, "approval", {"approval_id": req.id, "status": req.status, "decided_by": req.decided_by,
                                      "comment": req.comment})
        self.store.save(run)
        await self._enqueue(run.id)

    # ================================================================== execution
    def _limits(self, wf: WorkflowDef) -> dict[str, Any]:
        base = self.cfg.limits.model_dump()
        for k, v in (wf.limits or {}).items():
            if k in base and v is not None:
                base[k] = min(base[k], v)
        return base

    def _principal(self, run: RunState) -> Principal | None:
        if not run.token:
            return None
        try:
            p = get_token_service().verify(run.token)
        except AuthenticationError:
            return None
        if p.sub != run.owner:
            return None
        p.actor = f"agent:{run.workflow}"
        return p

    async def process(self, run_id: str) -> RunState:
        """Run steps until the run finishes, waits or yields. Safe to call from any worker."""
        lease_key = f"agents:lease:{run_id}"
        if not self.leases.set_if_absent(lease_key, {"since": time.time()}, ttl=self.cfg.workers.lease_seconds):
            return self.store.get(run_id)  # another worker holds it
        try:
            run = self.store.get(run_id)
            if run.status not in ("queued", "running"):
                return run
            token = set_trace_id(run.trace_id)
            try:
                return await self._execute(run)
            finally:
                from aiready.core.telemetry import _trace_id

                _trace_id.reset(token)
        finally:
            self.leases.delete(lease_key)

    async def _execute(self, run: RunState) -> RunState:
        principal = self._principal(run)
        if principal is None:
            run.status, run.pending = "waiting_user", {"kind": "user", "reason": "sign-in expired"}
            self._event(run, "run", {"status": run.status, "reason": "The user's sign-in expired; the user must "
                                                                     "resume the run."})
            self.store.save(run)
            return run
        wf = WorkflowDef.model_validate(run.definition)
        run.status = "running"
        self._event(run, "run", {"status": "running"})
        self.store.save(run)
        slice_start = time.monotonic()
        steps_this_slice = 0
        limits = run.limits
        while True:
            fresh = self.store.find(run.id)  # cancellation from another pod
            if fresh and fresh.status == "cancelled":
                return fresh
            elapsed = run.active_seconds + (time.monotonic() - slice_start)
            reason = None
            if run.step_count >= limits["max_steps"]:
                reason = f"step limit reached ({limits['max_steps']} steps)"
            elif elapsed >= limits["max_seconds"]:
                reason = f"time limit reached ({limits['max_seconds']:g}s)"
            elif run.tokens_used >= limits["max_tokens"]:
                reason = f"token limit reached ({run.tokens_used}/{limits['max_tokens']} tokens)"
            if reason:
                run.active_seconds = elapsed
                return self._close(run, principal, "stopped", stop_reason=reason)
            if not run.current_step:
                return self._close(run, principal, "completed")

            try:
                step = wf.step(run.current_step)
            except NotFound as exc:
                return self._close(run, principal, "failed", error=exc.message)
            remaining = max(0.1, limits["max_seconds"] - elapsed)
            outcome, error = await self._run_step(run, wf, step, principal, remaining)
            run.active_seconds = run.active_seconds + (time.monotonic() - slice_start)
            slice_start = time.monotonic()
            if error is not None:
                if step.on_error:
                    run.vars["last_error"] = error
                    run.current_step = step.on_error
                    self.store.save(run)
                    continue
                return self._close(run, principal, "failed", error=f"step '{step.id}' failed: {error}")
            assert outcome is not None
            run.step_count += 1
            run.tokens_used += outcome.tokens
            if outcome.output is not None or outcome.wait is None:
                run.outputs[step.id] = outcome.output
                run.last_step = step.id
            if outcome.wait:
                run.status = {"approval": "waiting_approval", "child": "waiting_child", "user": "waiting_user"}[outcome.wait]
                run.pending = outcome.pending
                self._event(run, "approval" if outcome.wait == "approval" else "run",
                            {"status": run.status, "step": step.id, **(outcome.pending or {})})
                self.store.save(run)
                if outcome.wait == "approval":
                    get_audit().for_principal(principal, "agents.run.wait", resource=run.workflow, run_id=run.id,
                                              decision="waiting_approval", step=step.id,
                                              approval_ids=(outcome.pending or {}).get("approval_ids"))
                if outcome.wait == "child":
                    await self._check_child_done(run)
                return self.store.get(run.id)
            if outcome.end:
                run.result = outcome.output if isinstance(outcome.output, dict) else {"answer": outcome.output}
                return self._close(run, principal, outcome.status or "completed")
            run.current_step = outcome.next or step.next
            self.store.save(run)  # checkpoint: any worker can continue from here
            steps_this_slice += 1
            if self.cfg.workers.steps_per_slice and steps_this_slice >= self.cfg.workers.steps_per_slice:
                run.status = "queued"
                self.store.save(run)
                await self._enqueue(run.id)  # yield the worker; whoever picks it up resumes from the checkpoint
                return run

    async def _run_step(self, run: RunState, wf: WorkflowDef, step: Any, principal: Principal,
                        remaining: float) -> tuple[StepOutcome | None, str | None]:
        handler = step_handler(step)
        attempts = max(1, step.retries if step.retries is not None else self.cfg.retry.max_attempts)
        timeout = min(step.timeout_seconds or remaining, remaining)
        last_error = ""
        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            ctx = StepContext(runtime=self, run=run, step=step, workflow=wf, principal=principal)
            with span("agents.step", run_id=run.id, step=step.id, type=step.type_name, attempt=attempt) as sp, \
                    metrics.time_stage(SERVICE, f"step.{step.type_name}"):
                try:
                    outcome = await asyncio.wait_for(handler.run(ctx), timeout=timeout)
                except TimeoutError:
                    last_error, retryable = f"timed out after {timeout:.1f}s", False
                except StepFailed as exc:
                    last_error, retryable = exc.message, exc.retryable
                except NON_RETRYABLE as exc:
                    last_error, retryable = exc.message, False
                except (UpstreamError, AIReadyError) as exc:
                    last_error, retryable = exc.message, True
                except Exception as exc:  # a bug in a step must stop the run cleanly, not the worker
                    log.exception("step %s of run %s raised", step.id, run.id)
                    last_error, retryable = f"{type(exc).__name__}: {exc}", True
                else:
                    sp.set(decision=outcome.decision, tokens=outcome.tokens)
                    self._record(run, step, outcome, attempt, time.perf_counter() - t0)
                    return outcome, None
                sp.error(last_error)
            self._event(run, "step", {"step": step.id, "type": step.type_name, "status": "error", "attempt": attempt,
                                      "error": last_error, "retrying": retryable and attempt < attempts})
            if not retryable or attempt >= attempts:
                break
            run.attempts[step.id] = attempt
            await asyncio.sleep(min(self.cfg.retry.backoff_seconds * 2 ** (attempt - 1),
                                    self.cfg.retry.max_backoff_seconds))
        return None, last_error

    def _record(self, run: RunState, step: Any, outcome: StepOutcome, attempt: int, seconds: float) -> None:
        limit = self.cfg.history_chars
        self._event(run, "step", {
            "step": step.id, "type": step.type_name, "status": "waiting" if outcome.wait else "ok",
            "attempt": attempt, "decision": outcome.decision, "tokens": outcome.tokens,
            "duration_ms": round(seconds * 1000, 1), "input": clip(outcome.input, limit),
            "output": clip(outcome.output, limit), "next": outcome.next or step.next})
        get_audit().record("agents.step", actor=run.owner, tenant=run.tenant, app=run.app,
                           on_behalf_of=f"agent:{run.workflow}", resource=f"{run.workflow}.{step.id}",
                           decision=outcome.decision, run_id=run.id, step_type=step.type_name,
                           tokens=outcome.tokens)

    def _finish(self, run: RunState, status: str, *, error: str | None = None, stop_reason: str | None = None) -> None:
        run.status = status  # type: ignore[assignment]
        run.error, run.stop_reason = error, stop_reason
        run.finished_at = time.time()
        run.pending = None if status in TERMINAL else run.pending
        run.token = None  # nothing left to do as the user: drop their token
        self._event(run, "done", {"status": status, "result": run.result, "error": error, "stop_reason": stop_reason,
                                  "steps": run.step_count, "tokens": run.tokens_used})

    def _close(self, run: RunState, principal: Principal, status: str, *, error: str | None = None,
               stop_reason: str | None = None) -> RunState:
        self._finish(run, status, error=error, stop_reason=stop_reason)
        self.store.save(run)
        get_audit().for_principal(principal, "agents.run.finish", resource=f"{run.workflow}@{run.version}",
                                  decision=status, run_id=run.id, steps=run.step_count, tokens=run.tokens_used,
                                  error=error, stop_reason=stop_reason)
        if run.parent_run_id:
            task = asyncio.ensure_future(self._wake_parent(run))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return run

    async def _wake_parent(self, child: RunState) -> None:
        parent = self.store.find(child.parent_run_id or "")
        if parent and parent.status == "waiting_child" and (parent.pending or {}).get("run_id") == child.id:
            parent.status, parent.pending = "queued", None
            self.store.save(parent)
            await self._enqueue(parent.id)

    async def _check_child_done(self, parent: RunState) -> None:
        """A fast child may finish before its parent saved ``waiting_child``: wake the parent then."""
        child = self.store.find((parent.pending or {}).get("run_id", ""))
        if child and child.status in TERMINAL:
            await self._wake_parent(child)
