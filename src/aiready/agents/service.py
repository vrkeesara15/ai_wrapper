"""In-process agent service (wiring name ``agents``): the API surface over ``AgentRuntime`` and the approval inbox.

The HTTP client (``client.py``) has the same method names and return types, so the assistant API and other
callers work the same whether the runtime is in-process (pilot) or a separate pod (production).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from aiready.agents.approvals import ApprovalRequest, get_approval_store
from aiready.agents.config import AgentsConfig
from aiready.agents.models import RunEvent, RunView
from aiready.agents.runtime import AgentRuntime
from aiready.agents.workflows import WorkflowRegistry
from aiready.core.config import get_config
from aiready.core.identity import Principal


class AgentService:
    def __init__(self, runtime: AgentRuntime | None = None, *, watch_config: bool = True):
        if runtime is None:
            cfg = get_config().section("agents", AgentsConfig)
            runtime = AgentRuntime(cfg)
        self.runtime = runtime
        if watch_config:
            get_config().subscribe(self._on_config)

    def _on_config(self, tree: dict[str, Any]) -> None:
        """New workflow versions and limits apply to new runs; running runs keep their snapshot."""
        cfg = AgentsConfig.model_validate(tree.get("agents") or {})
        self.runtime.cfg = cfg
        self.runtime.workflows = WorkflowRegistry(tree.get("workflows") or {}, pins=cfg.workflow_pins)

    # ------------------------------------------------------------------ runs
    async def start_run(self, workflow: str | None, input: dict[str, Any], principal: Principal, *,
                        session_id: str | None = None, version: str | None = None,
                        feature: str | None = None) -> RunView:
        run = await self.runtime.start(workflow or self.runtime.cfg.default_workflow, input, principal,
                                       session_id=session_id, version=version, feature=feature)
        return RunView.of(run)

    async def get_run(self, run_id: str, principal: Principal) -> RunView:
        return RunView.of(self.runtime.get(run_id, principal))

    async def list_runs(self, principal: Principal, *, status: str | None = None, limit: int = 50) -> list[RunView]:
        return [RunView.of(r) for r in self.runtime.list(principal, status=status, limit=limit)]

    async def resume_run(self, run_id: str, principal: Principal, *, input: dict[str, Any] | None = None) -> RunView:
        return RunView.of(await self.runtime.resume(run_id, principal, input=input))

    async def cancel_run(self, run_id: str, principal: Principal) -> RunView:
        return RunView.of(await self.runtime.cancel(run_id, principal))

    async def stream_events(self, run_id: str, principal: Principal, *, after: int = 0) -> AsyncIterator[RunEvent]:
        async for ev in self.runtime.events(run_id, principal, after=after):
            yield ev

    async def run_history(self, run_id: str, principal: Principal) -> list[RunEvent]:
        return [ev async for ev in self.runtime.events(run_id, principal, follow=False)]

    async def list_workflows(self) -> list[dict[str, Any]]:
        return self.runtime.workflows.list()

    # ------------------------------------------------------------------ approval inbox
    async def list_approvals(self, principal: Principal) -> list[ApprovalRequest]:
        return get_approval_store().list_pending(principal)

    async def list_my_approvals(self, principal: Principal, *, status: str | None = None) -> list[ApprovalRequest]:
        return get_approval_store().list_requested(principal, status)

    async def get_approval(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        return get_approval_store().get(approval_id, principal)

    async def decide_approval(self, approval_id: str, principal: Principal, *, approve: bool,
                              comment: str | None = None) -> ApprovalRequest:
        req = get_approval_store().decide(approval_id, principal, approve=approve, comment=comment)
        # the store notifies listeners asynchronously; resume here too so the caller sees the run move at once
        await self.runtime.on_approval_decided(req)
        return req

    async def aclose(self) -> None:
        await self.runtime.close()
