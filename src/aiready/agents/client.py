"""HTTP client for the agent runtime (``services.agents.url`` set). Same methods as ``AgentService``."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from aiready.agents.approvals import ApprovalRequest
from aiready.agents.models import RunEvent, RunView
from aiready.core.http import ServiceClient
from aiready.core.identity import Principal
from aiready.core.wiring import service_token


class AgentsClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None,
                 service: str = "assistant"):
        self._http = ServiceClient(base_url, service_token=service_token(service), timeout=timeout,
                                   transport=transport)

    async def start_run(self, workflow: str | None, input: dict[str, Any], principal: Principal, *,
                        session_id: str | None = None, version: str | None = None,
                        feature: str | None = None) -> RunView:
        data = await self._http.json("POST", "/v1/runs", principal=principal, retry=False,
                                     json={"workflow": workflow, "input": input, "session_id": session_id,
                                           "version": version, "feature": feature})
        return RunView.model_validate(data)

    async def get_run(self, run_id: str, principal: Principal) -> RunView:
        return RunView.model_validate(await self._http.json("GET", f"/v1/runs/{run_id}", principal=principal))

    async def list_runs(self, principal: Principal, *, status: str | None = None, limit: int = 50) -> list[RunView]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        data = await self._http.json("GET", "/v1/runs", principal=principal, params=params)
        return [RunView.model_validate(r) for r in data["runs"]]

    async def resume_run(self, run_id: str, principal: Principal, *, input: dict[str, Any] | None = None) -> RunView:
        data = await self._http.json("POST", f"/v1/runs/{run_id}/resume", principal=principal, retry=False,
                                     json={"input": input})
        return RunView.model_validate(data)

    async def cancel_run(self, run_id: str, principal: Principal) -> RunView:
        data = await self._http.json("POST", f"/v1/runs/{run_id}/cancel", principal=principal, retry=False)
        return RunView.model_validate(data)

    async def stream_events(self, run_id: str, principal: Principal, *, after: int = 0) -> AsyncIterator[RunEvent]:
        async with self._http.stream("GET", f"/v1/runs/{run_id}/events?after={after}", principal=principal) as r:
            if r.status_code >= 400:
                await r.aread()
                from aiready.core.errors import UpstreamError

                raise UpstreamError(f"event stream returned {r.status_code}", details={"body": r.text[:300]})
            data_lines: list[str] = []
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line == "" and data_lines:
                    yield RunEvent.model_validate(json.loads("\n".join(data_lines)))
                    data_lines = []

    async def run_history(self, run_id: str, principal: Principal) -> list[RunEvent]:
        data = await self._http.json("GET", f"/v1/runs/{run_id}/history", principal=principal)
        return [RunEvent.model_validate(e) for e in data["events"]]

    async def list_workflows(self) -> list[dict[str, Any]]:
        return (await self._http.json("GET", "/v1/workflows"))["workflows"]

    async def list_approvals(self, principal: Principal) -> list[ApprovalRequest]:
        data = await self._http.json("GET", "/v1/approvals", principal=principal)
        return [ApprovalRequest.model_validate(a) for a in data["approvals"]]

    async def list_my_approvals(self, principal: Principal, *, status: str | None = None) -> list[ApprovalRequest]:
        data = await self._http.json("GET", "/v1/approvals/mine", principal=principal,
                                     params={"status": status} if status else None)
        return [ApprovalRequest.model_validate(a) for a in data["approvals"]]

    async def get_approval(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        return ApprovalRequest.model_validate(
            await self._http.json("GET", f"/v1/approvals/{approval_id}", principal=principal))

    async def decide_approval(self, approval_id: str, principal: Principal, *, approve: bool,
                              comment: str | None = None) -> ApprovalRequest:
        action = "approve" if approve else "reject"
        data = await self._http.json("POST", f"/v1/approvals/{approval_id}/{action}", principal=principal,
                                     retry=False, json={"comment": comment})
        return ApprovalRequest.model_validate(data)

    async def aclose(self) -> None:
        await self._http.aclose()
