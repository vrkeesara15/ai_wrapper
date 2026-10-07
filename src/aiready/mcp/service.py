"""In-process tool service (wiring name ``tools``): registry + executor, rebuilt on configuration change."""

from __future__ import annotations

from typing import Any

import httpx

from aiready.agents.approvals import ApprovalStore
from aiready.core.config import get_config
from aiready.mcp.config import ToolsConfig
from aiready.mcp.executor import ToolExecutorService
from aiready.mcp.registry import ToolRegistry


class ToolService(ToolExecutorService):
    """``core.interfaces.ToolExecutor`` backed by the configured registry.

    ``transport`` lets tests (or an embedding host) route facade calls through an in-memory httpx transport;
    production leaves it ``None`` and real HTTP is used.
    """

    def __init__(self, cfg: ToolsConfig | None = None, *, transport: httpx.AsyncBaseTransport | None = None,
                 approvals: ApprovalStore | None = None, watch_config: bool = True):
        cfg = cfg or get_config().section("tools", ToolsConfig)
        super().__init__(ToolRegistry(cfg, transport=transport), cfg, approvals=approvals)
        if watch_config:
            get_config().subscribe(self._on_config)

    @classmethod
    def from_config(cls) -> "ToolService":
        """The in-process executor built from the ``tools`` configuration section."""
        return cls()

    def _on_config(self, tree: dict[str, Any]) -> None:
        """New routes/overrides take effect on the next call; the registry reloads lazily."""
        cfg = ToolsConfig.model_validate(tree.get("tools") or {})
        self.cfg = cfg
        self.registry.invalidate(cfg)

    async def reload(self) -> list[str]:
        await self.registry.load()
        return self.registry.errors
