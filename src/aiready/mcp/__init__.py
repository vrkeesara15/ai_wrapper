"""MCP server and tool executor (wiring name ``tools``, port 8140).

Wraps the application's existing APIs (through the API facade) as typed tools, exposed to any MCP-capable
assistant over Streamable HTTP and to the agent runtime over a small REST surface. Every call carries the
signed-in user's identity; writes, payments and deletions wait for a person.
"""

from __future__ import annotations

from typing import Any

from aiready.core.wiring import provider


@provider("tools", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.mcp.service import ToolService

    return ToolService()


@provider("tools", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.mcp.client import ToolsClient

    return ToolsClient(cfg["url"], timeout=float(cfg.get("timeout", 60.0)))
