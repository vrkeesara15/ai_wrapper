"""The query service as an MCP tool: definition and call adapter for the MCP server.

The MCP server (``aiready.mcp``) can list ``QUERY_TOOL`` and route calls to ``call_query_tool``; the call runs
as the signed-in user through ``connect("query")``, so the same validation, policy and audit apply.
"""

from __future__ import annotations

from typing import Any

from aiready.core.errors import AIReadyError
from aiready.core.identity import Principal
from aiready.core.interfaces import QueryRequest, ToolDefinition, ToolResult
from aiready.core.types import DataClass

QUERY_TOOL = ToolDefinition(
    name="query_data",
    description=("Answer a question from the application's data (read replica). Prefer `metric` with an approved "
                 "metric name; otherwise pass `question` and read-only SQL is generated, validated and limited. "
                 "The SQL that ran is returned for audit."),
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The business question in plain language"},
            "metric": {"type": "string", "description": "Name of an approved metric (preferred)"},
            "params": {"type": "object", "description": "Parameters of the approved metric"},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
        },
        "anyOf": [{"required": ["question"]}, {"required": ["metric"]}],
        "additionalProperties": False,
    },
    effect="read",
    data_class=DataClass.CONFIDENTIAL,
    timeout_seconds=30.0,
    source="builtin",
)


async def call_query_tool(arguments: dict[str, Any], principal: Principal) -> ToolResult:
    from aiready.core.wiring import connect

    try:
        req = QueryRequest.model_validate(arguments)
        result = await connect("query").query(req, principal)
    except AIReadyError as exc:
        return ToolResult(ok=False, error=f"{exc.code}: {exc.message}", content=exc.details or None)
    except ValueError as exc:  # pydantic validation of the arguments
        return ToolResult(ok=False, error=f"invalid arguments: {exc}")
    return ToolResult(ok=True, content=result.model_dump())
