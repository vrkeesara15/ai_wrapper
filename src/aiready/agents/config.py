"""Configuration of the agent runtime (section ``agents`` in ``config/agents.yaml``).

Approval routing lives in its own section ``approvals`` (see ``aiready.agents.approvals.ApprovalsConfig``)
because the MCP server reads it too.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Limits(BaseModel):
    """Caps for one run. Workflow-level limits may only lower these (the global values are ceilings)."""

    max_steps: int = 40
    max_seconds: float = 300.0  # active processing time; time spent waiting for a person does not count
    max_tokens: int = 60000
    max_tool_calls: int = 25
    max_subagent_depth: int = 2


class RetryConfig(BaseModel):
    max_attempts: int = 3  # per step, for transient errors (upstream / timeouts)
    backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0


class WorkersConfig(BaseModel):
    # inline: a run is processed by a task in the process that started it (pilot, tests, single pod).
    # pool:   `count` workers pull runs from the queue; any worker resumes any run from its checkpoint.
    mode: Literal["inline", "pool"] = "inline"
    count: int = 2
    poll_interval_seconds: float = 0.5
    lease_seconds: float = 120.0  # a run whose worker vanished is picked up again after this
    steps_per_slice: int = 0  # >0: re-queue a run after N steps so long runs share workers fairly


class AgentProfile(BaseModel):
    """A named agent: prompt, model route and tool allow-list used by `agent_loop` workflows."""

    description: str = ""
    system_prompt: str | None = None
    model: str | None = None
    tools: list[str] = Field(default_factory=lambda: ["*"])  # glob patterns over tool names


class AgentsConfig(BaseModel):
    limits: Limits = Field(default_factory=Limits)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    queue: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    lease_store: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    topic: str = "agent_runs"
    default_workflow: str = "retrieval_chain"
    default_model: str = "auto"
    system_prompt: str = (
        "You are an assistant working inside the user's application. You act with the user's permissions. "
        "Use the tools to look things up before answering. Actions that change data wait for a person to "
        "approve them; say so plainly when that happens. Cite sources with their [n] markers."
    )
    profiles: dict[str, AgentProfile] = Field(default_factory=dict)
    workflow_pins: dict[str, str] = Field(default_factory=dict)  # workflow -> version to use for new runs
    store_user_token: bool = True  # keep the user's short-lived token with the run so workers act as the user
    event_poll_seconds: float = 0.25
    history_chars: int = 4000  # max characters of a step input/output kept in run history
    tool_result_chars: int = 12000  # max characters of a tool result shown to the model
