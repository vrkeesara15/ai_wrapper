"""Configuration of the assistant API and widget (section ``assistant`` in ``config/assistant.yaml``)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """One host application that embeds the widget."""

    title: str = "Assistant"
    greeting: str = "Hi! Ask me about this application."
    workflow: str | None = None  # default workflow for this app
    features: dict[str, str] = Field(default_factory=dict)  # feature (screen / use case) -> workflow
    collection: str | None = None  # knowledge collection passed to retrieval
    allowed_origins: list[str] = Field(default_factory=list)  # exact origins or globs (https://*.acme.com)
    theme: dict[str, Any] = Field(default_factory=dict)  # colours etc. handed to the widget


class IdentityBridgeConfig(BaseModel):
    """Where the app session is exchanged for an AI-Ready token (the identity bridge, port 8150)."""

    url: str | None = None
    exchange_path: str = "/v1/token"
    validator: str | None = None  # bridge session validator to use (default: the bridge tries each in order)
    forward_cookies: list[str] = Field(default_factory=list)  # cookie names to forward; empty = all
    forward_headers: list[str] = Field(default_factory=lambda: ["X-CSRF-Token"])
    timeout_seconds: float = 10.0


class AssistantConfig(BaseModel):
    default_workflow: str = "retrieval_chain"
    apps: dict[str, AppConfig] = Field(default_factory=dict)
    allowed_origins: list[str] = Field(default_factory=list)  # allowed for every app
    identity_bridge: IdentityBridgeConfig = Field(default_factory=IdentityBridgeConfig)
    static_dir: str | None = None  # directory holding widget/ and webmcp/ (default: the repo's web/)
    feedback_namespace: str = "feedback"
    max_message_chars: int = 8000
    max_history_turns: int = 20
    stream_ping_seconds: int = 15
