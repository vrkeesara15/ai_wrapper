"""Configuration of the memory service (section ``memory`` in ``config/memory.yaml``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    summarize_over_tokens: int = 1500  # when the stored turns exceed this, older turns are summarised
    keep_recent_turns: int = 6  # newest turns always kept verbatim
    summary_route: str = "small"
    summary_max_words: int = 150
    summary_timeout_seconds: float = 5.0
    prompt: str = "memory.summarize"
    prompt_version: str | None = None
    idle_ttl_hours: float = 24.0  # a session untouched this long is dropped
    max_turn_chars: int = 8000


class LongTermConfig(BaseModel):
    # Long-term memory is stored ONLY with the user's consent and always with an expiry date.
    require_consent: bool = True
    default_ttl_days: int = 90
    max_ttl_days: int = 365
    max_items_per_user: int = 500  # oldest-expiring items are evicted beyond this (audited)
    screen_writes: bool = True  # screen facts through guardrails (blocks memory poisoning / secrets)
    share_across_apps: bool = False  # False: an app only recalls what was remembered in that app
    min_recall_score: float = 0.0
    allowed_kinds: list[str] = Field(default_factory=lambda: ["fact", "preference", "summary"])


class EmbeddingConfig(BaseModel):
    type: Literal["hash", "gateway"] = "hash"  # hash: offline, deterministic; gateway: the embedding model
    model: str = "embedding"
    dim: int = 384


class MemoryConfig(BaseModel):
    prompts_source: Literal["auto", "registry", "files"] = "auto"
    namespace_sessions: str = "memory_sessions"
    namespace_items: str = "memory_items"
    session: SessionConfig = Field(default_factory=SessionConfig)
    long_term: LongTermConfig = Field(default_factory=LongTermConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
