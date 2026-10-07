"""Configuration of the reranking service (top-level section ``rerank:``)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SelectionConfig(BaseModel):
    """How many candidates go in, how many come out. 50 in / 5 out are starting defaults, tuned per client
    against the evaluation set."""

    max_candidates: int = Field(50, ge=1)
    top_n: int = Field(5, ge=1)
    threshold: float | None = None  # drop passages scoring under this (scale depends on the model)


class RerankConfig(BaseModel):
    models: dict[str, dict[str, Any]] = Field(default_factory=lambda: {"reranker": {"type": "lexical"}})
    default_model: str = "reranker"
    defaults: SelectionConfig = Field(default_factory=SelectionConfig)
    apps: dict[str, SelectionConfig] = Field(default_factory=dict)  # per-application overrides
    timeout_seconds: float = Field(2.0, gt=0)  # slower than this -> fall back to search order
    # unavailable reranker -> search order (logged + metric) instead of an error
    fallback_on_error: bool = True
    policy_check: bool = True

    def selection(self, app: str | None) -> SelectionConfig:
        return self.apps.get(app or "", self.defaults)
