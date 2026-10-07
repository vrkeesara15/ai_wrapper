"""Configuration of the search service (top-level section ``search:``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FusionConfig(BaseModel):
    method: Literal["rrf", "weighted"] = "rrf"
    k: int = Field(60, ge=1)  # RRF constant
    weights: list[float] = Field(default_factory=lambda: [1.0, 1.0])  # [dense, keyword]


class SearchConfig(BaseModel):
    base_collection: str = "knowledge"  # collections are named <base>__<app>__<tenant>
    vector_store: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    keyword_index: dict[str, Any] = Field(default_factory=lambda: {"type": "bm25"})
    dense_enabled: bool = True
    keyword_enabled: bool = True
    top_k: int = Field(50, ge=1)  # candidates returned (sent on to the reranker)
    max_top_k: int = Field(200, ge=1)
    candidates_per_index: int = Field(100, ge=1)  # pulled from each index before fusion
    min_dense_score: float | None = None  # drop dense hits under this cosine similarity
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    embedding_model: str = "embedding"  # logical model the gateway uses for query (and missing chunk) vectors
    embed_missing_vectors: bool = True  # upsert embeds chunks that arrive without a vector (bulk priority)
    enforce_tenant_collections: bool = True  # a user may only search collections of their own tenant
    policy_check: bool = True  # data.read on the collection before every search
    snapshot_dir: str = "data/snapshots/search"
    return_vectors: bool = False
    max_query_chars: int = Field(4000, ge=1)
