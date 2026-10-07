"""Configuration models for the ``ingestion`` section (see ``config/ingestion.yaml``).

Every default runs locally with no infrastructure: an in-memory work queue, a local landing directory,
the shared DocStore (SQLite) for registry / lineage / runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.types import DataClass


class VerifyQuestion(BaseModel):
    """A test question run after each load. Passes when an expected document is in the top ``top_k`` hits."""

    question: str
    expect_doc_id: str | None = None  # exact document id
    expect_doc_contains: str | None = None  # or a substring of the document id / title / uri
    top_k: int = 5
    as_groups: list[str] | None = None  # access groups to search with (default: the expected doc's tags)


class SourceConfig(BaseModel):
    name: str
    connector: dict[str, Any]  # plugin spec: {type: filesystem, root: ...} or {class: pkg.mod:Cls, ...}
    collection: str = "default"
    enabled: bool = True
    schedule: str | None = None  # "every 15m", "interval:1h", "@daily", or a 5-field cron expression
    chunker: dict[str, Any] | None = None  # overrides ingestion.chunking.default for this source
    parsers: dict[str, dict[str, Any] | str] = Field(default_factory=dict)  # mime or .ext -> parser spec
    access_tags: list[str] = Field(default_factory=list)  # added to every document of the source
    require_access_tags: bool = True  # fail closed: a document without tags is not indexed
    data_class: DataClass = DataClass.INTERNAL  # floor sensitivity for the source
    detect_deletes: bool = True  # infer deletions from a full listing
    max_delete_fraction: float = 0.5  # refuse to delete more than this share of a source in one run (outage guard)
    landing: bool = True  # keep raw copies for replay
    verify: list[VerifyQuestion] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    default: dict[str, Any] = Field(default_factory=lambda: {"type": "heading", "max_tokens": 350, "overlap_tokens": 40})
    by_mime: dict[str, dict[str, Any]] = Field(default_factory=lambda: {
        "text/csv": {"type": "table_rows", "rows_per_chunk": 20},
    })


class CleaningConfig(BaseModel):
    pii_mode: Literal["mask", "tag", "off"] = "tag"  # mask: replace in text; tag: mark the chunk as personal data
    injection_action: Literal["flag", "drop"] = "flag"  # drop: the document is quarantined, not indexed
    injection_patterns: list[str] = Field(default_factory=list)  # extra regexes for indirect prompt injection
    boilerplate_patterns: list[str] = Field(default_factory=list)  # extra line regexes to strip
    repeated_line_ratio: float = 0.6  # a line repeated on more than this share of pages is a header/footer
    min_chars: int = 1  # documents shorter than this after cleaning are skipped


class EmbeddingConfig(BaseModel):
    model: str = "embedding"
    batch_size: int = 64
    priority: Literal["live", "bulk"] = "bulk"
    fingerprint: str | None = None  # embedding model version; changing it means a full re-embed
    include_title: bool = True  # prefix title and section to the embedded text (not to the stored text)


class QueueConfig(BaseModel):
    spec: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    topic: str = "ingestion.documents"
    max_attempts: int = 3
    concurrency: int = 4
    poll_timeout: float = 1.0


class LandingConfig(BaseModel):
    type: str = "local"
    path: str = "data/landing"
    bucket: str | None = None
    prefix: str = "landing/"
    keep_versions: int = 3


class AlertConfig(BaseModel):
    webhook_url: str | None = None  # POSTed a JSON body when verification fails or a document is dead-lettered
    log_level: str = "WARNING"


class IngestionConfig(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    landing: LandingConfig = Field(default_factory=LandingConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    admin_roles: list[str] = Field(default_factory=lambda: ["ingestion-admin"])
    default_collection: str = "default"
    scheduler_tick_seconds: float = 30.0

    def source(self, name: str) -> SourceConfig | None:
        return next((s for s in self.sources if s.name == name), None)


def load_config() -> IngestionConfig:
    from aiready.core.config import get_config

    return get_config().section("ingestion", IngestionConfig)
